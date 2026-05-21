"""Feature-space metrics: isotropy, rank, multimodality.

Anisotropy metric summary (all computed on L2-normalized CLS features):
┌──────────────────────┬──────────────────────────────────────┬───────────────┐
│ Metric               │ Definition                           │ ↑/↓ isotropic │
├──────────────────────┼──────────────────────────────────────┼───────────────┤
│ effective_rank       │ exp(H(λ/Σλ))  ∈ [1, D]              │ ↑             │
│ participation_ratio  │ 1 / (D · Σλ²)  ∈ (0, 1]             │ ↑             │
│ stable_rank          │ 1 / λ_max  (= Σλ/λ_max, normalized)  │ ↑             │
│ numerical_rank       │ #{s_i ≥ 1% · s_max}                 │ ↑             │
│ avg_cos_sim          │ mean pairwise cosine (subsample 2k)  │ ↓             │
│ std_cos_sim          │ std  pairwise cosine → multi-modal   │ ↑ multi-modal │
│ pct_var_top_p{p}     │ cumulative var% at top p% of dims    │ ↓             │
└──────────────────────┴──────────────────────────────────────┴───────────────┘

top-k% metrics use fractions of D so models with different dims are comparable:
  p=0.5  → top 0.5% of D  (≈4 for D=768, ≈5 for D=1024, ≈12 for D=2304)
  p=5    → top 5%  of D
  p=25   → top 25% of D
  p=50   → top 50% of D
"""
import numpy as np


def fps_sample(feats: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Farthest Point Sampling in embedding space. Returns k indices."""
    rng = np.random.default_rng(seed)
    n = len(feats)
    chosen = [int(rng.integers(n))]
    dists = np.full(n, np.inf)
    for _ in range(k - 1):
        d = ((feats - feats[chosen[-1]]) ** 2).sum(1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))
    return np.array(chosen)


def random_batches(N: int, batch_size: int = 256, n_batches: int = 20, seed: int = 0):
    """返回 n_batches 组随机采样索引 (不放回, 各组独立)."""
    rng = np.random.default_rng(seed)
    return [rng.choice(N, min(batch_size, N), replace=False) for _ in range(n_batches)]


def fps_batches(feats: np.ndarray, batch_size: int = 256, n_batches: int = 20, seed: int = 0):
    """FPS 顺序分区: 反复从剩余点中选 batch_size 个最远点."""
    N = len(feats)
    remaining = np.arange(N)
    batches = []
    for i in range(n_batches):
        if len(remaining) < batch_size:
            batches.append(remaining.copy())
            break
        # FPS on remaining subset
        sub_feats = feats[remaining]
        sub_idx = fps_sample(sub_feats, k=batch_size, seed=seed + i)
        batches.append(remaining[sub_idx])
        remaining = np.delete(remaining, sub_idx)
    return batches


def compute_knn_density(feats: np.ndarray, K: int = 50) -> np.ndarray:
    """kNN 密度: 1 / mean_knn_distance. 返回 (N,) 数组."""
    from sklearn.neighbors import BallTree
    tree = BallTree(feats)
    dists, _ = tree.query(feats, k=K + 1)  # 含自身
    mean_dist = dists[:, 1:].mean(axis=1)  # 排除自身距离0
    return 1.0 / (mean_dist + 1e-10)


def compute_knn_curvature(feats: np.ndarray, K: int = 50) -> np.ndarray:
    """kNN 曲率: 1 - lambda_max / sum_lambda (局部PCA各向异性).
    值高=局部弯曲, 值低=局部平坦. 返回 (N,) 数组."""
    from sklearn.neighbors import BallTree
    tree = BallTree(feats)
    _, indices = tree.query(feats, k=K + 1)
    N = len(feats)
    curvature = np.empty(N)
    for i in range(N):
        nbrs = feats[indices[i, 1:]]  # (K, D)
        nbrs_c = nbrs - nbrs.mean(axis=0)
        # 仅需最大奇异值和总方差
        s = np.linalg.svd(nbrs_c, compute_uv=False)
        lam = s ** 2
        curvature[i] = 1.0 - lam[0] / (lam.sum() + 1e-10)
    return curvature


def compute_anisotropy(feats: np.ndarray, max_components: int = 256) -> dict:
    """Compute full-dimensional isotropy + rank + multimodality metrics.

    top-k% metrics are parameterised by fraction of D so that models with
    different feature dims are directly comparable.

    max_components: cap on SVD rank. 256 is sufficient for all metrics.
    """
    from sklearn.utils.extmath import randomized_svd

    D = feats.shape[1]
    f = feats - feats.mean(0, keepdims=True)
    k = min(D, f.shape[0] - 1, max_components)
    _, s, _ = randomized_svd(f, n_components=k, random_state=0)

    lam = s ** 2
    lam = lam / lam.sum()                            # normalized eigenvalues

    eff_rank = float(np.exp(-(lam * np.log(lam + 1e-12)).sum()))
    pr = float(1.0 / (k * (lam ** 2).sum()))
    stable_rank = float(1.0 / lam[0])
    num_rank = int((s >= s[0] * 0.01).sum())

    # Pairwise cosine on 2k-subsample
    rng = np.random.default_rng(42)
    idx = rng.choice(len(feats), min(2000, len(feats)), replace=False)
    sub = feats[idx]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
    tri = (sub @ sub.T)[np.triu_indices(len(sub), k=1)]
    avg_cos = float(tri.mean())
    std_cos = float(tri.std())

    # top-p% of dims (fraction-based, cross-model comparable)
    cum = np.cumsum(lam)
    pct = {}
    for p in [0.5, 5, 25, 50]:
        n_pcs = max(1, int(round(D * p / 100)))
        n_pcs = min(n_pcs, len(cum))
        pct[f'pct_var_top_p{p}'] = float(cum[n_pcs - 1] * 100)
    # legacy absolute top-k (kept for backward compat with run_epochs logging)
    for t in [4, 10, 50, 100]:
        pct[f'pct_var_top{t}'] = float(cum[min(t, len(cum)) - 1] * 100)

    return dict(effective_rank=eff_rank, participation_ratio=pr,
                stable_rank=stable_rank, numerical_rank=num_rank,
                avg_cos_sim=avg_cos, std_cos_sim=std_cos,
                dim=D, n_components=k, eigenvalues=lam, **pct)
