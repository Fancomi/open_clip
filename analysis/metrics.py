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
    dists  = np.full(n, np.inf)
    for _ in range(k - 1):
        d = ((feats - feats[chosen[-1]]) ** 2).sum(1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))
    return np.array(chosen)


def compute_anisotropy(feats: np.ndarray, max_components: int = 256) -> dict:
    """Compute full-dimensional isotropy + rank + multimodality metrics.

    top-k% metrics are parameterised by fraction of D so that models with
    different feature dims are directly comparable.

    max_components: cap on SVD rank. 256 is sufficient for all metrics.
    """
    from sklearn.utils.extmath import randomized_svd

    D  = feats.shape[1]
    f  = feats - feats.mean(0, keepdims=True)
    k  = min(D, f.shape[0] - 1, max_components)
    _, s, _ = randomized_svd(f, n_components=k, random_state=0)

    lam = s ** 2
    lam = lam / lam.sum()                            # normalized eigenvalues

    eff_rank     = float(np.exp(-(lam * np.log(lam + 1e-12)).sum()))
    pr           = float(1.0 / (k * (lam ** 2).sum()))
    stable_rank  = float(1.0 / lam[0])
    num_rank     = int((s >= s[0] * 0.01).sum())

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
