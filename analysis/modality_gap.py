"""
Modality Gap Analysis — post-processing experiments on saved probe features.

Loads image + text features from a probe .npz file and evaluates four
post-processing variants:

  raw        : original l2-normalized features (no change)
  centered   : subtract each modality's own mean, then re-normalize
  gap_remove : project out the modality-direction vector, re-normalize
  whitened   : full PCA whitening on concatenated [image, text] pool

For each variant, computes:
  1. PCA-1 modality separation  (linear classifier accuracy + mean projection)
  2. Modality classifier accuracy (sklearn LogisticRegression)
  3. i2t / t2i Recall@1,5,10 on the provided probe pairs

Usage
-----
python analysis/modality_gap.py \
    --probe  logs/<run>/probe/step_001740.npz \
    [--split  proj_features]   # which key to use: features | proj_features (default: proj_features)
    [--out    analysis/research/modality_gap_out.json]

The probe .npz must have keys:
  features / proj_features   — [N, D_img] l2-norm image features
  txt_features               — [N, D_txt] l2-norm text features
"""

import argparse
import json
import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ─────────────────────────── helpers ────────────────────────────────────────

def to_torch(arr):
    return torch.from_numpy(arr.astype(np.float32))


def recall_at_k(img_feat, txt_feat, ks=(1, 5, 10)):
    """Compute i2t and t2i Recall@k (exact, CPU).

    img_feat: [N, D]  l2-normalized
    txt_feat: [N, D]  l2-normalized, aligned (img[i] ↔ txt[i])
    """
    N = img_feat.shape[0]
    # Full similarity matrix [N, N]
    S = img_feat @ txt_feat.T  # [N, N]

    results = {}
    for k in ks:
        # i2t: each image retrieves text
        ranks_i2t = (S >= S.diagonal().unsqueeze(1)).sum(dim=1).float()  # rank of the GT (1-indexed)
        results[f"i2t_R@{k}"] = (ranks_i2t <= k).float().mean().item() * 100

        # t2i: each text retrieves image
        ranks_t2i = (S.T >= S.diagonal().unsqueeze(1)).sum(dim=1).float()
        results[f"t2i_R@{k}"] = (ranks_t2i <= k).float().mean().item() * 100

    return results


def modality_classifier_accuracy(img_np, txt_np):
    """Fit a LogisticRegression to classify image vs text. Returns accuracy [0-100]."""
    X = np.concatenate([img_np, txt_np], axis=0).astype(np.float32)
    y = np.array([0] * len(img_np) + [1] * len(txt_np))
    # L2-normalize (may already be normalized, but centering can change norm)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-8)
    scaler = StandardScaler(with_std=False)  # only center, keep unit sphere structure
    X_s = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs', n_jobs=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_s, y)
    return clf.score(X_s, y) * 100


def pc1_modality_stats(img_np, txt_np):
    """PCA on concatenated features; return PC1 projection stats per modality."""
    X = np.concatenate([img_np, txt_np], axis=0).astype(np.float32)
    pca = PCA(n_components=1)
    proj = pca.fit_transform(X).ravel()  # [2N]
    img_proj = proj[:len(img_np)]
    txt_proj = proj[len(img_np):]
    # Ensure image is on the positive side (convention)
    if img_proj.mean() < txt_proj.mean():
        img_proj, txt_proj = -img_proj, -txt_proj
    return {
        "img_pc1_mean":  float(img_proj.mean()),
        "img_pc1_std":   float(img_proj.std()),
        "txt_pc1_mean":  float(txt_proj.mean()),
        "txt_pc1_std":   float(txt_proj.std()),
        "pc1_gap":       float(img_proj.mean() - txt_proj.mean()),
        "pc1_var_ratio": float(pca.explained_variance_ratio_[0]),
    }


# ─────────────────────────── post-processing transforms ─────────────────────

def variant_raw(img, txt):
    """No change. Both must already be l2-normalized."""
    return img, txt


def variant_centered(img, txt):
    """Subtract each modality's own mean, re-normalize."""
    img_c = F.normalize(img - img.mean(dim=0, keepdim=True), dim=-1)
    txt_c = F.normalize(txt - txt.mean(dim=0, keepdim=True), dim=-1)
    return img_c, txt_c


def variant_gap_remove(img, txt):
    """Project out the modality-direction unit vector, re-normalize.

    modality direction u = normalize(mean_img - mean_txt)
    x_new = normalize(x - (x·u) * u)
    """
    u = F.normalize((img.mean(0) - txt.mean(0)).unsqueeze(0), dim=-1)  # [1, D]
    img_p = F.normalize(img - (img @ u.T) * u, dim=-1)
    txt_p = F.normalize(txt - (txt @ u.T) * u, dim=-1)
    return img_p, txt_p


def variant_whitened(img, txt):
    """PCA-whitening on the joint pool, then re-normalize per modality."""
    joint = torch.cat([img, txt], dim=0).numpy().astype(np.float64)
    N_img = img.shape[0]
    mu = joint.mean(axis=0, keepdims=True)
    joint_c = joint - mu
    cov = (joint_c.T @ joint_c) / (len(joint) - 1)
    vals, vecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(vals)[::-1]
    vals, vecs = vals[idx], vecs[:, idx]
    # Whitening: W = V * diag(1/sqrt(λ + ε))
    eps = 1e-5
    W = vecs / np.sqrt(vals + eps)  # [D, D]
    img_w = torch.from_numpy(((img.numpy().astype(np.float64) - mu) @ W).astype(np.float32))
    txt_w = torch.from_numpy(((txt.numpy().astype(np.float64) - mu) @ W).astype(np.float32))
    img_w = F.normalize(img_w, dim=-1)
    txt_w = F.normalize(txt_w, dim=-1)
    return img_w, txt_w


VARIANTS = {
    "raw":        variant_raw,
    "centered":   variant_centered,
    "gap_remove": variant_gap_remove,
    "whitened":   variant_whitened,
}


# ─────────────────────────── main ────────────────────────────────────────────

def run(probe_path, split="proj_features", out_path=None):
    data = np.load(probe_path, allow_pickle=True)

    # ── load image features ──────────────────────────────────────────────────
    if split in data:
        img_np = data[split].astype(np.float32)
    elif "features" in data:
        img_np = data["features"].astype(np.float32)
        print(f"[warn] key '{split}' not found, falling back to 'features'")
    else:
        raise KeyError(f"Neither '{split}' nor 'features' found in {probe_path}")

    if "txt_features" not in data:
        raise KeyError(f"'txt_features' key not found in {probe_path}. "
                       "Re-run training with --probe-data to generate text features.")
    txt_np = data["txt_features"].astype(np.float32)

    assert img_np.shape[0] == txt_np.shape[0], "image/text feature count mismatch"
    N = img_np.shape[0]
    print(f"[info] Loaded {N} pairs  img_dim={img_np.shape[1]}  txt_dim={txt_np.shape[1]}")

    img_t = to_torch(img_np)
    txt_t = to_torch(txt_np)

    # ── re-normalize (probe may already be l2-norm, but re-do for safety) ───
    img_t = F.normalize(img_t, dim=-1)
    txt_t = F.normalize(txt_t, dim=-1)

    results = {}

    for name, fn in VARIANTS.items():
        print(f"\n── variant: {name} ──")
        img_v, txt_v = fn(img_t.clone(), txt_t.clone())

        img_np_v = img_v.numpy()
        txt_np_v = txt_v.numpy()

        # PCA stats
        pca_stats = pc1_modality_stats(img_np_v, txt_np_v)
        print(f"  PC1 gap={pca_stats['pc1_gap']:.4f}  var_ratio={pca_stats['pc1_var_ratio']:.4f}")
        print(f"  img_pc1={pca_stats['img_pc1_mean']:.4f}±{pca_stats['img_pc1_std']:.4f}  "
              f"txt_pc1={pca_stats['txt_pc1_mean']:.4f}±{pca_stats['txt_pc1_std']:.4f}")

        # Modality classifier
        clf_acc = modality_classifier_accuracy(img_np_v, txt_np_v)
        print(f"  modality_clf_acc={clf_acc:.1f}%")

        # Retrieval
        recall = recall_at_k(img_v, txt_v)
        for k_name, v in recall.items():
            print(f"  {k_name}={v:.2f}%")

        results[name] = {**pca_stats, "modality_clf_acc": clf_acc, **recall}

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n══ Summary ══")
    header = f"{'Variant':<14}  {'PC1_gap':>8}  {'clf%':>6}  {'i2t@1':>7}  {'t2i@1':>7}  {'i2t@5':>7}  {'t2i@5':>7}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<14}  {r['pc1_gap']:8.4f}  {r['modality_clf_acc']:6.1f}  "
              f"{r['i2t_R@1']:7.2f}  {r['t2i_R@1']:7.2f}  "
              f"{r['i2t_R@5']:7.2f}  {r['t2i_R@5']:7.2f}")

    # ── save ──────────────────────────────────────────────────────────────────
    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        meta = {"probe": probe_path, "split": split, "N": N}
        with open(out_path, "w") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)
        print(f"\n[saved] {out_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Modality Gap post-processing analysis")
    parser.add_argument("--probe",  required=True, help="Path to probe .npz file")
    parser.add_argument("--split",  default="proj_features",
                        choices=["features", "proj_features"],
                        help="Which image feature key to use (default: proj_features)")
    parser.add_argument("--out",    default=None, help="Output JSON path")
    args = parser.parse_args()
    run(args.probe, split=args.split, out_path=args.out)


if __name__ == "__main__":
    main()
