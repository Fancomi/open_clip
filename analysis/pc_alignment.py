"""PC alignment analysis: track how PCA principal directions change across training.

Metrics
-------
per_pc_cos   : |cos θ_i| for each PC i between two feature sets (sign-invariant)
               1.0 = same direction, 0.0 = orthogonal
subspace_cos : cosines of principal angles between the k-dim PCA subspaces
               (SVD of V_a.T @ V_b); ordered from largest to smallest
grassmann    : Grassmann distance = √Σ(arccos σ_i)²  ∈ [0, π√k/2]
               0.0 = identical subspaces, larger = more rotation

Comparison bases
----------------
vs_first  : alignment of each step against step 0 → accumulated drift
vs_final  : alignment of each step against the last checkpoint → convergence
vs_prev   : alignment between consecutive steps → local rotation speed
"""
import os
import logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA


# ── Core metric ──────────────────────────────────────────────────────────────

def compute_pc_alignment(feats_a: np.ndarray, feats_b: np.ndarray,
                         n_pcs: int = 16) -> dict:
    """Compute principal-angle metrics between PCA subspaces of two feature sets.

    Parameters
    ----------
    feats_a, feats_b : (N, D) float arrays (raw, un-normalised features)
    n_pcs            : number of principal components to compare

    Returns
    -------
    dict with keys:
        per_pc_cos   : (n_pcs,) |cos θ_i| per PC (corresponding pair)
        subspace_cos : (n_pcs,) principal-angle cosines (sorted descending)
        grassmann    : scalar Grassmann distance
        var_ratio_a  : (n_pcs,) explained variance ratio for set A
        var_ratio_b  : (n_pcs,) explained variance ratio for set B
    """
    k = min(n_pcs, feats_a.shape[1], feats_a.shape[0] - 1,
            feats_b.shape[0] - 1)
    pca_a = PCA(n_components=k).fit(feats_a)
    pca_b = PCA(n_components=k).fit(feats_b)

    Va = pca_a.components_  # (k, D) — rows are PCs
    Vb = pca_b.components_  # (k, D)

    # Per-PC cosine (sign-invariant): |dot(va_i, vb_i)|
    per_pc = np.abs((Va * Vb).sum(axis=1))  # (k,)

    # Subspace principal angles via SVD of the cross-Gram matrix
    M     = Va @ Vb.T          # (k, k)
    sigma = np.linalg.svd(M, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    grassmann = float(np.sqrt((np.arccos(sigma) ** 2).sum()))

    return dict(
        per_pc_cos   = per_pc,
        subspace_cos = sigma,
        grassmann    = grassmann,
        var_ratio_a  = pca_a.explained_variance_ratio_,
        var_ratio_b  = pca_b.explained_variance_ratio_,
    )


# ── Pipeline ─────────────────────────────────────────────────────────────────

def run_pc_alignment(args):
    """Load probe checkpoints and compute PC alignment across training.

    Expects args.probe_dir with step_XXXXXX.npz or epoch_XX.npz files.
    Saves plots to <log_root>/probe/plots/pc_alignment_*.png  (sibling of checkpoints/)
    """
    import re

    probe_dir = args.probe_dir
    n_pcs     = getattr(args, 'n_pcs', 16)
    # Place plots at <log_root>/probe/plots  (sibling of checkpoints/)
    plots_dir = os.path.join(probe_dir, '..', '..', 'probe', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # ── Collect and sort checkpoint files ────────────────────────────────────
    files = sorted(f for f in os.listdir(probe_dir) if f.endswith('.npz'))
    if not files:
        logging.error(f'No .npz files in {probe_dir}')
        return

    step_pat  = re.compile(r'step_(\d+)\.npz')
    epoch_pat = re.compile(r'epoch_(\d+)\.npz')
    matched = []
    for f in files:
        m = step_pat.match(f) or epoch_pat.match(f)
        if m:
            matched.append((int(m.group(1)), f))
    if not matched:
        logging.error('No step_*.npz / epoch_*.npz found')
        return
    matched.sort()
    step_ids = [s for s, _ in matched]
    id_label = 'Step' if step_pat.match(matched[0][1]) else 'Epoch'

    logging.info(f'[pc_align] {len(matched)} checkpoints, n_pcs={n_pcs}')

    # ── Load image features from each checkpoint ─────────────────────────────
    all_feats = []
    for sid, fname in matched:
        data = np.load(os.path.join(probe_dir, fname))
        key  = 'img' if 'img' in data else list(data.keys())[0]
        all_feats.append(data[key].astype(np.float32))
        logging.info(f'  {fname}: {all_feats[-1].shape}')

    # ── Compute alignment: vs_first, vs_final, vs_prev ───────────────────────
    n = len(all_feats)
    k = min(n_pcs, all_feats[0].shape[1])

    def _align_series(ref_feats, query_feats_list):
        grass, per_pc_mat, subsp_mat = [], [], []
        for f in query_feats_list:
            r = compute_pc_alignment(ref_feats, f, n_pcs=k)
            grass.append(r['grassmann'])
            per_pc_mat.append(r['per_pc_cos'])
            subsp_mat.append(r['subspace_cos'])
        return (np.array(grass),
                np.array(per_pc_mat),   # (T, k)
                np.array(subsp_mat))    # (T, k)

    grass_first, perpc_first, subsp_first = _align_series(all_feats[0],  all_feats)
    grass_final, perpc_final, subsp_final = _align_series(all_feats[-1], all_feats)

    # Consecutive (vs_prev): step i vs step i-1
    grass_prev, perpc_prev = [], []
    for i in range(1, n):
        r = compute_pc_alignment(all_feats[i - 1], all_feats[i], n_pcs=k)
        grass_prev.append(r['grassmann'])
        perpc_prev.append(r['per_pc_cos'])
    grass_prev  = np.array(grass_prev)
    perpc_prev  = np.array(perpc_prev)   # (n-1, k)

    # ── Plots ─────────────────────────────────────────────────────────────────
    _plot_grassmann(step_ids, grass_first, grass_final, grass_prev,
                    id_label, plots_dir)
    _plot_perpc_heatmap(step_ids, perpc_first, perpc_final,
                        id_label, plots_dir)
    _plot_perpc_lines(step_ids, perpc_first, grass_prev, id_label, plots_dir,
                      perpc_prev=perpc_prev)


# ── Visualisation helpers ─────────────────────────────────────────────────────

def _plot_grassmann(step_ids, grass_first, grass_final, grass_prev,
                    id_label, plots_dir):
    """Three Grassmann distance curves on one figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(step_ids, grass_first, marker='o', ms=4, lw=1.5, color='steelblue')
    ax.set_xlabel(id_label); ax.set_ylabel('Grassmann distance')
    ax.set_title(f'Subspace drift from {id_label} 0\n(0=identical, larger=more rotated)')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(step_ids, grass_final, marker='o', ms=4, lw=1.5, color='coral')
    ax.set_xlabel(id_label); ax.set_ylabel('Grassmann distance')
    ax.set_title(f'Convergence to final {id_label}\n(→0 means fully converged)')
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(step_ids[1:], grass_prev, marker='o', ms=4, lw=1.5, color='seagreen')
    ax.set_xlabel(id_label); ax.set_ylabel('Grassmann distance')
    ax.set_title(f'Consecutive-step rotation speed\n(larger = bigger jump)')
    ax.grid(True, alpha=0.3)

    fig.suptitle('PCA Subspace Stability (Grassmann Distance)', fontsize=11, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'pc_alignment_grassmann.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[pc_align] {save_path}')


def _plot_perpc_heatmap(step_ids, perpc_first, perpc_final, id_label, plots_dir):
    """Heatmap: rows = PCs, columns = steps, color = |cos θ| vs reference."""
    n_pcs = perpc_first.shape[1]
    fig, axes = plt.subplots(1, 2, figsize=(max(12, len(step_ids) * 0.6), n_pcs * 0.45 + 2))

    for ax, data, ref_lbl, cmap in [
        (axes[0], perpc_first.T, f'{id_label} 0',     'Blues'),
        (axes[1], perpc_final.T, f'final {id_label}', 'Oranges'),
    ]:
        im = ax.imshow(data, aspect='auto', vmin=0, vmax=1,
                       cmap=cmap, origin='lower')
        ax.set_xlabel(id_label)
        ax.set_ylabel('PC index')
        ax.set_xticks(range(len(step_ids)))
        ax.set_xticklabels(step_ids, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n_pcs))
        ax.set_yticklabels([f'PC{i+1}' for i in range(n_pcs)], fontsize=7)
        ax.set_title(f'Per-PC |cos θ| vs {ref_lbl}\n(1=same direction, 0=orthogonal)',
                     fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.03)

    fig.suptitle('Per-PC Direction Alignment', fontsize=11, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'pc_alignment_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[pc_align] {save_path}')


def _plot_perpc_lines(step_ids, perpc_first, grass_prev, id_label, plots_dir,
                      perpc_prev=None, top_k=8):
    """Line plot: top-k PCs alignment vs step_0 + Grassmann speed subplot."""
    k = min(top_k, perpc_first.shape[1])
    colors = cm.tab10(np.linspace(0, 0.9, k))

    nrows = 2 if perpc_prev is not None else 1
    fig, axes = plt.subplots(nrows, 1,
                             figsize=(max(10, len(step_ids) * 0.5), 4 * nrows))
    if nrows == 1:
        axes = [axes]

    ax = axes[0]
    for i, c in zip(range(k), colors):
        ax.plot(step_ids, perpc_first[:, i], marker='o', ms=3, lw=1.2,
                color=c, label=f'PC{i+1}')
    ax.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(id_label)
    ax.set_ylabel('|cos θ| vs step 0')
    ax.set_title(f'Top-{k} PC alignment with initial directions\n'
                 f'(1.0 = direction unchanged from {id_label} 0)', fontsize=9)
    ax.legend(ncol=4, fontsize=7, loc='lower left')
    ax.grid(True, alpha=0.3)

    if perpc_prev is not None:
        ax2 = axes[1]
        for i, c in zip(range(k), colors):
            ax2.plot(step_ids[1:], perpc_prev[:, i], marker='o', ms=3, lw=1.2,
                     color=c, label=f'PC{i+1}', alpha=0.8)
        ax2.axhline(1.0, color='gray', lw=0.8, ls='--', alpha=0.5)
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_xlabel(id_label)
        ax2.set_ylabel('|cos θ| (consecutive)')
        ax2.set_title(f'Consecutive-step PC alignment\n'
                      f'(1.0 = no rotation from previous {id_label.lower()})', fontsize=9)
        ax2.legend(ncol=4, fontsize=7, loc='lower left')
        ax2.grid(True, alpha=0.3)

    fig.suptitle('PCA Direction Stability', fontsize=11, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(plots_dir, 'pc_alignment_lines.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[pc_align] {save_path}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    ap = argparse.ArgumentParser(description='PC alignment analysis across training steps')
    ap.add_argument('--probe-dir', required=True,
                    help='Directory containing step_*.npz or epoch_*.npz files')
    ap.add_argument('--n-pcs', type=int, default=16,
                    help='Number of principal components to compare (default: 16)')
    args = ap.parse_args()
    run_pc_alignment(args)
