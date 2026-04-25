"""Visualization utilities for feature-space analysis."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from sklearn.decomposition import PCA

_FPS_MARKERS = ['*', 'D', '^', 's', 'P']
_FPS_COLORS  = ['red', 'lime', 'cyan', 'orange', 'magenta']

# ── PCA helpers ───────────────────────────────────────────────────────────────

def _fit_pca(feats_list, n):
    """Shared PCA if same dim; independent per-model otherwise."""
    dims = [f.shape[1] for f in feats_list]
    if len(set(dims)) == 1:
        pca = PCA(n_components=n).fit(np.concatenate(feats_list))
        return [pca] * len(feats_list), pca.explained_variance_ratio_
    return [PCA(n_components=n).fit(f) for f in feats_list], None


# ── Main plots ────────────────────────────────────────────────────────────────

def plot_scatter(feats_dict, title, save_path, n_pca=4, fps_indices=None):
    """Multi-axis PCA scatter.  Same-dim → shared PCA; mixed-dim → independent rows.
    fps_indices: highlight same samples (by index) across all model rows."""
    labels = list(feats_dict.keys())
    feats  = list(feats_dict.values())
    colors = cm.tab10(np.linspace(0, 0.9, len(labels)))
    pairs  = list(combinations(range(n_pca), 2))
    pcas, shared_var = _fit_pca(feats, n_pca)
    projs  = [pca.transform(f) for pca, f in zip(pcas, feats)]

    def _fps_on(ax, proj, pi, pj):
        if fps_indices is None:
            return
        for fi, (idx, mk, fc) in enumerate(zip(fps_indices, _FPS_MARKERS, _FPS_COLORS)):
            ax.scatter(proj[idx, pi], proj[idx, pj], marker=mk, s=120, color=fc,
                       edgecolors='black', linewidths=0.5, zorder=5,
                       label=f'FPS-{fi}' if (pi == 0 and pj == 1) else '')

    if shared_var is not None:
        # ── shared PCA: single row ─────────────────────────────────────────────
        ncols = len(pairs) + 1
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
        for col, (pi, pj) in enumerate(pairs):
            ax = axes[col]
            for label, proj, c in zip(labels, projs, colors):
                ax.scatter(proj[:, pi], proj[:, pj], s=3, alpha=0.3, color=c,
                           label=label if col == 0 else '', rasterized=True)
            _fps_on(ax, projs[0], pi, pj)
            ax.set_xlabel(f'PC{pi+1}'); ax.set_ylabel(f'PC{pj+1}')
            ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
            if col == 0:
                ax.legend(markerscale=4, fontsize=8)
        ax = axes[-1]
        ax.bar(range(1, n_pca + 1), shared_var * 100, color='steelblue')
        ax.set_xlabel('Component'); ax.set_ylabel('Variance explained (%)')
        ax.set_title('Explained variance', fontsize=9)
        fig.suptitle(title, fontsize=12, y=1.01)
    else:
        # ── independent PCA: one row per model, extra FPS column ──────────────
        nrows = len(labels)
        has_fps = fps_indices is not None
        ncols   = len(pairs) + 1 + (1 if has_fps else 0)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        for row, (label, pca, proj, feat, c) in enumerate(
                zip(labels, pcas, projs, feats, colors)):
            var = pca.explained_variance_ratio_
            for col, (pi, pj) in enumerate(pairs):
                ax = axes[row, col]
                ax.scatter(proj[:, pi], proj[:, pj], s=3, alpha=0.3,
                           color=c, rasterized=True)
                _fps_on(ax, proj, pi, pj)
                ax.set_xlabel(f'PC{pi+1}')
                ax.set_ylabel(f'{label}  (dim={feat.shape[1]})\nPC{pj+1}'
                              if col == 0 else f'PC{pj+1}')
                ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
            ax = axes[row, len(pairs)]
            ax.bar(range(1, n_pca + 1), var * 100, color=c)
            ax.set_xlabel('Component'); ax.set_ylabel('Var. explained (%)')
            ax.set_title(f'{label}  explained var.', fontsize=9)
            if has_fps:
                # Dedicated FPS panel: PC1 vs PC2, prominent markers
                ax = axes[row, -1]
                ax.scatter(proj[:, 0], proj[:, 1], s=2, alpha=0.15,
                           color=c, rasterized=True)
                for fi, (idx, mk, fc) in enumerate(
                        zip(fps_indices, _FPS_MARKERS, _FPS_COLORS)):
                    ax.scatter(proj[idx, 0], proj[idx, 1], marker=mk, s=200,
                               color=fc, edgecolors='black', linewidths=0.8,
                               zorder=6, label=f'FPS-{fi}')
                ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
                ax.set_title(f'{label}  FPS anchors', fontsize=9)
                if row == 0:
                    ax.legend(markerscale=1, fontsize=7,
                              title='Same sample\nacross models')
        fig.suptitle(title + '  [independent PCA]', fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {save_path}')


def plot_overlap(feats_a, feats_b, label_a, label_b, model_name, save_path):
    """Overlay two distributions in shared PCA space (all samples, rasterized)."""
    combined = np.concatenate([feats_a, feats_b])
    pca = PCA(n_components=2).fit(combined)
    pa, pb = pca.transform(feats_a), pca.transform(feats_b)
    d = np.linalg.norm(pa.mean(0) - pb.mean(0))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax in axes:
        ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color='steelblue',
                   label=label_a, rasterized=True)
        ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.3, color='coral',
                   label=label_b, rasterized=True)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(markerscale=4, fontsize=9)
    axes[0].set_title(f'{model_name}: {label_a} vs {label_b}', fontsize=10)
    axes[1].set_title(f'centroid dist (PC1-2) = {d:.3f}', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {save_path}')


def plot_anisotropy(metrics_dict: dict, save_path: str):
    """Bar charts + eigenvalue spectrum for all anisotropy metrics."""
    models  = list(metrics_dict.keys())
    colors  = cm.tab10(np.linspace(0, 0.9, len(models)))
    scalars = [
        ('effective_rank',      'Effective Rank'),
        ('participation_ratio', 'Participation Ratio'),
        ('stable_rank',         'Stable Rank (1/λ_max)'),
        ('numerical_rank',      'Numerical Rank (1% thr)'),
        ('avg_cos_sim',         'Avg Cosine Sim ↓'),
        ('std_cos_sim',         'Std Cosine Sim (multi-modal ↑)'),
        ('pct_var_top4',        'Var% top-4'),
        ('pct_var_top10',       'Var% top-10'),
        ('pct_var_top50',       'Var% top-50'),
        ('pct_var_top100',      'Var% top-100'),
    ]
    ncols = len(scalars) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 4.5))
    for ax, (key, lbl) in zip(axes[:-1], scalars):
        vals = [metrics_dict[m][key] for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=30, ha='right', fontsize=7)
        ax.set_title(lbl, fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.1f}' if v > 10 else f'{v:.3f}',
                    ha='center', va='bottom', fontsize=6)
    # Eigenvalue spectrum (log scale, top-100 PCs)
    ax = axes[-1]
    for m, c in zip(models, colors):
        eigs = metrics_dict[m]['eigenvalues'][:100]
        ax.plot(np.arange(1, len(eigs) + 1), eigs * 100,
                color=c, label=m, lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('PC index'); ax.set_ylabel('Variance % (log)')
    ax.set_title('Eigenvalue spectrum top-100 (log)', fontsize=8)
    ax.legend(fontsize=7)
    fig.suptitle('Feature Anisotropy & Rank Metrics', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {save_path}')


def plot_evolution(epoch_feats, epoch_ids, save_dir, n_traj=100, seed=42):
    """Epoch-by-epoch PCA scatter + trajectory plot."""
    import os
    pca   = PCA(n_components=2).fit(epoch_feats[-1])
    projs = [pca.transform(f) for f in epoch_feats]
    n     = len(epoch_ids)
    all_p = np.concatenate(projs)
    pad   = 0.05
    x0, x1 = all_p[:, 0].min(), all_p[:, 0].max()
    y0, y1 = all_p[:, 1].min(), all_p[:, 1].max()
    xp = (x1 - x0) * pad; yp = (y1 - y0) * pad
    xlim = (x0 - xp, x1 + xp); ylim = (y0 - yp, y1 + yp)

    ncols = min(5, n); nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (eid, proj) in enumerate(zip(epoch_ids, projs)):
        axes[i].scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.4,
                        c=np.arange(len(proj)), cmap='viridis', rasterized=True)
        axes[i].set_title(f'Epoch {eid}', fontsize=9)
        axes[i].set_xlim(xlim); axes[i].set_ylim(ylim); axes[i].axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Image Feature Distribution per Epoch  [PCA fitted on final epoch]', fontsize=10)
    plt.tight_layout()
    p = os.path.join(save_dir, 'epoch_evolution.png')
    plt.savefig(p, dpi=150); plt.close(); print(f'[viz] {p}')

    rng    = np.random.default_rng(seed)
    idx    = rng.choice(len(epoch_feats[0]), min(n_traj, len(epoch_feats[0])), replace=False)
    colors = cm.tab20(np.linspace(0, 1, len(idx)))
    alphas = np.linspace(0.10, 1.00, n); lws = np.linspace(0.3, 1.8, n)
    fig, ax = plt.subplots(figsize=(8, 7))
    for si, color in zip(idx, colors):
        pts = np.array([pr[si] for pr in projs])
        for t in range(len(pts) - 1):
            ax.plot(pts[t:t+2, 0], pts[t:t+2, 1], '-', color=color,
                    alpha=float(alphas[t + 1]), lw=float(lws[t + 1]))
        ax.scatter(pts[0, 0],  pts[0, 1],  color=color, s=12, marker='o',
                   alpha=float(alphas[0]), zorder=3)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=40, marker='*',
                   alpha=1.0, zorder=4)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(f'Sample Trajectories  N={len(idx)}\no=start  *=end  light→dark = early→late epoch', fontsize=9)
    ax.set_xlabel('PC1 (final epoch)'); ax.set_ylabel('PC2 (final epoch)')
    plt.tight_layout()
    p = os.path.join(save_dir, 'trajectory.png')
    plt.savefig(p, dpi=150); plt.close(); print(f'[viz] {p}')
