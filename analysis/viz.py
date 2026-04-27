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


def _tsne_proj(feats, subsample=2000, seed=42):
    """Run T-SNE on a subsample; return (N_sub, 2) projection + original indices."""
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(feats), min(subsample, len(feats)), replace=False)
    sub = feats[idx]
    emb = TSNE(n_components=2, perplexity=30, n_jobs=-1,
               random_state=seed).fit_transform(sub.astype(np.float32))
    return emb, idx


# ── Main plots ────────────────────────────────────────────────────────────────

def plot_scatter(feats_dict, title, save_path, n_pca=4, fps_indices=None,
                 with_tsne=True):
    """Multi-axis PCA scatter + optional T-SNE column.

    Same-dim → shared PCA row; mixed-dim → independent rows.
    fps_indices: highlight same samples (by index) across all model rows.
    with_tsne: append a T-SNE panel (2k subsample) per row.
    """
    labels = list(feats_dict.keys())
    feats  = list(feats_dict.values())
    colors = cm.tab10(np.linspace(0, 0.9, len(labels)))
    pairs  = list(combinations(range(n_pca), 2))
    pcas, shared_var = _fit_pca(feats, n_pca)
    projs  = [pca.transform(f) for pca, f in zip(pcas, feats)]

    # Pre-compute T-SNE (slow — done once per call)
    if with_tsne:
        tsne_projs = []
        for f in feats:
            emb, _ = _tsne_proj(f)
            tsne_projs.append(emb)

    def _fps_on(ax, proj, pi, pj):
        if fps_indices is None:
            return
        for fi, (idx, mk, fc) in enumerate(zip(fps_indices, _FPS_MARKERS, _FPS_COLORS)):
            ax.scatter(proj[idx, pi], proj[idx, pj], marker=mk, s=120, color=fc,
                       edgecolors='black', linewidths=0.5, zorder=5,
                       label=f'FPS-{fi}' if (pi == 0 and pj == 1) else '')

    tsne_col = 1 if with_tsne else 0

    if shared_var is not None:
        # ── shared PCA: single row ─────────────────────────────────────────────
        ncols = len(pairs) + 1 + tsne_col
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
        ax = axes[len(pairs)]
        ax.bar(range(1, n_pca + 1), shared_var * 100, color='steelblue')
        ax.set_xlabel('Component'); ax.set_ylabel('Variance explained (%)')
        ax.set_title('Explained variance', fontsize=9)
        if with_tsne:
            ax = axes[-1]
            for label, emb, c in zip(labels, tsne_projs, colors):
                ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.3, color=c,
                           label=label, rasterized=True)
            ax.set_title('T-SNE (2k subsample)', fontsize=9)
            ax.legend(markerscale=4, fontsize=8)
            ax.axis('off')
        fig.suptitle(title, fontsize=12, y=1.01)
    else:
        # ── independent PCA: one row per model ────────────────────────────────
        nrows = len(labels)
        has_fps = fps_indices is not None
        ncols   = len(pairs) + 1 + (1 if has_fps else 0) + tsne_col
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        for row, (label, pca, proj, feat, c) in enumerate(
                zip(labels, pcas, projs, feats, colors)):
            var = pca.explained_variance_ratio_
            dim = feat.shape[1]
            for col, (pi, pj) in enumerate(pairs):
                ax = axes[row, col]
                ax.scatter(proj[:, pi], proj[:, pj], s=3, alpha=0.3,
                           color=c, rasterized=True)
                _fps_on(ax, proj, pi, pj)
                ax.set_xlabel(f'PC{pi+1}')
                ax.set_ylabel(f'{label}  (D={dim})\nPC{pj+1}'
                              if col == 0 else f'PC{pj+1}')
                ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
            ax = axes[row, len(pairs)]
            ax.bar(range(1, n_pca + 1), var * 100, color=c)
            ax.set_xlabel('Component'); ax.set_ylabel('Var. explained (%)')
            ax.set_title(f'{label} (D={dim})  explained var.', fontsize=9)
            if has_fps:
                ax = axes[row, len(pairs) + 1]
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
            if with_tsne:
                ax = axes[row, -1]
                emb = tsne_projs[row]
                ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.3, color=c,
                           rasterized=True)
                ax.set_title(f'{label}  T-SNE', fontsize=9)
                ax.axis('off')
        fig.suptitle(title + '  [independent PCA]', fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {save_path}')


def plot_overlap(pa, pb, label_a, label_b, model_name, save_path,
                 a_on_top: bool = True, centroid_dist: float = None):
    """Single-panel scatter: one dataset drawn on top of the other.

    pa, pb: already-projected 2-D arrays (N×2).
    a_on_top=True  → draw B first (background), then A on top.
    a_on_top=False → draw A first (background), then B on top.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    if a_on_top:
        ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.3, color='coral',
                   label=label_b, rasterized=True)
        ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color='steelblue',
                   label=label_a, rasterized=True)
        on_top_lbl = label_a
    else:
        ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color='steelblue',
                   label=label_a, rasterized=True)
        ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.3, color='coral',
                   label=label_b, rasterized=True)
        on_top_lbl = label_b
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(markerscale=4, fontsize=9)
    dist_str = f'  centroid dist={centroid_dist:.3f}' if centroid_dist is not None else ''
    ax.set_title(f'{model_name}: {on_top_lbl} on top{dist_str}', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {save_path}')


def plot_aniso_evolution(step_ids, aniso_list, save_path, id_label='Step'):
    """Line plot of anisotropy metrics across training steps/epochs."""
    keys = [
        ('effective_rank',      'Effective Rank'),
        ('stable_rank',         'Stable Rank'),
        ('avg_cos_sim',         'Avg Cosine Sim ↓'),
        ('std_cos_sim',         'Std Cosine Sim'),
        ('pct_var_top_p0.5',    'Var% top-0.5% dims'),
        ('pct_var_top_p5',      'Var% top-5% dims'),
    ]
    ncols = 3; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.reshape(-1)
    xs = step_ids
    for ax, (key, lbl) in zip(axes, keys):
        ys = [m[key] for m in aniso_list]
        ax.plot(xs, ys, marker='o', ms=4, lw=1.5, color='steelblue')
        ax.set_xlabel(id_label); ax.set_title(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.annotate(f'{ys[0]:.2f}',  (xs[0],  ys[0]),  textcoords='offset points',
                    xytext=(4, 4),  fontsize=7, color='gray')
        ax.annotate(f'{ys[-1]:.2f}', (xs[-1], ys[-1]), textcoords='offset points',
                    xytext=(-20, 4), fontsize=7, color='steelblue')
    # Annotate dim (same for all checkpoints from same model)
    dim = aniso_list[0].get('dim')
    dim_str = f'  D={dim}' if dim else ''
    fig.suptitle(f'Anisotropy Evolution across {id_label}s{dim_str}', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {save_path}')


def plot_anisotropy(metrics_dict: dict, save_path: str):
    """Bar charts + eigenvalue spectrum. top-k uses fraction of D for fair comparison."""
    models  = list(metrics_dict.keys())
    colors  = cm.tab10(np.linspace(0, 0.9, len(models)))
    # Labels include D for each model
    model_labels = [f'{m}\n(D={metrics_dict[m].get("dim","?")})'
                    for m in models]
    scalars = [
        ('effective_rank',      'Effective Rank'),
        ('participation_ratio', 'Participation Ratio'),
        ('stable_rank',         'Stable Rank (1/λ_max)'),
        ('numerical_rank',      'Numerical Rank (1% thr)'),
        ('avg_cos_sim',         'Avg Cosine Sim ↓'),
        ('std_cos_sim',         'Std Cosine Sim (multi-modal ↑)'),
        ('pct_var_top_p0.5',    'Var% top-0.5% of D'),
        ('pct_var_top_p5',      'Var% top-5% of D'),
        ('pct_var_top_p25',     'Var% top-25% of D'),
        ('pct_var_top_p50',     'Var% top-50% of D'),
    ]
    ncols = len(scalars) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 5))
    for ax, (key, lbl) in zip(axes[:-1], scalars):
        vals = [metrics_dict[m].get(key, float('nan')) for m in models]
        bars = ax.bar(range(len(models)), vals, color=colors)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(model_labels, rotation=30, ha='right', fontsize=7)
        ax.set_title(lbl, fontsize=8)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{v:.1f}' if v > 10 else f'{v:.3f}',
                        ha='center', va='bottom', fontsize=6)
    # Eigenvalue spectrum (log scale, top-100 PCs, x-axis normalized by D)
    ax = axes[-1]
    for m, c in zip(models, colors):
        eigs = metrics_dict[m]['eigenvalues'][:100]
        D    = metrics_dict[m].get('dim', len(eigs))
        xs   = np.arange(1, len(eigs) + 1) / D * 100   # % of total dims
        ax.plot(xs, eigs * 100, color=c,
                label=f'{m} (D={D})', lw=1.2)
    ax.set_yscale('log')
    ax.set_xlabel('PC index (% of D)'); ax.set_ylabel('Variance % (log)')
    ax.set_title('Eigenvalue spectrum top-100 (log,\nx-axis = % of total dims)', fontsize=8)
    ax.legend(fontsize=7)
    fig.suptitle('Feature Anisotropy & Rank Metrics', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {save_path}')


def plot_evolution(step_feats, step_ids, save_dir, n_traj=100, seed=42,
                   id_label='Step', fps=4):
    """PCA scatter GIF + trajectory GIF + static trajectory plot.

    Outputs
    -------
    {id_label}_evolution.gif   : scatter snapshot per checkpoint
    trajectory.gif             : progressive trajectory animation
    trajectory.png             : static full-path overlay
    """
    import os
    from matplotlib.animation import FuncAnimation, PillowWriter

    pca   = PCA(n_components=2).fit(step_feats[-1])
    projs = [pca.transform(f) for f in step_feats]
    n     = len(step_ids)
    all_p = np.concatenate(projs)
    pad   = 0.05
    x0, x1 = all_p[:, 0].min(), all_p[:, 0].max()
    y0, y1 = all_p[:, 1].min(), all_p[:, 1].max()
    xp = (x1 - x0) * pad; yp = (y1 - y0) * pad
    xlim = (x0 - xp, x1 + xp); ylim = (y0 - yp, y1 + yp)

    colors_n = cm.viridis(np.linspace(0, 1, n))

    # ── GIF 1: scatter snapshot per checkpoint ─────────────────────────────
    fig_gif, ax_gif = plt.subplots(figsize=(5, 5))
    scat = ax_gif.scatter([], [], s=3, alpha=0.4, rasterized=True)
    ax_gif.set_xlim(xlim); ax_gif.set_ylim(ylim); ax_gif.axis('off')
    title_obj = ax_gif.set_title('', fontsize=10)

    def _init_scat():
        scat.set_offsets(np.empty((0, 2)))
        return (scat, title_obj)

    def _update_scat(frame):
        scat.set_offsets(projs[frame][:, :2])
        scat.set_color(colors_n[frame])
        title_obj.set_text(f'{id_label} {step_ids[frame]}  '
                           f'[PCA on final {id_label.lower()}]')
        return (scat, title_obj)

    anim = FuncAnimation(fig_gif, _update_scat, init_func=_init_scat,
                         frames=n, interval=1000 // fps, blit=True)
    gif_path = os.path.join(save_dir, f'{id_label.lower()}_evolution.gif')
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig_gif)
    print(f'[viz] {gif_path}')

    # ── Sample selection (shared across both trajectory outputs) ───────────
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(len(step_feats[0]), min(n_traj, len(step_feats[0])), replace=False)
    traj_colors = cm.tab20(np.linspace(0, 1, len(idx)))
    # pre-collect per-sample point arrays
    sample_pts = [np.array([pr[si] for pr in projs]) for si in idx]

    # ── GIF 2: trajectory progressive animation ────────────────────────────
    # frame t shows all trails from step 0 → t, plus current scatter
    fig_traj, ax_traj = plt.subplots(figsize=(7, 7))
    ax_traj.set_xlim(xlim); ax_traj.set_ylim(ylim)
    ax_traj.set_xlabel(f'PC1 (final {id_label.lower()})')
    ax_traj.set_ylabel(f'PC2 (final {id_label.lower()})')
    traj_title = ax_traj.set_title('', fontsize=9)

    line_artists  = []   # one Line2D per sample
    start_scats   = []   # start-point dot per sample
    cur_scats     = []   # current-position dot per sample
    for pts, c in zip(sample_pts, traj_colors):
        ln, = ax_traj.plot([], [], '-', color=c, alpha=0.6, lw=0.8)
        sc_start = ax_traj.scatter(pts[0, 0], pts[0, 1], color=c, s=10,
                                   marker='o', alpha=0.5, zorder=3)
        sc_cur   = ax_traj.scatter([], [], color=c, s=40, marker='*',
                                   zorder=5, edgecolors='black', linewidths=0.3)
        line_artists.append(ln)
        start_scats.append(sc_start)
        cur_scats.append(sc_cur)

    def _init_traj():
        for ln, sc in zip(line_artists, cur_scats):
            ln.set_data([], [])
            sc.set_offsets(np.empty((0, 2)))
        traj_title.set_text('')
        return line_artists + cur_scats + [traj_title]

    def _update_traj(frame):
        for pts, ln, sc in zip(sample_pts, line_artists, cur_scats):
            ln.set_data(pts[:frame + 1, 0], pts[:frame + 1, 1])
            sc.set_offsets(pts[frame:frame + 1, :2])
        traj_title.set_text(
            f'{id_label} {step_ids[frame]}  '
            f'N={len(idx)} samples  o=start  *=current')
        return line_artists + cur_scats + [traj_title]

    anim_traj = FuncAnimation(fig_traj, _update_traj, init_func=_init_traj,
                              frames=n, interval=1000 // fps, blit=True)
    traj_gif = os.path.join(save_dir, 'trajectory.gif')
    anim_traj.save(traj_gif, writer=PillowWriter(fps=fps))
    plt.close(fig_traj)
    print(f'[viz] {traj_gif}')

    # ── Static trajectory: full paths overlaid ────────────────────────────
    alphas = np.linspace(0.10, 1.00, n); lws = np.linspace(0.3, 1.8, n)
    fig, ax = plt.subplots(figsize=(8, 7))
    for pts, color in zip(sample_pts, traj_colors):
        for t in range(len(pts) - 1):
            ax.plot(pts[t:t+2, 0], pts[t:t+2, 1], '-', color=color,
                    alpha=float(alphas[t + 1]), lw=float(lws[t + 1]))
        ax.scatter(pts[0, 0],  pts[0, 1],  color=color, s=12, marker='o',
                   alpha=float(alphas[0]), zorder=3)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=40, marker='*',
                   alpha=1.0, zorder=4)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(f'Sample Trajectories  N={len(idx)}\n'
                 f'o=start  *=end  light→dark = early→late {id_label.lower()}', fontsize=9)
    ax.set_xlabel(f'PC1 (final {id_label.lower()})'); ax.set_ylabel(f'PC2 (final {id_label.lower()})')
    plt.tight_layout()
    traj_path = os.path.join(save_dir, 'trajectory.png')
    plt.savefig(traj_path, dpi=150); plt.close(); print(f'[viz] {traj_path}')
