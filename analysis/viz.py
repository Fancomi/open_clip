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

# ── Pair-link mode (modality gap): same shape, modality-specific color + line ─
# Colors must contrast with cloud colors (#0055FF, #FF2200) and white background
_PAIR_COL_IMG  = '#00CC44'   # vivid green  — image FPS markers
_PAIR_COL_TXT  = '#AA00FF'   # violet       — text  FPS markers
_PAIR_LINE_COL = '#222222'   # near-black   — connecting line

# ── PCA helpers ───────────────────────────────────────────────────────────────

def _fit_pca(feats_list, n):
    """Shared PCA if same dim; independent per-model otherwise."""
    dims = [f.shape[1] for f in feats_list]
    if len(set(dims)) == 1:
        pca = PCA(n_components=n).fit(np.concatenate(feats_list))
        return [pca] * len(feats_list), pca.explained_variance_ratio_
    return [PCA(n_components=n).fit(f) for f in feats_list], None


def _tsne_proj(feats, subsample=2000, seed=42, force_indices=None):
    """Run T-SNE on a subsample; return (N_sub, 2) projection + original indices.

    force_indices: list/array of indices that MUST be in the subsample (e.g. FPS).
    """
    from sklearn.manifold import TSNE
    rng = np.random.default_rng(seed)
    n   = len(feats)
    if force_indices is not None and len(force_indices) > 0:
        forced   = np.unique(np.asarray(force_indices, dtype=int))
        pool     = np.setdiff1d(np.arange(n), forced)
        n_extra  = max(0, min(subsample - len(forced), len(pool)))
        extra    = rng.choice(pool, n_extra, replace=False) if n_extra > 0 else np.array([], dtype=int)
        idx      = np.concatenate([forced, extra]).astype(int)
    else:
        idx = rng.choice(n, min(subsample, n), replace=False)
    sub = feats[idx]
    emb = TSNE(n_components=2, perplexity=30, n_jobs=-1,
               random_state=seed).fit_transform(sub.astype(np.float32))
    return emb, idx


# ── Main plots ────────────────────────────────────────────────────────────────

def plot_scatter(feats_dict, title, save_path, n_pca=4, fps_indices=None,
                 with_tsne=True, colors=None, fps_pair_link=False):
    """Multi-axis PCA scatter + optional T-SNE column.

    fps_pair_link : when True (modality gap mode, exactly 2 models in shared PCA),
                    FPS samples are rendered as same-shape / modality-color pairs
                    with a line connecting image↔text of the same entity.
                    Image FPS = green (#00CC44), Text FPS = violet (#AA00FF).
    """
    labels = list(feats_dict.keys())
    feats  = list(feats_dict.values())
    if colors is None:
        colors = cm.tab10(np.linspace(0, 0.9, len(labels)))
    pairs  = list(combinations(range(n_pca), 2))
    pcas, shared_var = _fit_pca(feats, n_pca)
    projs  = [pca.transform(f) for pca, f in zip(pcas, feats)]

    # FPS helper: flat array of all fps indices (for forcing into T-SNE subsample)
    fps_flat = np.array(fps_indices, dtype=int) if fps_indices is not None else None

    # Pre-compute T-SNE — force FPS indices into every subsample
    if with_tsne:
        tsne_data = []   # list of (emb, sub_idx) per model
        for f in feats:
            emb, sub_idx = _tsne_proj(f, force_indices=fps_flat)
            tsne_data.append((emb, sub_idx))

    def _fps_on_pca(ax, proj, pi, pj, add_label=False):
        """Mark FPS points on a PCA panel (standard multi-model mode)."""
        if fps_indices is None:
            return
        for fi, (idx, mk, fc) in enumerate(zip(fps_indices, _FPS_MARKERS, _FPS_COLORS)):
            ax.scatter(proj[idx, pi], proj[idx, pj], marker=mk, s=120, color=fc,
                       edgecolors='black', linewidths=0.5, zorder=5,
                       label=f'FPS-{fi}' if add_label else '')

    def _fps_pair_on_pca(ax, pi, pj, add_legend=False):
        """Pair-link mode: same shape per FPS sample, green=image / violet=text + line."""
        if fps_indices is None or len(projs) < 2:
            return
        p_img_all = projs[0]   # image projection
        p_txt_all = projs[1]   # text  projection
        for fi, (fps_i, mk) in enumerate(zip(fps_indices, _FPS_MARKERS)):
            xi, yi = p_img_all[fps_i, pi], p_img_all[fps_i, pj]
            xt, yt = p_txt_all[fps_i, pi], p_txt_all[fps_i, pj]
            ax.plot([xi, xt], [yi, yt], '-', color=_PAIR_LINE_COL,
                    lw=1.0, alpha=0.65, zorder=4)
            ax.scatter(xi, yi, marker=mk, s=160, color=_PAIR_COL_IMG,
                       edgecolors='black', linewidths=0.6, zorder=6,
                       label='Image FPS' if (add_legend and fi == 0) else '')
            ax.scatter(xt, yt, marker=mk, s=160, color=_PAIR_COL_TXT,
                       edgecolors='black', linewidths=0.6, zorder=6,
                       label='Text FPS' if (add_legend and fi == 0) else '')

    def _fps_on_tsne(ax, emb, sub_idx, add_label=False):
        """Mark FPS points on a T-SNE panel using index remapping (standard mode)."""
        if fps_indices is None:
            return
        idx_map = {int(orig): pos for pos, orig in enumerate(sub_idx)}
        for fi, (fps_i, mk, fc) in enumerate(zip(fps_indices, _FPS_MARKERS, _FPS_COLORS)):
            pos = idx_map.get(int(fps_i))
            if pos is not None:
                ax.scatter(emb[pos, 0], emb[pos, 1], marker=mk, s=120, color=fc,
                           edgecolors='black', linewidths=0.5, zorder=5,
                           label=f'FPS-{fi}' if add_label else '')

    def _fps_pair_on_tsne(ax, add_legend=False):
        """Pair-link mode on T-SNE: green=image, violet=text, line between pairs."""
        if fps_indices is None or len(tsne_data) < 2:
            return
        emb0, idx0 = tsne_data[0]   # image T-SNE
        emb1, idx1 = tsne_data[1]   # text  T-SNE
        map0 = {int(o): p for p, o in enumerate(idx0)}
        map1 = {int(o): p for p, o in enumerate(idx1)}
        for fi, (fps_i, mk) in enumerate(zip(fps_indices, _FPS_MARKERS)):
            p0 = map0.get(int(fps_i))
            p1 = map1.get(int(fps_i))
            if p0 is None or p1 is None:
                continue
            xi, yi = emb0[p0, 0], emb0[p0, 1]
            xt, yt = emb1[p1, 0], emb1[p1, 1]
            ax.plot([xi, xt], [yi, yt], '-', color=_PAIR_LINE_COL,
                    lw=1.0, alpha=0.65, zorder=4)
            ax.scatter(xi, yi, marker=mk, s=160, color=_PAIR_COL_IMG,
                       edgecolors='black', linewidths=0.6, zorder=6,
                       label='Image FPS' if (add_legend and fi == 0) else '')
            ax.scatter(xt, yt, marker=mk, s=160, color=_PAIR_COL_TXT,
                       edgecolors='black', linewidths=0.6, zorder=6,
                       label='Text FPS' if (add_legend and fi == 0) else '')

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
            if fps_pair_link:
                _fps_pair_on_pca(ax, pi, pj, add_legend=(col == 0))
            else:
                for mi, proj in enumerate(projs):
                    _fps_on_pca(ax, proj, pi, pj, add_label=(col == 0 and mi == 0))
            ax.set_xlabel(f'PC{pi+1}'); ax.set_ylabel(f'PC{pj+1}')
            ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
            if col == 0:
                ax.legend(markerscale=1, fontsize=8)
        ax = axes[len(pairs)]
        ax.bar(range(1, n_pca + 1), shared_var * 100, color='steelblue')
        ax.set_xlabel('Component'); ax.set_ylabel('Variance explained (%)')
        ax.set_title('Explained variance', fontsize=9)
        if with_tsne:
            ax = axes[-1]
            for mi, (label, (emb, sub_idx), c) in enumerate(
                    zip(labels, tsne_data, colors)):
                ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.3, color=c,
                           label=label, rasterized=True)
            if fps_pair_link:
                _fps_pair_on_tsne(ax, add_legend=True)
            else:
                for mi, (_, (emb, sub_idx)) in enumerate(zip(labels, tsne_data)):
                    _fps_on_tsne(ax, emb, sub_idx, add_label=(mi == 0))
            ax.set_title('T-SNE (2k subsample)', fontsize=9)
            ax.legend(markerscale=1, fontsize=8)
            ax.axis('off')
        fig.suptitle(title, fontsize=12, y=1.01)
    else:
        # ── independent PCA: one row per model ────────────────────────────────
        nrows   = len(labels)
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
                _fps_on_pca(ax, proj, pi, pj, add_label=(col == 0 and row == 0))
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
                emb, sub_idx = tsne_data[row]
                ax = axes[row, -1]
                ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.3, color=c,
                           rasterized=True)
                _fps_on_tsne(ax, emb, sub_idx, add_label=(row == 0))
                ax.set_title(f'{label}  T-SNE', fontsize=9)
                ax.axis('off')
                if row == 0 and has_fps:
                    ax.legend(markerscale=2, fontsize=7)
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
    _COL_A = '#0055FF'   # pure blue  — image
    _COL_B = '#FF2200'   # pure red   — text
    if a_on_top:
        ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.35, color=_COL_B,
                   label=label_b, rasterized=True)
        ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.35, color=_COL_A,
                   label=label_a, rasterized=True)
        on_top_lbl = label_a
    else:
        ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.35, color=_COL_A,
                   label=label_a, rasterized=True)
        ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.35, color=_COL_B,
                   label=label_b, rasterized=True)
        on_top_lbl = label_b
    ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
    ax.legend(markerscale=1, fontsize=9)
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

    # ── GIF 1: T-SNE scatter snapshot per checkpoint ───────────────────────
    # Run T-SNE independently for each checkpoint (2k subsample for speed)
    print(f'[viz] computing T-SNE for {n} checkpoints …')
    tsne_projs = []
    for i, f in enumerate(step_feats):
        emb, _ = _tsne_proj(f, subsample=2000, seed=seed)
        tsne_projs.append(emb)
        if (i + 1) % 5 == 0 or i == n - 1:
            print(f'[viz]   T-SNE {i+1}/{n}')

    # Compute a stable axis range across all frames
    all_t = np.concatenate(tsne_projs)
    pad   = 0.05
    tx0, tx1 = all_t[:, 0].min(), all_t[:, 0].max()
    ty0, ty1 = all_t[:, 1].min(), all_t[:, 1].max()
    tp = max((tx1 - tx0), (ty1 - ty0)) * pad
    tlim_x = (tx0 - tp, tx1 + tp); tlim_y = (ty0 - tp, ty1 + tp)

    fig_gif, ax_gif = plt.subplots(figsize=(5, 5))
    scat = ax_gif.scatter([], [], s=3, alpha=0.4, rasterized=True)
    ax_gif.set_xlim(tlim_x); ax_gif.set_ylim(tlim_y); ax_gif.axis('off')
    title_obj = ax_gif.set_title('', fontsize=10)

    def _init_scat():
        scat.set_offsets(np.empty((0, 2)))
        return (scat, title_obj)

    def _update_scat(frame):
        scat.set_offsets(tsne_projs[frame])
        scat.set_color(colors_n[frame])
        pct = (frame + 1) / n * 100
        title_obj.set_text(f'{id_label} {step_ids[frame]}  ({pct:.0f}%)  [T-SNE 2k]')
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

    # ── GIF 2: sliding-window trajectory (last `trail` steps) ────────────────
    # frame t shows trails from max(0, t-trail+1) → t  (rolling window)
    # + step label and % of training completed
    trail = 10
    fig_traj, ax_traj = plt.subplots(figsize=(7, 7))
    ax_traj.set_xlim(xlim); ax_traj.set_ylim(ylim)
    ax_traj.set_xlabel(f'PC1 (final {id_label.lower()})')
    ax_traj.set_ylabel(f'PC2 (final {id_label.lower()})')
    traj_title = ax_traj.set_title('', fontsize=9)

    # Each sample needs `trail-1` line segments; pre-create them
    # seg_artists[i] = list of Line2D, one per segment slot in the window
    seg_artists = []
    cur_scats   = []
    for pts, c in zip(sample_pts, traj_colors):
        segs = [ax_traj.plot([], [], '-', color=c, lw=1.2)[0]
                for _ in range(trail - 1)]
        sc_cur = ax_traj.scatter([], [], color=c, s=50, marker='*',
                                 zorder=5, edgecolors='black', linewidths=0.4)
        seg_artists.append(segs)
        cur_scats.append(sc_cur)

    all_line_artists = [seg for segs in seg_artists for seg in segs]

    def _init_traj():
        for segs in seg_artists:
            for seg in segs:
                seg.set_data([], [])
                seg.set_alpha(0.0)
        for sc in cur_scats:
            sc.set_offsets(np.empty((0, 2)))
        traj_title.set_text('')
        return all_line_artists + cur_scats + [traj_title]

    def _update_traj(frame):
        win_start = max(0, frame - trail + 1)
        win_pts   = list(range(win_start, frame + 1))   # indices in window
        w         = len(win_pts)
        # alpha ramp: oldest segment is faintest
        alphas = np.linspace(0.08, 0.85, w) if w > 1 else [0.85]

        for pts, segs in zip(sample_pts, seg_artists):
            # fill segments that are active in the window
            for slot in range(trail - 1):
                seg = segs[slot]
                seg_idx = win_start + slot          # index of segment start
                if slot < w - 1 and seg_idx + 1 <= frame:
                    seg.set_data(pts[seg_idx:seg_idx + 2, 0],
                                 pts[seg_idx:seg_idx + 2, 1])
                    seg.set_alpha(float(alphas[slot]))
                else:
                    seg.set_data([], [])
                    seg.set_alpha(0.0)
        for pts, sc in zip(sample_pts, cur_scats):
            sc.set_offsets(pts[frame:frame + 1, :2])

        pct = (frame + 1) / n * 100
        traj_title.set_text(
            f'{id_label} {step_ids[frame]}  ({pct:.0f}% of training)  '
            f'N={len(idx)}  trail={trail}  *=current')
        return all_line_artists + cur_scats + [traj_title]

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
