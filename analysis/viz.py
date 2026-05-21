"""Visualization utilities for feature-space analysis."""
import logging
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
    """Line plot of anisotropy metrics across training steps/epochs.

    aniso_list entries are backbone CLS metrics (the primary feature space).
    """
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
        ax.plot(xs, ys, marker='o', ms=3, lw=1.5, color='steelblue')
        ax.annotate(f'{ys[0]:.2f}',  (xs[0],  ys[0]),  textcoords='offset points',
                    xytext=(4, 4),   fontsize=7, color='gray')
        ax.annotate(f'{ys[-1]:.2f}', (xs[-1], ys[-1]), textcoords='offset points',
                    xytext=(-20, 4), fontsize=7, color='steelblue')
        ax.set_xlabel(id_label)
        ax.set_title(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)

    dim_str = f'  (backbone CLS, D={aniso_list[0].get("dim", "?")})'
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


# ── Evolution & trajectory shared helpers ────────────────────────────────────

def _compute_lims(projs, txt_projs=None, pad=0.05):
    """Compute shared xlim/ylim from all projection arrays."""
    all_p = np.concatenate(projs + (txt_projs if txt_projs else []))
    x0, x1 = all_p[:, 0].min(), all_p[:, 0].max()
    y0, y1 = all_p[:, 1].min(), all_p[:, 1].max()
    xp, yp = (x1 - x0) * pad, (y1 - y0) * pad
    return (x0 - xp, x1 + xp), (y0 - yp, y1 + yp)


def _make_evolution_gif(projs, txt_projs, step_ids, xlim, ylim,
                        save_path, id_label, method, fps=4, fps_indices=None):
    """Scatter GIF: one frame per checkpoint, shared coordinate system.

    fps_indices : optional array of 5 int indices for FPS pair-link overlay.
                  When provided and txt_projs is not None, draws green/violet
                  marker pairs connected by lines on each frame.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    _C_IMG, _C_TXT = '#0055FF', '#FF2200'
    n = len(step_ids)
    has_fps = fps_indices is not None and txt_projs is not None

    fig, ax = plt.subplots(figsize=(5, 5))
    scat_img = ax.scatter([], [], s=3, alpha=0.35, color=_C_IMG,
                          label='Image', rasterized=True)
    scat_txt = (ax.scatter([], [], s=3, alpha=0.35, color=_C_TXT,
                           label='Text', rasterized=True)
                if txt_projs is not None else None)
    ax.set_xlim(xlim); ax.set_ylim(ylim); ax.axis('off')

    # FPS pair-link artists
    fps_lines, fps_img_scats, fps_txt_scats = [], [], []
    if has_fps:
        for i, mk in enumerate(_FPS_MARKERS[:len(fps_indices)]):
            ln, = ax.plot([], [], '-', color=_PAIR_LINE_COL,
                          lw=1.0, alpha=0.65, zorder=4)
            si = ax.scatter([], [], marker=mk, s=160, color=_PAIR_COL_IMG,
                            edgecolors='black', linewidths=0.6, zorder=6)
            st = ax.scatter([], [], marker=mk, s=160, color=_PAIR_COL_TXT,
                            edgecolors='black', linewidths=0.6, zorder=6)
            fps_lines.append(ln)
            fps_img_scats.append(si)
            fps_txt_scats.append(st)

    # Legend with controlled marker sizes (avoid markerscale bloating FPS markers)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', color='w', markerfacecolor=_C_IMG,
                      markersize=6, label='Image'),
               Line2D([], [], marker='o', color='w', markerfacecolor=_C_TXT,
                      markersize=6, label='Text')]
    if has_fps:
        handles += [Line2D([], [], marker='*', color='w', markerfacecolor=_PAIR_COL_IMG,
                           markeredgecolor='black', markersize=8, label='Img FPS'),
                    Line2D([], [], marker='*', color='w', markerfacecolor=_PAIR_COL_TXT,
                           markeredgecolor='black', markersize=8, label='Txt FPS')]
    if scat_txt is not None or has_fps:
        ax.legend(handles=handles, fontsize=7, loc='lower right')
    title_obj = ax.set_title('', fontsize=10)
    artists = ([scat_img] + ([scat_txt] if scat_txt else [])
               + fps_lines + fps_img_scats + fps_txt_scats + [title_obj])

    def _init():
        scat_img.set_offsets(np.empty((0, 2)))
        if scat_txt is not None:
            scat_txt.set_offsets(np.empty((0, 2)))
        for ln in fps_lines:
            ln.set_data([], [])
        for sc in fps_img_scats + fps_txt_scats:
            sc.set_offsets(np.empty((0, 2)))
        return artists

    def _update(frame):
        scat_img.set_offsets(projs[frame][:, :2])
        if scat_txt is not None:
            scat_txt.set_offsets(txt_projs[frame][:, :2])
        if has_fps:
            for i, fi in enumerate(fps_indices):
                xi, yi = projs[frame][fi, 0], projs[frame][fi, 1]
                xt, yt = txt_projs[frame][fi, 0], txt_projs[frame][fi, 1]
                fps_lines[i].set_data([xi, xt], [yi, yt])
                fps_img_scats[i].set_offsets([[xi, yi]])
                fps_txt_scats[i].set_offsets([[xt, yt]])
        sfx = '+Text' if txt_projs is not None else ''
        title_obj.set_text(
            f'{id_label} {step_ids[frame]}  ({(frame+1)/n*100:.0f}%)  [{method}{sfx}]')
        return artists

    anim = FuncAnimation(fig, _update, init_func=_init,
                         frames=n, interval=1000 // fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'[viz] {save_path}')


def _make_trajectory_gif(sample_pts, traj_colors, step_ids, xlim, ylim,
                         save_path, id_label, fps=4, trail=10):
    """Sliding-window trajectory animation."""
    from matplotlib.animation import FuncAnimation, PillowWriter
    n = len(step_ids)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(f'PC1 (final {id_label.lower()})')
    ax.set_ylabel(f'PC2 (final {id_label.lower()})')
    title_obj = ax.set_title('', fontsize=9)

    seg_artists, cur_scats = [], []
    for pts, c in zip(sample_pts, traj_colors):
        segs = [ax.plot([], [], '-', color=c, lw=1.2)[0] for _ in range(trail - 1)]
        sc = ax.scatter([], [], color=c, s=50, marker='*',
                        zorder=5, edgecolors='black', linewidths=0.4)
        seg_artists.append(segs)
        cur_scats.append(sc)

    all_lines = [seg for segs in seg_artists for seg in segs]

    def _init():
        for segs in seg_artists:
            for seg in segs:
                seg.set_data([], []); seg.set_alpha(0.0)
        for sc in cur_scats:
            sc.set_offsets(np.empty((0, 2)))
        title_obj.set_text('')
        return all_lines + cur_scats + [title_obj]

    def _update(frame):
        win_start = max(0, frame - trail + 1)
        w = frame - win_start + 1
        alphas = np.linspace(0.08, 0.85, w) if w > 1 else [0.85]
        for pts, segs in zip(sample_pts, seg_artists):
            for slot in range(trail - 1):
                seg = segs[slot]
                seg_idx = win_start + slot
                if slot < w - 1 and seg_idx + 1 <= frame:
                    seg.set_data(pts[seg_idx:seg_idx + 2, 0], pts[seg_idx:seg_idx + 2, 1])
                    seg.set_alpha(float(alphas[slot]))
                else:
                    seg.set_data([], []); seg.set_alpha(0.0)
        for pts, sc in zip(sample_pts, cur_scats):
            sc.set_offsets(pts[frame:frame + 1, :2])
        title_obj.set_text(
            f'{id_label} {step_ids[frame]}  ({(frame+1)/n*100:.0f}%)  '
            f'N={len(sample_pts)}  trail={trail}  *=current')
        return all_lines + cur_scats + [title_obj]

    anim = FuncAnimation(fig, _update, init_func=_init,
                         frames=n, interval=1000 // fps, blit=True)
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'[viz] {save_path}')


def _draw_static_trajectory(sample_pts, traj_colors, step_ids, xlim, ylim,
                            save_path, id_label, axis_prefix='PC',
                            fps_indices=None, all_projs=None, all_txt_projs=None):
    """Static trajectory: full paths overlaid, light→dark = early→late.

    fps_indices / all_projs / all_txt_projs : when all provided, overlay FPS
    image (green) and text (violet) trajectories with final-checkpoint pair link.
    """
    n = len(step_ids)
    alphas = np.linspace(0.10, 1.00, n)
    lws    = np.linspace(0.3,  1.8,  n)
    fig, ax = plt.subplots(figsize=(8, 7))
    for pts, color in zip(sample_pts, traj_colors):
        for t in range(len(pts) - 1):
            ax.plot(pts[t:t + 2, 0], pts[t:t + 2, 1], '-', color=color,
                    alpha=float(alphas[t + 1]), lw=float(lws[t + 1]))
        ax.scatter(pts[0, 0],  pts[0, 1],  color=color, s=12, marker='o',
                   alpha=float(alphas[0]), zorder=3)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=40, marker='*',
                   alpha=1.0, zorder=4)

    # FPS pair-link overlay
    if fps_indices is not None and all_projs is not None and all_txt_projs is not None:
        for i, (fi, mk) in enumerate(zip(fps_indices, _FPS_MARKERS[:len(fps_indices)])):
            img_path = np.array([all_projs[t][fi, :2] for t in range(n)])
            txt_path = np.array([all_txt_projs[t][fi, :2] for t in range(n)])
            # Image trajectory (green)
            for t in range(n - 1):
                ax.plot(img_path[t:t + 2, 0], img_path[t:t + 2, 1], '-',
                        color=_PAIR_COL_IMG, alpha=float(alphas[t + 1]), lw=2.0)
            # Text trajectory (violet)
            for t in range(n - 1):
                ax.plot(txt_path[t:t + 2, 0], txt_path[t:t + 2, 1], '-',
                        color=_PAIR_COL_TXT, alpha=float(alphas[t + 1]), lw=2.0)
            # Final-checkpoint connecting line + markers
            ax.plot([img_path[-1, 0], txt_path[-1, 0]],
                    [img_path[-1, 1], txt_path[-1, 1]],
                    '-', color=_PAIR_LINE_COL, lw=1.0, alpha=0.65, zorder=5)
            ax.scatter(img_path[-1, 0], img_path[-1, 1], marker=mk, s=160,
                       color=_PAIR_COL_IMG, edgecolors='black', linewidths=0.6,
                       zorder=7)
            ax.scatter(txt_path[-1, 0], txt_path[-1, 1], marker=mk, s=160,
                       color=_PAIR_COL_TXT, edgecolors='black', linewidths=0.6,
                       zorder=7)

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(f'Sample Trajectories  N={len(sample_pts)}\n'
                 f'o=start  *=end  light→dark = early→late {id_label.lower()}', fontsize=9)
    ax.set_xlabel(f'{axis_prefix}1'); ax.set_ylabel(f'{axis_prefix}2')
    if fps_indices is not None and all_txt_projs is not None:
        from matplotlib.lines import Line2D
        handles = [Line2D([], [], marker='*', color='w', markerfacecolor=_PAIR_COL_IMG,
                          markeredgecolor='black', markersize=8, label='Img FPS'),
                   Line2D([], [], marker='*', color='w', markerfacecolor=_PAIR_COL_TXT,
                          markeredgecolor='black', markersize=8, label='Txt FPS')]
        ax.legend(handles=handles, fontsize=7, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150); plt.close()
    print(f'[viz] {save_path}')


# ── Main evolution plots ─────────────────────────────────────────────────────

def plot_evolution(step_feats, step_ids, save_dir, n_traj=100, seed=42,
                   id_label='Step', fps=4, txt_feats=None, fps_indices=None):
    """PCA scatter GIF + trajectory GIF + static trajectory plot.

    txt_feats : optional list of (N, D) text feature arrays, one per checkpoint.
                When provided, the scatter GIF shows both modalities in the same
                PCA space (image=blue, text=red).  Trajectory tracks image only.
    fps_indices : optional array of 5 int indices for FPS pair-link overlay.

    Outputs
    -------
    {id_label}_evolution.gif   : scatter snapshot per checkpoint
    trajectory.gif             : progressive trajectory animation
    trajectory.png             : static full-path overlay
    """
    import os

    # PCA fitted on final checkpoint (joint img+txt when text available)
    fit_data = (np.concatenate([step_feats[-1], txt_feats[-1]])
                if txt_feats is not None else step_feats[-1])
    pca   = PCA(n_components=4).fit(fit_data)
    projs = [pca.transform(f) for f in step_feats]
    txt_projs = ([pca.transform(f) for f in txt_feats]
                 if txt_feats is not None else None)

    xlim, ylim = _compute_lims(projs, txt_projs)

    # GIF: PCA scatter snapshot per checkpoint
    gif_path = os.path.join(save_dir, f'{id_label.lower()}_evolution.gif')
    _make_evolution_gif(projs, txt_projs, step_ids, xlim, ylim,
                        gif_path, id_label, 'PCA', fps, fps_indices=fps_indices)

    # Trajectory GIF + static plot
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(step_feats[0]), min(n_traj, len(step_feats[0])), replace=False)
    traj_colors = cm.tab20(np.linspace(0, 1, len(idx)))
    sample_pts = [np.array([pr[si] for pr in projs]) for si in idx]

    _make_trajectory_gif(sample_pts, traj_colors, step_ids, xlim, ylim,
                         os.path.join(save_dir, 'trajectory.gif'), id_label, fps)
    _draw_static_trajectory(sample_pts, traj_colors, step_ids, xlim, ylim,
                            os.path.join(save_dir, 'trajectory.png'), id_label,
                            fps_indices=fps_indices, all_projs=projs,
                            all_txt_projs=txt_projs)


# ── UMAP evolution & trajectory ───────────────────────────────────────────────


def plot_umap_evolution(step_feats, step_ids, save_dir,
                        n_traj=100, seed=42, id_label='Step',
                        fps=4, txt_feats=None,
                        n_neighbors=15, min_dist=0.1,
                        subsample=2000, fps_indices=None):
    """UMAP evolution GIF + static trajectory plot (GPU-accelerated via cuML).

    Outputs
    -------
    umap_evolution.gif    : cloud snapshot per checkpoint
    umap_trajectory.png   : static full-path overlay for n_traj fixed samples
    """
    import os
    from cuml.manifold import UMAP as cuUMAP

    n_ckpt  = len(step_ids)
    N       = len(step_feats[0])
    rng     = np.random.default_rng(seed)
    has_txt = txt_feats is not None

    # Fixed trajectory indices
    traj_n   = min(n_traj, N)
    traj_idx = rng.choice(N, traj_n, replace=False)

    # Fit on final checkpoint
    fit_img = step_feats[-1].astype(np.float32)
    fit_data = (np.concatenate([fit_img, txt_feats[-1].astype(np.float32)])
                if has_txt else fit_img)
    logging.info(f'[umap-gpu] fit on final ckpt  {len(fit_data)} pts  '
                 f'n_neighbors={n_neighbors}')

    reducer = cuUMAP(n_components=2, n_neighbors=n_neighbors,
                     min_dist=min_dist, random_state=seed, verbose=False)
    reducer.fit(fit_data)
    logging.info('[umap-gpu] fit done')

    # Transform every checkpoint
    projs, txt_projs = [], []
    for i, feats in enumerate(step_feats):
        logging.info(f'[umap-gpu] transform ckpt {i+1}/{n_ckpt}  img  {feats.shape}')
        projs.append(np.asarray(reducer.transform(feats.astype(np.float32))))
        if has_txt:
            logging.info(f'[umap-gpu] transform ckpt {i+1}/{n_ckpt}  txt  {txt_feats[i].shape}')
            txt_projs.append(np.asarray(reducer.transform(txt_feats[i].astype(np.float32))))
    txt_projs = txt_projs if has_txt else None

    xlim, ylim = _compute_lims(projs, txt_projs)

    # GIF: scatter snapshot per checkpoint
    gif_path = os.path.join(save_dir, 'umap_evolution.gif')
    _make_evolution_gif(projs, txt_projs, step_ids, xlim, ylim,
                        gif_path, id_label, 'UMAP', fps, fps_indices=fps_indices)

    # Static trajectory
    traj_colors = cm.tab20(np.linspace(0, 1, traj_n))
    sample_pts  = [np.array([projs[t][traj_idx[i]] for t in range(n_ckpt)])
                   for i in range(traj_n)]
    _draw_static_trajectory(sample_pts, traj_colors, step_ids, xlim, ylim,
                            os.path.join(save_dir, 'umap_trajectory.png'),
                            id_label, axis_prefix='UMAP',
                            fps_indices=fps_indices, all_projs=projs,
                            all_txt_projs=txt_projs)



# ── Extended analysis: PC pairs overlay + batch GIF + extremes ───────────────


def plot_pc_pairs_allmodels(feats_dict, save_path, n_pcs=12, extremes=None):
    """每个模型独立 PCA, stride-2 PC pair. Layout: rows=models, cols=6 PC pairs.

    feats_dict : {model_name: (N, D)}
    extremes   : {model: {'high_density': idx5, ...}} or None
    """
    labels = list(feats_dict.keys())
    n_pairs = n_pcs // 2
    nrows, ncols = len(labels), n_pairs
    colors = cm.tab10(np.linspace(0, 0.9, nrows))

    _EXT_MK = {'high_density': ('D', '#CC0000', 'High Density'),
               'low_density': ('D', '#0044CC', 'Low Density'),
               'high_curvature': ('^', '#CC0000', 'High Curvature'),
               'low_curvature': ('^', '#0044CC', 'Low Curvature')}

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 3.5 * nrows))
    axes = np.array(axes).reshape(nrows, ncols)

    for row, (name, c) in enumerate(zip(labels, colors)):
        feats = feats_dict[name]
        pca = PCA(n_components=n_pcs).fit(feats)
        vr = pca.explained_variance_ratio_
        proj = pca.transform(feats)

        for col in range(n_pairs):
            pi, pj = col * 2, col * 2 + 1
            ax = axes[row, col]
            ax.scatter(proj[:, pi], proj[:, pj], s=2, alpha=0.3,
                       color=c, rasterized=True)
            if extremes and name in extremes:
                for cat, (mk, mc, lbl) in _EXT_MK.items():
                    idxs = extremes[name].get(cat)
                    if idxs is None:
                        continue
                    ax.scatter(proj[idxs, pi], proj[idxs, pj], marker=mk, s=100,
                               color=mc, edgecolors='black', linewidths=0.5, zorder=5)
            ax.set_xlabel(f'PC{pi+1} ({vr[pi]*100:.1f}%)', fontsize=7)
            if col == 0:
                ax.set_ylabel(f'{name}\nPC{pj+1} ({vr[pj]*100:.1f}%)', fontsize=8)
            else:
                ax.set_ylabel(f'PC{pj+1} ({vr[pj]*100:.1f}%)', fontsize=7)
            ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle('Vision Encoder Image Features — PC Pairs (per-model PCA)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'[viz] {save_path}')


def plot_batch_gif(feats, batches, save_path, n_pcs=12, method='Random', model='', fps=4):
    """Batch sampling GIF: 2×3 panels (6 PC pairs), 20 frames.

    Each frame: all points in gray, batch highlighted, center + 1σ ellipse.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Ellipse as MplEllipse

    pca = PCA(n_components=n_pcs).fit(feats)
    proj = pca.transform(feats)  # (N, n_pcs)
    vr = pca.explained_variance_ratio_

    n_pairs = n_pcs // 2
    ncols, nrows = 3, (n_pairs + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    # Static background (gray)
    bg_scats = []
    for idx in range(n_pairs):
        pi, pj = idx * 2, idx * 2 + 1
        ax = axes_flat[idx]
        ax.scatter(proj[:, pi], proj[:, pj], s=1, alpha=0.12,
                   color='#999999', rasterized=True)
        ax.set_xlabel(f'PC{pi+1} ({vr[pi]*100:.1f}%)', fontsize=7)
        ax.set_ylabel(f'PC{pj+1} ({vr[pj]*100:.1f}%)', fontsize=7)
        ax.tick_params(labelsize=6)
        # Set fixed limits
        pad = 0.05
        xr = proj[:, pi].max() - proj[:, pi].min()
        yr = proj[:, pj].max() - proj[:, pj].min()
        ax.set_xlim(proj[:, pi].min() - pad * xr, proj[:, pi].max() + pad * xr)
        ax.set_ylim(proj[:, pj].min() - pad * yr, proj[:, pj].max() + pad * yr)
    for i in range(n_pairs, len(axes_flat)):
        axes_flat[i].set_visible(False)

    # Dynamic artists per panel
    batch_scats = []
    center_scats = []
    ellipses = []
    for idx in range(n_pairs):
        ax = axes_flat[idx]
        sc = ax.scatter([], [], s=6, alpha=0.85, color='#E63946', rasterized=True)
        ct = ax.scatter([], [], marker='X', s=200, color='black', zorder=6)
        ell = MplEllipse((0, 0), 0, 0, fill=False, color='black', lw=1.5, ls='--', zorder=5)
        ax.add_patch(ell)
        batch_scats.append(sc)
        center_scats.append(ct)
        ellipses.append(ell)

    title_obj = fig.suptitle('', fontsize=10)
    n_frames = len(batches)

    def _update(frame):
        batch_idx = batches[frame]
        batch_proj = proj[batch_idx]
        for idx in range(n_pairs):
            pi, pj = idx * 2, idx * 2 + 1
            bp = batch_proj[:, [pi, pj]]
            batch_scats[idx].set_offsets(bp)
            center = bp.mean(axis=0)
            center_scats[idx].set_offsets([center])
            # 1-sigma ellipse from covariance
            cov = np.cov(bp.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            order = eigvals.argsort()[::-1]
            eigvals, eigvecs = eigvals[order], eigvecs[:, order]
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            w, h = 2 * np.sqrt(eigvals)
            ellipses[idx].set_center(center)
            ellipses[idx].width = w
            ellipses[idx].height = h
            ellipses[idx].angle = angle
        title_obj.set_text(
            f'{model} — {method} Batch {frame+1}/{n_frames}  (n={len(batch_idx)})')
        return batch_scats + center_scats + ellipses + [title_obj]

    anim = FuncAnimation(fig, _update, frames=n_frames, interval=1000 // fps, blit=False)
    plt.tight_layout()
    anim.save(save_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f'[viz] {save_path}')


def plot_extremes_single(feats, extremes, save_path, model_name, n_pcs=12, feat_type='Image'):
    """单模型极端点可视化: 6 PC pair panels, 标注 density/curvature extremes.

    extremes: {'high_density': idx5, 'low_density': idx5,
               'high_curvature': idx5, 'low_curvature': idx5}
    """
    pca = PCA(n_components=n_pcs).fit(feats)
    proj = pca.transform(feats)
    vr = pca.explained_variance_ratio_

    _MK = {'high_density': ('D', '#CC0000', 'High Density'),
            'low_density': ('D', '#0044CC', 'Low Density'),
            'high_curvature': ('^', '#CC0000', 'High Curvature'),
            'low_curvature': ('^', '#0044CC', 'Low Curvature')}

    n_pairs = n_pcs // 2
    ncols, nrows = 3, (n_pairs + 2) // 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes_flat = np.array(axes).reshape(-1)

    for idx in range(n_pairs):
        pi, pj = idx * 2, idx * 2 + 1
        ax = axes_flat[idx]
        ax.scatter(proj[:, pi], proj[:, pj], s=2, alpha=0.15,
                   color='#999999', rasterized=True)
        for cat, (mk, mc, lbl) in _MK.items():
            idxs = extremes.get(cat)
            if idxs is None:
                continue
            ax.scatter(proj[idxs, pi], proj[idxs, pj], marker=mk, s=140,
                       color=mc, edgecolors='black', linewidths=0.6, zorder=5,
                       label=lbl if idx == 0 else '')
            # 标注序号
            for rank, i in enumerate(idxs):
                ax.annotate(str(rank + 1), (proj[i, pi], proj[i, pj]),
                            fontsize=6, ha='center', va='bottom',
                            xytext=(0, 4), textcoords='offset points')
        ax.set_xlabel(f'PC{pi+1} ({vr[pi]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC{pj+1} ({vr[pj]*100:.1f}%)', fontsize=8)
        ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc='best')

    for i in range(n_pairs, len(axes_flat)):
        axes_flat[i].set_visible(False)
    fig.suptitle(f'{model_name} {feat_type} — Density & Curvature Extremes (K=50)', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'[viz] {save_path}')


def plot_crop_probe(img_feats, crops_feats, fps_idx, out_path):
    """Show where DINOv3-style global/local crops land vs the full-image features.

    Layout : ceil(n_models / 3) rows × 3 cols, PC1 vs PC2 per panel.
    Each panel:
      - Gray cloud  : full image distribution (all N samples)
      - Per-sample colour (5 colours from tab10):
          ●  circle         : FPS original image
          ▲  triangle-up    : global crop  (224px, scale 0.32–1.0)
          ▼  triangle-down  : local  crop  (96px,  scale 0.05–0.32)
        Three points of the same sample are connected by a thin line.

    Args:
        img_feats   : dict {model_name: (N, D) ndarray}  — full distributions
        crops_feats : dict {model_name: {'orig':(5,D), 'global':(5,D), 'local':(5,D)}}
        fps_idx     : array-like of 5 sample indices (for panel annotation)
        out_path    : save path (.png)
    """
    models = [k for k in img_feats if k in crops_feats]
    n_models = len(models)
    ncols = 3
    nrows = (n_models + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
    axes = np.array(axes).reshape(-1)

    # 5 distinct, high-contrast sample colours
    sample_colors = [plt.cm.tab10(i) for i in range(5)]

    _MARKERS = {'orig': 'o', 'global': '^', 'local': 'v'}
    _SIZES   = {'orig': 100, 'global': 100, 'local': 100}
    _LABELS  = {'orig': 'Original', 'global': 'Global crop (224px)', 'local': 'Local crop (96px)'}

    for ax_i, name in enumerate(models):
        ax = axes[ax_i]
        full = img_feats[name]

        # Independent PCA per model (same as image_allmodels independent-PCA branch)
        pca = PCA(n_components=2).fit(full)
        var = pca.explained_variance_ratio_
        full_proj = pca.transform(full)

        # Background cloud
        ax.scatter(full_proj[:, 0], full_proj[:, 1], s=2, alpha=0.12,
                   color='#999999', rasterized=True)

        cd = crops_feats[name]
        projs = {k: pca.transform(cd[k]) for k in ('orig', 'global', 'local')}

        for si in range(5):
            c = sample_colors[si]
            # Connect the three variants with a thin line
            xs = [projs[k][si, 0] for k in ('orig', 'global', 'local')]
            ys = [projs[k][si, 1] for k in ('orig', 'global', 'local')]
            ax.plot(xs, ys, '-', color=c, alpha=0.55, linewidth=1.2, zorder=3)

            for key in ('orig', 'global', 'local'):
                ax.scatter(projs[key][si, 0], projs[key][si, 1],
                           s=_SIZES[key], marker=_MARKERS[key],
                           color=c, edgecolors='white', linewidths=0.6, zorder=5)

        ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)', fontsize=8)
        ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)', fontsize=8)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)

    # Shared legend (placed in last used panel)
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker=_MARKERS[k], color='gray', linestyle='None',
               markersize=9, label=_LABELS[k], markeredgecolor='white')
        for k in ('orig', 'global', 'local')
    ] + [Line2D([0], [0], color=sample_colors[i], linestyle='-',
                linewidth=1.5, label=f'Sample #{fps_idx[i]}') for i in range(5)]
    axes[len(models) - 1].legend(handles=legend_handles, fontsize=7,
                                  loc='lower right', markerscale=1)

    # Hide unused panels
    for ax_i in range(len(models), len(axes)):
        axes[ax_i].set_visible(False)

    fig.suptitle(
        'Crop Probe: Original vs DINOv3-style Global / Local Crop\n'
        '●=original  ▲=global crop (224px, scale 0.32–1.0)  '
        '▼=local crop (96px, scale 0.05–0.32)',
        fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[viz] {out_path}')
