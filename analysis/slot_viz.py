"""Visualization helpers for slot frequency and feature overlays."""
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .metrics import compute_knn_curvature, compute_knn_density


def fit_umap_embedding(feats, n_neighbors=15, min_dist=0.1, seed=0,
                       cache_path=None):
    """Fit UMAP on *feats* and return the 2-D embedding.

    If *cache_path* is given and exists, load from disk (skip fitting).
    If *cache_path* is given and doesn't exist, fit and save to disk.

    Falls back to PCA if umap-learn is not importable.
    Returns (embedding_2d, reducer_tag) where reducer_tag is 'umap' or 'pca'.
    """
    if cache_path and os.path.exists(cache_path):
        logging.info(f'[slots] loading UMAP embedding from cache {cache_path}')
        return np.load(cache_path), 'umap'

    try:
        import umap as umap_lib
        logging.info(f'[slots] fitting UMAP n={feats.shape[0]} n_neighbors={n_neighbors} ...')
        reducer = umap_lib.UMAP(n_components=2, n_neighbors=n_neighbors,
                                min_dist=min_dist, random_state=seed,
                                low_memory=True)
        emb = reducer.fit_transform(feats).astype(np.float32)
        tag = 'umap'
        logging.info(f'[slots] UMAP done  shape={emb.shape}')
    except ImportError:
        logging.warning('[slots] umap-learn not found, falling back to PCA')
        emb = PCA(n_components=2, random_state=seed).fit_transform(feats).astype(np.float32)
        tag = 'pca'

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.save(cache_path, emb)
        logging.info(f'[slots] saved embedding to {cache_path}')
    return emb, tag


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _sanitize_name(name):
    keep = [c if c.isalnum() or c in ('-', '_') else '_' for c in str(name)]
    return ''.join(keep).strip('_') or 'empty'


def plot_slot_frequency_bars(freqs, save_dir, top_n=50):
    """Plot one Top-N bar chart per slot type."""
    _ensure_dir(save_dir)
    paths = []
    for slot, freq in freqs.items():
        items = list(freq.items())[:top_n]
        if not items:
            logging.warning(f'[slots] skip empty frequency bar: {slot}')
            continue
        words, counts = zip(*items)
        fig_h = max(4, 0.22 * len(words) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        y = np.arange(len(words))
        ax.barh(y, counts, color='#4C78A8')
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Words')
        ax.set_title(f'Top {len(words)} {slot} by frequency')
        for yi, count in zip(y, counts):
            ax.text(count, yi, f' {count}', va='center', fontsize=7)
        fig.tight_layout()
        out = os.path.join(save_dir, f'slot_freq_top_{_sanitize_name(slot)}.png')
        fig.savefig(out, dpi=160, bbox_inches='tight')
        plt.close(fig)
        logging.info(f'[slots] wrote {out}')
        paths.append(out)
    return paths


def plot_slot_frequency_hist(freqs, save_dir, bins=50):
    """Plot one word-frequency histogram per slot type."""
    _ensure_dir(save_dir)
    paths = []
    for slot, freq in freqs.items():
        counts = np.array(list(freq.values()), dtype=float)
        if counts.size == 0:
            logging.warning(f'[slots] skip empty frequency histogram: {slot}')
            continue
        n_bins = min(int(bins), max(1, int(counts.max())))
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(counts, bins=n_bins, color='#72B7B2', edgecolor='white')
        ax.set_xlabel('Word frequency')
        ax.set_ylabel('Number of words')
        ax.set_title(f'Frequency distribution of {slot}')
        ax.grid(axis='y', alpha=0.25)
        fig.tight_layout()
        out = os.path.join(save_dir, f'slot_freq_dist_{_sanitize_name(slot)}.png')
        fig.savefig(out, dpi=160, bbox_inches='tight')
        plt.close(fig)
        logging.info(f'[slots] wrote {out}')
        paths.append(out)
    return paths


def _sample_with_forced(n, max_points, seed, forced=None):
    all_idx = np.arange(n, dtype=int)
    if max_points is None or max_points <= 0 or n <= max_points:
        return all_idx
    rng = np.random.default_rng(seed)
    forced = np.unique(np.asarray(forced if forced is not None else [], dtype=int))
    forced = forced[(forced >= 0) & (forced < n)]
    if len(forced) >= max_points:
        return np.sort(forced)
    pool = np.setdiff1d(all_idx, forced, assume_unique=False)
    extra = rng.choice(pool, max_points - len(forced), replace=False)
    return np.sort(np.concatenate([forced, extra]))


def _subsample_word_indices(indices, max_points, seed):
    idx = np.asarray(indices, dtype=int)
    if max_points is None or max_points <= 0 or len(idx) <= max_points:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, max_points, replace=False))


def _compute_metric(feats, metric, k):
    if metric == 'density':
        return compute_knn_density(feats, K=k), 'kNN density'
    if metric == 'curvature':
        return compute_knn_curvature(feats, K=k), 'kNN curvature'
    raise ValueError(f'unknown metric={metric}')


def plot_slot_feature_overlay(feats, word_to_indices, words, save_path, slot_type,
                              model_name='Feature', metric='density', k=50,
                              max_points_per_word=200, metric_max_points=0,
                              background_max_points=0, seed=0,
                              embedding_2d=None, reducer_tag='umap',
                              high_words=None, low_words=None):
    """Plot selected slot-word samples on a 2-D feature distribution.

    *embedding_2d* — pre-computed 2-D projection for all N features (same row
    order as *feats*).  Pass the result of fit_umap_embedding() so UMAP is
    fitted only once per overlay run.  When None, falls back to PCA fitted on
    the metric subset.

    *high_words* / *low_words* — lists that determine marker shape:
      high → circle 'o'  (common, many points)
      low  → star   '*'  (rare, few points — easier to spot)
    When not provided, all words use 'o'.
    """
    feats = np.asarray(feats)
    if feats.ndim != 2:
        raise ValueError(f'features must be 2D, got shape={feats.shape}')
    n = feats.shape[0]
    if n < 3:
        raise ValueError(f'need at least 3 feature rows, got {n}')

    high_set = set(high_words) if high_words else set()
    low_set  = set(low_words)  if low_words  else set()

    word_idxs = {}
    forced = []
    for wi, word in enumerate(words):
        idx = np.asarray(word_to_indices.get(word, []), dtype=int)
        idx = idx[(idx >= 0) & (idx < n)]
        idx = np.unique(idx)
        if len(idx) == 0:
            logging.warning(f'[slots] no matched feature rows for word={word}')
            continue
        shown = _subsample_word_indices(idx, max_points_per_word, seed + wi)
        word_idxs[word] = (idx, shown)
        forced.extend(shown.tolist())
    if not word_idxs:
        raise ValueError(f'no selected words matched feature rows for slot={slot_type}')
    forced = np.unique(np.asarray(forced, dtype=int))

    # ── Compute metric values ─────────────────────────────────────────────────
    metric_idx = (_sample_with_forced(n, metric_max_points, seed, forced)
                  if metric_max_points else np.arange(n))
    if len(metric_idx) < 2:
        raise ValueError(f'need at least 2 rows to compute {metric}, got {len(metric_idx)}')
    if len(metric_idx) < n:
        logging.info(f'[slots] compute {metric} on subset {len(metric_idx)}/{n}')
    k_eff = max(1, min(int(k), len(metric_idx) - 1))
    metric_vals, metric_label = _compute_metric(feats[metric_idx], metric, k=k_eff)

    # ── 2-D projection ────────────────────────────────────────────────────────
    if embedding_2d is not None:
        emb = np.asarray(embedding_2d)
        assert emb.shape == (n, 2), f'embedding_2d shape mismatch: {emb.shape} vs ({n}, 2)'
        xlabel, ylabel = f'{reducer_tag.upper()} 1', f'{reducer_tag.upper()} 2'
    else:
        # Fallback: PCA fitted on metric subset
        pca = PCA(n_components=2, random_state=seed).fit(feats[metric_idx])
        emb = pca.transform(feats)
        var = pca.explained_variance_ratio_
        xlabel = f'PC1 ({var[0] * 100:.1f}%)'
        ylabel = f'PC2 ({var[1] * 100:.1f}%)'
        reducer_tag = 'pca'

    # ── Background subset ─────────────────────────────────────────────────────
    if len(metric_idx) < n:
        bg_idx = metric_idx
        bg_c   = metric_vals
    else:
        bg_idx = (_sample_with_forced(n, background_max_points, seed + 17, forced)
                  if background_max_points else np.arange(n))
        bg_vals = np.full(n, np.nan, dtype=float)
        bg_vals[metric_idx] = metric_vals
        bg_c = bg_vals[bg_idx]
    if len(bg_idx) < n:
        logging.info(f'[slots] draw background subset {len(bg_idx)}/{n}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(emb[bg_idx, 0], emb[bg_idx, 1], s=5, alpha=0.35,
                    c=bg_c, cmap='viridis', rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label)

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(word_idxs))))
    for color, (word, (all_idx, shown_idx)) in zip(colors, word_idxs.items()):
        label = (f'{word} ({len(shown_idx)}/{len(all_idx)})'
                 if len(shown_idx) < len(all_idx) else f'{word} ({len(all_idx)})')
        # Shape: star for low-frequency, circle for high-frequency (or unknown)
        if word in low_set:
            marker, ms, lw = '*', 130, 0.4
            label = f'★ {label}'
        else:
            marker, ms, lw = 'o', 36, 0.35
        ax.scatter(emb[shown_idx, 0], emb[shown_idx, 1],
                   s=ms, alpha=0.9, color=color, marker=marker,
                   edgecolors='black', linewidths=lw, label=label, zorder=5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{slot_type} — {model_name} ({metric})')
    ax.legend(fontsize=7, loc='best', markerscale=0.9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'[slots] wrote {save_path}')

    return {
        'save_path': save_path,
        'slot_type': slot_type,
        'metric': metric,
        'reducer': reducer_tag,
        'n_features': int(n),
        'metric_points': int(len(metric_idx)),
        'background_points': int(len(bg_idx)),
        'words': {
            word: {'matched': int(len(all_idx)), 'shown': int(len(shown_idx))}
            for word, (all_idx, shown_idx) in word_idxs.items()
        },
    }
