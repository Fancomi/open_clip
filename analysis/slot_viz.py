"""Visualization helpers for slot frequency and feature overlays."""
import logging
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from .metrics import compute_knn_curvature, compute_knn_density


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _sanitize_name(name):
    keep = [c if c.isalnum() or c in ('-', '_') else '_' for c in str(name)]
    return ''.join(keep).strip('_') or 'empty'


def plot_slot_frequency_bars(freqs, save_dir, top_n=50, order='top', min_count=1):
    """Plot one Top-N (order='top') or Bottom-N (order='bottom') bar chart per slot type."""
    if order not in ('top', 'bottom'):
        raise ValueError(f"order must be 'top' or 'bottom', got {order!r}")
    _ensure_dir(save_dir)
    paths = []
    for slot, freq in freqs.items():
        pool = [(w, int(c)) for w, c in freq.items() if int(c) >= min_count]
        if order == 'bottom':
            items = sorted(pool, key=lambda x: (x[1], x[0]))[:top_n]
            label = f'Bottom {len(items)}'
        else:
            items = sorted(pool, key=lambda x: (-x[1], x[0]))[:top_n]
            label = f'Top {len(items)}'
        if not items:
            logging.warning(f'[slots] skip empty frequency bar: {slot}')
            continue
        words, counts = zip(*items)
        fig_h = max(4, 0.22 * len(words) + 1.5)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        y = np.arange(len(words))
        ax.barh(y, counts, color='#E45756' if order == 'bottom' else '#4C78A8')
        ax.set_yticks(y)
        ax.set_yticklabels(words, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Frequency')
        ax.set_ylabel('Words')
        ax.set_title(f'{label} {slot} by frequency (min_count={min_count})')
        for yi, count in zip(y, counts):
            ax.text(count, yi, f' {count}', va='center', fontsize=7)
        fig.tight_layout()
        out = os.path.join(save_dir, f'slot_freq_{order}_{_sanitize_name(slot)}.png')
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
                              background_max_points=0, seed=0):
    """Plot selected slot-word samples on a PCA feature distribution."""
    feats = np.asarray(feats)
    if feats.ndim != 2:
        raise ValueError(f'features must be 2D, got shape={feats.shape}')
    n = feats.shape[0]
    if n < 3:
        raise ValueError(f'need at least 3 feature rows, got {n}')

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

    metric_idx = _sample_with_forced(n, metric_max_points, seed, forced) if metric_max_points else np.arange(n)
    if len(metric_idx) < 2:
        raise ValueError(f'need at least 2 rows to compute {metric}, got {len(metric_idx)}')
    if len(metric_idx) < n:
        logging.info(f'[slots] compute {metric} on subset {len(metric_idx)}/{n}, forced highlights included')
    k_eff = max(1, min(int(k), len(metric_idx) - 1))
    metric_vals, metric_label = _compute_metric(feats[metric_idx], metric, k=k_eff)
    pca = PCA(n_components=2, random_state=seed).fit(feats[metric_idx])

    if len(metric_idx) < n:
        bg_idx = metric_idx
        bg_c = metric_vals
    else:
        bg_idx = _sample_with_forced(n, background_max_points, seed + 17, forced) if background_max_points else np.arange(n)
        bg_vals = np.full(n, np.nan, dtype=float)
        bg_vals[metric_idx] = metric_vals
        bg_c = bg_vals[bg_idx]
    if len(bg_idx) < n:
        logging.info(f'[slots] draw background subset {len(bg_idx)}/{n}, forced highlights included')
    bg_proj = pca.transform(feats[bg_idx])

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(bg_proj[:, 0], bg_proj[:, 1], s=5, alpha=0.35,
                    c=bg_c, cmap='viridis', rasterized=True)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label)

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(word_idxs))))
    for color, (word, (all_idx, shown_idx)) in zip(colors, word_idxs.items()):
        proj = pca.transform(feats[shown_idx])
        label = f'{word} ({len(shown_idx)}/{len(all_idx)})' if len(shown_idx) < len(all_idx) else f'{word} ({len(all_idx)})'
        ax.scatter(proj[:, 0], proj[:, 1], s=34, alpha=0.9, color=color,
                   edgecolors='black', linewidths=0.35, label=label, zorder=5)

    var = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({var[0] * 100:.1f}%)')
    ax.set_ylabel(f'PC2 ({var[1] * 100:.1f}%)')
    ax.set_title(f'{slot_type} on {model_name} feature distribution ({metric})')
    ax.legend(fontsize=7, loc='best', markerscale=1)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=170, bbox_inches='tight')
    plt.close(fig)
    logging.info(f'[slots] wrote {save_path}')

    return {
        'save_path': save_path,
        'slot_type': slot_type,
        'metric': metric,
        'n_features': int(n),
        'metric_points': int(len(metric_idx)),
        'background_points': int(len(bg_idx)),
        'words': {
            word: {'matched': int(len(all_idx)), 'shown': int(len(shown_idx))}
            for word, (all_idx, shown_idx) in word_idxs.items()
        },
    }
