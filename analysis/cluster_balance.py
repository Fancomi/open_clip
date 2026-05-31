#!/usr/bin/env python3
"""CC3M 特征空间聚类长尾分析: 对不同 teacher 模型做 K-Means++ 聚类, 可视化簇大小分布。

用法:
    python analysis/cluster_balance.py [--teachers siglip2 eva02] [--k 5000] [--force]

产出: /root/.../datas/cc3m-tsv/feature_probe/cluster_balance/
    - cluster_sizes_{teacher}.png   — 簇大小分布 (log-log rank plot + histogram)
    - cluster_balance_summary.png   — 所有 teacher 的 Lorenz 曲线对比
    - cluster_stats.json            — 数值统计
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_FEAT_DIR = f'{_BASE}/datas/cc3m-tsv/features'
_OUT_DIR = f'{_BASE}/datas/cc3m-tsv/feature_probe/cluster_balance'
TEACHERS = ['pe_core', 'dinov3', 'siglip2', 'datacomp', 'dfn2b', 'eva02', 'laion2b', 'metaclip']


def cluster_and_stats(feats_path, K, device):
    """K-Means++ 聚类并返回簇大小数组。"""
    feats = np.load(feats_path)
    N, D = feats.shape
    feats_t = torch.from_numpy(feats.astype(np.float32)).to(device)
    feats_t = torch.nn.functional.normalize(feats_t, dim=1)

    # K-Means++ init
    centroids = torch.empty(K, D, device=device)
    centroids[0] = feats_t[torch.randint(N, (1,), device=device)]
    min_dist = torch.full((N,), float('inf'), device=device)
    for k in range(1, K):
        dist = 1 - (feats_t @ centroids[k - 1].unsqueeze(1)).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
        probs = min_dist ** 2
        probs /= probs.sum()
        centroids[k] = feats_t[torch.multinomial(probs, 1).item()]

    # 20 iterations
    for _ in range(20):
        assignments = torch.empty(N, dtype=torch.long, device=device)
        for i in range(0, N, 32768):
            end = min(i + 32768, N)
            assignments[i:end] = (feats_t[i:end] @ centroids.T).argmax(dim=1)
        new_c = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device)
        new_c.scatter_add_(0, assignments.unsqueeze(1).expand(N, D), feats_t)
        counts.scatter_add_(0, assignments, torch.ones(N, device=device))
        valid = counts > 0
        new_c[valid] /= counts[valid].unsqueeze(1)
        new_c[~valid] = centroids[~valid]
        centroids = torch.nn.functional.normalize(new_c, dim=1)

    # Final assignment
    assignments = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, 32768):
        end = min(i + 32768, N)
        assignments[i:end] = (feats_t[i:end] @ centroids.T).argmax(dim=1)

    sizes = np.bincount(assignments.cpu().numpy(), minlength=K)
    del feats_t, centroids
    torch.cuda.empty_cache()
    return sizes[sizes > 0]


def compute_stats(sizes):
    """计算均衡性指标。"""
    sorted_s = np.sort(sizes)
    n = len(sorted_s)
    gini = (2 * np.sum(np.arange(1, n + 1) * sorted_s) - (n + 1) * sorted_s.sum()) / (n * sorted_s.sum())
    return {
        'n_clusters': int(n),
        'mean': float(sizes.mean()),
        'median': float(np.median(sizes)),
        'std': float(sizes.std()),
        'min': int(sizes.min()),
        'max': int(sizes.max()),
        'max_min_ratio': float(sizes.max() / max(sizes.min(), 1)),
        'p10': float(np.percentile(sizes, 10)),
        'p90': float(np.percentile(sizes, 90)),
        'p90_p10_ratio': float(np.percentile(sizes, 90) / max(np.percentile(sizes, 10), 1)),
        'gini': float(gini),
        'cov': float(sizes.std() / sizes.mean()),
    }


def plot_per_teacher(sizes, teacher, out_dir):
    """单 teacher: rank-size plot (log-log) + histogram。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sorted_s = np.sort(sizes)[::-1]

    # Rank-size (log-log)
    ax = axes[0]
    ax.loglog(np.arange(1, len(sorted_s) + 1), sorted_s, lw=1.5, color='#2196F3')
    ax.axhline(sizes.mean(), ls='--', color='gray', lw=1, label=f'mean={sizes.mean():.0f}')
    ax.set_xlabel('Cluster rank')
    ax.set_ylabel('Cluster size')
    ax.set_title(f'{teacher} — Rank-Size (log-log)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Histogram
    ax = axes[1]
    ax.hist(sizes, bins=100, color='#4CAF50', alpha=0.7, edgecolor='none')
    ax.axvline(sizes.mean(), ls='--', color='red', lw=1.5, label=f'mean={sizes.mean():.0f}')
    ax.axvline(np.median(sizes), ls=':', color='orange', lw=1.5, label=f'median={np.median(sizes):.0f}')
    ax.set_xlabel('Cluster size')
    ax.set_ylabel('Count')
    ax.set_title(f'{teacher} — Cluster Size Distribution')
    ax.set_xlim(0, np.percentile(sizes, 99) * 1.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out_dir, f'cluster_sizes_{teacher}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_lorenz_all(all_sizes, out_dir):
    """所有 teacher 的 Lorenz 曲线对比 + Gini 系数。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_sizes)))

    for (teacher, sizes), color in zip(all_sizes.items(), colors):
        sorted_s = np.sort(sizes)
        cumsum = np.cumsum(sorted_s) / sorted_s.sum()
        x = np.linspace(0, 1, len(cumsum))
        gini = compute_stats(sizes)['gini']
        ax.plot(x, cumsum, lw=1.5, color=color, label=f'{teacher} (Gini={gini:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Perfect equality')
    ax.set_xlabel('Cumulative fraction of clusters (sorted by size)')
    ax.set_ylabel('Cumulative fraction of samples')
    ax.set_title('CC3M Cluster Balance — Lorenz Curves (K=5000)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    path = os.path.join(out_dir, 'cluster_balance_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--teachers', nargs='+', default=TEACHERS)
    parser.add_argument('--k', type=int, default=5000)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    os.makedirs(_OUT_DIR, exist_ok=True)
    stats_path = os.path.join(_OUT_DIR, 'cluster_stats.json')

    # Load existing stats if not force
    all_stats = {}
    if os.path.exists(stats_path) and not args.force:
        all_stats = json.load(open(stats_path))

    all_sizes = {}
    for teacher in args.teachers:
        feat_path = os.path.join(_FEAT_DIR, f'{teacher}.npy')
        if not os.path.exists(feat_path):
            log.warning(f'No features for {teacher}, skip')
            continue

        cache = os.path.join(_OUT_DIR, f'sizes_{teacher}.npy')
        if os.path.exists(cache) and not args.force:
            log.info(f'Loading cached sizes: {teacher}')
            sizes = np.load(cache)
        else:
            log.info(f'Clustering {teacher} (K={args.k})...')
            t0 = time.time()
            sizes = cluster_and_stats(feat_path, args.k, args.device)
            np.save(cache, sizes)
            log.info(f'  done in {time.time()-t0:.0f}s, {len(sizes)} non-empty clusters')

        all_sizes[teacher] = sizes
        stats = compute_stats(sizes)
        all_stats[teacher] = stats
        log.info(f'  {teacher}: Gini={stats["gini"]:.3f}, CoV={stats["cov"]:.2f}, '
                 f'max/min={stats["max_min_ratio"]:.0f}x, P90/P10={stats["p90_p10_ratio"]:.1f}x')

        # Per-teacher plot
        plot_per_teacher(sizes, teacher, _OUT_DIR)

    # Summary Lorenz curve
    if len(all_sizes) > 1:
        plot_lorenz_all(all_sizes, _OUT_DIR)

    # Save stats
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    log.info(f'Stats saved: {stats_path}')
    log.info(f'Plots saved: {_OUT_DIR}/')


if __name__ == '__main__':
    main()
