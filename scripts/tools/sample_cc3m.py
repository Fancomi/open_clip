#!/usr/bin/env python3
"""从 CC3M 中采样 80K 子集: FPS (多样性最大化) 或 K-Means (均匀覆盖)。

用法:
    python scripts/tools/sample_cc3m.py --teacher siglip2 --method fps --n-samples 80000
    python scripts/tools/sample_cc3m.py --teacher eva02 --method kmeans --n-samples 80000
    python scripts/tools/sample_cc3m.py --method random --n-samples 80000
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from open_clip_train.curriculum import (
    _gpu_fps, _EXTERNAL_CLIPS, _DINOV3_DIR, _PE_CORE_CKPT,
)
from open_clip_train.probe_hook import _ImgDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_CC3M_TSV = f'{_BASE}/datas/cc3m-tsv/annotations/clip_train.tsv'
_FEAT_DIR = f'{_BASE}/datas/cc3m-tsv/features'
_OUT_DIR = f'{_BASE}/datas/cc3m-tsv/subsets'

TEACHERS = ['pe_core', 'dinov3', 'siglip2', 'datacomp', 'dfn2b', 'eva02', 'laion2b', 'metaclip']


def read_cc3m_tsv():
    """读取 CC3M TSV，返回 (paths, captions) 列表。"""
    paths, captions = [], []
    with open(_CC3M_TSV) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                paths.append(parts[0])
                captions.append(parts[1])
    log.info(f'CC3M: {len(paths)} samples loaded')
    return paths, captions


@torch.no_grad()
def extract_features(teacher, paths, device='cuda:0', num_workers=16):
    """高效特征提取: 多 workers + batch 512 + 进度日志 + 磁盘缓存。"""
    os.makedirs(_FEAT_DIR, exist_ok=True)
    cache = os.path.join(_FEAT_DIR, f'{teacher}.npy')
    if os.path.exists(cache):
        log.info(f'Loading cached features: {cache}')
        return np.load(cache)

    log.info(f'Extracting features: teacher={teacher}, N={len(paths)}')
    t0 = time.time()

    if teacher == 'dinov3':
        from transformers import AutoModel
        from torchvision import transforms
        model = AutoModel.from_pretrained(_DINOV3_DIR, trust_remote_code=True).eval().to(device)
        preproc = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        def forward_fn(imgs):
            return model(imgs).last_hidden_state[:, 0].float()
    else:
        import open_clip
        if teacher == 'pe_core':
            model_name, ckpt = 'PE-Core-B-16', _PE_CORE_CKPT
        else:
            model_name, ckpt = _EXTERNAL_CLIPS[teacher]
        model, _, preproc = open_clip.create_model_and_transforms(model_name, pretrained=ckpt)
        model = model.eval().to(device)
        vis = model.visual
        use_trunk = hasattr(vis, 'trunk') and hasattr(vis.trunk, 'forward_features')

        def forward_fn(imgs):
            if use_trunk:
                return vis.trunk.forward_features(imgs)[:, 0].float()
            proj_bak = getattr(vis, 'proj', None)
            if hasattr(vis, 'proj'):
                vis.proj = None
            out = model.encode_image(imgs, normalize=False).float()
            if hasattr(vis, 'proj'):
                vis.proj = proj_bak
            return out

    dl = DataLoader(_ImgDataset(paths, preproc), batch_size=512,
                    num_workers=num_workers, pin_memory=True, prefetch_factor=4)
    feats = []
    total = len(dl)
    for i, (imgs, _) in enumerate(dl):
        feats.append(forward_fn(imgs.to(device)).cpu())
        if (i + 1) % 200 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) * 512 / elapsed
            eta = (total - i - 1) * 512 / rate
            log.info(f'  [{teacher}] {i+1}/{total} batches, {rate:.0f} img/s, ETA {eta/60:.1f}min')

    del model
    torch.cuda.empty_cache()
    feats_np = torch.cat(feats, 0).numpy().astype(np.float16)
    np.save(cache, feats_np)
    log.info(f'Features saved: {cache} ({feats_np.shape}, {time.time()-t0:.0f}s)')
    return feats_np


def sample_fps(features, n_samples, device='cuda:0'):
    """FPS 采样: 取 FPS 排序的前 n_samples 个索引。"""
    log.info(f'FPS sampling: {len(features)} -> {n_samples}')
    feats_t = torch.from_numpy(features.astype(np.float32)).to(device)
    order = _gpu_fps(feats_t, device)
    del feats_t
    torch.cuda.empty_cache()
    return order[:n_samples]


def sample_kmeans(features, n_samples, device='cuda:0', n_iter=30, batch_size=65536):
    """GPU K-Means++ 分层采样 (文献最优方案)。

    方法: K-Means++ 初始化 → 迭代收敛 → 按簇大小比例分配名额 + 簇内随机采样。
    参考: Meta FAIR 2024 "Automatic Data Curation" — random from balanced clusters 最优。
    """
    N, D = features.shape
    K_base = min(5000, n_samples // 10)
    log.info(f'K-Means++ stratified: N={N}, K={K_base} clusters, target={n_samples}')
    t0 = time.time()

    feats = torch.from_numpy(features.astype(np.float32)).to(device)
    feats = torch.nn.functional.normalize(feats, dim=1)

    # ── K-Means++ 初始化 ──
    centroids = torch.empty(K_base, D, device=device)
    centroids[0] = feats[torch.randint(N, (1,), device=device)]
    min_dist = torch.full((N,), float('inf'), device=device)

    for k in range(1, K_base):
        # 更新最小距离 (用 1 - cos_sim 作为距离)
        sim = feats @ centroids[k - 1].unsqueeze(1)  # (N, 1)
        dist = 1 - sim.squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
        # 按距离^2 概率采样下一个质心
        probs = min_dist ** 2
        probs /= probs.sum()
        idx = torch.multinomial(probs, 1).item()
        centroids[k] = feats[idx]
        if (k + 1) % 1000 == 0:
            log.info(f'  K-Means++ init: {k+1}/{K_base}')

    log.info(f'  K-Means++ init done in {time.time()-t0:.0f}s')

    # ── 迭代优化 ──
    for it in range(n_iter):
        assignments = torch.empty(N, dtype=torch.long, device=device)
        chunk = 32768
        for i in range(0, N, chunk):
            end = min(i + chunk, N)
            sim = feats[i:end] @ centroids.T
            assignments[i:end] = sim.argmax(dim=1)

        # Mini-batch 质心更新
        if batch_size < N:
            idx = torch.randperm(N, device=device)[:batch_size]
            sub_feats, sub_assign = feats[idx], assignments[idx]
        else:
            sub_feats, sub_assign = feats, assignments

        new_centroids = torch.zeros_like(centroids)
        counts = torch.zeros(K_base, device=device)
        new_centroids.scatter_add_(0, sub_assign.unsqueeze(1).expand_as(sub_feats), sub_feats)
        counts.scatter_add_(0, sub_assign, torch.ones(len(sub_assign), device=device))

        valid = counts > 0
        new_centroids[valid] /= counts[valid].unsqueeze(1)
        new_centroids[~valid] = centroids[~valid]
        centroids = torch.nn.functional.normalize(new_centroids, dim=1)

        if (it + 1) % 10 == 0:
            log.info(f'  iter {it+1}/{n_iter}, empty: {(~valid).sum().item()}/{K_base}')

    # ── 最终分配 ──
    assignments = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, 32768):
        end = min(i + 32768, N)
        assignments[i:end] = (feats[i:end] @ centroids.T).argmax(dim=1)

    # ── 分层随机采样: 按簇大小比例分配, 簇内随机 ──
    log.info('K-Means: stratified random sampling...')
    assignments_cpu = assignments.cpu().numpy()
    cluster_sizes = np.bincount(assignments_cpu, minlength=K_base)
    nonzero = cluster_sizes > 0
    log.info(f'  non-empty clusters: {nonzero.sum()}/{K_base}')

    # 按比例分配名额 (每个非空簇至少 1)
    total_valid = cluster_sizes[nonzero].sum()
    quotas = np.zeros(K_base, dtype=int)
    quotas[nonzero] = np.maximum(1, (cluster_sizes[nonzero] / total_valid * n_samples).astype(int))
    # 调整总数
    diff = quotas.sum() - n_samples
    if diff > 0:
        large_clusters = np.argsort(-quotas)
        for i in range(diff):
            quotas[large_clusters[i % len(large_clusters)]] -= 1
    elif diff < 0:
        large_clusters = np.argsort(-cluster_sizes)
        for i in range(-diff):
            quotas[large_clusters[i % len(large_clusters)]] += 1

    rng = np.random.default_rng(42)
    selected = []
    for c in range(K_base):
        if cluster_sizes[c] == 0 or quotas[c] == 0:
            continue
        members = np.where(assignments_cpu == c)[0]
        k = min(quotas[c], len(members))
        selected.append(rng.choice(members, size=k, replace=False))

    selected = np.concatenate(selected)[:n_samples]
    del feats, centroids
    torch.cuda.empty_cache()
    log.info(f'K-Means done in {time.time()-t0:.0f}s, selected {len(selected)}')
    return selected


def sample_random(n_total, n_samples):
    """随机采样。"""
    rng = np.random.default_rng(42)
    return rng.choice(n_total, size=n_samples, replace=False)


def sample_kmeans_uniform(features, n_samples, device='cuda:0', n_iter=30):
    """对齐 Meta FAIR 2024 (arxiv:2405.15613) 的均匀簇采样。

    核心差异 vs sample_kmeans:
      - 簇间均匀分配 (每簇等量) 而非按比例分配
      - K 值按 n_samples 决定: K = n_samples / 10 (每簇贡献 ~10 样本)
    FAIR 结论: uniform from balanced clusters > proportional > nearest-centroid
    """
    N, D = features.shape
    K = min(n_samples // 10, N // 20)  # 每簇 ~10 样本, 至少每簇 20 成员
    per_cluster = max(1, n_samples // K)
    log.info(f'K-Means uniform: N={N}, K={K}, {per_cluster}/cluster, target={n_samples}')
    t0 = time.time()

    feats = torch.from_numpy(features.astype(np.float32)).to(device)
    feats = torch.nn.functional.normalize(feats, dim=1)

    # K-Means++ 初始化
    centroids = torch.empty(K, D, device=device)
    centroids[0] = feats[torch.randint(N, (1,), device=device)]
    min_dist = torch.full((N,), float('inf'), device=device)
    for k in range(1, K):
        sim = feats @ centroids[k - 1].unsqueeze(1)
        min_dist = torch.minimum(min_dist, 1 - sim.squeeze(1))
        probs = min_dist ** 2
        probs /= probs.sum()
        centroids[k] = feats[torch.multinomial(probs, 1).item()]
        if (k + 1) % 2000 == 0:
            log.info(f'  init: {k+1}/{K}')
    log.info(f'  init done in {time.time()-t0:.0f}s')

    # 迭代
    chunk = 32768
    for it in range(n_iter):
        assignments = torch.empty(N, dtype=torch.long, device=device)
        for i in range(0, N, chunk):
            assignments[i:min(i+chunk, N)] = (feats[i:min(i+chunk, N)] @ centroids.T).argmax(1)
        new_c = torch.zeros_like(centroids)
        counts = torch.zeros(K, device=device)
        new_c.scatter_add_(0, assignments.unsqueeze(1).expand(-1, D), feats)
        counts.scatter_add_(0, assignments, torch.ones(N, device=device))
        valid = counts > 0
        new_c[valid] /= counts[valid].unsqueeze(1)
        new_c[~valid] = centroids[~valid]
        centroids = torch.nn.functional.normalize(new_c, dim=1)
        if (it + 1) % 10 == 0:
            log.info(f'  iter {it+1}/{n_iter}, empty: {(~valid).sum().item()}/{K}')

    # 最终分配
    assignments = torch.empty(N, dtype=torch.long, device=device)
    for i in range(0, N, chunk):
        assignments[i:min(i+chunk, N)] = (feats[i:min(i+chunk, N)] @ centroids.T).argmax(1)

    # 均匀采样: 每簇取相同数量 (FAIR 核心方法)
    assignments_cpu = assignments.cpu().numpy()
    cluster_sizes = np.bincount(assignments_cpu, minlength=K)
    nonzero_ids = np.where(cluster_sizes > 0)[0]
    log.info(f'  non-empty: {len(nonzero_ids)}/{K}')

    rng = np.random.default_rng(42)
    selected = []
    quota = max(1, n_samples // len(nonzero_ids))  # 每簇均匀分配
    for c in nonzero_ids:
        members = np.where(assignments_cpu == c)[0]
        k = min(quota, len(members))
        selected.append(rng.choice(members, size=k, replace=False))

    selected = np.concatenate(selected)
    # 如果超出 n_samples 则随机截断
    if len(selected) > n_samples:
        selected = rng.choice(selected, size=n_samples, replace=False)
    elif len(selected) < n_samples:
        # 不够则从大簇补充
        remaining = n_samples - len(selected)
        sel_set = set(selected.tolist())
        pool = np.array([i for i in range(N) if i not in sel_set])
        selected = np.concatenate([selected, rng.choice(pool, size=remaining, replace=False)])

    del feats, centroids
    torch.cuda.empty_cache()
    log.info(f'K-Means uniform done in {time.time()-t0:.0f}s, selected {len(selected)}')
    return selected


def write_tsv(paths, captions, indices, output_path):
    """写出子集 TSV。"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('filepath\tcaption\n')
        for i in indices:
            f.write(f'{paths[i]}\t{captions[i]}\n')
    log.info(f'Written {len(indices)} samples -> {output_path}')


def main():
    parser = argparse.ArgumentParser(description='CC3M subset sampling')
    parser.add_argument('--teacher', choices=TEACHERS + ['random'], required=True)
    parser.add_argument('--method', choices=['fps', 'kmeans', 'random'], required=True)
    parser.add_argument('--n-samples', type=int, default=80000)
    parser.add_argument('--max-images', type=int, default=0,
                        help='Limit source images for smoke test (0=all)')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', help='Output TSV path (auto-generated if not specified)')
    args = parser.parse_args()

    if args.method == 'random' and args.teacher != 'random':
        parser.error('--method random requires --teacher random')

    paths, captions = read_cc3m_tsv()
    if args.max_images > 0:
        paths, captions = paths[:args.max_images], captions[:args.max_images]
        log.info(f'Smoke mode: limited to {len(paths)} images')

    # 输出路径
    if args.output:
        out_path = args.output
    else:
        os.makedirs(_OUT_DIR, exist_ok=True)
        tag = f'{args.method}_{args.teacher}' if args.method != 'random' else 'random'
        out_path = os.path.join(_OUT_DIR, f'{tag}_{args.n_samples // 1000}k.tsv')

    if os.path.exists(out_path):
        log.info(f'Output already exists: {out_path}, skipping')
        return

    if args.method == 'random':
        indices = sample_random(len(paths), args.n_samples)
    else:
        features = extract_features(args.teacher, paths, args.device, args.workers)
        if args.method == 'fps':
            indices = sample_fps(features, args.n_samples, args.device)
        else:
            indices = sample_kmeans(features, args.n_samples, args.device)

    write_tsv(paths, captions, indices, out_path)


if __name__ == '__main__':
    main()
