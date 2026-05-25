"""Curriculum learning: 每 epoch 开始前按特征空间度量重排样本顺序。

支持 5 种策略:
  fps            — Farthest Point Sampling (最大化多样性)
  density_high   — kNN 密度排序, 高密度优先 (简单样本优先)
  density_low    — kNN 密度排序, 低密度优先 (困难样本优先)
  curvature_high — kNN 曲率排序, 高曲率优先 (决策边界优先)
  curvature_low  — kNN 曲率排序, 低曲率优先 (平坦区域优先)

Epoch 0 特征源可选: dinov3 / pe_core / self (当前模型)
Epoch 1+ 始终用当前模型 checkpoint.

所有排序计算均使用 GPU 加速 (torch.cdist kNN, GPU FPS)。
"""
import logging
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import Sampler, DataLoader

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_TIMM = f'{_BASE}/models/timm'
_DINOV3_DIR = f'{_BASE}/models/dino/dinov3-vitb16-pretrain-lvd1689m'
_PE_CORE_CKPT = f'{_TIMM}/PE-Core-B-16/open_clip_model.safetensors'
_EXTERNAL_CLIPS = {
    'pe_core_always': ('PE-Core-B-16', _PE_CORE_CKPT),
    'siglip2': ('ViT-B-16-SigLIP2', f'{_TIMM}/ViT-B-16-SigLIP2/open_clip_model.safetensors'),
    'datacomp': ('ViT-B-16', f'{_TIMM}/DataComp-XL-B-16/open_clip_pytorch_model.bin'),
    'dfn2b': ('ViT-B-16', f'{_TIMM}/DFN2B-ViT-B-16/open_clip_pytorch_model.bin'),
    'eva02': ('EVA02-B-16', f'{_TIMM}/EVA02-B-16/open_clip_model.safetensors'),
    'laion2b': ('ViT-B-16', f'{_TIMM}/LAION2B-B-16/open_clip_model.safetensors'),
    'metaclip': ('ViT-B-16-quickgelu', f'{_TIMM}/MetaCLIP-FullCC-B-16/open_clip_model.safetensors'),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Sampler
# ═══════════════════════════════════════════════════════════════════════════════

class OrderedDistributedSampler(Sampler):
    """按预计算顺序分发索引的分布式 Sampler (连续块划分)。"""

    def __init__(self, ordered_indices, num_replicas, rank, drop_last=True):
        self.num_replicas = num_replicas
        self.rank = rank
        total = len(ordered_indices)
        self.num_samples = total // num_replicas if drop_last else -(-total // num_replicas)
        self.total_size = self.num_samples * num_replicas
        idx = list(ordered_indices)
        if len(idx) < self.total_size:
            idx += idx[:self.total_size - len(idx)]
        self.indices = idx[:self.total_size]

    def __iter__(self):
        start = self.rank * self.num_samples
        return iter(self.indices[start:start + self.num_samples])

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 特征提取 (仅 rank 0 执行)
# ═══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _extract_with_dinov3(paths, preprocess, device):
    from transformers import AutoModel
    from open_clip_train.probe_hook import _ImgDataset
    logging.info('[curriculum] Loading DINOv3 for epoch-0 features...')
    model = AutoModel.from_pretrained(_DINOV3_DIR, trust_remote_code=True).eval().to(device)
    dl = DataLoader(_ImgDataset(paths, preprocess), batch_size=256, num_workers=4, pin_memory=True)
    feats = []
    for imgs, _ in dl:
        h = model(imgs.to(device)).last_hidden_state.float()
        feats.append(h[:, 0].cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats, 0).numpy()


@torch.no_grad()
def _extract_with_open_clip(init_name, paths, device):
    import os
    import open_clip
    from open_clip_train.probe_hook import _ImgDataset

    if init_name == 'pe_core':
        model_name, ckpt = 'PE-Core-B-16', _PE_CORE_CKPT
    elif init_name == 'random_init':
        model_name, ckpt = 'ViT-B-16', None
    else:
        model_name, ckpt = _EXTERNAL_CLIPS[init_name]
    if ckpt is not None and not os.path.exists(ckpt):
        raise FileNotFoundError(f'[curriculum] checkpoint not found for {init_name}: {ckpt}')

    logging.info(f'[curriculum] Loading external CLIP features: {init_name} ({model_name})')
    model, _, preproc = open_clip.create_model_and_transforms(model_name, pretrained=ckpt)
    if init_name == 'random_init':
        torch.manual_seed(0)
        for p in model.visual.parameters():
            if p.ndim > 1:
                torch.nn.init.xavier_uniform_(p)
            else:
                torch.nn.init.zeros_(p)
    model = model.eval().to(device)
    vis = model.visual
    use_trunk = hasattr(vis, 'trunk') and hasattr(vis.trunk, 'forward_features')
    dl = DataLoader(_ImgDataset(paths, preproc), batch_size=256, num_workers=4, pin_memory=True)
    feats = []
    for imgs, _ in dl:
        imgs = imgs.to(device)
        if use_trunk:
            out = vis.trunk.forward_features(imgs)
            cls = out[:, 0]
        else:
            proj_bak = getattr(vis, 'proj', None)
            if hasattr(vis, 'proj'):
                vis.proj = None
            cls = model.encode_image(imgs, normalize=False)
            if hasattr(vis, 'proj'):
                vis.proj = proj_bak
        feats.append(cls.float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(feats, 0).numpy()


def _extract_with_pe_core(paths, preprocess, device):
    return _extract_with_open_clip('pe_core', paths, device)


@torch.no_grad()
def _extract_with_self(model, paths, preprocess, device):
    """用当前训练模型提取 backbone CLS (多 worker 加速)。"""
    from open_clip_train.probe_hook import _ImgDataset
    m = model.module if hasattr(model, 'module') else model
    if hasattr(m, 'clip_model'):
        m = m.clip_model
    visual = getattr(m, 'visual', None)
    has_trunk = (visual is not None and hasattr(visual, 'trunk')
                 and hasattr(visual.trunk, 'forward_features'))
    dl = DataLoader(_ImgDataset(paths, preprocess), batch_size=256, num_workers=8, pin_memory=True)
    m.eval()
    feats = []
    for imgs, _ in dl:
        if has_trunk:
            out = visual.trunk.forward_features(imgs.to(device))
            feats.append(out[:, 0].float().cpu())
        else:
            feats.append(m.encode_image(imgs.to(device), normalize=True).float().cpu())
    m.train()
    return torch.cat(feats, 0).numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# GPU 加速排序策略
# ═══════════════════════════════════════════════════════════════════════════════

def _gpu_knn(feats_t, K, device):
    """GPU batched kNN: 返回 (N, K) 距离和索引。"""
    N = feats_t.shape[0]
    chunk = 4096
    all_dists = torch.empty(N, K, device=device)
    all_idx = torch.empty(N, K, dtype=torch.long, device=device)
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        d = torch.cdist(feats_t[i:end], feats_t)  # (chunk, N)
        # 排除自身
        d[torch.arange(end - i, device=device), torch.arange(i, end, device=device)] = float('inf')
        topk = d.topk(K, dim=1, largest=False)
        all_dists[i:end] = topk.values
        all_idx[i:end] = topk.indices
    return all_dists, all_idx


def _gpu_fps(feats_t, device):
    """分层 FPS: 随机分组→质心 FPS→按质心顺序展开。O(N + K^2), K=n_clusters。"""
    N = feats_t.shape[0]
    n_clusters = min(2000, N)
    cluster_size = N // n_clusters

    perm = torch.randperm(N, device=device)
    # 计算每组质心
    # reshape 为 (n_clusters, cluster_size, D) 取 mean — 尾部余数归入最后一组
    main = feats_t[perm[:n_clusters * cluster_size]].view(n_clusters, cluster_size, -1)
    centroids = main.mean(dim=1)  # (n_clusters, D)

    # 对质心做精确 FPS
    chosen_c = torch.empty(n_clusters, dtype=torch.long, device=device)
    chosen_c[0] = 0
    dists_c = torch.full((n_clusters,), float('inf'), device=device)
    for step in range(1, n_clusters):
        d = torch.sum((centroids - centroids[chosen_c[step - 1]]) ** 2, dim=1)
        dists_c = torch.minimum(dists_c, d)
        chosen_c[step] = torch.argmax(dists_c)

    # 按质心 FPS 顺序展开各组
    order = []
    for ci in chosen_c:
        start = int(ci) * cluster_size
        end = start + cluster_size
        order.append(perm[start:end])
    # 尾部余数
    if n_clusters * cluster_size < N:
        order.append(perm[n_clusters * cluster_size:])
    return torch.cat(order).cpu().numpy()


def _gpu_density(feats_t, K, device):
    """kNN 密度: 1 / mean_knn_distance (GPU)。返回 numpy。"""
    dists, _ = _gpu_knn(feats_t, K, device)
    density = 1.0 / (dists.mean(dim=1) + 1e-10)
    return density.cpu().numpy()


def _gpu_curvature(feats_t, K, device):
    """kNN 曲率: 1 - λ_max/Σλ (power iteration on Gram matrix, GPU)。返回 numpy。"""
    _, indices = _gpu_knn(feats_t, K, device)
    N = feats_t.shape[0]
    curvature = torch.empty(N, device=device)
    chunk = 8192
    for i in range(0, N, chunk):
        end = min(i + chunk, N)
        nbrs = feats_t[indices[i:end]]  # (batch, K, D)
        nbrs_c = nbrs - nbrs.mean(dim=1, keepdim=True)
        total_var = (nbrs_c ** 2).sum(dim=(1, 2))  # Σλ = ||X_c||_F^2
        gram = torch.bmm(nbrs_c, nbrs_c.transpose(1, 2))  # (batch, K, K)
        # Power iteration (5 steps) 估计 λ_max
        v = torch.randn(end - i, K, 1, device=device)
        for _ in range(5):
            v = torch.bmm(gram, v)
            v = v / (v.norm(dim=1, keepdim=True) + 1e-10)
        lam_max = (torch.bmm(gram, v) * v).sum(dim=(1, 2))
        curvature[i:end] = 1.0 - lam_max / (total_var + 1e-10)
    return curvature.cpu().numpy()


def _compute_order_tensor(feats_t, strategy, k, device):
    """按策略计算排序, feats_t 已在 GPU。返回 int64 numpy。"""
    if strategy == 'fps':
        order = _gpu_fps(feats_t, device)
    elif strategy == 'fps_reverse':
        order = _gpu_fps(feats_t, device)[::-1].copy()
    elif strategy == 'density_high':
        order = np.argsort(-_gpu_density(feats_t, k, device))
    elif strategy == 'density_low':
        order = np.argsort(_gpu_density(feats_t, k, device))
    elif strategy == 'curvature_high':
        order = np.argsort(-_gpu_curvature(feats_t, k, device))
    elif strategy == 'curvature_low':
        order = np.argsort(_gpu_curvature(feats_t, k, device))
    else:
        raise ValueError(f'Unknown curriculum strategy: {strategy}')
    return order.astype(np.int64)


def compute_curriculum_order(features_np, strategy, k, device):
    """按策略计算排序 (GPU 加速), 返回有序索引 numpy 数组。"""
    logging.info(f'[curriculum] Computing order: strategy={strategy}, N={len(features_np)}, K={k}')
    t0 = time.time()
    feats_t = torch.from_numpy(features_np).to(device=device, dtype=torch.float32)
    order = _compute_order_tensor(feats_t, strategy, k, device)
    del feats_t
    torch.cuda.empty_cache()
    logging.info(f'[curriculum] Order computed in {time.time()-t0:.1f}s')
    return order


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

def _save_base_loader(train_info):
    if not hasattr(train_info, '_curriculum_base_dataloader'):
        train_info._curriculum_base_dataloader = train_info.dataloader
        train_info._curriculum_base_sampler = train_info.sampler


def restore_default_order(data, args, epoch=None):
    """恢复 curriculum 介入前的随机/DistributedSampler DataLoader。"""
    train_info = data.get('train') if isinstance(data, dict) else None
    if train_info is None or not hasattr(train_info, '_curriculum_base_dataloader'):
        return
    if train_info.dataloader is not train_info._curriculum_base_dataloader:
        train_info.dataloader = train_info._curriculum_base_dataloader
        train_info.sampler = train_info._curriculum_base_sampler
        if getattr(args, 'rank', 0) == 0:
            logging.info(f'[curriculum] Restored default random sampler at epoch {epoch}')


def apply_curriculum(model, data, epoch, args, preprocess_val, device):
    """Epoch 开始前: 提取特征 → 计算排序 → 替换 DataLoader/shard 顺序。

    - CSV 数据集: 样本级精确排序 (替换 sampler)
    - WebDataset: shard 级近似排序 (替换 shard list 顺序, 禁用 shuffle)
    """
    if not getattr(args, 'curriculum_strategy', None):
        return
    train_info = data['train']

    if args.dataset_type == 'webdataset':
        _apply_curriculum_wds(model, data, epoch, args, preprocess_val, device)
    else:
        _apply_curriculum_csv(model, data, epoch, args, preprocess_val, device)


def _rank_range(n, world_size, rank):
    start = (n * rank) // world_size
    end = (n * (rank + 1)) // world_size
    return start, end


def _extract_feature_block(model, paths, preprocess, device, args, epoch):
    init = args.curriculum_init
    if init == 'dinov3' and epoch == 0:
        return _extract_with_dinov3(paths, preprocess, device)
    if init == 'pe_core' and epoch == 0:
        return _extract_with_pe_core(paths, preprocess, device)
    if init in _EXTERNAL_CLIPS or init == 'random_init':
        return _extract_with_open_clip(init, paths, device)
    return _extract_with_self(model, paths, preprocess, device)


def _apply_curriculum_csv(model, data, epoch, args, preprocess_val, device):
    """CSV 数据集: 样本级精确排序。分布式提特征，rank0 排序。"""
    train_info = data['train']
    _save_base_loader(train_info)
    dataset = train_info._curriculum_base_dataloader.dataset
    paths = dataset.images
    N = len(paths)

    start, end = _rank_range(N, args.world_size, args.rank)
    local_paths = paths[start:end]
    if args.rank == 0:
        logging.info(f'[curriculum] Epoch {epoch}: extracting features ({N} samples, distributed) ...')
    t0 = time.time()
    local_np = _extract_feature_block(model, local_paths, preprocess_val, device, args, epoch)
    local = torch.from_numpy(local_np).to(device=device, dtype=torch.float16)
    local_n = torch.tensor([local.shape[0]], device=device, dtype=torch.long)

    if args.distributed:
        sizes = [torch.zeros_like(local_n) for _ in range(args.world_size)]
        dist.all_gather(sizes, local_n)
        sizes = [int(s.item()) for s in sizes]
        max_n = max(sizes)
        if local.shape[0] < max_n:
            pad = torch.zeros(max_n - local.shape[0], local.shape[1], device=device, dtype=local.dtype)
            local = torch.cat([local, pad], dim=0)
        gathered = [torch.empty_like(local) for _ in range(args.world_size)] if args.rank == 0 else None
        dist.gather(local, gather_list=gathered, dst=0)
        if args.rank == 0:
            features_t = torch.cat([g[:sizes[i]] for i, g in enumerate(gathered)], dim=0).float()
            logging.info(f'[curriculum] Features gathered in {time.time()-t0:.1f}s, shape={tuple(features_t.shape)}')
            t1 = time.time()
            ordered = _compute_order_tensor(features_t, args.curriculum_strategy, args.curriculum_k, device)
            logging.info(f'[curriculum] Order computed in {time.time()-t1:.1f}s')
            del features_t, gathered
        else:
            ordered = np.empty(N, dtype=np.int64)
        del local
    else:
        features_t = local.float()
        ordered = _compute_order_tensor(features_t, args.curriculum_strategy, args.curriculum_k, device)
        del features_t, local

    torch.cuda.empty_cache()

    if args.distributed:
        t = torch.from_numpy(ordered).to(device)
        dist.broadcast(t, src=0)
        ordered = t.cpu().numpy()

    sampler = OrderedDistributedSampler(
        ordered.tolist(), num_replicas=args.world_size, rank=args.rank, drop_last=True)
    old_dl = train_info._curriculum_base_dataloader
    new_dl = DataLoader(
        dataset, batch_size=old_dl.batch_size, num_workers=old_dl.num_workers,
        pin_memory=True, sampler=sampler, drop_last=True)
    new_dl.num_samples = old_dl.num_samples
    new_dl.num_batches = len(new_dl)
    train_info.dataloader = new_dl
    train_info.sampler = sampler
    if args.rank == 0:
        logging.info(f'[curriculum] Epoch {epoch}: {new_dl.num_batches} batches, '
                     f'first 5 indices={ordered[:5].tolist()}')


@torch.no_grad()
def _extract_shard_centroids(shard_urls, preprocess, model_or_mode, device, args):
    """对每个 shard 采样 N 张图提取特征, 返回 (n_shards, D) 质心矩阵。"""
    import tarfile, io
    from PIL import Image

    n_per_shard = 32  # 每 shard 采样 32 张算质心

    # 加载外部模型 (如需要)
    ext_model = None
    if model_or_mode == 'dinov3':
        from transformers import AutoModel
        ext_model = AutoModel.from_pretrained(_DINOV3_DIR, trust_remote_code=True).eval().to(device)
    elif model_or_mode == 'pe_core':
        import open_clip
        ext_model, _, _ = open_clip.create_model_and_transforms('PE-Core-B-16', pretrained=_PE_CORE_CKPT)
        ext_model = ext_model.eval().to(device)

    centroids = []
    for si, url in enumerate(shard_urls):
        # 直接从 tar 读取前 n_per_shard 张图
        imgs = []
        try:
            with tarfile.open(url) as tf:
                for member in tf:
                    if member.name.endswith(('.jpg', '.png', '.jpeg', '.webp')):
                        f = tf.extractfile(member)
                        if f:
                            img = Image.open(io.BytesIO(f.read())).convert('RGB')
                            imgs.append(preprocess(img))
                            if len(imgs) >= n_per_shard:
                                break
        except Exception:
            pass

        if not imgs:
            centroids.append(torch.zeros(768, device=device))
            continue

        batch = torch.stack(imgs).to(device)
        if model_or_mode == 'dinov3':
            h = ext_model(batch).last_hidden_state.float()
            feat = h[:, 0].mean(dim=0)
        elif model_or_mode == 'pe_core':
            out = ext_model.visual.trunk.forward_features(batch)
            feat = out[:, 0].float().mean(dim=0)
        else:  # self (model object)
            m = model_or_mode
            visual = getattr(m, 'visual', None)
            if visual and hasattr(visual, 'trunk'):
                out = visual.trunk.forward_features(batch)
                feat = out[:, 0].float().mean(dim=0)
            else:
                feat = m.encode_image(batch, normalize=True).float().mean(dim=0)
        centroids.append(feat)

        if (si + 1) % 100 == 0:
            logging.info(f'[curriculum] shard centroids: {si+1}/{len(shard_urls)}')

    if ext_model is not None:
        del ext_model
        torch.cuda.empty_cache()

    return torch.stack(centroids)  # (n_shards, D)


def _apply_curriculum_wds(model, data, epoch, args, preprocess_val, device):
    """WebDataset: shard 级排序。修改 pipeline 中 SimpleShardList.urls 为排序后的列表。"""
    import braceexpand
    train_info = data['train']

    shard_urls = sorted(braceexpand.braceexpand(args.train_data))
    n_shards = len(shard_urls)

    if args.rank == 0:
        logging.info(f'[curriculum] WDS epoch {epoch}: computing shard order ({n_shards} shards)...')
        t0 = time.time()

        if epoch == 0 and args.curriculum_init == 'dinov3':
            mode = 'dinov3'
        elif epoch == 0 and args.curriculum_init == 'pe_core':
            mode = 'pe_core'
        else:
            m = model.module if hasattr(model, 'module') else model
            if hasattr(m, 'clip_model'):
                m = m.clip_model
            mode = m

        centroids = _extract_shard_centroids(shard_urls, preprocess_val, mode, device, args)
        shard_order = _compute_shard_order(centroids, args.curriculum_strategy, device)
        logging.info(f'[curriculum] WDS shard order computed in {time.time()-t0:.1f}s, '
                     f'first 5 shards={shard_order[:5].tolist()}')
    else:
        shard_order = np.empty(n_shards, dtype=np.int64)

    if args.distributed:
        t = torch.from_numpy(shard_order.astype(np.int64)).to(device)
        dist.broadcast(t, src=0)
        shard_order = t.cpu().numpy()

    # 替换 pipeline 中 SimpleShardList 的 urls, 并禁用 shard/sample shuffle
    ordered_urls = [shard_urls[i] for i in shard_order]
    inner_pipeline = train_info.dataloader.pipeline[0].dataset.pipeline
    for stage in inner_pipeline:
        if hasattr(stage, 'urls'):
            stage.urls = ordered_urls
        # 禁用 detshuffle2 (shard shuffle) — 设 bufsize=1 使其 pass-through
        if hasattr(stage, 'bufsize') and hasattr(stage, 'epoch'):
            stage.bufsize = 1
            stage.initial = 1

    if args.rank == 0:
        logging.info(f'[curriculum] WDS epoch {epoch}: shard order applied')


def _compute_shard_order(centroids, strategy, device):
    """对 shard 质心计算排序。"""
    n = centroids.shape[0]
    feats_t = centroids.to(device=device, dtype=torch.float32)

    if strategy in ('fps', 'fps_reverse'):
        # 对 576 个质心做精确 FPS (O(576^2) = 瞬间)
        chosen = torch.empty(n, dtype=torch.long, device=device)
        chosen[0] = 0
        dists = torch.full((n,), float('inf'), device=device)
        for step in range(1, n):
            d = torch.sum((feats_t - feats_t[chosen[step - 1]]) ** 2, dim=1)
            dists = torch.minimum(dists, d)
            chosen[step] = torch.argmax(dists)
        order = chosen.cpu().numpy()
        if strategy == 'fps_reverse':
            order = order[::-1].copy()
    elif strategy == 'density_high':
        dists, _ = _gpu_knn(feats_t, min(50, n - 1), device)
        density = 1.0 / (dists.mean(dim=1) + 1e-10)
        order = torch.argsort(density, descending=True).cpu().numpy()
    elif strategy == 'density_low':
        dists, _ = _gpu_knn(feats_t, min(50, n - 1), device)
        density = 1.0 / (dists.mean(dim=1) + 1e-10)
        order = torch.argsort(density).cpu().numpy()
    elif strategy == 'curvature_high':
        curv = torch.from_numpy(_gpu_curvature(feats_t, min(50, n - 1), device)).to(device)
        order = torch.argsort(curv, descending=True).cpu().numpy()
    elif strategy == 'curvature_low':
        curv = torch.from_numpy(_gpu_curvature(feats_t, min(50, n - 1), device)).to(device)
        order = torch.argsort(curv).cpu().numpy()
    else:
        raise ValueError(f'Unknown strategy: {strategy}')
    return order.astype(np.int64)
