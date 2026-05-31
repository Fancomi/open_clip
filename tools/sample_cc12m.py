#!/usr/bin/env python3
"""从 CC12M-wds 提取特征并采样子集。

CC12M 是未经精细筛选的 web 图文对 (~11M), 比 CC3M 更脏。
用于验证: 在高噪声 + 低保留率场景下, FPS/K-Means 是否优于 Random。

用法:
    # Phase 1: 提取特征 (单 GPU, ~30min/teacher)
    python tools/sample_cc12m.py extract --teacher siglip2

    # Phase 2: 采样
    python tools/sample_cc12m.py sample --teacher siglip2 --method fps --n-samples 50000

    # Phase 3: 导出选中图片为 TSV (从 tar 中抽取)
    python tools/sample_cc12m.py export --tsv subsets/fps_siglip2_50k.tsv
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from open_clip_train.curriculum import _EXTERNAL_CLIPS, _DINOV3_DIR, _PE_CORE_CKPT

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_CC12M_DIR = f'{_BASE}/datas/cc12m-wds'
_FEAT_DIR = f'{_BASE}/datas/cc12m-wds/features'
_SUBSET_DIR = f'{_BASE}/datas/cc12m-wds/subsets'
_META_PATH = f'{_BASE}/datas/cc12m-wds/metadata.npz'  # paths + captions index

TEACHERS = ['pe_core', 'dinov3', 'siglip2', 'datacomp', 'dfn2b', 'eva02', 'laion2b', 'metaclip']


class WdsImageDataset(IterableDataset):
    """流式读取 CC12M wds tar, 返回 (image_tensor, sample_key)。跳过损坏图片。"""
    def __init__(self, tar_pattern, transform):
        import webdataset as wds
        self.pipeline = (
            wds.WebDataset(tar_pattern, shardshuffle=False, handler=wds.warn_and_continue)
            .decode('pil', handler=wds.warn_and_continue)
            .to_tuple('jpg;png', '__key__', handler=wds.warn_and_continue)
            .map_tuple(transform, lambda x: x)
        )

    def __iter__(self):
        return iter(self.pipeline)


def build_metadata():
    """首次运行: 扫描所有 tar 构建 key→(shard, caption) 索引。"""
    if os.path.exists(_META_PATH):
        log.info(f'Metadata exists: {_META_PATH}')
        return
    import webdataset as wds
    import glob

    tars = sorted(glob.glob(f'{_CC12M_DIR}/cc12m-train-*.tar'))
    log.info(f'Building metadata from {len(tars)} shards...')
    keys, captions, shard_ids = [], [], []
    for si, tar_path in enumerate(tars):
        ds = wds.WebDataset(tar_path, shardshuffle=False).decode('utf-8')
        for sample in ds:
            key = sample['__key__']
            txt = sample.get('.txt', sample.get('txt', ''))
            keys.append(key)
            captions.append(txt)
            shard_ids.append(si)
        if (si + 1) % 100 == 0:
            log.info(f'  scanned {si+1}/{len(tars)} shards, {len(keys)} samples')

    log.info(f'Total: {len(keys)} samples')
    np.savez_compressed(_META_PATH, keys=np.array(keys), captions=np.array(captions),
                        shard_ids=np.array(shard_ids, dtype=np.int16))


@torch.no_grad()
def extract(teacher, device='cuda:0'):
    """流式提取 CC12M 全量特征 → 缓存 npy (float16)。"""
    import glob
    os.makedirs(_FEAT_DIR, exist_ok=True)
    cache = os.path.join(_FEAT_DIR, f'{teacher}.npy')
    if os.path.exists(cache):
        log.info(f'Features exist: {cache}')
        return

    tar_pattern = sorted(glob.glob(f'{_CC12M_DIR}/cc12m-train-*.tar'))
    log.info(f'Extracting {teacher} from {len(tar_pattern)} shards...')
    t0 = time.time()

    # 加载模型
    if teacher == 'dinov3':
        from transformers import AutoModel
        from torchvision import transforms
        model = AutoModel.from_pretrained(_DINOV3_DIR, trust_remote_code=True).eval().to(device)
        preproc = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        def fwd(imgs): return model(imgs).last_hidden_state[:, 0].float()
    else:
        import open_clip
        if teacher == 'pe_core':
            mn, ckpt = 'PE-Core-B-16', _PE_CORE_CKPT
        else:
            mn, ckpt = _EXTERNAL_CLIPS[teacher]
        model, _, preproc = open_clip.create_model_and_transforms(mn, pretrained=ckpt)
        model = model.eval().to(device)
        vis = model.visual
        use_trunk = hasattr(vis, 'trunk') and hasattr(vis.trunk, 'forward_features')
        def fwd(imgs):
            if use_trunk:
                return vis.trunk.forward_features(imgs)[:, 0].float()
            proj_bak = getattr(vis, 'proj', None)
            if hasattr(vis, 'proj'): vis.proj = None
            out = model.encode_image(imgs, normalize=False).float()
            if hasattr(vis, 'proj'): vis.proj = proj_bak
            return out

    # 流式提取
    ds = WdsImageDataset(tar_pattern, preproc)
    dl = DataLoader(ds, batch_size=512, num_workers=16, prefetch_factor=4)
    feats = []
    count = 0
    for imgs, _ in dl:
        feats.append(fwd(imgs.to(device)).cpu())
        count += imgs.shape[0]
        if count % (512 * 200) == 0:
            elapsed = time.time() - t0
            rate = count / elapsed
            log.info(f'  [{teacher}] {count:,} samples, {rate:.0f} img/s')

    del model
    torch.cuda.empty_cache()
    feats_np = torch.cat(feats, 0).numpy().astype(np.float16)
    np.save(cache, feats_np)
    log.info(f'Saved: {cache} ({feats_np.shape}, {time.time()-t0:.0f}s)')


def sample(teacher, method, n_samples, device='cuda:0'):
    """采样 (复用 sample_cc3m 的 FPS/K-Means 逻辑)。"""
    from sample_cc3m import sample_fps, sample_kmeans, sample_kmeans_uniform, sample_random

    os.makedirs(_SUBSET_DIR, exist_ok=True)
    tag = f'{method}_{teacher}' if method != 'random' else 'random'
    out_path = os.path.join(_SUBSET_DIR, f'{tag}_{n_samples // 1000}k.indices.npy')
    if os.path.exists(out_path):
        log.info(f'Exists: {out_path}')
        return

    if method == 'random':
        feat_file = os.path.join(_FEAT_DIR, f'{TEACHERS[0]}.npy')
        n_total = np.load(feat_file, mmap_mode='r').shape[0]
        indices = sample_random(n_total, n_samples)
    else:
        feat_path = os.path.join(_FEAT_DIR, f'{teacher}.npy')
        features = np.load(feat_path)
        if method == 'fps':
            indices = sample_fps(features, n_samples, device)
        elif method == 'kmeans':
            indices = sample_kmeans(features, n_samples, device)
        else:  # kmeans_uniform
            indices = sample_kmeans_uniform(features, n_samples, device)

    np.save(out_path, indices)
    log.info(f'Saved {len(indices)} indices -> {out_path}')


def export_all():
    """一次遍历所有 tar，批量导出全部 .indices.npy 为 TSV + 图片文件。

    共享图片目录避免重复写入。不 decode (raw bytes 直写) 最大化吞吐。
    跳过率 <0.01%，index 偏移对 50K 采样无实际影响。
    """
    import glob
    import webdataset as wds

    os.makedirs(_SUBSET_DIR, exist_ok=True)
    idx_files = sorted(glob.glob(f'{_SUBSET_DIR}/*.indices.npy'))
    if not idx_files:
        log.info('No .indices.npy found, nothing to export.')
        return

    # 加载所有 subset
    subsets = {}  # tag -> {'indices': set, 'tsv': str, 'rows': []}
    union = set()
    for f in idx_files:
        tag = os.path.basename(f).replace('.indices.npy', '')
        tsv_path = os.path.join(_SUBSET_DIR, f'{tag}.tsv')
        if os.path.exists(tsv_path):
            log.info(f'Skip (exists): {tsv_path}')
            continue
        indices = set(np.load(f).tolist())
        subsets[tag] = {'indices': indices, 'tsv': tsv_path, 'rows': []}
        union |= indices

    if not subsets:
        log.info('All TSVs exist, nothing to do.')
        return

    img_dir = os.path.join(_SUBSET_DIR, '_images')
    os.makedirs(img_dir, exist_ok=True)
    log.info(f'Exporting {len(subsets)} subsets, union={len(union)} samples -> {img_dir}')

    tar_pattern = sorted(glob.glob(f'{_CC12M_DIR}/cc12m-train-*.tar'))
    ds = (
        wds.WebDataset(tar_pattern, shardshuffle=False, handler=wds.warn_and_continue)
        .to_tuple('jpg;png', 'txt', '__key__', handler=wds.warn_and_continue)
    )

    global_idx = 0
    exported = 0
    t0 = time.time()
    for img_bytes, txt_bytes, key in ds:
        if global_idx in union:
            txt = txt_bytes.decode('utf-8', errors='ignore') if isinstance(txt_bytes, bytes) else txt_bytes
            img_path = os.path.join(img_dir, f'{key}.jpg')
            if not os.path.exists(img_path):
                with open(img_path, 'wb') as fw:
                    fw.write(img_bytes)
            for tag, info in subsets.items():
                if global_idx in info['indices']:
                    info['rows'].append((img_path, txt))
            exported += 1
            if exported % 50000 == 0:
                elapsed = time.time() - t0
                log.info(f'  exported {exported}/{len(union)} ({elapsed:.0f}s, {exported/elapsed:.0f}/s)')
        global_idx += 1

    for tag, info in subsets.items():
        with open(info['tsv'], 'w') as f:
            f.write('filepath\tcaption\n')
            for path, cap in info['rows']:
                f.write(f'{path}\t{cap}\n')
        log.info(f'TSV: {info["tsv"]} ({len(info["rows"])} rows)')
    log.info(f'Export done: {exported} unique in {time.time()-t0:.0f}s')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    p_ext = sub.add_parser('extract')
    p_ext.add_argument('--teacher', choices=TEACHERS, required=True)
    p_ext.add_argument('--device', default='cuda:0')

    p_smp = sub.add_parser('sample')
    p_smp.add_argument('--teacher', choices=TEACHERS + ['random'], required=True)
    p_smp.add_argument('--method', choices=['fps', 'kmeans', 'kmeans_uniform', 'random'], required=True)
    p_smp.add_argument('--n-samples', type=int, default=50000)
    p_smp.add_argument('--device', default='cuda:0')

    sub.add_parser('export_all', help='一次遍历导出全部 .indices.npy 为 TSV')

    args = parser.parse_args()

    if args.cmd == 'extract':
        extract(args.teacher, args.device)
    elif args.cmd == 'sample':
        sample(args.teacher, args.method, args.n_samples, args.device)
    elif args.cmd == 'export_all':
        export_all()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
