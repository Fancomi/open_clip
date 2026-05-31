#!/usr/bin/env python3
"""批量评估已训练 checkpoint 在指定 val set 上的 retrieval 性能。"""
import argparse
import glob
import logging
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import open_clip
from open_clip_train.probe_hook import _ImgDataset, _TxtDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def load_model(ckpt_path, device):
    """加载 PE-Core-B-16-dinov3 + checkpoint state_dict。"""
    model, _, preproc = open_clip.create_model_and_transforms('PE-Core-B-16-dinov3')
    tokenizer = open_clip.get_tokenizer('PE-Core-B-16-dinov3')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    sd = ckpt['state_dict']
    # 去掉 module. 前缀 (DDP)
    sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.eval().to(device)
    return model, preproc, tokenizer


def read_val_tsv(tsv_path):
    """读取 val TSV (filepath\tcaption)。"""
    paths, captions = [], []
    with open(tsv_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.rstrip('\n').split('\t', 1)
            if len(parts) == 2:
                paths.append(parts[0])
                captions.append(parts[1])
    return paths, captions


@torch.no_grad()
def compute_retrieval(model, preproc, tokenizer, paths, captions, device, batch_size=256):
    """计算 i2t 和 t2i R@1/5/10。"""
    # Image features
    img_dl = DataLoader(_ImgDataset(paths, preproc), batch_size=batch_size,
                        num_workers=16, pin_memory=True)
    img_feats = []
    for imgs, _ in img_dl:
        img_feats.append(model.encode_image(imgs.to(device), normalize=True).cpu())
    img_feats = torch.cat(img_feats, 0)  # (N, D)

    # Text features
    txt_dl = DataLoader(_TxtDataset(captions, tokenizer), batch_size=batch_size,
                        num_workers=4, pin_memory=True)
    txt_feats = []
    for toks, _ in txt_dl:
        txt_feats.append(model.encode_text(toks.to(device), normalize=True).cpu())
    txt_feats = torch.cat(txt_feats, 0)  # (N, D)

    # Similarity
    sim = img_feats @ txt_feats.T  # (N, N)
    N = sim.shape[0]

    # i2t: for each image, rank texts
    i2t_ranks = (sim.argsort(dim=1, descending=True) == torch.arange(N).unsqueeze(1)).nonzero()[:, 1].float()
    # t2i: for each text, rank images
    t2i_ranks = (sim.T.argsort(dim=1, descending=True) == torch.arange(N).unsqueeze(1)).nonzero()[:, 1].float()

    results = {}
    for prefix, ranks in [('i2t', i2t_ranks), ('t2i', t2i_ranks)]:
        for k in [1, 5, 10]:
            results[f'{prefix}_R@{k}'] = (ranks < k).float().mean().item()
        results[f'{prefix}_median_rank'] = ranks.median().item()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val-tsv', required=True)
    parser.add_argument('--log-dir', default='logs', help='Directory containing experiment logs')
    parser.add_argument('--pattern', default='cc3m_sample_*_0527_1956',
                        help='Glob pattern for experiment dirs')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    paths, captions = read_val_tsv(args.val_tsv)
    log.info(f'Val set: {len(paths)} samples from {args.val_tsv}')

    exp_dirs = sorted(glob.glob(os.path.join(args.log_dir, args.pattern)))
    log.info(f'Found {len(exp_dirs)} experiments')

    print(f"{'Tag':<30} {'i2t_R@1':>8} {'i2t_R@5':>8} {'t2i_R@1':>8} {'t2i_R@5':>8}")
    print('-' * 70)

    for exp_dir in exp_dirs:
        tag = os.path.basename(exp_dir).replace('cc3m_sample_', '').replace('_0527_1956', '')
        ckpt = os.path.join(exp_dir, 'checkpoints', 'epoch_20.pt')
        if not os.path.exists(ckpt):
            log.warning(f'No checkpoint: {ckpt}')
            continue

        model, preproc, tokenizer = load_model(ckpt, args.device)
        results = compute_retrieval(model, preproc, tokenizer, paths, captions, args.device)
        print(f"{tag:<30} {results['i2t_R@1']:>8.4f} {results['i2t_R@5']:>8.4f} "
              f"{results['t2i_R@1']:>8.4f} {results['t2i_R@5']:>8.4f}")

        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
