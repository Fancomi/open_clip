"""
Per-step / per-epoch feature probe: extract image features and save to disk.
Called from train.py (step-granularity) or main.py (epoch fallback).

File naming:
  step-based  →  step_XXXXXX.npz   (6-digit zero-padded global optimizer step)
  epoch-based →  epoch_XX.npz      (kept for backward compat / epoch-only mode)
"""
import os
import logging
import numpy as np
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class _ImgDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths, self.transform = paths, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert('RGB')), i


@torch.no_grad()
def extract_image_features(model, paths, preprocess, device, batch_size=256):
    """Extract l2-normalized image features via model.encode_image(normalize=True)."""
    dl = DataLoader(_ImgDataset(paths, preprocess), batch_size=batch_size,
                    num_workers=0, pin_memory=False)
    model.eval()
    feats = []
    for i, (imgs, _) in enumerate(dl):
        feats.append(model.encode_image(imgs.to(device), normalize=True).cpu().float().numpy())
        if (i + 1) % 5 == 0 or (i + 1) == len(dl):
            logging.info(f'[probe] extracting {(i+1)*batch_size}/{len(paths)} ...')
    model.train()
    return np.concatenate(feats, 0)


def run_probe(model, epoch, args, preprocess_val, step=None):
    """Extract image features for probe_data TSV and save npz.

    step: global optimizer step. If provided, file is named step_XXXXXX.npz;
          otherwise falls back to epoch_XX.npz (legacy mode).
    """
    if not getattr(args, 'probe_data', None):
        return
    model = model.module if hasattr(model, 'module') else model
    model = model.clip_model if hasattr(model, 'clip_model') else model
    probe_dir = getattr(args, 'probe_dir', None) or os.path.join(args.checkpoint_path, 'probe')
    os.makedirs(probe_dir, exist_ok=True)
    df = pd.read_csv(args.probe_data, sep='\t')
    paths = df['filepath'].tolist()
    device = next(model.parameters()).device
    feats = extract_image_features(model, paths, preprocess_val, device)
    if step is not None:
        out = os.path.join(probe_dir, f'step_{step:06d}.npz')
        logging.info(f'[probe] step={step}  shape={feats.shape}  -> {out}')
    else:
        out = os.path.join(probe_dir, f'epoch_{epoch:02d}.npz')
        logging.info(f'[probe] epoch={epoch}  shape={feats.shape}  -> {out}')
    np.savez_compressed(out, features=feats, paths=np.array(paths))
