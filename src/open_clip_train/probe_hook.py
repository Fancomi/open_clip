"""
Per-step / per-epoch feature probe: extract image + text features and save to disk.
Called from train.py (step-granularity) or main.py (epoch fallback).

File naming:
  step-based  →  step_XXXXXX.npz   (6-digit zero-padded global optimizer step)
  epoch-based →  epoch_XX.npz      (kept for backward compat / epoch-only mode)

npz keys:
  features      — (N, D) l2-normalised image features   [always present]
  txt_features  — (N, D) l2-normalised text  features   [present when model has text tower]
  paths         — image file paths
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


class _TxtDataset(Dataset):
    def __init__(self, captions, tokenize_fn):
        self.captions   = captions
        self.tokenize_fn = tokenize_fn

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        return self.tokenize_fn([self.captions[i]])[0], i


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
            logging.info(f'[probe] img {(i+1)*batch_size}/{len(paths)} ...')
    model.train()
    return np.concatenate(feats, 0)


@torch.no_grad()
def extract_text_features(model, captions, device, batch_size=512):
    """Extract l2-normalized text features via model.encode_text(normalize=True).
    Returns None if model has no text tower."""
    try:
        from open_clip import tokenize
    except ImportError:
        return None
    if not hasattr(model, 'encode_text'):
        return None
    # Respect the model's context_length (e.g. PE-Core uses 32, not the default 77)
    ctx_len = getattr(model, 'context_length', 77)
    tokenize_fn = lambda texts: tokenize(texts, context_length=ctx_len)
    dl = DataLoader(_TxtDataset(captions, tokenize_fn), batch_size=batch_size,
                    num_workers=0, pin_memory=False)
    model.eval()
    feats = []
    try:
        for i, (tokens, _) in enumerate(dl):
            feats.append(model.encode_text(tokens.to(device), normalize=True).cpu().float().numpy())
            if (i + 1) % 10 == 0 or (i + 1) == len(dl):
                logging.info(f'[probe] txt {(i+1)*batch_size}/{len(captions)} ...')
    except Exception as e:
        logging.warning(f'[probe] text extraction failed: {e}')
        return None
    model.train()
    return np.concatenate(feats, 0)


def run_probe(model, epoch, args, preprocess_val, step=None):
    """Extract image (+ text when available) features for probe_data TSV and save npz.

    step: global optimizer step. If provided, file is named step_XXXXXX.npz;
          otherwise falls back to epoch_XX.npz (legacy mode).
    """
    if not getattr(args, 'probe_data', None):
        return
    model = model.module if hasattr(model, 'module') else model
    model = model.clip_model if hasattr(model, 'clip_model') else model
    probe_dir = getattr(args, 'probe_dir', None) or os.path.join(args.checkpoint_path, 'probe')
    os.makedirs(probe_dir, exist_ok=True)

    df      = pd.read_csv(args.probe_data, sep='\t')
    paths   = df['filepath'].tolist()
    caps    = df['caption'].tolist() if 'caption' in df.columns else None
    device  = next(model.parameters()).device

    img_feats = extract_image_features(model, paths, preprocess_val, device)
    txt_feats = extract_text_features(model, caps, device) if caps is not None else None

    if step is not None:
        out = os.path.join(probe_dir, f'step_{step:06d}.npz')
        logging.info(f'[probe] step={step}  img={img_feats.shape}'
                     + (f'  txt={txt_feats.shape}' if txt_feats is not None else '')
                     + f'  -> {out}')
    else:
        out = os.path.join(probe_dir, f'epoch_{epoch:02d}.npz')
        logging.info(f'[probe] epoch={epoch}  img={img_feats.shape}'
                     + (f'  txt={txt_feats.shape}' if txt_feats is not None else '')
                     + f'  -> {out}')

    save_kwargs = dict(features=img_feats, paths=np.array(paths))
    if txt_feats is not None:
        save_kwargs['txt_features'] = txt_feats
    np.savez_compressed(out, **save_kwargs)
