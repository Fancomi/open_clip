"""
Per-step / per-epoch feature probe: extract image + text features and save to disk.
Called from train.py (step-granularity) or main.py (epoch fallback).

File naming:
  step-based  →  step_XXXXXX.npz   (6-digit zero-padded global optimizer step)
  epoch-based →  epoch_XX.npz      (kept for backward compat / epoch-only mode)

npz keys:
  features      — (N, D_bb)   raw backbone CLS / summary token  [always]
                  PE-Core: trunk CLS [B, 768]  (before Linear 768→1024 projection)
                  Other CLIP (SigLIP2, ViT-B/16, …): encode_image output [B, D]
  proj_features — (N, D_proj) l2-norm projected CLIP image features  [when TimmModel trunk]
                  PE-Core only: projection head output [B, 1024]
                  Used exclusively for modality-gap / contrastive dynamics (step_evolution GIF).
  txt_features  — (N, D_proj) l2-norm projected text features         [when text tower exists]
  paths         — image file paths

Why this split:
  features     = backbone CLS  → the representation actually consumed by downstream VLMs.
                 Geometry analysis (anisotropy, image_allmodels, crop_probe) lives here.
  proj_features = post-projection CLIP space → where contrastive loss acts.
                 Only step_evolution (image-text modality gap) uses this.
"""
import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
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
        self.captions    = captions
        self.tokenize_fn = tokenize_fn

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, i):
        return self.tokenize_fn([self.captions[i]])[0], i


@torch.no_grad()
def extract_backbone_cls(model, paths, preprocess, device, batch_size=256):
    """Extract backbone CLS token (primary feature, before any CLIP projection head).

    For TimmModel trunks (PE-Core, SigLIP2 with empty head, etc.):
      Uses trunk.forward_features(x)[:, 0, :] → backbone CLS [B, D_backbone].
      PE-Core: D_backbone=768  (Eva ViT-B, before Linear 768→1024).

    For all other architectures (no .trunk):
      Falls back to encode_image(normalize=True) → whatever dim the model exposes.
      This is the correct path for plain ViT CLIP models.

    Returns: (backbone_cls, proj_cls)
      backbone_cls — always present, [B, D_backbone], raw (un-normalized)
      proj_cls     — only when trunk was used AND trunk dim ≠ encode_image dim, else None
                     This is the projected [B, 1024] CLIP-space feature (l2-normalized).
    """
    visual = getattr(model, 'visual', None)
    has_trunk = (visual is not None
                 and hasattr(visual, 'trunk')
                 and hasattr(visual.trunk, 'forward_features'))

    dl = DataLoader(_ImgDataset(paths, preprocess), batch_size=batch_size,
                    num_workers=0, pin_memory=False)
    model.eval()
    bb_feats   = []
    proj_feats = [] if has_trunk else None

    for i, (imgs, _) in enumerate(dl):
        imgs = imgs.to(device)
        if has_trunk:
            trunk_out = visual.trunk.forward_features(imgs)
            # [B, N, D] — CLS token at position 0, raw (un-normalized)
            bb = trunk_out[:, 0, :].float()
            bb_feats.append(bb.cpu().numpy())
            # Also capture projected output to detect if head is non-trivial
            proj = model.encode_image(imgs, normalize=True).float()
            proj_feats.append(proj.cpu().numpy())
        else:
            bb = model.encode_image(imgs, normalize=True).float()
            bb_feats.append(bb.cpu().numpy())
        if (i + 1) % 5 == 0 or (i + 1) == len(dl):
            logging.info(f'[probe] img {min((i+1)*batch_size, len(paths))}/{len(paths)} ...')

    model.train()
    backbone_cls = np.concatenate(bb_feats, 0)

    if proj_feats is not None:
        proj_arr = np.concatenate(proj_feats, 0)
        # Only expose proj when dimensions differ (meaningful projection head exists)
        if proj_arr.shape[1] != backbone_cls.shape[1]:
            return backbone_cls, proj_arr
    return backbone_cls, None


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
                logging.info(f'[probe] txt {min((i+1)*batch_size, len(captions))}/{len(captions)} ...')
    except Exception as e:
        logging.warning(f'[probe] text extraction failed: {e}')
        return None
    model.train()
    return np.concatenate(feats, 0)


def run_probe(model, epoch, args, preprocess_val, step=None):
    """Extract features for probe_data TSV and save npz.

    npz keys saved:
      features      — backbone CLS [B, D_backbone]  (always; primary geometry feature)
      proj_features — projected CLIP [B, 1024]       (PE-Core only; for step_evolution GIF)
      txt_features  — projected text [B, D_proj]     (when text tower present)
      paths         — image file paths

    step: global optimizer step. If provided, file is named step_XXXXXX.npz;
          otherwise falls back to epoch_XX.npz (legacy mode).
    """
    if not getattr(args, 'probe_data', None):
        return
    model = model.module if hasattr(model, 'module') else model
    model = model.clip_model if hasattr(model, 'clip_model') else model
    probe_dir = getattr(args, 'probe_dir', None) or os.path.join(args.checkpoint_path, 'probe')
    os.makedirs(probe_dir, exist_ok=True)

    df    = pd.read_csv(args.probe_data, sep='\t')
    paths = df['filepath'].tolist()
    caps  = df['caption'].tolist() if 'caption' in df.columns else None
    device = next(model.parameters()).device

    bb_cls, proj_cls = extract_backbone_cls(model, paths, preprocess_val, device)
    txt_feats        = extract_text_features(model, caps, device) if caps is not None else None

    if step is not None:
        out = os.path.join(probe_dir, f'step_{step:06d}.npz')
        logging.info(f'[probe] step={step}  features(bb)={bb_cls.shape}'
                     + (f'  proj_features={proj_cls.shape}' if proj_cls is not None else '')
                     + (f'  txt={txt_feats.shape}' if txt_feats is not None else '')
                     + f'  -> {out}')
    else:
        out = os.path.join(probe_dir, f'epoch_{epoch:02d}.npz')
        logging.info(f'[probe] epoch={epoch}  features(bb)={bb_cls.shape}'
                     + (f'  proj_features={proj_cls.shape}' if proj_cls is not None else '')
                     + (f'  txt={txt_feats.shape}' if txt_feats is not None else '')
                     + f'  -> {out}')

    save_kwargs = dict(features=bb_cls, paths=np.array(paths))
    if proj_cls is not None:
        save_kwargs['proj_features'] = proj_cls
    if txt_feats is not None:
        save_kwargs['txt_features'] = txt_feats
    np.savez_compressed(out, **save_kwargs)
