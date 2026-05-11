"""Feature extraction and NPZ cache management.

ALL cached image features are raw (un-normalized) global CLS tokens:
  PE-Core raw   → trunk.forward_features → CLS [B, 768]  (before projection head)
  PE-Core proj  → encode_image(normalize=True) → [B, 1024]  (CLIP space, modality_gap only)
  SigLIP2 img   → encode_image(normalize=True) → [B, 768]   (head=Sequential(), effectively CLS)
  DINOv3 / EUPE → forward_features → x_norm_clstoken        (model-normalized, no extra L2)
  RADIO         → model forward    → output.summary          (raw)
  TIPSv2 image  → encode_image     → cls_token               (raw)
  Text features → always L2-normalized (CLIP-aligned text space)

NOTE: if you have old npz caches with L2-normalized image features, delete them
      (or pass --force) before re-running pretrained mode.
"""
import os, logging
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .models import DEVICE

# ── Transforms (all resize to 224×224 for uniform batch stacking) ─────────────
_DINO_TF  = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
_RADIO_TF = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor()])
_EUPE_TF  = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
                        T.ToTensor(),
                        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
_TIPS_TF  = T.ToTensor()   # resizes inline to 448×448
_TIPS_SZ  = 448
_TIPS_CHUNK = 8             # TIPSv2 sub-batch: 1024 patches @ 448² → OOM guard

# ── DINOv3-style crop augmentation (RandomResizedCrop applied to PIL) ─────────
_GC_CROP = T.RandomResizedCrop(224, scale=(0.32, 1.0),  ratio=(3/4, 4/3),
                                interpolation=T.InterpolationMode.BICUBIC)
_LC_CROP = T.RandomResizedCrop(96,  scale=(0.05, 0.32), ratio=(3/4, 4/3),
                                interpolation=T.InterpolationMode.BICUBIC)

# ── Required npz for cache-hit ─────────────────────────────────────────────────
_REQUIRED_NPZS = {
    'pe_img':    'pe_core_img.npz',  'pe_txt':   'pe_core_txt.npz',
    'sig2_img':  'siglip2_img.npz',  'sig2_txt': 'siglip2_txt.npz',
    'dino_img':  'dinov3_img.npz',   'radio_img':'radio_img.npz',
    'tips_img':  'tips_img.npz',     'tips_txt': 'tips_txt.npz',
}
_OPTIONAL_NPZS = {'eupe_img': 'eupe_img.npz'}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _npz(path, force):
    """Load cached npz features if available."""
    if not force and os.path.exists(path):
        logging.info(f'[cache] {os.path.basename(path)}')
        return np.load(path)['features']
    return None


def load_from_cache(out_dir, force=False):
    """Return feature dict if ALL required npzs exist, else None.
    Optional (eupe) is loaded if present."""
    if force:
        return None
    result = {}
    for key, fname in _REQUIRED_NPZS.items():
        p = os.path.join(out_dir, fname)
        if not os.path.exists(p):
            logging.info(f'[cache] Miss: {fname} — will run inference')
            return None
        result[key] = np.load(p)['features']
    for key, fname in _OPTIONAL_NPZS.items():
        p = os.path.join(out_dir, fname)
        if os.path.exists(p):
            result[key] = np.load(p)['features']
    total = sum(v.shape[0] for v in result.values())
    logging.info(f'[cache] Hit: {len(result)} files, {total} rows — skipping inference')
    return result


# ── Batch image/text extractors ───────────────────────────────────────────────

def extract_clip_img(model, paths, preproc, out_path, force=False, bs=256):
    """Extract L2-normalized CLIP projection features (encode_image output).
    Used exclusively for modality_gap plots — NOT for anisotropy/scatter."""
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    from open_clip_train.probe_hook import extract_backbone_cls
    # extract_backbone_cls returns (backbone_cls, proj_cls);
    # for CLIP projection space we want proj_cls (normalized 1024-dim).
    # For models without a projection head (proj_cls is None), fall back to backbone_cls.
    _, proj = extract_backbone_cls(model, paths, preproc, next(model.parameters()).device, bs)
    if proj is None:
        dev = next(model.parameters()).device
        feats = []
        for i in range(0, len(paths), bs):
            x = torch.stack([preproc(Image.open(p).convert('RGB'))
                             for p in paths[i:i+bs]]).to(dev)
            feats.append(model.encode_image(x, normalize=True).detach().cpu().float().numpy())
        feat = np.concatenate(feats)
    else:
        feat = proj
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_pe_core_img_raw(model, paths, preproc, out_path, force=False, bs=256):
    """PE-Core backbone CLS **before** the Linear(768→1024) projection head.

    PE-Core's TimmModel has:
      trunk (Eva ViT-B) → forward_features → [B, N, 768]  (CLS at index 0)
      head  Sequential(Linear(768 → 1024))                 ← CLIP projection

    For VLM fine-tuning (and consistent geometry comparison with DINOv3/EUPE),
    we want the 768-dim backbone CLS, not the 1024-dim projected CLIP space.
    The projected 1024-dim version is kept separately for modality_gap plots.
    """
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.stack([preproc(Image.open(p).convert('RGB'))
                         for p in paths[i:i+bs]]).to(DEVICE)
        trunk_out = model.visual.trunk.forward_features(x)   # [B, N, 768]
        cls = trunk_out[:, 0, :].float()                      # CLS token, raw
        feats.append(cls.cpu().numpy())
        if (i // bs + 1) % 5 == 0 or i + bs >= len(paths):
            logging.info(f'  [PE-Core raw] {min(i+bs, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_clip_txt(model, tok, caps, out_path, force=False, bs=512):
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    feats = [model.encode_text(tok(caps[i:i+bs]).to(DEVICE), normalize=True)
             .cpu().float().numpy() for i in range(0, len(caps), bs)]
    feat = np.concatenate(feats)
    np.savez_compressed(out_path, features=feat)
    return feat


@torch.no_grad()
def extract_dinov3_img(model, paths, out_path, force=False, bs=128):
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.stack([_DINO_TF(Image.open(p).convert('RGB'))
                         for p in paths[i:i+bs]]).to(DEVICE)
        cls = model.forward_features(x)['x_norm_clstoken']
        feats.append(cls.cpu().float().numpy())
        if (i // bs + 1) % 5 == 0 or i + bs >= len(paths):
            logging.info(f'  [DINOv3] {min(i+bs, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_radio_img(model, cond, paths, out_path, force=False, bs=128):
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(paths), bs):
        x = torch.stack([_RADIO_TF(Image.open(p).convert('RGB'))
                         for p in paths[i:i+bs]]).to(DEVICE)
        if cond is not None:
            x = cond(x)
        out = model(x)
        s = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'summary', out[0])
        feats.append(s.cpu().float().numpy())
        if (i // bs + 1) % 5 == 0 or i + bs >= len(paths):
            logging.info(f'  [RADIO] {min(i+bs, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_eupe_img(model, paths, out_path, force=False, bs=128):
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    feats, dev = [], DEVICE.type
    for i in range(0, len(paths), bs):
        x = torch.stack([_EUPE_TF(Image.open(p).convert('RGB'))
                         for p in paths[i:i+bs]]).to(DEVICE)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != 'cpu')):
            cls = model.forward_features(x)['x_norm_clstoken']
        feats.append(cls.cpu().float().numpy())
        if (i // bs + 1) % 5 == 0 or i + bs >= len(paths):
            logging.info(f'  [EUPE] {min(i+bs, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_tips_img(model, paths, out_path, force=False, bs=8):
    """bs=8: TIPSv2 448×448 → 1024 patches, attention O(1024²·bs) — keep small."""
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    feats, dev = [], DEVICE.type
    for i in range(0, len(paths), bs):
        imgs = [_TIPS_TF(Image.open(p).convert('RGB')
                         .resize((_TIPS_SZ, _TIPS_SZ), Image.BICUBIC))
                for p in paths[i:i+bs]]
        x = torch.stack(imgs).to(DEVICE)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != 'cpu')):
            out = model.encode_image(x)
        cls = out.cls_token.squeeze(1).float()
        feats.append(cls.cpu().numpy())
        if (i // bs + 1) % 20 == 0 or i + bs >= len(paths):
            logging.info(f'  [TIPSv2] {min(i+bs, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_tips_txt(model, tok, caps, out_path, force=False, bs=512):
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(caps), bs):
        ids, pads = tok.tokenize(caps[i:i+bs], max_len=model.config.max_len)
        ids  = torch.from_numpy(ids).to(DEVICE)
        pads = torch.from_numpy(pads).to(DEVICE)
        feats.append(F.normalize(model.encode_text(ids, pads), dim=-1)
                     .cpu().float().numpy())
    feat = np.concatenate(feats)
    np.savez_compressed(out_path, features=feat)
    return feat



# ── DINOv3-style crop generation ──────────────────────────────────────────────

def make_dino_crops(fps_paths, seed=42):
    """Generate one global crop (224px) and one local crop (96px) per FPS path.

    Crop sizes match DINOv3 training augmentation:
      global crop : RandomResizedCrop(224, scale=(0.32, 1.0))  ← teacher/student view
      local  crop : RandomResizedCrop(96,  scale=(0.05, 0.32)) ← student-only view

    Local crops are returned at 96px native size; each model's PIL extractor
    handles the resize internally (e.g. _DINO_TF resizes anything to 224).

    Returns:
        orig_imgs    : list[PIL] — full original images
        global_crops : list[PIL] — global crops (224px)
        local_crops  : list[PIL] — local  crops  (96px)
    """
    torch.manual_seed(seed)
    orig, gc, lc = [], [], []
    for p in fps_paths:
        img = Image.open(p).convert('RGB')
        orig.append(img)
        gc.append(_GC_CROP(img))
        lc.append(_LC_CROP(img))
    return orig, gc, lc


# ── Per-model PIL-based feature extractors (used by crop_probe mode) ──────────
# All accept a list of PIL images of any size; internal transforms handle resize.

@torch.no_grad()
def extract_dinov3_pil(model, pil_imgs):
    """DINOv3 CLS features from PIL images (any size → 224 via _DINO_TF)."""
    x = torch.stack([_DINO_TF(img) for img in pil_imgs]).to(DEVICE)
    cls = model.forward_features(x)['x_norm_clstoken']
    return cls.cpu().float().numpy()


@torch.no_grad()
def extract_clip_pil(model, preproc, pil_imgs):
    """CLIP-style image features from PIL images (preproc handles resize)."""
    x = torch.stack([preproc(img) for img in pil_imgs]).to(DEVICE)
    return model.encode_image(x, normalize=True).detach().cpu().float().numpy()


@torch.no_grad()
def extract_pe_core_pil_raw(model, preproc, pil_imgs):
    """PE-Core backbone raw CLS from PIL images (before Linear 768→1024 head)."""
    x = torch.stack([preproc(img) for img in pil_imgs]).to(DEVICE)
    trunk_out = model.visual.trunk.forward_features(x)   # [B, N, 768]
    cls = trunk_out[:, 0, :].float()
    return cls.cpu().numpy()


@torch.no_grad()
def extract_radio_pil(model, cond, pil_imgs):
    """RADIO features from PIL images (any size → 224 via _RADIO_TF)."""
    x = torch.stack([_RADIO_TF(img) for img in pil_imgs]).to(DEVICE)
    if cond is not None:
        x = cond(x)
    out = model(x)
    s = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'summary', out[0])
    return s.cpu().float().numpy()


@torch.no_grad()
def extract_eupe_pil(model, pil_imgs):
    """EUPE features from PIL images (any size → 224 via _EUPE_TF)."""
    x = torch.stack([_EUPE_TF(img) for img in pil_imgs]).to(DEVICE)
    dev = DEVICE.type
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != 'cpu')):
        cls = model.forward_features(x)['x_norm_clstoken']
    return cls.cpu().float().numpy()


@torch.no_grad()
def extract_tips_pil(model, pil_imgs):
    """TIPSv2 features from PIL images (any size → 448 inline)."""
    dev = DEVICE.type
    chunks = []
    for i in range(0, len(pil_imgs), _TIPS_CHUNK):
        imgs = [_TIPS_TF(img.resize((_TIPS_SZ, _TIPS_SZ), Image.BICUBIC))
                for img in pil_imgs[i:i + _TIPS_CHUNK]]
        x = torch.stack(imgs).to(DEVICE)
        with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != 'cpu')):
            out = model.encode_image(x)
        chunks.append(out.cls_token.squeeze(1).float().cpu().numpy())
    return np.concatenate(chunks)
