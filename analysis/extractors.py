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
    feat = _npz(out_path, force)
    if feat is not None:
        return feat
    from open_clip_train.probe_hook import extract_image_features
    feat = extract_image_features(model, paths, preproc, DEVICE, bs)
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


# ── WDS (streaming) extractor ─────────────────────────────────────────────────

def extract_wds_features(
        pe_model, pe_preproc, pe_tok,
        sig2_model, sig2_preproc, sig2_tok,
        dino_model, radio_model, radio_cond, eupe_model,
        tips_model, tips_tok,
        pattern, out_dir, max_samples=100_000, force=False, bs=64):
    import webdataset as wds

    npz_map = {
        'pe_img':   'pe_core_img.npz',  'pe_txt':   'pe_core_txt.npz',
        'sig2_img': 'siglip2_img.npz',  'sig2_txt': 'siglip2_txt.npz',
        'dino_img': 'dinov3_img.npz',   'radio_img':'radio_img.npz',
        'eupe_img': 'eupe_img.npz',     'tips_img': 'tips_img.npz',
        'tips_txt': 'tips_txt.npz',
    }
    active_map = {
        'pe_img': pe_model,    'pe_txt':   pe_model,
        'sig2_img': sig2_model,'sig2_txt': sig2_model,
        'dino_img': dino_model,'radio_img':radio_model,
        'eupe_img': eupe_model,'tips_img': tips_model,
        'tips_txt': tips_model and tips_tok,
    }
    active = {k: os.path.join(out_dir, v)
              for k, v in npz_map.items() if active_map[k]}

    if not force and all(os.path.exists(p) for p in active.values()):
        logging.info('[cache] All wds npz found — loading ...')
        return {k: np.load(p)['features'] for k, p in active.items()}

    acc  = {k: [] for k in active}
    bufs = {'imgs': [], 'cap': []}
    count = 0

    @torch.no_grad()
    def _flush():
        imgs, caps = bufs['imgs'], bufs['cap']
        if 'pe_img' in active or 'pe_txt' in active:
            pb = torch.stack([pe_preproc(im) for im in imgs]).to(DEVICE)
            if 'pe_img' in active:
                acc['pe_img'].append(
                    pe_model.encode_image(pb, normalize=True).cpu().float().numpy())
            if 'pe_txt' in active:
                acc['pe_txt'].append(
                    pe_model.encode_text(pe_tok(caps).to(DEVICE), normalize=True)
                    .cpu().float().numpy())
        if 'sig2_img' in active or 'sig2_txt' in active:
            sb = torch.stack([sig2_preproc(im) for im in imgs]).to(DEVICE)
            if 'sig2_img' in active:
                acc['sig2_img'].append(
                    sig2_model.encode_image(sb, normalize=True).cpu().float().numpy())
            if 'sig2_txt' in active:
                acc['sig2_txt'].append(
                    sig2_model.encode_text(sig2_tok(caps).to(DEVICE), normalize=True)
                    .cpu().float().numpy())
        if 'dino_img' in active:
            dx = torch.stack([_DINO_TF(im) for im in imgs]).to(DEVICE)
            acc['dino_img'].append(
                dino_model.forward_features(dx)['x_norm_clstoken']
                .cpu().float().numpy())
        if 'radio_img' in active:
            rx = torch.stack([_RADIO_TF(im) for im in imgs]).to(DEVICE)
            if radio_cond is not None:
                rx = radio_cond(rx)
            out = radio_model(rx)
            s = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'summary', out[0])
            acc['radio_img'].append(s.cpu().float().numpy())
        if 'eupe_img' in active:
            ex = torch.stack([_EUPE_TF(im) for im in imgs]).to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                enabled=(DEVICE.type != 'cpu')):
                eout = eupe_model.forward_features(ex)
            acc['eupe_img'].append(
                eout['x_norm_clstoken'].cpu().float().numpy())
        if 'tips_img' in active:
            tiles = [_TIPS_TF(im.resize((_TIPS_SZ, _TIPS_SZ), Image.BICUBIC)) for im in imgs]
            chunks = []
            for ci in range(0, len(tiles), _TIPS_CHUNK):
                tx = torch.stack(tiles[ci:ci+_TIPS_CHUNK]).to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                    enabled=(DEVICE.type != 'cpu')):
                    tout = tips_model.encode_image(tx)
                chunks.append(
                    tout.cls_token.squeeze(1).float().cpu().numpy())
            acc['tips_img'].append(np.concatenate(chunks))
        if 'tips_txt' in active:
            ids, pads = tips_tok.tokenize(caps, max_len=tips_model.config.max_len)
            ids  = torch.from_numpy(ids).to(DEVICE)
            pads = torch.from_numpy(pads).to(DEVICE)
            acc['tips_txt'].append(
                F.normalize(tips_model.encode_text(ids, pads), dim=-1)
                .cpu().float().numpy())

    ds = (wds.WebDataset(pattern, shardshuffle=False)
          .decode('pil').to_tuple('jpg', 'txt'))
    for img, cap in ds:
        bufs['imgs'].append(img); bufs['cap'].append(cap); count += 1
        if len(bufs['imgs']) == bs:
            _flush(); bufs = {'imgs': [], 'cap': []}
            logging.info(f'  wds {count}/{max_samples}')
        if count >= max_samples:
            break
    if bufs['imgs']:
        _flush()

    result = {k: np.concatenate(v) for k, v in acc.items() if v}
    for k, p in active.items():
        if k in result:
            np.savez_compressed(p, features=result[k])
    logging.info('wds done: ' + ', '.join(f'{k}{v.shape}' for k, v in result.items()))
    return result


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
    return model.encode_image(x, normalize=True).cpu().float().numpy()


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
