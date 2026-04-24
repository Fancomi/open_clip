#!/usr/bin/env python3
"""
Feature space analysis for pretrained models (Exp1) and training dynamics (Exp2).

Exp1 - pretrained comparison (COCO):
    python feature_probe.py --mode pretrained

Exp1 - pretrained comparison (CC3M, wds streaming):
    python feature_probe.py --mode pretrained --data-type wds \
        --data '/path/to/wds/{00000..00280}.tar' \
        --out-dir /path/to/LLaVA-ReCap-CC3M/feature_probe \
        --max-samples 100000

Exp2 - epoch evolution (after training with --probe-data):
    python feature_probe.py --mode epochs --probe-dir LOGS/RUN/checkpoints/probe

Exp3 - COCO vs CC3M distribution overlap (needs both caches):
    python feature_probe.py --mode overlap \
        --coco-dir  /path/coco/feature_probe/pretrained \
        --cc3m-dir  /path/cc3m/feature_probe/pretrained

Exp4 - Anisotropy metrics (needs cached npz):
    python feature_probe.py --mode anisotropy \
        --aniso-dir /path/to/feature_probe/pretrained
"""
import argparse, glob, logging, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from PIL import Image
from sklearn.decomposition import PCA
import torchvision.transforms as T

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_DEFAULTS = dict(
    data      = f'{_BASE}/datas/coco/annotations/karpathy_1cap.tsv',
    pe_ckpt   = f'{_BASE}/models/timm/PE-Core-B-16/open_clip_model.safetensors',
    sig2_ckpt = f'{_BASE}/models/timm/ViT-B-16-SigLIP2/open_clip_model.safetensors',
    dino_repo = f'{_BASE}/vision_encoder/dinov3',
    dino_ckpt = f'{_BASE}/models/dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    radio     = f'{_BASE}/models/C-RADIOv4-SO400M',
    eupe_repo = f'{_BASE}/vision_encoder/EUPE',
    eupe_ckpt = f'{_BASE}/models/EUPE-ViT-B/EUPE-ViT-B.pt',
    tips      = f'{_BASE}/models/tipsv2-b14',
    out_dir   = f'{_BASE}/datas/coco/feature_probe',
    coco_dir  = f'{_BASE}/datas/coco/feature_probe/pretrained',
    cc3m_dir  = f'{_BASE}/datas/LLaVA-ReCap-CC3M/feature_probe/pretrained',
)

_HF_CACHE = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')


# ─── Model loading ────────────────────────────────────────────────────────────

def load_pe_core(ckpt_path):
    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms('PE-Core-B-16', pretrained=ckpt_path)
    tok = open_clip.get_tokenizer('PE-Core-B-16')
    return model.eval().to(DEVICE), preproc, tok


def load_siglip2(ckpt_path):
    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms('ViT-B-16-SigLIP2', pretrained=ckpt_path)
    tok = open_clip.get_tokenizer('ViT-B-16-SigLIP2')
    return model.eval().to(DEVICE), preproc, tok


def load_dinov3(dino_repo, dino_ckpt):
    logging.info(f'Loading DINOv3 from hub {dino_repo} ...')
    model = torch.hub.load(dino_repo, 'dinov3_vitb16', source='local', pretrained=False)
    sd = torch.load(dino_ckpt, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    return model.eval().to(DEVICE)


def load_radio(radio_path):
    from transformers import AutoModel
    logging.info(f'Loading C-RADIOv4 from {radio_path} ...')
    model = AutoModel.from_pretrained(radio_path, trust_remote_code=True)
    conditioner = model.input_conditioner if hasattr(model, 'input_conditioner') else None
    return model.eval().to(DEVICE), conditioner


def load_eupe(eupe_repo, eupe_ckpt):
    if not os.path.exists(os.path.join(eupe_repo, 'hubconf.py')):
        logging.warning(f'EUPE repo not found at {eupe_repo}, skipping')
        return None
    logging.info(f'Loading EUPE from hub {eupe_repo} ...')
    try:
        model = torch.hub.load(eupe_repo, 'eupe_vitb16', source='local',
                               weights=eupe_ckpt, trust_repo=True)
        return model.eval().to(DEVICE)
    except Exception as e:
        logging.warning(f'EUPE load failed: {e}')
        return None


def load_tips(tips_path):
    """Load TIPSv2 via local cached modules; returns (model, tokenizer_fn)."""
    import shutil, json
    from safetensors.torch import load_file as sf_load
    cache_dir = os.path.join(_HF_CACHE, 'tipsv2_hyphen_b14')
    for fname in ('image_encoder.py', 'text_encoder.py'):
        dst = os.path.join(cache_dir, fname)
        src = os.path.join(tips_path, fname)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(src, dst)
    if _HF_CACHE not in sys.path:
        sys.path.insert(0, _HF_CACHE)
    from transformers_modules.tipsv2_hyphen_b14.configuration_tips import TIPSv2Config
    from transformers_modules.tipsv2_hyphen_b14.modeling_tips import TIPSv2Model
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tips_te', os.path.join(cache_dir, 'text_encoder.py'))
    te_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(te_mod)
    tok_path = os.path.join(tips_path, 'tokenizer.model')
    tokenizer = te_mod.Tokenizer(tok_path)

    cfg_raw = json.load(open(os.path.join(tips_path, 'config.json')))
    skip = {'_name_or_path', 'transformers_version', 'auto_map', 'architectures',
            'model_type', 'torch_dtype'}
    cfg = TIPSv2Config(**{k: v for k, v in cfg_raw.items() if k not in skip})
    logging.info(f'Loading TIPSv2 from {tips_path} ...')
    model = TIPSv2Model(cfg)
    sd = sf_load(os.path.join(tips_path, 'model.safetensors'))
    model.load_state_dict(sd, strict=True)
    return model.eval().to(DEVICE), tokenizer


# ─── Feature extraction ───────────────────────────────────────────────────────

def _npz_cache(path, force):
    if not force and os.path.exists(path):
        logging.info(f'[cache] Loading {path}')
        return np.load(path)['features']
    return None


def extract_clip_img(model, paths, preproc, out_path, force=False, batch_size=256):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    from open_clip_train.probe_hook import extract_image_features
    feat = extract_image_features(model, paths, preproc, DEVICE, batch_size)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_clip_txt(model, tok, captions, out_path, force=False, batch_size=512):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(captions), batch_size):
        t = tok(captions[i:i+batch_size]).to(DEVICE)
        feats.append(model.encode_text(t, normalize=True).cpu().float().numpy())
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat)
    return feat


extract_pe_img = extract_clip_img
extract_pe_txt = extract_clip_txt


_DINO_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
_DINO_PATCH = 16


@torch.no_grad()
def extract_dinov3_img(model, paths, out_path, force=False, batch_size=128):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [_DINO_TRANSFORM(Image.open(p).convert('RGB')) for p in paths[i:i+batch_size]]
        x = torch.stack(batch).to(DEVICE)
        out = model.forward_features(x)
        cls = F.normalize(out['x_norm_clstoken'], dim=-1)
        feats.append(cls.cpu().float().numpy())
        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(paths):
            logging.info(f'  [DINOv3] {min(i+batch_size, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


_RADIO_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
])
_RADIO_PATCH = 16


@torch.no_grad()
def extract_radio_img(model, conditioner, paths, out_path, force=False, batch_size=128):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [_RADIO_TRANSFORM(Image.open(p).convert('RGB')) for p in paths[i:i+batch_size]]
        x = torch.stack(batch).to(DEVICE)
        if conditioner is not None:
            x = conditioner(x)
        out = model(x)
        summary = out[0] if isinstance(out, (tuple, list)) else getattr(out, 'summary', out[0])
        cls = F.normalize(summary, dim=-1)
        feats.append(cls.cpu().float().numpy())
        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(paths):
            logging.info(f'  [RADIO] {min(i+batch_size, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


_EUPE_TRANSFORM = T.Compose([
    T.Resize((224, 224), interpolation=T.InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])
_EUPE_PATCH = 16


@torch.no_grad()
def extract_eupe_img(model, paths, out_path, force=False, batch_size=128):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    dev_type = DEVICE.type
    for i in range(0, len(paths), batch_size):
        batch = [_EUPE_TRANSFORM(Image.open(p).convert('RGB')) for p in paths[i:i+batch_size]]
        x = torch.stack(batch).to(DEVICE)
        with torch.autocast(device_type=dev_type, dtype=torch.bfloat16,
                            enabled=(dev_type != 'cpu')):
            out = model.forward_features(x)
        cls = F.normalize(out['x_norm_clstoken'], dim=-1)
        feats.append(cls.cpu().float().numpy())
        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(paths):
            logging.info(f'  [EUPE] {min(i+batch_size, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


_TIPS_TRANSFORM = T.ToTensor()
_TIPS_SIZE = 448


@torch.no_grad()
def extract_tips_img(model, paths, out_path, force=False, batch_size=64):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(paths), batch_size):
        batch = [_TIPS_TRANSFORM(Image.open(p).convert('RGB').resize(
            (_TIPS_SIZE, _TIPS_SIZE), Image.BICUBIC)) for p in paths[i:i+batch_size]]
        x = torch.stack(batch).to(DEVICE)
        out = model.encode_image(x)
        cls = F.normalize(out.cls_token.squeeze(1), dim=-1)
        feats.append(cls.cpu().float().numpy())
        if (i // batch_size + 1) % 5 == 0 or i + batch_size >= len(paths):
            logging.info(f'  [TIPSv2] {min(i+batch_size, len(paths))}/{len(paths)}')
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_tips_txt(model, tokenizer, captions, out_path, force=False, batch_size=512):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(captions), batch_size):
        batch = captions[i:i+batch_size]
        ids, pads = tokenizer.tokenize(batch, max_len=model.config.max_len)
        ids  = torch.from_numpy(ids).to(DEVICE)
        pads = torch.from_numpy(pads).to(DEVICE)
        feats.append(F.normalize(model.encode_text(ids, pads), dim=-1).cpu().float().numpy())
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat)
    return feat


@torch.no_grad()
def extract_wds_features(
        pe_model, pe_preproc, pe_tok,
        sig2_model, sig2_preproc, sig2_tok,
        dino_model, radio_model, radio_cond, eupe_model,
        tips_model, tips_tok,
        pattern, out_dir, max_samples=100000, force=False, batch_size=64):
    """Stream wds; extract all active models' features in one pass."""
    import webdataset as wds

    npz_map = {
        'pe_img':    os.path.join(out_dir, 'pe_core_img.npz'),
        'pe_txt':    os.path.join(out_dir, 'pe_core_txt.npz'),
        'sig2_img':  os.path.join(out_dir, 'siglip2_img.npz'),
        'sig2_txt':  os.path.join(out_dir, 'siglip2_txt.npz'),
        'dino_img':  os.path.join(out_dir, 'dinov3_img.npz'),
        'radio_img': os.path.join(out_dir, 'radio_img.npz'),
        'eupe_img':  os.path.join(out_dir, 'eupe_img.npz'),
        'tips_img':  os.path.join(out_dir, 'tips_img.npz'),
        'tips_txt':  os.path.join(out_dir, 'tips_txt.npz'),
    }
    _active_map = {
        'pe_img':    pe_model    is not None,
        'pe_txt':    pe_model    is not None,
        'sig2_img':  sig2_model  is not None,
        'sig2_txt':  sig2_model  is not None,
        'dino_img':  dino_model  is not None,
        'radio_img': radio_model is not None,
        'eupe_img':  eupe_model  is not None,
        'tips_img':  tips_model  is not None,
        'tips_txt':  tips_model  is not None and tips_tok is not None,
    }
    active = {k: p for k, p in npz_map.items() if _active_map[k]}
    if not force and all(os.path.exists(p) for p in active.values()):
        logging.info('[cache] Loading all wds features from cache ...')
        return {k: np.load(p)['features'] for k, p in active.items()}

    acc  = {k: [] for k in active}
    bufs = {k: [] for k in ['imgs', 'cap']}
    count = 0

    ds = (wds.WebDataset(pattern, shardshuffle=False)
          .decode('pil')
          .to_tuple('jpg', 'txt'))

    def _flush():
        imgs = bufs['imgs']
        caps = bufs['cap']
        with torch.no_grad():
            if 'pe_img' in active or 'pe_txt' in active:
                pb = torch.stack([pe_preproc(im) for im in imgs]).to(DEVICE)
                if 'pe_img' in active:
                    acc['pe_img'].append(pe_model.encode_image(pb, normalize=True).cpu().float().numpy())
                if 'pe_txt' in active:
                    pt = pe_tok(caps).to(DEVICE)
                    acc['pe_txt'].append(pe_model.encode_text(pt, normalize=True).cpu().float().numpy())

            if 'sig2_img' in active or 'sig2_txt' in active:
                sb = torch.stack([sig2_preproc(im) for im in imgs]).to(DEVICE)
                if 'sig2_img' in active:
                    acc['sig2_img'].append(sig2_model.encode_image(sb, normalize=True).cpu().float().numpy())
                if 'sig2_txt' in active:
                    st = sig2_tok(caps).to(DEVICE)
                    acc['sig2_txt'].append(sig2_model.encode_text(st, normalize=True).cpu().float().numpy())

            if 'dino_img' in active:
                dx = torch.stack([_DINO_TRANSFORM(im) for im in imgs]).to(DEVICE)
                dout = dino_model.forward_features(dx)
                acc['dino_img'].append(F.normalize(dout['x_norm_clstoken'], dim=-1).cpu().float().numpy())

            if 'radio_img' in active:
                rx = torch.stack([_RADIO_TRANSFORM(im) for im in imgs]).to(DEVICE)
                if radio_cond is not None:
                    rx = radio_cond(rx)
                rout = radio_model(rx)
                summary = rout[0] if isinstance(rout, (tuple, list)) else getattr(rout, 'summary', rout[0])
                acc['radio_img'].append(F.normalize(summary, dim=-1).cpu().float().numpy())

            if 'eupe_img' in active:
                ex = torch.stack([_EUPE_TRANSFORM(im) for im in imgs]).to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                    enabled=(DEVICE.type != 'cpu')):
                    eout = eupe_model.forward_features(ex)
                acc['eupe_img'].append(F.normalize(eout['x_norm_clstoken'], dim=-1).cpu().float().numpy())

            if 'tips_img' in active:
                tx = torch.stack([_TIPS_TRANSFORM(
                    im.resize((_TIPS_SIZE, _TIPS_SIZE), Image.BICUBIC)) for im in imgs]).to(DEVICE)
                tout = tips_model.encode_image(tx)
                acc['tips_img'].append(F.normalize(tout.cls_token.squeeze(1), dim=-1).cpu().float().numpy())

            if 'tips_txt' in active:
                ids, pads = tips_tok.tokenize(caps, max_len=tips_model.config.max_len)
                ids  = torch.from_numpy(ids).to(DEVICE)
                pads = torch.from_numpy(pads).to(DEVICE)
                acc['tips_txt'].append(F.normalize(
                    tips_model.encode_text(ids, pads), dim=-1).cpu().float().numpy())

    for img, cap in ds:
        bufs['imgs'].append(img)
        bufs['cap'].append(cap)
        count += 1
        if len(bufs['imgs']) == batch_size:
            _flush()
            bufs = {k: [] for k in bufs}
            logging.info(f'  wds extracted {count}/{max_samples} ...')
        if count >= max_samples:
            break
    if bufs['imgs']:
        _flush()

    result = {k: np.concatenate(v, 0) for k, v in acc.items() if v}
    for k, p in active.items():
        if k in result:
            np.savez_compressed(p, features=result[k])
    logging.info('  wds done: ' + ', '.join(f'{k} {v.shape}' for k, v in result.items()))
    return result


# ─── Analysis tools ───────────────────────────────────────────────────────────

def fps_sample(feats: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    """Farthest Point Sampling in embedding space. Returns k indices."""
    rng = np.random.default_rng(seed)
    n = len(feats)
    chosen = [int(rng.integers(n))]
    dists  = np.full(n, np.inf)
    for _ in range(k - 1):
        d = ((feats - feats[chosen[-1]]) ** 2).sum(axis=1)
        dists = np.minimum(dists, d)
        chosen.append(int(np.argmax(dists)))
    return np.array(chosen)


def compute_anisotropy(feats: np.ndarray) -> dict:
    """
    Full-dimensional anisotropy metrics (all singular values, not just top-4).

    Returns:
      effective_rank   : exp(entropy of normalized eigenvalue distribution)  [1, D]
                         higher = more isotropic
      participation_ratio: (Σλ)² / (D·Σλ²)  ∈ (0,1]   higher = more isotropic
      avg_cos_sim      : mean pairwise cosine similarity (subsample 2000)
                         lower = more isotropic / spread out
      pct_var_top{k}   : % variance explained by top-k PCs  (k=4,10,50,100)
    """
    # Center features
    f = feats - feats.mean(axis=0, keepdims=True)
    # Covariance eigenvalues via SVD on centered matrix
    # For large N, use randomized SVD capped at D components
    D = f.shape[1]
    k_svd = min(D, f.shape[0], 512)   # cap for speed
    from sklearn.utils.extmath import randomized_svd
    _, s, _ = randomized_svd(f, n_components=k_svd, random_state=0)
    lam = s ** 2  # eigenvalues of scatter matrix (proportional to cov)
    lam = lam / lam.sum()

    # Effective rank
    H   = -(lam * np.log(lam + 1e-12)).sum()
    eff_rank = float(np.exp(H))

    # Participation ratio
    pr = float(lam.sum() ** 2 / (k_svd * (lam ** 2).sum()))

    # Average cosine similarity (subsample 2000)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(feats), size=min(2000, len(feats)), replace=False)
    sub = feats[idx]
    sub = sub / (np.linalg.norm(sub, axis=1, keepdims=True) + 1e-8)
    cos = sub @ sub.T
    # Exclude diagonal
    n_sub = len(sub)
    avg_cos = float((cos.sum() - n_sub) / (n_sub * (n_sub - 1)))

    # % variance by top-k PCs
    cum = np.cumsum(lam)
    pct = {}
    for topk in [4, 10, 50, 100]:
        pct[f'pct_var_top{topk}'] = float(cum[min(topk, len(cum)) - 1] * 100)

    return dict(
        effective_rank=eff_rank,
        participation_ratio=pr,
        avg_cos_sim=avg_cos,
        n_components=k_svd,
        eigenvalues=lam,
        **pct,
    )


# ─── Visualization helpers ────────────────────────────────────────────────────

_FPS_MARKERS = ['*', 'D', '^', 's', 'P']
_FPS_COLORS  = ['red', 'lime', 'cyan', 'orange', 'magenta']
_FPS_SIZE    = 120


def _fit_pca(feats_list, n):
    dims = [f.shape[1] for f in feats_list]
    if len(set(dims)) == 1:
        pca = PCA(n_components=n).fit(np.concatenate(feats_list, 0))
        return [pca] * len(feats_list), pca.explained_variance_ratio_
    else:
        pcas = [PCA(n_components=n).fit(f) for f in feats_list]
        return pcas, None


def plot_scatter(feats_dict, title, save_path, n_pca=4, fps_indices=None):
    """
    Multi-axis PCA scatter.
    fps_indices: if given, highlight these sample indices with consistent markers
                 across all subplots (same marker = same sample).
    Same-dim → shared PCA; diff-dim → independent per-model rows.
    """
    labels = list(feats_dict.keys())
    feats  = list(feats_dict.values())
    colors = cm.tab10(np.linspace(0, 0.9, len(labels)))
    pairs  = list(combinations(range(n_pca), 2))
    pcas, shared_var = _fit_pca(feats, n_pca)
    projs  = [pca.transform(f) for pca, f in zip(pcas, feats)]
    shared = shared_var is not None
    ncols  = len(pairs) + 1

    def _draw_fps(ax, proj, pi, pj):
        if fps_indices is None:
            return
        for fi, (idx, mk, fc) in enumerate(zip(fps_indices, _FPS_MARKERS, _FPS_COLORS)):
            ax.scatter(proj[idx, pi], proj[idx, pj],
                       marker=mk, s=_FPS_SIZE, color=fc,
                       edgecolors='black', linewidths=0.5,
                       zorder=5, label=f'FPS-{fi}' if (pi == 0 and pj == 1) else '')

    if shared:
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
        for col, (pi, pj) in enumerate(pairs):
            ax = axes[col]
            for label, proj, c in zip(labels, projs, colors):
                ax.scatter(proj[:, pi], proj[:, pj], s=3, alpha=0.3,
                           color=c, rasterized=True,
                           label=label if col == 0 else '')
            _draw_fps(ax, projs[0], pi, pj)   # fps anchored to first model's proj
            ax.set_xlabel(f'PC{pi+1}'); ax.set_ylabel(f'PC{pj+1}')
            ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
            if col == 0:
                ax.legend(markerscale=4, fontsize=8, loc='best')
        ax = axes[-1]
        ax.bar(range(1, n_pca + 1), shared_var * 100, color='steelblue')
        ax.set_xlabel('Component'); ax.set_ylabel('Variance explained (%)')
        ax.set_title('Explained variance', fontsize=9)
        fig.suptitle(title, fontsize=12, y=1.01)
    else:
        nrows = len(labels)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        for row, (label, pca, proj, feat, c) in enumerate(
                zip(labels, pcas, projs, feats, colors)):
            var = pca.explained_variance_ratio_
            for col, (pi, pj) in enumerate(pairs):
                ax = axes[row, col]
                ax.scatter(proj[:, pi], proj[:, pj], s=3, alpha=0.3,
                           color=c, rasterized=True)
                # project fps indices (computed in PE space) into this model's PCA
                _draw_fps(ax, proj, pi, pj)
                ax.set_xlabel(f'PC{pi+1}')
                ax.set_ylabel(f'{label}  (dim={feat.shape[1]})\nPC{pj+1}'
                              if col == 0 else f'PC{pj+1}')
                ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
            ax = axes[row, -1]
            ax.bar(range(1, n_pca + 1), var * 100, color=c)
            ax.set_xlabel('Component'); ax.set_ylabel('Var. explained (%)')
            ax.set_title(f'{label}  explained var.', fontsize=9)
        fig.suptitle(title + '  [independent PCA axes]', fontsize=12, y=1.01)
        # FPS legend in top-right corner of first row
        if fps_indices is not None:
            handles = [plt.scatter([], [], marker=mk, s=_FPS_SIZE, color=fc,
                                   edgecolors='black', label=f'FPS-{fi}')
                       for fi, (mk, fc) in enumerate(zip(_FPS_MARKERS, _FPS_COLORS))]
            fig.legend(handles=handles, loc='upper right', fontsize=9, ncol=len(handles))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved: {save_path}')


def plot_overlap(feats_a, feats_b, label_a, label_b, model_name, save_path, n_pca=2):
    """
    Overlay two distributions (e.g. COCO vs CC3M) in shared PCA space.
    PCA fitted on combined data.
    """
    combined = np.concatenate([feats_a, feats_b], 0)
    pca = PCA(n_components=n_pca).fit(combined)
    pa  = pca.transform(feats_a)
    pb  = pca.transform(feats_b)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, (pi, pj) in zip(axes, [(0, 1)] * 2):
        ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color='steelblue',
                   label=label_a, rasterized=True)
        ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.3, color='coral',
                   label=label_b, rasterized=True)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(markerscale=4, fontsize=9)
    axes[0].set_title(f'{model_name}: {label_a} vs {label_b}', fontsize=10)
    # centroid distance in PCA space
    d = np.linalg.norm(pa.mean(0) - pb.mean(0))
    axes[1].set_title(f'centroid dist (PC1-2) = {d:.3f}', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved: {save_path}')


def plot_anisotropy(metrics_dict: dict, save_path: str):
    """
    Bar charts comparing anisotropy metrics across models.
    metrics_dict: {model_name: compute_anisotropy(...) result}
    """
    models  = list(metrics_dict.keys())
    colors  = cm.tab10(np.linspace(0, 0.9, len(models)))
    n_m     = len(models)

    scalar_keys = ['effective_rank', 'participation_ratio', 'avg_cos_sim',
                   'pct_var_top4', 'pct_var_top10', 'pct_var_top50', 'pct_var_top100']
    labels_nice = ['Effective Rank', 'Participation Ratio', 'Avg Cosine Sim',
                   'Var% top-4', 'Var% top-10', 'Var% top-50', 'Var% top-100']

    fig, axes = plt.subplots(1, len(scalar_keys) + 1,
                             figsize=(3.5 * (len(scalar_keys) + 1), 4.5))

    for ax, key, lbl in zip(axes[:-1], scalar_keys, labels_nice):
        vals = [metrics_dict[m][key] for m in models]
        bars = ax.bar(range(n_m), vals, color=colors)
        ax.set_xticks(range(n_m)); ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
        ax.set_title(lbl, fontsize=9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.2f}', ha='center', va='bottom', fontsize=7)

    # Eigenvalue spectrum (normalized, top-100)
    ax = axes[-1]
    for m, c in zip(models, colors):
        eigs = metrics_dict[m]['eigenvalues'][:100]
        ax.plot(np.arange(1, len(eigs) + 1), eigs * 100, color=c, label=m, lw=1.2)
    ax.set_xlabel('PC index'); ax.set_ylabel('Variance % (of total)')
    ax.set_title('Eigenvalue spectrum (top-100)', fontsize=9)
    ax.legend(fontsize=7)

    fig.suptitle('Feature Anisotropy Metrics', fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved: {save_path}')


def plot_evolution(epoch_feats, epoch_ids, save_dir, n_traj=100, seed=42):
    """Evolution grid + trajectory. PCA fitted on final epoch as reference."""
    pca   = PCA(n_components=2).fit(epoch_feats[-1])
    projs = [pca.transform(f) for f in epoch_feats]
    n     = len(epoch_ids)

    all_pts = np.concatenate(projs, 0)
    pad = 0.05
    x0, x1 = all_pts[:, 0].min(), all_pts[:, 0].max()
    y0, y1 = all_pts[:, 1].min(), all_pts[:, 1].max()
    xp, yp = (x1 - x0) * pad, (y1 - y0) * pad
    xlim = (x0 - xp, x1 + xp)
    ylim = (y0 - yp, y1 + yp)

    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (eid, proj) in enumerate(zip(epoch_ids, projs)):
        axes[i].scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.4,
                        c=np.arange(len(proj)), cmap='viridis', rasterized=True)
        axes[i].set_title(f'Epoch {eid}', fontsize=9)
        axes[i].set_xlim(xlim); axes[i].set_ylim(ylim)
        axes[i].axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Image Feature Distribution per Epoch  [PCA fitted on final epoch]', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'epoch_evolution.png'), dpi=150)
    plt.close()
    logging.info(f'Saved: {os.path.join(save_dir, "epoch_evolution.png")}')

    rng    = np.random.default_rng(seed)
    idx    = rng.choice(len(epoch_feats[0]), size=min(n_traj, len(epoch_feats[0])), replace=False)
    colors = cm.tab20(np.linspace(0, 1, len(idx)))
    alphas = np.linspace(0.10, 1.00, n)
    lws    = np.linspace(0.3,  1.8,  n)

    fig, ax = plt.subplots(figsize=(8, 7))
    for si, color in zip(idx, colors):
        pts = np.array([p[si] for p in projs])
        for t in range(len(pts) - 1):
            ax.plot(pts[t:t+2, 0], pts[t:t+2, 1], '-',
                    color=color, alpha=float(alphas[t + 1]), lw=float(lws[t + 1]))
        ax.scatter(pts[0, 0],  pts[0, 1],  color=color, s=12, zorder=3,
                   marker='o', alpha=float(alphas[0]))
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=40, zorder=4,
                   marker='*', alpha=1.0)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(f'Sample Trajectories  N={len(idx)}\n'
                 f'o=start  ★=end  light→dark = early→late epoch', fontsize=10)
    ax.set_xlabel('PC1  (final epoch)'); ax.set_ylabel('PC2  (final epoch)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory.png'), dpi=150)
    plt.close()
    logging.info(f'Saved: {os.path.join(save_dir, "trajectory.png")}')


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_pretrained(args):
    out = os.path.join(args.out_dir, 'pretrained')
    os.makedirs(out, exist_ok=True)

    if args.data_type == 'wds':
        pe_model,   pe_preproc,   pe_tok   = load_pe_core(args.pe_ckpt)
        sig2_model, sig2_preproc, sig2_tok = load_siglip2(args.sig2_ckpt)
        dino_model  = load_dinov3(args.dino_repo, args.dino_ckpt)
        radio_model, radio_cond = load_radio(args.radio)
        eupe_model  = load_eupe(args.eupe_repo, args.eupe_ckpt)
        tips_model, tips_tok = load_tips(args.tips)

        feats = extract_wds_features(
            pe_model, pe_preproc, pe_tok,
            sig2_model, sig2_preproc, sig2_tok,
            dino_model, radio_model, radio_cond, eupe_model, tips_model, tips_tok,
            args.data, out, max_samples=args.max_samples, force=args.force,
        )
        pe_img, pe_txt     = feats['pe_img'],   feats['pe_txt']
        sig2_img, sig2_txt = feats['sig2_img'], feats['sig2_txt']
        dino_img           = feats['dino_img']
        radio_img          = feats.get('radio_img')
        eupe_img           = feats.get('eupe_img')
        tips_img           = feats.get('tips_img')
        tips_txt           = feats.get('tips_txt')

    else:
        # ── TSV mode (COCO) ──────────────────────────────────────────────────
        df = pd.read_csv(args.data, sep='\t')
        paths, captions = df['filepath'].tolist(), df['caption'].tolist()

        logging.info('Loading PE-Core ...')
        pe_model, pe_preproc, pe_tok = load_pe_core(args.pe_ckpt)
        pe_img = extract_clip_img(pe_model, paths, pe_preproc,
                                  os.path.join(out, 'pe_core_img.npz'), args.force)
        pe_txt = extract_clip_txt(pe_model, pe_tok, captions,
                                  os.path.join(out, 'pe_core_txt.npz'), args.force)
        del pe_model; torch.cuda.empty_cache()

        logging.info('Loading SigLIP2 ...')
        sig2_model, sig2_preproc, sig2_tok = load_siglip2(args.sig2_ckpt)
        sig2_img = extract_clip_img(sig2_model, paths, sig2_preproc,
                                    os.path.join(out, 'siglip2_img.npz'), args.force)
        sig2_txt = extract_clip_txt(sig2_model, sig2_tok, captions,
                                    os.path.join(out, 'siglip2_txt.npz'), args.force)
        del sig2_model; torch.cuda.empty_cache()

        logging.info('Loading DINOv3 ...')
        dino_model = load_dinov3(args.dino_repo, args.dino_ckpt)
        dino_img = extract_dinov3_img(dino_model, paths,
                                      os.path.join(out, 'dinov3_img.npz'), args.force)
        del dino_model; torch.cuda.empty_cache()

        logging.info('Loading C-RADIOv4 ...')
        radio_model, radio_cond = load_radio(args.radio)
        radio_img = extract_radio_img(radio_model, radio_cond, paths,
                                      os.path.join(out, 'radio_img.npz'), args.force)
        del radio_model; torch.cuda.empty_cache()

        eupe_model = load_eupe(args.eupe_repo, args.eupe_ckpt)
        if eupe_model is not None:
            eupe_img = extract_eupe_img(eupe_model, paths,
                                        os.path.join(out, 'eupe_img.npz'), args.force)
            del eupe_model; torch.cuda.empty_cache()
        else:
            eupe_img = None

        logging.info('Loading TIPSv2 ...')
        tips_model, tips_tok = load_tips(args.tips)
        tips_img = extract_tips_img(tips_model, paths,
                                    os.path.join(out, 'tips_img.npz'), args.force)
        tips_txt = extract_tips_txt(tips_model, tips_tok, captions,
                                    os.path.join(out, 'tips_txt.npz'), args.force)
        del tips_model; torch.cuda.empty_cache()

    # ── FPS on PE image features ──────────────────────────────────────────────
    fps_idx = fps_sample(pe_img, k=5)
    logging.info(f'FPS anchor indices (PE space): {fps_idx.tolist()}')

    # ── Plots ─────────────────────────────────────────────────────────────────
    # 1. PE-Core modality gap
    plot_scatter(
        {'PE-Core Image': pe_img, 'PE-Core Text': pe_txt},
        'PE-Core: Image vs Text Feature Distribution',
        os.path.join(out, 'pe_core_modality_gap.png'), n_pca=args.n_pca,
    )
    # 2. SigLIP2 modality gap
    plot_scatter(
        {'SigLIP2 Image': sig2_img, 'SigLIP2 Text': sig2_txt},
        'SigLIP2: Image vs Text Feature Distribution',
        os.path.join(out, 'siglip2_modality_gap.png'), n_pca=args.n_pca,
    )
    # 3. TIPSv2 modality gap
    plot_scatter(
        {'TIPSv2 Image': tips_img, 'TIPSv2 Text': tips_txt},
        'TIPSv2: Image vs Text Feature Distribution',
        os.path.join(out, 'tips_modality_gap.png'), n_pca=args.n_pca,
    )
    # 4. All-model image comparison with FPS tracking
    img_feats = {k: v for k, v in [
        ('DINOv3',  dino_img),
        ('RADIO',   radio_img),
        ('EUPE',    eupe_img),
        ('TIPSv2',  tips_img),
        ('PE-Core', pe_img),
        ('SigLIP2', sig2_img),
    ] if v is not None}
    plot_scatter(
        img_feats,
        'Vision Encoder Image Feature Comparison  (★ = FPS anchors from PE space)',
        os.path.join(out, 'image_allmodels.png'), n_pca=args.n_pca,
        fps_indices=fps_idx,
    )
    # 5. Anisotropy
    aniso = {name: compute_anisotropy(feat) for name, feat in img_feats.items()}
    plot_anisotropy(aniso, os.path.join(out, 'anisotropy.png'))
    # Print summary table
    logging.info('\n=== Anisotropy summary ===')
    header = f"{'Model':<12} {'EffRank':>8} {'PR':>6} {'AvgCos':>8} " \
             f"{'top4%':>7} {'top10%':>7} {'top50%':>7} {'top100%':>8}"
    logging.info(header)
    for name, m in aniso.items():
        logging.info(f"{name:<12} {m['effective_rank']:8.1f} {m['participation_ratio']:6.4f} "
                     f"{m['avg_cos_sim']:8.4f} {m['pct_var_top4']:7.1f} "
                     f"{m['pct_var_top10']:7.1f} {m['pct_var_top50']:7.1f} "
                     f"{m['pct_var_top100']:8.1f}")


def run_overlap(args):
    """Load cached COCO + CC3M npz, plot per-model distribution overlap."""
    # Model → npz filename mapping
    npz_files = {
        'PE-Core':  ('pe_core_img.npz',  'pe_core_img.npz'),
        'SigLIP2':  ('siglip2_img.npz',  'siglip2_img.npz'),
        'DINOv3':   ('dinov3_img.npz',   'dinov3_img.npz'),
        'RADIO':    ('radio_img.npz',    'radio_img.npz'),
        'EUPE':     ('eupe_img.npz',     'eupe_img.npz'),
        'TIPSv2':   ('tips_img.npz',     'tips_img.npz'),
    }
    out = os.path.join(args.cc3m_dir, '..', 'overlap')
    os.makedirs(out, exist_ok=True)

    available = {}
    for model, (coco_f, cc3m_f) in npz_files.items():
        cp = os.path.join(args.coco_dir, coco_f)
        mp = os.path.join(args.cc3m_dir, cc3m_f)
        if os.path.exists(cp) and os.path.exists(mp):
            available[model] = (cp, mp)
        else:
            logging.info(f'  skip {model}: missing cache ({cp} or {mp})')

    assert available, 'No cached npz pairs found. Run pretrained mode for both COCO and CC3M first.'

    # One combined grid: rows = models, single PC1 vs PC2 panel
    n_models = len(available)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, (model, (cp, mp)) in zip(axes, available.items()):
        fa = np.load(cp)['features']
        fb = np.load(mp)['features']
        # subsample for speed
        rng = np.random.default_rng(42)
        fa = fa[rng.choice(len(fa), min(5000, len(fa)), replace=False)]
        fb = fb[rng.choice(len(fb), min(5000, len(fb)), replace=False)]
        combined = np.concatenate([fa, fb], 0)
        pca = PCA(n_components=2).fit(combined)
        pa, pb = pca.transform(fa), pca.transform(fb)
        ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color='steelblue',
                   label='COCO', rasterized=True)
        ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.3, color='coral',
                   label='CC3M', rasterized=True)
        d = np.linalg.norm(pa.mean(0) - pb.mean(0))
        ax.set_title(f'{model}\ncentroid dist={d:.3f}', fontsize=9)
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
        ax.legend(markerscale=4, fontsize=8)

        # Also save individual plot
        plot_overlap(fa, fb, 'COCO', 'CC3M', model,
                     os.path.join(out, f'overlap_{model.lower()}.png'))

    fig.suptitle('COCO vs CC3M Feature Distribution Overlap (PC1 vs PC2)', fontsize=11)
    plt.tight_layout()
    grid_path = os.path.join(out, 'overlap_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved: {grid_path}')


def run_anisotropy(args):
    """Compute anisotropy from cached npz in a single directory."""
    npz_map = {
        'PE-Core':  'pe_core_img.npz',
        'SigLIP2':  'siglip2_img.npz',
        'DINOv3':   'dinov3_img.npz',
        'RADIO':    'radio_img.npz',
        'EUPE':     'eupe_img.npz',
        'TIPSv2':   'tips_img.npz',
    }
    metrics = {}
    for name, fname in npz_map.items():
        p = os.path.join(args.aniso_dir, fname)
        if not os.path.exists(p):
            logging.info(f'  skip {name}: {p} not found')
            continue
        f = np.load(p)['features']
        logging.info(f'  computing {name}  shape={f.shape} ...')
        metrics[name] = compute_anisotropy(f)

    assert metrics, f'No npz found in {args.aniso_dir}'
    plot_anisotropy(metrics, os.path.join(args.aniso_dir, 'anisotropy.png'))

    logging.info('\n=== Anisotropy summary ===')
    header = f"{'Model':<12} {'EffRank':>8} {'PR':>6} {'AvgCos':>8} " \
             f"{'top4%':>7} {'top10%':>7} {'top50%':>7} {'top100%':>8}"
    logging.info(header)
    for name, m in metrics.items():
        logging.info(f"{name:<12} {m['effective_rank']:8.1f} {m['participation_ratio']:6.4f} "
                     f"{m['avg_cos_sim']:8.4f} {m['pct_var_top4']:7.1f} "
                     f"{m['pct_var_top10']:7.1f} {m['pct_var_top50']:7.1f} "
                     f"{m['pct_var_top100']:8.1f}")


def run_epochs(args):
    files = sorted(glob.glob(os.path.join(args.probe_dir, 'epoch_*.npz')))
    assert files, f'No epoch_*.npz found in {args.probe_dir}'
    epoch_ids, epoch_feats = [], []
    for f in files:
        eid = int(os.path.splitext(os.path.basename(f))[0].split('_')[1])
        epoch_ids.append(eid)
        epoch_feats.append(np.load(f)['features'])
        logging.info(f'  epoch {eid:02d}: {epoch_feats[-1].shape}')
    out = os.path.join(args.probe_dir, 'plots')
    os.makedirs(out, exist_ok=True)
    plot_evolution(epoch_feats, epoch_ids, out, n_traj=args.n_traj)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', choices=['pretrained', 'epochs', 'overlap', 'anisotropy'],
                   required=True)
    # pretrained
    p.add_argument('--data',        default=_DEFAULTS['data'])
    p.add_argument('--data-type',   choices=['tsv', 'wds'], default='tsv')
    p.add_argument('--out-dir',     default=_DEFAULTS['out_dir'])
    p.add_argument('--pe-ckpt',     default=_DEFAULTS['pe_ckpt'])
    p.add_argument('--sig2-ckpt',   default=_DEFAULTS['sig2_ckpt'])
    p.add_argument('--dino-repo',   default=_DEFAULTS['dino_repo'])
    p.add_argument('--dino-ckpt',   default=_DEFAULTS['dino_ckpt'])
    p.add_argument('--radio',       default=_DEFAULTS['radio'])
    p.add_argument('--eupe-repo',   default=_DEFAULTS['eupe_repo'])
    p.add_argument('--eupe-ckpt',   default=_DEFAULTS['eupe_ckpt'])
    p.add_argument('--tips',        default=_DEFAULTS['tips'])
    p.add_argument('--max-samples', type=int, default=100000)
    p.add_argument('--n-pca',       type=int, default=4)
    p.add_argument('--force',       action='store_true')
    # epochs
    p.add_argument('--probe-dir',   default=None)
    p.add_argument('--n-traj',      type=int, default=100)
    # overlap
    p.add_argument('--coco-dir',    default=_DEFAULTS['coco_dir'])
    p.add_argument('--cc3m-dir',    default=_DEFAULTS['cc3m_dir'])
    # anisotropy (standalone)
    p.add_argument('--aniso-dir',   default=_DEFAULTS['coco_dir'])
    args = p.parse_args()

    if args.mode == 'pretrained':
        run_pretrained(args)
    elif args.mode == 'epochs':
        assert args.probe_dir, '--probe-dir required for epochs mode'
        run_epochs(args)
    elif args.mode == 'overlap':
        run_overlap(args)
    elif args.mode == 'anisotropy':
        run_anisotropy(args)


if __name__ == '__main__':
    main()
