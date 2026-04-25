"""Multi-model patch-feature PCA visualization.

Compares patch-level PCA RGB maps for 4 models:
  DINOv3 / C-RADIOv4 / TIPSv2 / EUPE

Usage (from repo root):
  python -m analysis.pca_demo --img_dir <dir> --num_images 6 --output_dir <out>
"""
import argparse, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from sklearn.decomposition import PCA

from .models import load_dinov3, load_radio, load_eupe, load_tips, DEVICE

# ── Image prep ────────────────────────────────────────────────────────────────

def _align(img: Image.Image, ps: int) -> Image.Image:
    W, H = img.size
    W, H = (W // ps) * ps, (H // ps) * ps
    return img.crop((0, 0, W, H)) if (W, H) != img.size else img


def _short_edge(img: Image.Image, short: int) -> Image.Image:
    W, H = img.size
    s = short / min(W, H)
    return img.resize((int(W * s), int(H * s)), Image.BICUBIC)


# ── Patch extractors (return H_p × W_p × D numpy) ────────────────────────────

_DINO_TF = T.Compose([T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
_TIPS_SZ = 448


@torch.no_grad()
def _patch_dinov3(img, info):
    ps = 16; img = _align(img, ps); W, H = img.size
    x  = _DINO_TF(img).unsqueeze(0).to(DEVICE)
    t  = info['model'].forward_features(x)['x_norm_patchtokens'][0].cpu().float().numpy()
    return t.reshape(H // ps, W // ps, -1)


@torch.no_grad()
def _patch_radio(img, info):
    ps = 16; img = _align(img, ps); W, H = img.size
    x  = T.ToTensor()(img).unsqueeze(0).to(DEVICE)
    if info['cond'] is not None:
        x = info['cond'](x)
    out = info['model'](x)
    sp = out[1] if isinstance(out, (tuple, list)) else \
         getattr(out, 'spatial_features', getattr(out, 'patch_features', None))
    return sp[0].cpu().float().numpy().reshape(H // ps, W // ps, -1)


@torch.no_grad()
def _patch_eupe(img, info):
    if info is None:
        return None
    ps = 16; img = _align(img, ps); W, H = img.size
    tf = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    x  = tf(img).unsqueeze(0).to(DEVICE)
    dev = DEVICE.type
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != 'cpu')):
        t = info['model'].forward_features(x)['x_norm_patchtokens']
    return t[0].cpu().float().numpy().reshape(H // ps, W // ps, -1)


@torch.no_grad()
def _patch_tips(img, info):
    ps = 14; img = img.resize((_TIPS_SZ, _TIPS_SZ), Image.BICUBIC)
    x  = T.ToTensor()(img).unsqueeze(0).to(DEVICE)
    dev = DEVICE.type
    with torch.autocast(device_type=dev, dtype=torch.bfloat16, enabled=(dev != 'cpu')):
        out = info['model'].encode_image(x)
    tokens = out.patch_tokens[0].cpu().float().numpy()
    N = _TIPS_SZ // ps
    return tokens.reshape(N, N, -1)


# ── Global PCA per model ──────────────────────────────────────────────────────

def _pca_rgb(feats_list: list) -> list:
    """Fit global PCA across all images for one model; returns normalized RGB maps."""
    valid = [(i, f) for i, f in enumerate(feats_list) if f is not None]
    if not valid:
        return [None] * len(feats_list)
    shapes = [(f.shape[0], f.shape[1]) for _, f in valid]
    all_t  = np.concatenate([f.reshape(-1, f.shape[-1]) for _, f in valid])
    pca    = PCA(n_components=3)
    mapped = pca.fit_transform(all_t)
    lo, hi = mapped.min(0, keepdims=True), mapped.max(0, keepdims=True)
    mapped = (mapped - lo) / (hi - lo + 1e-8)
    result, cursor = [None] * len(feats_list), 0
    for (list_i, _), (H, W) in zip(valid, shapes):
        n = H * W
        result[list_i] = mapped[cursor:cursor + n].reshape(H, W, 3)
        cursor += n
    return result


def _to_rgb(pca_map: np.ndarray, size) -> Image.Image:
    return Image.fromarray((pca_map * 255).astype(np.uint8)).resize(size, Image.NEAREST)


# ── Visualization ─────────────────────────────────────────────────────────────

def _draw_grid(images, all_pca, model_names, path):
    n_imgs = len(images); n_cols = 1 + len(model_names)
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_imgs))
    gs  = gridspec.GridSpec(n_imgs, n_cols, figure=fig, hspace=0.04, wspace=0.04)
    for j, t in enumerate(['Original'] + model_names):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(t, fontsize=10, pad=6); ax.axis('off')
    for i, img in enumerate(images):
        ax = fig.add_subplot(gs[i, 0]); ax.imshow(img); ax.axis('off')
        for j, pca_maps in enumerate(all_pca):
            ax = fig.add_subplot(gs[i, 1 + j])
            pm = pca_maps[i]
            if pm is not None:
                ax.imshow(_to_rgb(pm, img.size))
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                        fontsize=12, color='gray', transform=ax.transAxes)
            ax.axis('off')
    plt.savefig(path, bbox_inches='tight', dpi=150); plt.close()
    print(f'[pca_demo] {path}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _BASE = '/root/paddlejob/workspace/env_run/penghaotian'
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir',    default=f'{_BASE}/datas/coco/images/val2014')
    parser.add_argument('--num_images', type=int, default=6)
    parser.add_argument('--output_dir', default='analysis/pca_demo')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--resize',     type=int, default=None,
                        help='Resize short edge before extracting')
    parser.add_argument('--skip_dinov3', action='store_true')
    parser.add_argument('--skip_radio',  action='store_true')
    parser.add_argument('--skip_eupe',   action='store_true')
    parser.add_argument('--skip_tips',   action='store_true')
    args = parser.parse_args()

    random.seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    infos = {}
    if not args.skip_dinov3:
        infos['DINOv3'] = {'model': load_dinov3()}
    if not args.skip_radio:
        m, c = load_radio()
        infos['RADIO']  = {'model': m, 'cond': c}
    if not args.skip_tips:
        m, _ = load_tips()           # discard tokenizer
        infos['TIPSv2'] = {'model': m}
    if not args.skip_eupe:
        eu = load_eupe()
        if eu is not None:
            infos['EUPE'] = {'model': eu}

    extract = {
        'DINOv3': _patch_dinov3, 'RADIO': _patch_radio,
        'TIPSv2': _patch_tips,   'EUPE':  _patch_eupe,
    }

    img_paths = list(__import__('pathlib').Path(args.img_dir).glob('*.jpg'))
    assert img_paths, f'No .jpg in {args.img_dir}'
    img_paths = random.sample(img_paths, min(args.num_images, len(img_paths)))
    print(f'[pca_demo] {len(img_paths)} images, models: {list(infos)}')

    images_orig, raw_feats = [], {k: [] for k in infos}
    for p in img_paths:
        img = Image.open(p).convert('RGB')
        if args.resize:
            img = _short_edge(img, args.resize)
        images_orig.append(img)
        for k, info in infos.items():
            try:
                raw_feats[k].append(extract[k](img, info))
            except Exception as e:
                print(f'  [{k}] ERROR: {e}')
                raw_feats[k].append(None)

    model_names = list(infos)
    all_pca = [_pca_rgb(raw_feats[k]) for k in model_names]
    _draw_grid(images_orig, all_pca, model_names,
               os.path.join(args.output_dir, 'comparison_grid.png'))
    print(f'[pca_demo] done → {args.output_dir}')


if __name__ == '__main__':
    main()
