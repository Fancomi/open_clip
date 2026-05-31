"""Layer-wise feature probe for ViT-based vision encoders.

Extracts per-layer CLS token (or patch mean) features from a COCO TSV using
forward hooks on each transformer block, then computes anisotropy metrics and
PCA scatter maps for every layer.

Supported model families
------------------------
  dinov3    DinoVisionTransformer  → model.blocks[i]  → x_norm_clstoken (last layer)
  pe_core   PE-Core (Eva trunk)   → model.visual.trunk.blocks[i]
  siglip2   SigLIP2 (TimmModel)   → model.visual.trunk.blocks[i]
  eupe      EUPE (same as dinov3) → model.blocks[i]

Usage (from repo root):
  python -m analysis.layer_probe --model dinov3 --data <tsv> --out-dir <dir>
  python -m analysis.layer_probe --model pe_core --data <tsv> --out-dir <dir>
"""
import argparse, glob, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from sklearn.decomposition import PCA
from torch.utils.data import Dataset, DataLoader

from .models import DEVICE, load_pe_core, load_siglip2, load_dinov3, load_eupe, CKPT
from .metrics import compute_anisotropy

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_DEFAULT_DATA = f'{_BASE}/datas/coco/annotations/karpathy_1cap.tsv'
_TIMM = f'{_BASE}/models/timm'
_OPENCLIP_CKPTS = {
    'datacomp': ('ViT-B-16',           f'{_TIMM}/DataComp-XL-B-16/open_clip_pytorch_model.bin'),
    'dfn2b':    ('ViT-B-16',           f'{_TIMM}/DFN2B-ViT-B-16/open_clip_pytorch_model.bin'),
    'eva02':    ('EVA02-B-16',         f'{_TIMM}/EVA02-B-16/open_clip_model.safetensors'),
    'laion2b':  ('ViT-B-16',           f'{_TIMM}/LAION2B-B-16/open_clip_model.safetensors'),
    'metaclip': ('ViT-B-16-quickgelu', f'{_TIMM}/MetaCLIP-FullCC-B-16/open_clip_model.safetensors'),
}


# ── Dataset ──────────────────────────────────────────────────────────────────

class _ImgDS(Dataset):
    def __init__(self, paths, tf):
        self.paths, self.tf = paths, tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.tf(Image.open(self.paths[i]).convert('RGB')), i


def _read_paths(data, max_samples):
    """Read image paths from COCO TSV (with header) or CC3M TSV/glob (no header)."""
    import pandas as pd
    files = sorted(glob.glob(data)) if any(c in data for c in '*?[') else [data]
    paths = []
    for fp in files:
        need = None if max_samples is None else max_samples - len(paths)
        if need is not None and need <= 0:
            break
        df = pd.read_csv(fp, sep='\t', nrows=need, dtype=str)
        if 'filepath' in df.columns:
            vals = df['filepath'].tolist()
        else:
            df = pd.read_csv(fp, sep='\t', header=None, nrows=need, dtype=str)
            vals = df.iloc[:, 0].tolist()
        paths.extend(vals)
    return paths if max_samples is None else paths[:max_samples]


# ── Hook manager ─────────────────────────────────────────────────────────────

class LayerHookManager:
    """Registers one forward hook per block; captures the block output tensor."""

    def __init__(self, blocks):
        self._hooks = []
        self._outputs = {}   # layer_idx → last batch output
        for idx, block in enumerate(blocks):
            self._hooks.append(
                block.register_forward_hook(self._make_hook(idx))
            )

    def _make_hook(self, idx):
        def _hook(module, input, output):
            # output may be tuple (x, ...) — take first element
            t = output[0] if isinstance(output, (tuple, list)) else output
            self._outputs[idx] = t.detach()
        return _hook

    def clear(self):
        self._outputs.clear()

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Feature extraction ────────────────────────────────────────────────────────

def _seq_first(tensor):
    """Heuristic for OpenCLIP transformer outputs: (tokens, batch, dim)."""
    return tensor.ndim == 3 and tensor.shape[0] > tensor.shape[1]


def _extract_cls(tensor):
    """Extract CLS from (B,N,D) or (N,B,D); fallback to 2D output."""
    if tensor.ndim == 3:
        out = tensor[0] if _seq_first(tensor) else tensor[:, 0]
        return out.cpu().float().numpy()
    return tensor.cpu().float().numpy()


def _extract_patch_mean(tensor):
    """Patch mean from (B,N,D) or (N,B,D), skipping CLS token."""
    if tensor.ndim == 3:
        out = tensor[1:].mean(0) if _seq_first(tensor) else tensor[:, 1:].mean(1)
        return out.cpu().float().numpy()
    return tensor.cpu().float().numpy()


@torch.no_grad()
def extract_layer_features(model, blocks, paths, preprocess,
                            batch_size=64, token='cls'):
    """Return dict: layer_idx → (N, D) array of per-image features."""
    dl = DataLoader(_ImgDS(paths, preprocess), batch_size=batch_size,
                    num_workers=4, pin_memory=True)
    manager = LayerHookManager(blocks)
    extract_fn = _extract_cls if token == 'cls' else _extract_patch_mean

    layer_feats = {i: [] for i in range(len(blocks))}
    model.eval()
    dev = DEVICE.type

    for imgs, _ in dl:
        manager.clear()
        with torch.autocast(device_type=dev, dtype=torch.bfloat16,
                            enabled=(dev != 'cpu')):
            _ = model(imgs.to(DEVICE))
        for idx, out in manager._outputs.items():
            layer_feats[idx].append(extract_fn(out))

    manager.remove()
    return {i: np.concatenate(v, 0) for i, v in layer_feats.items()}


# ── Model loaders returning (model, blocks, preprocess) ──────────────────────

def _dinov3_setup(repo=None, ckpt=None):
    wrapper = load_dinov3(repo, ckpt).to(DEVICE)
    m = wrapper.m  # HF DINOv3ViTModel; wrapper only exposes forward_features.
    blocks = list(m.model.layer.children())
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return m, blocks, tf


def _pe_core_setup(ckpt=None):
    import open_clip
    m, _, prep = open_clip.create_model_and_transforms(
        'PE-Core-B-16', pretrained=ckpt or CKPT['pe_core'])
    blocks = list(m.visual.trunk.blocks.children())
    return m.eval().to(DEVICE), blocks, prep


def _siglip2_setup(ckpt=None):
    import open_clip
    m, _, prep = open_clip.create_model_and_transforms(
        'ViT-B-16-SigLIP2', pretrained=ckpt or CKPT['siglip2'])
    # SigLIP2 uses TimmModel (same as PE-Core) → trunk.blocks, not transformer.resblocks
    blocks = list(m.visual.trunk.blocks.children())
    return m.eval().to(DEVICE), blocks, prep


def _eupe_setup(repo=None, ckpt=None):
    m = load_eupe(repo, ckpt)
    assert m is not None, 'EUPE model not available'
    m = m.to(DEVICE)
    tf = T.Compose([T.Resize((224, 224)), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    return m, list(m.blocks.children()), tf


def _openclip_setup(model_key):
    import open_clip
    arch, ckpt = _OPENCLIP_CKPTS[model_key]
    m, _, prep = open_clip.create_model_and_transforms(arch, pretrained=ckpt)
    vis = m.visual
    if hasattr(vis, 'trunk'):
        blocks = list(vis.trunk.blocks.children())
    else:
        blocks = list(vis.transformer.resblocks.children())
    return m.to(DEVICE), blocks, prep


_SETUP = {
    'dinov3':  _dinov3_setup,
    'pe_core': _pe_core_setup,
    'siglip2': _siglip2_setup,
    'eupe':    _eupe_setup,
    **{k: (lambda key: (lambda: _openclip_setup(key)))(k) for k in _OPENCLIP_CKPTS},
}


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_layer_anisotropy(layer_ids, aniso_list, save_path, model_name):
    """Line chart of key anisotropy metrics vs. layer depth."""
    keys = [
        ('effective_rank', 'Effective Rank'),
        ('stable_rank',    'Stable Rank'),
        ('avg_cos_sim',    'Avg Cosine Sim'),
        ('std_cos_sim',    'Std Cosine Sim'),
        ('pct_var_top4',   'Var% top-4'),
        ('pct_var_top10',  'Var% top-10'),
    ]
    ncols = 3; nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.reshape(-1)
    for ax, (key, lbl) in zip(axes, keys):
        ys = [m[key] for m in aniso_list]
        ax.plot(layer_ids, ys, marker='o', ms=5, lw=1.8, color='steelblue')
        ax.set_xlabel('Layer'); ax.set_title(lbl, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(layer_ids)
    fig.suptitle(f'{model_name}: Per-Layer Anisotropy (CLS token)', fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[layer_probe] {save_path}')


def plot_layer_coord_variance(layer_ids, layer_feats, save_path, model_name, token='cls'):
    """Raw feature-coordinate variance per layer.

    Left column: all layers share one y-scale for absolute comparison.
    Right column: each layer uses its own y-scale to show active coordinates.
    """
    vars_by_layer = {lid: np.var(layer_feats[lid], axis=0, dtype=np.float64)
                     for lid in layer_ids}
    D = vars_by_layer[layer_ids[0]].shape[0]
    ymax = max(float(v.max()) for v in vars_by_layer.values())
    global_ylim = ymax * 1.05 if ymax > 0 else 1.0

    n = len(layer_ids)
    fig_h = max(1.05 * n, 6)
    fig, axes = plt.subplots(n, 2, figsize=(24, fig_h), sharex=True)
    axes = np.atleast_2d(axes)
    x = np.arange(D)
    colors = cm.viridis(np.linspace(0.15, 0.95, n))

    for row, (lid, c) in enumerate(zip(layer_ids, colors)):
        v = vars_by_layer[lid]
        local_max = float(v.max())
        local_ylim = local_max * 1.05 if local_max > 0 else 1.0
        for col, (ax, ylim) in enumerate(zip(axes[row], [global_ylim, local_ylim])):
            ax.bar(x, v, width=1.0, color=c, linewidth=0)
            ax.set_xlim(-1, D)
            ax.set_ylim(0, ylim)
            ax.grid(axis='y', alpha=0.18, linewidth=0.5)
            ax.tick_params(axis='y', labelsize=6)
            if col == 0:
                ax.set_ylabel(f'L{lid}', rotation=0, labelpad=18, fontsize=8)
            else:
                ax.text(0.995, 0.82, f'max={local_max:.3g}', transform=ax.transAxes,
                        ha='right', va='center', fontsize=6, color='dimgray')
    for ax in axes[:-1].reshape(-1):
        ax.tick_params(labelbottom=False)
    axes[0, 0].set_title(f'Global y-scale (max over all layers = {ymax:.4g})', fontsize=9)
    axes[0, 1].set_title('Per-layer y-scale (each row normalized to its own max)', fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(f'Raw feature coordinate dimension (0..{D-1})')
    fig.text(0.006, 0.5, 'Variance across samples', va='center', rotation='vertical')
    fig.suptitle(f'{model_name}: Layer-wise Raw Coordinate Variance ({token})',
                 fontsize=11, y=1.0)
    plt.tight_layout(rect=(0.02, 0, 1, 0.985))
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[layer_probe] {save_path}')



def plot_layer_pca_grid(layer_ids, layer_feats, save_path, model_name, max_layers=16):
    """PCA scatter grid: each panel is one layer, PC1 vs PC2."""
    ids_to_show = layer_ids[:max_layers]
    n = len(ids_to_show)
    ncols = min(4, n); nrows = (n + ncols - 1) // ncols
    colors_n = cm.viridis(np.linspace(0, 1, n))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    # Fit shared PCA across all layers if same dim; else per-layer
    all_dims = set(layer_feats[i].shape[1] for i in ids_to_show)
    if len(all_dims) == 1:
        combined = np.concatenate([layer_feats[i] for i in ids_to_show])
        pca_shared = PCA(n_components=2).fit(combined)
        projs = {i: pca_shared.transform(layer_feats[i]) for i in ids_to_show}
        pca_label = 'shared PCA'
    else:
        projs = {i: PCA(n_components=2).fit_transform(layer_feats[i]) for i in ids_to_show}
        pca_label = 'per-layer PCA'

    for ax_i, (lid, c) in enumerate(zip(ids_to_show, colors_n)):
        ax = axes[ax_i]
        p = projs[lid]
        ax.scatter(p[:, 0], p[:, 1], s=3, alpha=0.4, color=c, rasterized=True)
        ax.set_title(f'Layer {lid}', fontsize=8)
        ax.axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle(f'{model_name}: Per-Layer CLS Feature PCA  [{pca_label}]',
                 fontsize=10, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[layer_probe] {save_path}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import pandas as pd
    p = argparse.ArgumentParser()
    p.add_argument('--model',    required=True, choices=list(_SETUP.keys()))
    p.add_argument('--data',     default=_DEFAULT_DATA)
    p.add_argument('--out-dir',  default='analysis/layer_probe_out')
    p.add_argument('--max-samples', type=int, default=None,
                   help='Max images to process (default: all images in TSV)')
    p.add_argument('--token',    choices=['cls', 'patch_mean'], default='cls')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--force',    action='store_true')
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    cache_name = f'{args.model}_layers.npz' if args.token == 'cls' else f'{args.model}_{args.token}_layers.npz'
    cache = os.path.join(args.out_dir, cache_name)

    if os.path.exists(cache) and not args.force:
        logging.info(f'[layer_probe] loading cache {cache}')
        d = np.load(cache)
        layer_feats = {int(k.replace('layer_', '')): d[k] for k in d.files}
    else:
        paths = _read_paths(args.data, args.max_samples)
        logging.info(f'[layer_probe] {args.model}  {len(paths)} images  data={args.data}')

        setup_fn = _SETUP[args.model]
        model, blocks, preprocess = setup_fn()
        logging.info(f'[layer_probe] {len(blocks)} transformer blocks found')

        layer_feats = extract_layer_features(
            model, blocks, paths, preprocess,
            batch_size=args.batch_size, token=args.token)
        np.savez_compressed(cache,
                            **{f'layer_{i}': v for i, v in layer_feats.items()})
        logging.info(f'[layer_probe] cached → {cache}')

    layer_ids  = sorted(layer_feats.keys())
    logging.info(f'[layer_probe] {len(layer_ids)} layers, '
                 f'feature dim={layer_feats[layer_ids[0]].shape[1]}')

    # Anisotropy per layer
    logging.info('[layer_probe] computing per-layer anisotropy...')
    aniso_list = []
    for lid in layer_ids:
        m = compute_anisotropy(layer_feats[lid])
        aniso_list.append(m)
        logging.info(f'  layer {lid:2d}: EffRank={m["effective_rank"]:.1f}  '
                     f'AvgCos={m["avg_cos_sim"]:.4f}  top4%={m["pct_var_top4"]:.1f}')

    plot_layer_anisotropy(layer_ids, aniso_list,
                          os.path.join(args.out_dir, f'{args.model}_layer_anisotropy.png'),
                          model_name=args.model.upper())
    plot_layer_coord_variance(layer_ids, layer_feats,
                              os.path.join(args.out_dir, f'{args.model}_layer_coord_variance.png'),
                              model_name=args.model.upper(), token=args.token)
    plot_layer_pca_grid(layer_ids, layer_feats,
                        os.path.join(args.out_dir, f'{args.model}_layer_pca.png'),
                        model_name=args.model.upper())


if __name__ == '__main__':
    main()
