"""多 CLIP 模型 FPS batch 采样探索: 验证 PE-Core 的 FPS 收敛特性是否为 CLIP 范式通用.

Usage:
  python -m analysis.clip_fps_probe [--max-samples N] [--force]

输出: feature_probe/pretrained/clip_fps_compare/
  - batch_fps_<model>.gif   (每模型 FPS batch GIF)
  - image_allmodels.png     (所有模型 PC pairs 对比)
"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

import math
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_TIMM = f'{_BASE}/models/timm'
_DATA_TSV = f'{_BASE}/datas/coco/annotations/karpathy_1cap.tsv'
_OUT_DIR = f'{_BASE}/datas/coco/feature_probe/pretrained/clip_fps_compare'

# 模型配置: (open_clip_model_name, pretrained_path)
CLIP_MODELS = {
    'PE-Core':    ('PE-Core-B-16',       f'{_TIMM}/PE-Core-B-16/open_clip_model.safetensors'),
    'SigLIP2':    ('ViT-B-16-SigLIP2',   f'{_TIMM}/ViT-B-16-SigLIP2/open_clip_model.safetensors'),
    'DataComp':   ('ViT-B-16',           f'{_TIMM}/DataComp-XL-B-16/open_clip_pytorch_model.bin'),
    'DFN2B':      ('ViT-B-16',           f'{_TIMM}/DFN2B-ViT-B-16/open_clip_pytorch_model.bin'),
    'EVA02':      ('EVA02-B-16',         f'{_TIMM}/EVA02-B-16/open_clip_model.safetensors'),
    'LAION2B':    ('ViT-B-16',           f'{_TIMM}/LAION2B-B-16/open_clip_model.safetensors'),
    'MetaCLIP':   ('ViT-B-16-quickgelu', f'{_TIMM}/MetaCLIP-FullCC-B-16/open_clip_model.safetensors'),
}


@torch.no_grad()
def _extract_visual_features(model, preproc, paths, max_samples, device):
    """提取 backbone CLS (projection 前, 768-dim for ViT-B/16)."""
    from PIL import Image

    model = model.eval().to(device)
    vis = model.visual
    use_trunk = hasattr(vis, 'trunk')

    feats = []
    bs = 64
    n = min(len(paths), max_samples)
    for i in range(0, n, bs):
        imgs = []
        for p in paths[i:i + bs]:
            try:
                imgs.append(preproc(Image.open(p).convert('RGB')))
            except Exception:
                imgs.append(preproc(Image.new('RGB', (224, 224))))
        batch = torch.stack(imgs).to(device)
        with torch.amp.autocast('cuda'):
            if use_trunk:
                out = vis.trunk.forward_features(batch)
                cls = out[:, 0]
            else:
                proj_bak = vis.proj
                vis.proj = None
                cls = model.encode_image(batch, normalize=False)
                vis.proj = proj_bak
        feats.append(cls.float().cpu().numpy())
        if (i // bs) % 20 == 0:
            logging.info(f'    {min(i + bs, n)}/{n}')
    return np.concatenate(feats)


@torch.no_grad()
def _extract_backbone_cls(model_name, pretrained, paths, max_samples, device):
    """加载 open_clip 权重模型并提取 projection 前 backbone CLS."""
    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    feats = _extract_visual_features(model, preproc, paths, max_samples, device)
    del model
    torch.cuda.empty_cache()
    return feats


def _uniform_init_visual(vis, seed=0):
    """均匀随机初始化视觉塔权重；LayerNorm 保持标准初始化."""
    torch.manual_seed(seed)

    def _init(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            fan_in = nn.init._calculate_correct_fan(m.weight, 'fan_in')
            bound = 1.0 / math.sqrt(fan_in)
            nn.init.uniform_(m.weight, -bound, bound)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            bound = 1.0 / math.sqrt(m.embedding_dim)
            nn.init.uniform_(m.weight, -bound, bound)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    vis.apply(_init)
    for name, p in vis.named_parameters(recurse=False):
        if name in {'class_embedding', 'positional_embedding', 'proj'}:
            bound = 1.0 / math.sqrt(p.shape[-1])
            nn.init.uniform_(p, -bound, bound)


@torch.no_grad()
def _extract_random_init(paths, max_samples, device):
    """随机均匀初始化 ViT-B/16, 提取 projection 前 backbone CLS."""
    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms('ViT-B-16', pretrained=None)
    _uniform_init_visual(model.visual, seed=0)
    feats = _extract_visual_features(model, preproc, paths, max_samples, device)
    del model
    torch.cuda.empty_cache()
    return feats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max-samples', type=int, default=5000)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(_OUT_DIR, exist_ok=True)

    # 加载图片路径
    df = pd.read_csv(_DATA_TSV, sep='\t')
    paths = df['filepath'].tolist()[:args.max_samples]
    logging.info(f'数据: {len(paths)} 张图片')

    # 提取/加载特征
    all_feats = {}
    for display_name, (model_name, ckpt_path) in CLIP_MODELS.items():
        npz_path = os.path.join(_OUT_DIR, f'{display_name.lower()}_img.npz')
        if os.path.exists(npz_path) and not args.force:
            logging.info(f'[cache] {display_name} — 从缓存加载')
            all_feats[display_name] = np.load(npz_path)['features']
            continue
        if not os.path.exists(ckpt_path):
            logging.warning(f'[skip] {display_name} — 权重不存在: {ckpt_path}')
            continue
        logging.info(f'[extract] {display_name} ({model_name}) ...')
        feats = _extract_backbone_cls(model_name, ckpt_path, paths, args.max_samples, device)
        np.savez_compressed(npz_path, features=feats)
        all_feats[display_name] = feats
        logging.info(f'  shape={feats.shape} → {npz_path}')

    # 随机初始化 baseline: 同样 ViT-B/16, 均匀初始化视觉塔
    rand_path = os.path.join(_OUT_DIR, 'random_init_img.npz')
    if os.path.exists(rand_path) and not args.force:
        logging.info('[cache] RandomInit — 从缓存加载')
        all_feats['RandomInit'] = np.load(rand_path)['features']
    else:
        logging.info('[extract] RandomInit (uniform ViT-B/16) ...')
        feats = _extract_random_init(paths, args.max_samples, device)
        np.savez_compressed(rand_path, features=feats)
        all_feats['RandomInit'] = feats
        logging.info(f'  shape={feats.shape} → {rand_path}')

    if not all_feats:
        logging.error('没有可用模型特征')
        return

    # 生成 FPS batch GIF + PC pairs 图
    from .metrics import fps_batches, compute_knn_density
    from .viz import plot_batch_gif, plot_pc_pairs_allmodels

    logging.info(f'[viz] 生成 FPS batch GIF ({len(all_feats)} 模型) ...')
    for name, feats in all_feats.items():
        slug = name.lower().replace('-', '_')
        gif_path = os.path.join(_OUT_DIR, f'batch_fps_{slug}.gif')
        if os.path.exists(gif_path) and not args.force:
            logging.info(f'  [skip] {name} GIF 已存在')
            continue
        fps_b = fps_batches(feats, batch_size=256, n_batches=20)
        plot_batch_gif(feats, fps_b, gif_path, method='FPS', model=name)

    # PC pairs 对比图
    logging.info('[viz] 生成 PC pairs 对比图 ...')
    plot_pc_pairs_allmodels(all_feats, os.path.join(_OUT_DIR, 'image_allmodels.png'))

    # ── 密度直方图: 各模型 kNN 密度分布对比 ──────────────────────────────────
    logging.info('[analysis] 计算 kNN 密度分布 ...')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    densities = {}
    for name, feats in all_feats.items():
        densities[name] = compute_knn_density(feats, K=50)
        logging.info(f'  {name}: density range=[{densities[name].min():.4f}, {densities[name].max():.4f}]')

    # 密度直方图 (所有模型叠加)
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, d in densities.items():
        # 归一化到 [0,1] 方便对比形状
        d_norm = (d - d.min()) / (d.max() - d.min() + 1e-10)
        ax.hist(d_norm, bins=80, alpha=0.5, label=name, density=True)
    ax.set_xlabel('Normalized kNN Density')
    ax.set_ylabel('Probability Density')
    ax.set_title('kNN Density Distribution (K=50) — Backbone CLS Features')
    ax.legend(fontsize=8)
    plt.tight_layout()
    hist_path = os.path.join(_OUT_DIR, 'density_histogram.png')
    plt.savefig(hist_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'[viz] {hist_path}')

    # ── FPS batch centroid 轨迹 ──────────────────────────────────────────────
    logging.info('[analysis] 计算 FPS batch centroid 轨迹 ...')
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes_flat = axes.reshape(-1)

    for idx, (name, feats) in enumerate(all_feats.items()):
        if idx >= len(axes_flat):
            break
        ax = axes_flat[idx]
        pca = PCA(n_components=2).fit(feats)
        proj = pca.transform(feats)

        # 全局 centroid
        global_center = proj.mean(axis=0)

        # FPS batches centroid 轨迹
        fps_b = fps_batches(feats, batch_size=256, n_batches=20)
        centroids = []
        for batch_idx in fps_b:
            batch_proj = proj[batch_idx]
            centroids.append(batch_proj.mean(axis=0))
        centroids = np.array(centroids)

        # 背景点云
        ax.scatter(proj[:, 0], proj[:, 1], s=1, alpha=0.08, color='#999999', rasterized=True)
        # Centroid 轨迹 (颜色从浅到深 = batch 1→20)
        colors = plt.cm.viridis(np.linspace(0.2, 1.0, len(centroids)))
        for i in range(len(centroids) - 1):
            ax.plot(centroids[i:i+2, 0], centroids[i:i+2, 1], '-',
                    color=colors[i+1], lw=2.0, alpha=0.8)
        ax.scatter(centroids[:, 0], centroids[:, 1], c=range(len(centroids)),
                   cmap='viridis', s=60, edgecolors='black', linewidths=0.5, zorder=5)
        # 标注起止
        ax.scatter(*centroids[0], marker='o', s=150, color='lime',
                   edgecolors='black', linewidths=1, zorder=6, label='Batch 1')
        ax.scatter(*centroids[-1], marker='*', s=200, color='red',
                   edgecolors='black', linewidths=1, zorder=6, label='Batch 20')
        # 全局中心
        ax.scatter(*global_center, marker='+', s=200, color='black',
                   linewidths=2, zorder=6, label='Global Center')
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        if idx == 0:
            ax.legend(fontsize=7, loc='best')

    # 隐藏多余面板
    for i in range(len(all_feats), len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle('FPS Batch Centroid Trajectory (PC1 vs PC2)\n'
                 'light→dark = batch 1→20, ○=start, ★=end, +=global center',
                 fontsize=11)
    plt.tight_layout()
    traj_path = os.path.join(_OUT_DIR, 'centroid_trajectory.png')
    plt.savefig(traj_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'[viz] {traj_path}')

    # ── Centroid-to-center 距离曲线 ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, feats in all_feats.items():
        center = feats.mean(axis=0)
        fps_b = fps_batches(feats, batch_size=256, n_batches=20)
        dists = []
        for batch_idx in fps_b:
            batch_center = feats[batch_idx].mean(axis=0)
            dists.append(float(np.linalg.norm(batch_center - center)))
        ax.plot(range(1, len(dists)+1), dists, marker='o', ms=4, lw=1.5, label=name)
    ax.set_xlabel('FPS Batch Index')
    ax.set_ylabel('Distance: Batch Centroid → Global Center')
    ax.set_title('FPS Batch Centroid Convergence to Global Center\n'
                 '(decreasing = converging to center)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    conv_path = os.path.join(_OUT_DIR, 'centroid_convergence.png')
    plt.savefig(conv_path, dpi=150, bbox_inches='tight'); plt.close()
    print(f'[viz] {conv_path}')

    logging.info(f'[done] 输出目录: {_OUT_DIR}')


if __name__ == '__main__':
    main()
