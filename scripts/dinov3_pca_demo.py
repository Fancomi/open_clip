"""
DINOv3 ViT-B/16 PCA 可视化 Demo（原始分辨率版）

用法:
    python scripts/dinov3_pca_demo.py \
        --model_path /root/paddlejob/workspace/env_run/penghaotian/models/dinov3-vitb16-pretrain-lvd1689m \
        --img_dir    /root/paddlejob/workspace/env_run/penghaotian/datas/coco/images/val2014 \
        --num_images 6 \
        --output_dir /tmp/dinov3_pca_output

说明:
  - 默认使用原始图像分辨率（--native_res），仅将边长对齐至 patch_size=16 的倍数（floor 裁剪）
  - 可用 --resize 指定固定短边长度（如 480），同样按 16 对齐
  - last_hidden_state layout: [CLS(1), REG(4), PATCH(H_p*W_p)]
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoImageProcessor, AutoModel


# ─── 配置 ──────────────────────────────────────────────────────────────────────
PATCH_SIZE = 16
PATCH_START = 5  # skip CLS(1) + REG(4)


# ─── 工具函数 ──────────────────────────────────────────────────────────────────

def load_model(model_path: str, device: torch.device):
    print(f"[INFO] Loading DINOv3 from {model_path} ...")
    # do_resize=False：保留原始分辨率，只做归一化
    processor = AutoImageProcessor.from_pretrained(
        model_path, trust_remote_code=True, do_resize=False
    )
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval().to(device)
    print("[INFO] Model loaded.")
    return processor, model


def prepare_image(img: Image.Image, resize: int = None) -> Image.Image:
    """
    可选短边 resize，然后将 H/W 向下对齐到 patch_size 的整数倍。
    返回处理后的 PIL Image。
    """
    img = img.convert("RGB")
    if resize is not None:
        W, H = img.size
        scale = resize / min(W, H)
        img = img.resize((int(W * scale), int(H * scale)), Image.BICUBIC)
    W, H = img.size
    W16 = (W // PATCH_SIZE) * PATCH_SIZE
    H16 = (H // PATCH_SIZE) * PATCH_SIZE
    if W16 != W or H16 != H:
        img = img.crop((0, 0, W16, H16))
    return img


@torch.no_grad()
def extract_patch_features(img: Image.Image, processor, model, device) -> np.ndarray:
    """
    返回 (H_patch, W_patch, D) 的 patch 特征。
    H_patch = H / 16, W_patch = W / 16
    """
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model(**inputs)
    # last_hidden_state: (1, 1+4+H_p*W_p, D)
    patch_tokens = out.last_hidden_state[0, PATCH_START:, :]  # (H_p*W_p, D)
    patch_tokens = patch_tokens.cpu().float().numpy()
    W, H = img.size
    H_p, W_p = H // PATCH_SIZE, W // PATCH_SIZE
    return patch_tokens.reshape(H_p, W_p, -1)  # (H_p, W_p, D)


def pca_on_patches(patch_feats: np.ndarray, n_components: int = 3) -> np.ndarray:
    """
    patch_feats: (H, W, D)
    返回 (H, W, n_components)，值域 [0, 1]（min-max 归一化后）。
    """
    H, W, D = patch_feats.shape
    tokens = patch_feats.reshape(-1, D)          # (H*W, D)

    pca = PCA(n_components=n_components)
    pca_out = pca.fit_transform(tokens)          # (H*W, 3)

    # 每个主成分独立 min-max 到 [0,1]
    lo = pca_out.min(axis=0, keepdims=True)
    hi = pca_out.max(axis=0, keepdims=True)
    pca_out = (pca_out - lo) / (hi - lo + 1e-8)

    return pca_out.reshape(H, W, n_components)


def pca_across_images(patch_feats_list: list, n_components: int = 3) -> list:
    """
    跨多张图像做一次全局 PCA，保证颜色语义一致。
    patch_feats_list: list of (H, W, D)
    返回 list of (H, W, 3)
    """
    shapes = [f.shape[:2] for f in patch_feats_list]
    tokens_all = np.concatenate([f.reshape(-1, f.shape[-1]) for f in patch_feats_list], axis=0)

    pca = PCA(n_components=n_components)
    pca_all = pca.fit_transform(tokens_all)

    lo = pca_all.min(axis=0, keepdims=True)
    hi = pca_all.max(axis=0, keepdims=True)
    pca_all = (pca_all - lo) / (hi - lo + 1e-8)

    results = []
    idx = 0
    for H, W in shapes:
        n = H * W
        results.append(pca_all[idx: idx + n].reshape(H, W, n_components))
        idx += n
    return results


def visualize_grid(images_orig: list, pca_maps: list, output_path: str):
    """
    每行：原图 | PCA RGB 上采样 | 三个单独主成分
    """
    n = len(images_orig)
    fig = plt.figure(figsize=(20, 4 * n))
    gs = gridspec.GridSpec(n, 5, figure=fig, hspace=0.05, wspace=0.05)

    col_titles = ["Original", "PCA (PC1-3 as RGB)", "PC1", "PC2", "PC3"]
    for j, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(title, fontsize=12, pad=8)
        ax.axis("off")

    for i, (img, pca_map) in enumerate(zip(images_orig, pca_maps)):
        H_p, W_p = pca_map.shape[:2]
        # 上采样到原图尺寸（最近邻）
        pca_rgb = Image.fromarray((pca_map * 255).astype(np.uint8)).resize(
            img.size, resample=Image.NEAREST
        )

        ax0 = fig.add_subplot(gs[i, 0])
        ax0.imshow(img)
        ax0.axis("off")

        ax1 = fig.add_subplot(gs[i, 1])
        ax1.imshow(pca_rgb)
        ax1.axis("off")

        for c in range(3):
            ax = fig.add_subplot(gs[i, 2 + c])
            ax.imshow(pca_map[:, :, c], cmap="viridis", interpolation="nearest")
            ax.axis("off")

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved → {output_path}")


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="/root/paddlejob/workspace/env_run/penghaotian/models/dinov3-vitb16-pretrain-lvd1689m",
    )
    parser.add_argument(
        "--img_dir",
        default="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/images/val2014",
    )
    parser.add_argument("--num_images", type=int, default=6, help="随机采样的图像数量")
    parser.add_argument("--output_dir", default="/tmp/dinov3_pca_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="短边 resize 到指定像素后再推理（None=原始分辨率）",
    )
    parser.add_argument(
        "--global_pca",
        action="store_true",
        default=True,
        help="跨图像全局PCA（颜色语义一致），默认开启",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    processor, model = load_model(args.model_path, device)

    # 随机采样图像
    img_paths = list(Path(args.img_dir).glob("*.jpg"))
    assert img_paths, f"No jpg images found in {args.img_dir}"
    img_paths = random.sample(img_paths, min(args.num_images, len(img_paths)))
    print(f"[INFO] Processing {len(img_paths)} images ...")

    images_orig, patch_feats_list = [], []
    for p in img_paths:
        img_raw = Image.open(p).convert("RGB")
        img = prepare_image(img_raw, resize=args.resize)
        images_orig.append(img)
        feats = extract_patch_features(img, processor, model, device)
        patch_feats_list.append(feats)
        print(f"  {p.name}  orig={img_raw.size}  used={img.size}  patches={feats.shape[:2]}")

    # PCA
    if args.global_pca:
        print("[INFO] Running global PCA across all images ...")
        pca_maps = pca_across_images(patch_feats_list, n_components=3)
    else:
        print("[INFO] Running per-image PCA ...")
        pca_maps = [pca_on_patches(f) for f in patch_feats_list]

    # 绘图
    grid_path = str(output_dir / "pca_grid.png")
    visualize_grid(images_orig, pca_maps, grid_path)

    # 同时保存每张图的单独对比图
    for i, (img, pca_map) in enumerate(zip(images_orig, pca_maps)):
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        pca_rgb = Image.fromarray((pca_map * 255).astype(np.uint8)).resize(
            img.size, resample=Image.NEAREST
        )
        titles = ["Original", "PCA RGB", "PC1", "PC2", "PC3"]
        data = [img, pca_rgb, pca_map[:, :, 0], pca_map[:, :, 1], pca_map[:, :, 2]]
        cmaps = [None, None, "viridis", "viridis", "viridis"]
        for ax, title, d, cm in zip(axes, titles, data, cmaps):
            ax.imshow(d, cmap=cm, interpolation="nearest")
            ax.set_title(title, fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        single_path = str(output_dir / f"single_{i:02d}_{img_paths[i].stem}.png")
        plt.savefig(single_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[INFO] Saved → {single_path}")

    print(f"\n[DONE] All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
