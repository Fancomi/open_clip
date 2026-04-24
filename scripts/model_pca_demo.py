"""
多模型 Patch 特征 PCA 可视化 Demo

对比以下 4 个模型的 patch 特征 PCA 可视化：
  - DINOv3-ViT-B/16   (torch.hub.load, x_norm_patchtokens, patch_size=16, ImageNet norm)
  - C-RADIOv4-SO400M  (AutoModel, spatial_features[B,196,C], patch_size=16)
  - EUPE-ViT-B/16     (torch.hub.load + local EUPE repo, x_norm_patchtokens, patch_size=16)
  - TIPSv2-ViT-B/14   (local module load, patch_tokens[B,1024,D], img_size=448, patch_size=14)

用法:
    python scripts/model_pca_demo.py \
        --img_dir /root/paddlejob/workspace/env_run/penghaotian/datas/coco/images/val2014 \
        --num_images 6 \
        --output_dir /tmp/model_pca_output \
        [--skip_eupe]   # EUPE repo 不可用时跳过

输出:
    comparison_grid.png   — 全部图像 × 全部模型对比网格
    comparison_XX_*.png   — 每张图单独保存
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image
from sklearn.decomposition import PCA

# ─── 路径默认值 ────────────────────────────────────────────────────────────────
_BASE = "/root/paddlejob/workspace/env_run/penghaotian"
MODEL_BASE   = f"{_BASE}/models"
DINOV3_REPO  = f"{_BASE}/vision_encoder/dinov3"
DINOV3_CKPT  = f"{MODEL_BASE}/dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
RADIO_PATH   = f"{MODEL_BASE}/C-RADIOv4-SO400M"
EUPE_CKPT    = f"{MODEL_BASE}/EUPE-ViT-B/EUPE-ViT-B.pt"
EUPE_REPO    = f"{_BASE}/vision_encoder/EUPE"
TIPS_PATH    = f"{MODEL_BASE}/tipsv2-b14"
HF_CACHE     = os.path.expanduser("~/.cache/huggingface/modules/transformers_modules")


# ─── 图像工具 ──────────────────────────────────────────────────────────────────

def align_to_patch(img: Image.Image, patch_size: int) -> Image.Image:
    W, H = img.size
    W = (W // patch_size) * patch_size
    H = (H // patch_size) * patch_size
    return img.crop((0, 0, W, H)) if (W, H) != img.size else img


def resize_short_edge(img: Image.Image, short: int) -> Image.Image:
    W, H = img.size
    scale = short / min(W, H)
    return img.resize((int(W * scale), int(H * scale)), Image.BICUBIC)


# ─── DINOv3 ────────────────────────────────────────────────────────────────────
# torch.hub.load → forward_features → x_norm_patchtokens (B,196,768)
# ImageNet normalization, patch_size=16, token layout: CLS+REG4+PATCH

_DINOV3_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def load_dinov3(device):
    print(f"[INFO] Loading DINOv3 via hub from {DINOV3_REPO} ...")
    model = torch.hub.load(DINOV3_REPO, 'dinov3_vitb16', source='local', pretrained=False)
    sd = torch.load(DINOV3_CKPT, map_location='cpu')
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    print("[INFO] DINOv3 loaded.")
    return {"model": model, "patch_size": 16}


@torch.no_grad()
def extract_dinov3(img: Image.Image, info: dict, device) -> np.ndarray:
    """Returns (H_p, W_p, D) patch features."""
    ps = info["patch_size"]
    img = align_to_patch(img, ps)
    W, H = img.size
    x = _DINOV3_TRANSFORM(img).unsqueeze(0).to(device)
    out = info["model"].forward_features(x)
    tokens = out["x_norm_patchtokens"][0].cpu().float().numpy()  # (H_p*W_p, D)
    return tokens.reshape(H // ps, W // ps, -1)


# ─── C-RADIOv4 ────────────────────────────────────────────────────────────────
# AutoModel.from_pretrained (trust_remote_code), uses input_conditioner
# output: RadioOutput (summary[B,C], spatial[B,H_p*W_p,C]), patch_size=16

def load_radio(device):
    from transformers import AutoModel
    print(f"[INFO] Loading C-RADIOv4 from {RADIO_PATH} ...")
    model = AutoModel.from_pretrained(RADIO_PATH, trust_remote_code=True)
    model.eval().to(device)
    conditioner = model.input_conditioner if hasattr(model, 'input_conditioner') else None
    print("[INFO] C-RADIOv4 loaded.")
    return {"model": model, "conditioner": conditioner, "patch_size": 16}


@torch.no_grad()
def extract_radio(img: Image.Image, info: dict, device) -> np.ndarray:
    """Returns (H_p, W_p, D) patch features."""
    ps = info["patch_size"]
    img = align_to_patch(img, ps)
    W, H = img.size
    x = T.ToTensor()(img).unsqueeze(0).to(device)
    if info["conditioner"] is not None:
        x = info["conditioner"](x)
    out = info["model"](x)
    # RadioOutput: (summary, spatial_features)
    if isinstance(out, (tuple, list)):
        spatial = out[1]
    else:
        spatial = getattr(out, 'spatial_features',
                  getattr(out, 'patch_features',
                  out.last_hidden_state[:, 1:, :]))
    tokens = spatial[0].cpu().float().numpy()   # (H_p*W_p, C)
    return tokens.reshape(H // ps, W // ps, -1)


# ─── EUPE ─────────────────────────────────────────────────────────────────────
# torch.hub.load from facebookresearch/eupe repo
# forward_features → x_norm_patchtokens (B,196,768)
# ImageNet normalization, patch_size=16

_EUPE_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def load_eupe(device, skip_eupe=False):
    if skip_eupe:
        print("[INFO] EUPE skipped by --skip_eupe flag")
        return None
    if not (Path(EUPE_REPO) / "hubconf.py").exists():
        print(f"[WARN] EUPE repo not found at {EUPE_REPO}, skipping")
        return None
    print(f"[INFO] Loading EUPE via hub from {EUPE_REPO} ...")
    try:
        model = torch.hub.load(EUPE_REPO, 'eupe_vitb16', source='local',
                               weights=EUPE_CKPT, trust_repo=True)
        model.eval().to(device)
        print("[INFO] EUPE loaded.")
        return {"model": model, "patch_size": 16}
    except Exception as e:
        print(f"[WARN] EUPE load failed: {e}")
        return None


@torch.no_grad()
def extract_eupe(img: Image.Image, info: dict, device) -> np.ndarray:
    """Returns (H_p, W_p, D) patch features. info=None → returns None."""
    if info is None:
        return None
    ps = info["patch_size"]
    img = align_to_patch(img, ps)
    W, H = img.size
    x = _EUPE_TRANSFORM(img).unsqueeze(0).to(device)
    dev_type = device.type if hasattr(device, 'type') else 'cuda'
    with torch.autocast(device_type=dev_type, dtype=torch.bfloat16,
                        enabled=(dev_type != 'cpu')):
        out = info["model"].forward_features(x)
    tokens = out["x_norm_patchtokens"][0].cpu().float().numpy()
    return tokens.reshape(H // ps, W // ps, -1)


# ─── TIPSv2 ───────────────────────────────────────────────────────────────────
# Loaded directly from local module files (bypass HF remote path resolution)
# encode_image → patch_tokens (B,1024,768)
# Input: [0,1] (just ToTensor), img_size=448, patch_size=14

def _load_tips_module():
    """Import TIPSv2 from local cache (handles transformers 5.x path issue)."""
    cache_dir = Path(HF_CACHE) / "tipsv2_hyphen_b14"
    # Ensure sibling files are present in cache
    for fname in ("image_encoder.py", "text_encoder.py"):
        dst = cache_dir / fname
        src = Path(TIPS_PATH) / fname
        if not dst.exists() and src.exists():
            import shutil
            shutil.copy(src, dst)
    # Load via sys.path trick
    sys.path.insert(0, str(Path(HF_CACHE)))
    from transformers_modules.tipsv2_hyphen_b14.configuration_tips import TIPSv2Config
    from transformers_modules.tipsv2_hyphen_b14.modeling_tips import TIPSv2Model
    return TIPSv2Config, TIPSv2Model


def load_tips(device):
    from safetensors.torch import load_file
    import json
    print(f"[INFO] Loading TIPSv2 from {TIPS_PATH} ...")
    TIPSv2Config, TIPSv2Model = _load_tips_module()
    cfg_raw = json.load(open(f"{TIPS_PATH}/config.json"))
    skip_keys = {"_name_or_path", "transformers_version", "auto_map", "architectures",
                 "model_type", "torch_dtype"}
    cfg = TIPSv2Config(**{k: v for k, v in cfg_raw.items() if k not in skip_keys})
    model = TIPSv2Model(cfg)
    sd = load_file(f"{TIPS_PATH}/model.safetensors")
    model.load_state_dict(sd, strict=True)
    model.eval().to(device)
    print("[INFO] TIPSv2 loaded.")
    return {"model": model, "patch_size": 14, "img_size": 448}


_TIPS_TRANSFORM = T.ToTensor()   # [0,1], no normalization per TIPSv2 spec


@torch.no_grad()
def extract_tips(img: Image.Image, info: dict, device) -> np.ndarray:
    """Returns (H_p, W_p, D) patch features."""
    ps = info["patch_size"]
    img_size = info["img_size"]
    img_r = img.resize((img_size, img_size), Image.BICUBIC)
    x = _TIPS_TRANSFORM(img_r).unsqueeze(0).to(device)
    out = info["model"].encode_image(x)
    tokens = out.patch_tokens[0].cpu().float().numpy()  # (H_p*W_p, D)
    H_p = W_p = img_size // ps
    return tokens.reshape(H_p, W_p, -1)


# ─── PCA 工具 ─────────────────────────────────────────────────────────────────

def pca_rgb_across_images(feats_list: list, n_components: int = 3) -> list:
    """
    跨图像全局 PCA，保证颜色语义一致。
    None 条目保持 None 原样输出。
    """
    valid_idx = [i for i, f in enumerate(feats_list) if f is not None]
    if not valid_idx:
        return [None] * len(feats_list)

    shapes = [(feats_list[i].shape[0], feats_list[i].shape[1]) for i in valid_idx]
    tokens_all = np.concatenate(
        [feats_list[i].reshape(-1, feats_list[i].shape[-1]) for i in valid_idx], 0
    )

    pca = PCA(n_components=n_components)
    pca_all = pca.fit_transform(tokens_all)
    lo = pca_all.min(axis=0, keepdims=True)
    hi = pca_all.max(axis=0, keepdims=True)
    pca_all = (pca_all - lo) / (hi - lo + 1e-8)

    results = [None] * len(feats_list)
    idx = 0
    for list_i, (H, W) in zip(valid_idx, shapes):
        n = H * W
        results[list_i] = pca_all[idx: idx + n].reshape(H, W, n_components)
        idx += n
    return results


def pca_to_rgb_img(pca_map: np.ndarray, target_size) -> Image.Image:
    """Upsample (H_p, W_p, 3) → PIL RGB at target_size."""
    return Image.fromarray((pca_map * 255).astype(np.uint8)).resize(
        target_size, resample=Image.NEAREST
    )


# ─── 可视化 ───────────────────────────────────────────────────────────────────

def visualize_comparison_grid(images_orig, all_pca_maps, model_names, output_path):
    """rows = images, cols = Original | model PCA maps."""
    n_imgs = len(images_orig)
    n_cols = 1 + len(model_names)
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_imgs))
    gs = gridspec.GridSpec(n_imgs, n_cols, figure=fig, hspace=0.04, wspace=0.04)

    col_titles = ["Original"] + model_names
    for j, title in enumerate(col_titles):
        ax = fig.add_subplot(gs[0, j])
        ax.set_title(title, fontsize=11, pad=6)
        ax.axis("off")

    for i, img in enumerate(images_orig):
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(img)
        ax.axis("off")
        for j, pca_maps in enumerate(all_pca_maps):
            ax = fig.add_subplot(gs[i, 1 + j])
            pm = pca_maps[i]
            if pm is not None:
                ax.imshow(pca_to_rgb_img(pm, img.size))
            else:
                ax.text(0.5, 0.5, "N/A", ha='center', va='center',
                        fontsize=14, color='gray', transform=ax.transAxes)
            ax.axis("off")

    plt.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[INFO] Saved → {output_path}")


def visualize_single(img, pca_maps_per_model, model_names, output_path):
    """One row: Original | each model's PCA RGB."""
    valid = [(n, pm) for n, pm in zip(model_names, pca_maps_per_model) if pm is not None]
    n_cols = 1 + len(valid)
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 4))
    if n_cols == 1:
        axes = [axes]
    axes[0].imshow(img)
    axes[0].set_title("Original", fontsize=10)
    axes[0].axis("off")
    for ax, (name, pm) in zip(axes[1:], valid):
        ax.imshow(pca_to_rgb_img(pm, img.size))
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved → {output_path}")


# ─── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir",
        default=f"{_BASE}/datas/coco/images/val2014")
    parser.add_argument("--num_images", type=int, default=6)
    parser.add_argument("--output_dir", default="/tmp/model_pca_output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resize", type=int, default=None,
        help="短边 resize 到指定像素（None=原始分辨率）")
    parser.add_argument("--skip_dinov3", action="store_true")
    parser.add_argument("--skip_radio",  action="store_true")
    parser.add_argument("--skip_eupe",   action="store_true",
        help="跳过 EUPE（repo 不可用时使用）")
    parser.add_argument("--skip_tips",   action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    # ── 加载模型 ──
    model_infos = {}
    if not args.skip_dinov3:
        model_infos["DINOv3"] = load_dinov3(device)
    if not args.skip_radio:
        model_infos["RADIO"]  = load_radio(device)
    if not args.skip_tips:
        model_infos["TIPSv2"] = load_tips(device)
    # EUPE last (may need git clone)
    eupe_info = load_eupe(device, skip_eupe=args.skip_eupe)
    if eupe_info is not None:
        model_infos["EUPE"] = eupe_info

    extract_fn = {
        "DINOv3": extract_dinov3,
        "RADIO":  extract_radio,
        "EUPE":   extract_eupe,
        "TIPSv2": extract_tips,
    }

    # ── 随机采样图像 ──
    img_paths = list(Path(args.img_dir).glob("*.jpg"))
    assert img_paths, f"No jpg found in {args.img_dir}"
    img_paths = random.sample(img_paths, min(args.num_images, len(img_paths)))
    print(f"[INFO] Using {len(img_paths)} images")

    # ── 提取 patch 特征 ──
    images_orig = []
    all_raw_feats = {name: [] for name in model_infos}

    for p in img_paths:
        img = Image.open(p).convert("RGB")
        if args.resize:
            img = resize_short_edge(img, args.resize)
        images_orig.append(img)
        print(f"  {p.name}  size={img.size}")
        for name, info in model_infos.items():
            try:
                feats = extract_fn[name](img, info, device)
                all_raw_feats[name].append(feats)
                if feats is not None:
                    print(f"    [{name}] patches={feats.shape[:2]} dim={feats.shape[2]}")
            except Exception as e:
                print(f"    [{name}] ERROR: {e}")
                all_raw_feats[name].append(None)

    # ── 每模型独立全局 PCA ──
    model_names = list(model_infos.keys())
    all_pca_maps = []
    for name in model_names:
        print(f"[INFO] PCA for {name} ...")
        all_pca_maps.append(pca_rgb_across_images(all_raw_feats[name]))

    # ── 绘图 ──
    visualize_comparison_grid(
        images_orig, all_pca_maps, model_names,
        str(output_dir / "comparison_grid.png")
    )
    for i, (img, p) in enumerate(zip(images_orig, img_paths)):
        per_model_pca = [all_pca_maps[j][i] for j in range(len(model_names))]
        visualize_single(img, per_model_pca, model_names,
                         str(output_dir / f"comparison_{i:02d}_{p.stem}.png"))

    print(f"\n[DONE] All outputs in: {output_dir}")


if __name__ == "__main__":
    main()
