"""
DINOv3 权重加载验证脚本（正规三路对比版）

验证逻辑：
  Reference（ground truth）:
    官方原始 .pth  →  官方 hub 代码（原生 key，零 remap，strict=True）

  待验证：
    safetensors  →  transformers AutoModel

  两路用同一个 pixel_values tensor 做前向，对比 patch token 数值。
  完全独立，无循环依赖。
"""

import sys
import torch
from safetensors import safe_open
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

DINOV3_REPO   = "/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/dinov3"
PTH_PATH      = "/root/paddlejob/workspace/env_run/penghaotian/models/dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
HF_MODEL_PATH = "/root/paddlejob/workspace/env_run/penghaotian/models/dino/dinov3-vitb16-pretrain-lvd1689m"
TEST_IMAGE    = "/root/paddlejob/workspace/env_run/penghaotian/datas/coco/images/val2014/COCO_val2014_000000000042.jpg"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATCH_SIZE = 16
IMAGE_SIZE = 448


# ─── Reference: 官方 .pth 直接 load 进 hub 代码 ───────────────────────────────

def load_hub_reference(pth_path: str) -> torch.nn.Module:
    sys.path.insert(0, DINOV3_REPO)
    model = torch.hub.load(DINOV3_REPO, "dinov3_vitb16", source="local", pretrained=False)
    ckpt = torch.load(pth_path, map_location="cpu")
    result = model.load_state_dict(ckpt, strict=True)   # strict=True，不允许任何 key 缺失
    print(f"  [OK] hub reference loaded strictly. missing={result.missing_keys}, unexpected={result.unexpected_keys}")
    return model.eval().to(DEVICE)


# ─── 待验证: transformers AutoModel ───────────────────────────────────────────

def load_hf_model(model_path: str):
    proc  = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True, do_resize=False)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(DEVICE)
    return proc, model


# ─── Step 5: 准备图像 ─────────────────────────────────────────────────────────

def prepare_tensors(image_path: str):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    scale = IMAGE_SIZE / min(w, h)
    nw = ((int(w * scale)) // PATCH_SIZE) * PATCH_SIZE
    nh = ((int(h * scale)) // PATCH_SIZE) * PATCH_SIZE
    img = img.resize((nw, nh), Image.BICUBIC)

    # 两个模型共用同一个归一化后的 tensor（用 hf processor 做归一化）
    # 这样完全排除预处理差异，只验证模型权重
    return img


# ─── Step 6: 前向对比 ─────────────────────────────────────────────────────────

def compare(hub_model, hf_proc, hf_model, img_pil):
    # 两个模型共用同一个 pixel_values tensor
    inputs = hf_proc(images=img_pil, return_tensors="pt")
    pv = inputs["pixel_values"].to(DEVICE)
    print(f"      Shared pixel_values shape: {pv.shape}")

    # --- hub forward ---
    with torch.no_grad():
        hub_feats = hub_model.forward_features(pv)
    # x_norm_patchtokens: (1, N_patch, D)  — post-norm patch tokens
    hub_patch = hub_feats["x_norm_patchtokens"][0].cpu().float()

    # --- hf forward ---
    with torch.no_grad():
        out_hf = hf_model(pixel_values=pv)
    # last_hidden_state: (1, 1+4+N_patch, D)，跳过 CLS(1)+REG(4)
    hf_patch = out_hf.last_hidden_state[0, 5:, :].cpu().float()

    print(f"\n{'='*60}")
    print(f"  hub  patch tokens : {hub_patch.shape}")
    print(f"  hf   patch tokens : {hf_patch.shape}")

    assert hub_patch.shape == hf_patch.shape, \
        f"Shape mismatch! hub={hub_patch.shape} hf={hf_patch.shape}"

    diff = (hub_patch - hf_patch).abs()
    cos  = torch.nn.functional.cosine_similarity(hub_patch, hf_patch, dim=-1)

    print(f"\n  Absolute diff  max={diff.max():.6f}  mean={diff.mean():.6f}")
    print(f"  Cosine sim     min={cos.min():.6f}  mean={cos.mean():.6f}")

    # 阈值：cos > 0.9999 且 mean_abs_diff < 1e-3（允许 rope 实现差异带来的小误差）
    ok_cos  = cos.mean().item() > 0.9999
    ok_diff = diff.mean().item() < 1e-3

    print(f"\n{'='*60}")
    if ok_cos and ok_diff:
        print("  RESULT: PASS  -- HF model numerically matches hub reference.")
    else:
        if not ok_cos:
            print(f"  RESULT: FAIL  -- Cosine sim too low ({cos.mean():.6f})")
        if not ok_diff:
            print(f"  RESULT: FAIL  -- Mean abs diff too high ({diff.mean():.6f})")
    print(f"{'='*60}\n")

    return hub_patch, hf_patch


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("[1/4] Loading hub reference (official .pth, strict=True) ...")
    hub_model = load_hub_reference(PTH_PATH)

    print("[2/4] Loading HF transformers model (safetensors) ...")
    hf_proc, hf_model = load_hf_model(HF_MODEL_PATH)

    print("[3/4] Preparing test image ...")
    img_pil = prepare_tensors(TEST_IMAGE)
    print(f"      Image size used: {img_pil.size}")

    print("[4/4] Running forward pass comparison ...")
    hub_out, hf_out = compare(hub_model, hf_proc, hf_model, img_pil)
