#!/usr/bin/env python3
"""诊断：VISReg 是否缓解了 ClearCLIP 指出的"残差通道主导"病灶。

ClearCLIP (ECCV'24, arXiv:2407.12442) 发现：CLIP 最后一层的残差分量 X_res 在**某个
通道上有极端峰值**，远超其他通道；因 X_sum ≈ X_res，最终 patch 特征被这个全局方向
主导，导致各位置特征方向趋同、局部可区分性差（稠密任务糊）。作者归因于 image-level
对比监督把信息压在"残差潜空间的全局方向"上。

这本质是**各向异性**——正是 SIGReg/VISReg 要治的病，只是视角不同（预训练期正则 vs
推理期删残差）。可检验假设：**VISReg 训出的模型残差主导现象更弱**。

本脚本对最后一个 EvaBlock 分解：
    X_res  = z_in                       (残差旁路)
    X_attn = attn(norm1(z_in))          (注意力分量)
    X_sum  = X_res + X_attn             (残差相加后)
统计（仅 patch token，去掉 CLS）：
  1. 通道范数分布的**峰值主导度** dom = max_c ‖X[:,c]‖ / mean_c ‖X[:,c]‖
     —— ClearCLIP 的核心观测量，越大越病
  2. 通道范数的 top1 占比、基尼系数（分布集中度的补充刻画）
  3. **patch 间平均余弦相似度** —— 特征方向趋同度，越高越难区分（直接对应分割糊）
  4. cos(X_sum, X_res) —— X_sum 被残差主导的程度

对比 SIGReg ckpt 与 VISReg ckpt，同一批真实图像输入。

用法：
  python scripts/tools/diag_residual_dominance.py \
      --ckpt-a logs/visreg_cc3m_A_sigreg_0723_0911/checkpoints/epoch_7.pt \
      --ckpt-b logs/visreg_sweep_E_s1sh1_0723_2059/checkpoints/epoch_7.pt \
      --label-a SIGReg --label-b VISReg --n-images 256
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

import torch

import open_clip
from open_clip.factory import load_checkpoint


def gini(x: torch.Tensor) -> float:
    """基尼系数（0=完全均匀, →1=极度集中）。x 为非负一维张量。"""
    x = x.flatten().sort().values
    n = x.numel()
    idx = torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
    return float((2 * idx - n - 1).mul(x).sum() / (n * x.sum()))


@torch.no_grad()
def analyze(model, images, device):
    """分解最后一个 block 的残差/注意力分量并统计。"""
    trunk = model.visual.trunk
    blocks = trunk.blocks
    last = blocks[-1]

    captured = {}

    def hook(mod, inputs, output):
        # EvaBlock.forward: x = x + drop_path1(attn(norm1(x))) ...
        captured["z_in"] = inputs[0].detach()

    h = last.register_forward_hook(hook, with_kwargs=False)
    # 只跑到 trunk 的 forward_features 即可拿到最后一层输入
    _ = trunk.forward_features(images.to(device))
    h.remove()

    z_in = captured["z_in"]                       # [B, 1+N, D]
    # 复现最后一层的 attn 分支（rot_pos_emb 等按 block 默认签名调用）
    x_norm = last.norm1(z_in)
    try:
        x_attn = last.attn(x_norm)
    except TypeError:
        # 某些 EvaBlock.attn 需要 rope/attn_mask 参数，退化为 None
        x_attn = last.attn(x_norm, rope=None, attn_mask=None)

    X_res = z_in[:, 1:]                           # 去 CLS，仅 patch token
    X_attn = x_attn[:, 1:]
    X_sum = X_res + X_attn

    out = {}
    for name, X in (("X_res", X_res), ("X_attn", X_attn), ("X_sum", X_sum)):
        Xf = X.float()
        # 通道范数：对 (B, N) 展平后按通道取 L2
        ch = Xf.reshape(-1, Xf.shape[-1]).norm(dim=0)       # [D]
        dom = float(ch.max() / ch.mean())
        top1 = float(ch.max() / ch.sum())
        g = gini(ch)
        # patch 间平均余弦相似度（每图内部两两，抽样避免 O(N²) 爆内存）
        Xn = torch.nn.functional.normalize(Xf, dim=-1)
        sim = torch.einsum("bnd,bmd->bnm", Xn, Xn)
        N = sim.shape[-1]
        off = ~torch.eye(N, dtype=torch.bool, device=sim.device)
        mean_cos = float(sim[:, off].mean())
        out[name] = dict(dominance=dom, top1_share=top1, gini=g, patch_cos=mean_cos)

    # X_sum 被残差主导的程度
    out["cos_sum_res"] = float(torch.nn.functional.cosine_similarity(
        X_sum.float().flatten(0, 1), X_res.float().flatten(0, 1), dim=-1).mean())
    out["norm_ratio_res_attn"] = float(X_res.float().norm() / X_attn.float().norm())
    return out


def build(ckpt, device):
    model = open_clip.create_model("PE-Core-B-16-dinov3", pretrained=None, output_dict=True)
    # ckpt 是 CLIPLeJEPA 包装后的 state_dict（key 前缀 clip_model.），剥掉前缀
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    if any(k.startswith("clip_model.") for k in sd):
        sd = {k[len("clip_model."):]: v for k, v in sd.items() if k.startswith("clip_model.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded {ckpt.split('/')[-3]}: missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", required=True)
    ap.add_argument("--ckpt-b", required=True)
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--n-images", type=int, default=256)
    ap.add_argument("--tsv", default="/root/paddlejob/workspace/env_run/penghaotian/"
                                    "datas/coco/annotations/karpathy_1cap.tsv")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    device = torch.device(args.device)

    # 真实图像批（COCO probe tsv 的 filepath 列）
    import pandas as pd
    from PIL import Image
    df = pd.read_csv(args.tsv, sep="\t").head(args.n_images)
    col = "filepath" if "filepath" in df.columns else df.columns[0]
    _, _, preprocess = open_clip.create_model_and_transforms(
        "PE-Core-B-16-dinov3", pretrained=None)
    imgs = []
    for p in df[col].tolist():
        try:
            imgs.append(preprocess(Image.open(str(p)).convert("RGB")))
        except Exception:
            continue
    images = torch.stack(imgs)
    print(f"=> {len(images)} images from {args.tsv}")

    results = {}
    for label, ckpt in ((args.label_a, args.ckpt_a), (args.label_b, args.ckpt_b)):
        print(f"=> analyzing {label}")
        model = build(ckpt, device)
        results[label] = analyze(model, images, device)
        del model
        torch.cuda.empty_cache()

    la, lb = args.label_a, args.label_b
    print("\n" + "=" * 78)
    print(f"{'metric':<34}{la:>14}{lb:>14}{'Δ(B-A)':>14}")
    print("-" * 78)
    for comp in ("X_res", "X_attn", "X_sum"):
        for k in ("dominance", "top1_share", "gini", "patch_cos"):
            a, b = results[la][comp][k], results[lb][comp][k]
            print(f"{comp + '.' + k:<34}{a:>14.4f}{b:>14.4f}{b - a:>+14.4f}")
        print("-" * 78)
    for k in ("cos_sum_res", "norm_ratio_res_attn"):
        a, b = results[la][k], results[lb][k]
        print(f"{k:<34}{a:>14.4f}{b:>14.4f}{b - a:>+14.4f}")
    print("=" * 78)
    print("\n判读：X_res.dominance / top1_share / gini 越小 = 残差通道主导越弱（越健康）")
    print("      X_sum.patch_cos 越小 = patch 特征方向越可区分（利于稠密任务）")
    print("      cos_sum_res 越小 = 最终特征被残差旁路主导越少")


if __name__ == "__main__":
    main()
