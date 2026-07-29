#!/usr/bin/env python3
"""在真实训练好的 checkpoint 上测量 shape loss 的逐切片离散度与各向异性。

回答的问题：投影降维后各切片的 shape 差异有多大？会打架吗？真实特征上是什么量级？

用真实 cc3m 图像跑 backbone 拿 cls raw 特征 [N, D]，然后：
  1. 逐切片 shape loss 的离散度（min/max/std，max/min 倍数）
  2. 特征协方差谱的条件数（各向异性程度）
  3. K（切片数）对 loss 估计波动的影响

用法:
  python scripts/tools/probe_shape_slices.py \
      --ckpt logs/<run>/checkpoints/epoch_7.pt --n 4096
"""
import argparse
import math
import warnings

warnings.filterwarnings("ignore")

import torch
import pandas as pd
from PIL import Image

import open_clip
from open_clip.model import CLIPLeJEPA


def gauss_quantiles(n, device, dtype):
    u = torch.arange(1, n + 1, device=device, dtype=torch.float32) / (n + 1)
    return (torch.erfinv(2 * u - 1) * math.sqrt(2.0)).to(dtype)


def per_slice_shape_loss(z, W):
    """复刻 VISReg.shape：逐维标准化(detach) → 投影 → 排序 → 比高斯分位数。
    返回每个切片各自的 loss [K]。"""
    zc = z - z.mean(dim=0, keepdim=True)
    std = (zc.pow(2).mean(dim=0) + 1e-6).sqrt()
    zn = zc / std
    p = (zn @ W).sort(dim=0).values                       # [N, K]
    q = gauss_quantiles(z.shape[0], z.device, z.dtype).unsqueeze(1)
    return (p - q).pow(2).mean(dim=0)                      # [K]


@torch.no_grad()
def extract_cls(ckpt, tsv, n, batch, device):
    """加载 ckpt，跑真实图像，返回 cls raw 特征 [n, D]。"""
    model = open_clip.create_model("PE-Core-B-16-dinov3", pretrained=None, output_dict=True)
    model = CLIPLeJEPA(clip_model=model, sigreg_target="cls", output_dict=True)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    sd = {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded ckpt (missing={len(missing)} unexpected={len(unexpected)})")
    model = model.to(device).eval()

    _, _, preprocess = open_clip.create_model_and_transforms("PE-Core-B-16-dinov3", pretrained=None)
    df = pd.read_csv(tsv, sep="\t", nrows=n * 2)
    paths = df["filepath"].tolist()

    feats, used = [], 0
    buf = []
    for p in paths:
        if used >= n:
            break
        try:
            buf.append(preprocess(Image.open(p).convert("RGB")))
        except Exception:
            continue
        if len(buf) == batch:
            x = torch.stack(buf).to(device)
            _, cls = model._get_image_raw(x)
            feats.append(cls.float().cpu())
            used += len(buf)
            buf = []
    if buf and used < n:
        x = torch.stack(buf).to(device)
        _, cls = model._get_image_raw(x)
        feats.append(cls.float().cpu())
    z = torch.cat(feats)[:n]
    print(f"  extracted cls raw: {tuple(z.shape)}")
    return z


def report(z, tag):
    D = z.shape[1]
    print(f"\n===== {tag}  (N={z.shape[0]}, D={D}) =====")

    # 1. 逐切片离散度
    W = torch.nn.functional.normalize(torch.randn(D, 256), dim=0)
    L = per_slice_shape_loss(z, W)
    print(f"  逐切片 shape loss: min={L.min():.5f}  max={L.max():.5f}  "
          f"mean={L.mean():.5f}  std={L.std():.5f}  max/min={L.max()/L.min():.0f}x")

    # 2. 协方差谱 / 各向异性
    zc = z - z.mean(0)
    C = (zc.T @ zc) / (z.shape[0] - 1)
    e = torch.linalg.eigvalsh(C).clamp(min=1e-12)
    print(f"  协方差特征值: min={e.min():.3e}  max={e.max():.3e}  条件数={e.max()/e.min():.1f}")
    print(f"  有效秩(谱熵): {torch.exp(-(e/e.sum()*(e/e.sum()).log()).sum()).item():.1f} / {D}")

    # 3. K 对估计波动的影响
    print("  K 对 loss 估计波动的影响:")
    for K in (16, 64, 256, 1024):
        vals = [per_slice_shape_loss(z, torch.nn.functional.normalize(torch.randn(D, K), dim=0)).mean().item()
                for _ in range(20)]
        v = torch.tensor(vals)
        print(f"    K={K:>5}: 均值={v.mean():.5f}  跨次 std={v.std():.6f}  相对波动={(v.std()/v.mean()*100):.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tsv", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations/clip_train.tsv")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    print(f"=> ckpt: {args.ckpt}")
    z = extract_cls(args.ckpt, args.tsv, args.n, args.batch, torch.device(args.device))
    report(z, args.tag or args.ckpt.split("/")[-3])

    # 对照：同形状的理想各向同性高斯
    report(torch.randn_like(z), "对照组：理想各向同性高斯 N(0,I)")


if __name__ == "__main__":
    main()
