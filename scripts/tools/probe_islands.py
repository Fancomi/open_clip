#!/usr/bin/env python3
"""在真实训练好的 ckpt 上验证「多岛结构」与三项正则的检测能力。

回答：真实 CLIP 特征是否是多岛的？center/scale/shape 三项分别能否检测到它？

做法：
  1. 用 ckpt 跑真实 cc3m 图像，拿 cls raw 特征 [N, D]
  2. 多岛证据：k-means 聚类后的簇内/簇间距离比、有效秩、最近邻同簇率
  3. 三项正则在真实特征上的实际数值
  4. 对照：同均值同协方差的高斯（打散岛结构，保留二阶统计）
     → 若三项数值几乎不变，说明三项都「看不见」多岛结构

用法:
  python scripts/tools/probe_islands.py --ckpt logs/<run>/checkpoints/epoch_7.pt --n 8192
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


# ---------- 三项正则（与 loss.py:VISReg 同口径） ----------
def gauss_q(n, device, dtype):
    u = torch.arange(1, n + 1, device=device, dtype=torch.float32) / (n + 1)
    return (torch.erfinv(2 * u - 1) * math.sqrt(2.0)).to(dtype)


def three_terms(z, K=256, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    D = z.shape[1]
    mu = z.mean(0, keepdim=True)
    center = mu.pow(2).mean().item()
    zc = z - mu
    std = (zc.pow(2).mean(0) + 1e-6).sqrt()
    scale = (std - 1.0).pow(2).mean().item()
    W = torch.randn(D, K, generator=g)
    W = W / W.norm(dim=0, keepdim=True)
    p = ((zc / std) @ W).sort(dim=0).values
    shape = (p - gauss_q(z.shape[0], z.device, z.dtype).unsqueeze(1)).pow(2).mean().item()
    return center, scale, shape


def spectrum(z):
    zc = z - z.mean(0)
    C = (zc.T @ zc) / (z.shape[0] - 1)
    e = torch.linalg.eigvalsh(C).clamp(min=1e-12)
    p = e / e.sum()
    erank = torch.exp(-(p * p.log()).sum()).item()
    return (e.max() / e.min()).item(), erank


# ---------- 多岛性度量 ----------
def kmeans(z, k, iters=30, seed=0):
    g = torch.Generator(device=z.device).manual_seed(seed)
    idx = torch.randperm(z.shape[0], generator=g, device=z.device)[:k]
    C = z[idx].clone()
    for _ in range(iters):
        a = torch.cdist(z, C).argmin(1)
        for j in range(k):
            m = a == j
            if m.any():
                C[j] = z[m].mean(0)
    return a, C


def island_stats(z, k=50):
    """簇内/簇间距离比 + 最近邻同簇率。比值越小、同簇率越高 = 越'多岛'。"""
    a, C = kmeans(z, k)
    within = torch.stack([(z[a == j] - C[j]).norm(dim=1).mean()
                          for j in range(k) if (a == j).any()]).mean()
    cd = torch.cdist(C, C)
    between = cd[~torch.eye(len(C), dtype=torch.bool, device=z.device)].mean()
    # 最近邻同簇率（随机取 1000 个点）
    sub = torch.randperm(z.shape[0])[:1000]
    d = torch.cdist(z[sub], z)
    d.scatter_(1, sub.unsqueeze(1).to(d.device), float("inf"))
    nn = d.argmin(1)
    same = (a[sub] == a[nn]).float().mean().item()
    return within.item(), between.item(), (within / between).item(), same


@torch.no_grad()
def extract(ckpt, tsv, n, batch, device):
    m = open_clip.create_model("PE-Core-B-16-dinov3", pretrained=None, output_dict=True)
    m = CLIPLeJEPA(clip_model=m, sigreg_target="cls", output_dict=True)
    if ckpt != "untrained":
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        sd = {k[len("module."):]: v for k, v in sd.items() if k.startswith("module.")}
        m.load_state_dict(sd, strict=False)
    m = m.to(device).eval()
    _, _, pp = open_clip.create_model_and_transforms("PE-Core-B-16-dinov3", pretrained=None)
    df = pd.read_csv(tsv, sep="\t", nrows=n * 2)
    feats, buf, used = [], [], 0
    for p in df["filepath"].tolist():
        if used >= n:
            break
        try:
            buf.append(pp(Image.open(p).convert("RGB")))
        except Exception:
            continue
        if len(buf) == batch:
            _, c = m._get_image_raw(torch.stack(buf).to(device))
            feats.append(c.float().cpu()); used += len(buf); buf = []
    return torch.cat(feats)[:n]


def report(z, tag, k):
    print(f"\n===== {tag} =====")
    ce, sc, sh = three_terms(z)
    cond, er = spectrum(z)
    wi, be, ratio, same = island_stats(z, k)
    print(f"  三项正则  center={ce:.5f}  scale={sc:.5f}  shape={sh:.5f}")
    print(f"  谱        条件数={cond:.1f}  有效秩={er:.0f}/{z.shape[1]}")
    print(f"  多岛性    簇内均距={wi:.2f}  簇间均距={be:.2f}  比值={ratio:.3f}  最近邻同簇率={same*100:.1f}%")
    return dict(center=ce, scale=sc, shape=sh, cond=cond, erank=er, ratio=ratio, same=same)


def gaussianize(z):
    """生成同均值、同协方差的高斯样本：打散岛结构，保留全部二阶统计。"""
    mu = z.mean(0)
    zc = z - mu
    C = (zc.T @ zc) / (z.shape[0] - 1)
    C = C + 1e-5 * torch.eye(C.shape[0])
    L = torch.linalg.cholesky(C)
    return torch.randn_like(z) @ L.T + mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tsv", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations/clip_train.tsv")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--k", type=int, default=50, help="k-means 簇数")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    print(f"=> ckpt: {args.ckpt}")
    z = extract(args.ckpt, args.tsv, args.n, args.batch, torch.device(args.device))
    print(f"   cls raw: {tuple(z.shape)}")
    tag = args.tag or args.ckpt.split("/")[-3]

    real = report(z, f"{tag}｜真实特征", args.k)
    fake = report(gaussianize(z), f"{tag}｜同协方差高斯（岛被打散）", args.k)

    print(f"\n----- 关键对比：打散岛结构后，三项变化了多少 -----")
    for kk in ("center", "scale", "shape"):
        r, f = real[kk], fake[kk]
        print(f"  {kk:<7} 真实={r:.5f}  打散后={f:.5f}  变化={((f-r)/max(r,1e-9)*100):+7.1f}%")
    print(f"  {'同簇率':<7} 真实={real['same']*100:.1f}%  打散后={fake['same']*100:.1f}%"
          f"   ← 岛结构确实被打散")
    print("\n若三项变化都很小 ⟹ 三项都『看不见』多岛结构（它们只约束一/二阶统计与边缘分布）")


if __name__ == "__main__":
    main()
