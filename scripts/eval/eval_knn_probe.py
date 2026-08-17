#!/usr/bin/env python3
"""IN-1k k-NN probe：不依赖文本的纯图像特征质量评测。

为什么需要它：现有三个评测（COCO 5cap / IN-1k zero-shot / Urban-1k）
**全部用文本当分类器或 query**，所以它们测的是"图文对齐"，无法区分
"图像塔变好了"还是"文本塔更会对齐了"。

k-NN probe 把文本完全移出链路：冻结骨干 → 提特征 → 近邻投票。
零训练、无超参（除 k）、DINO 系的标准图像侧指标。

两个特征层都测（差异本身说明投影头丢了多少视觉信息）：
  backbone : trunk CLS token，投影头之前的原始视觉特征（PE-Core: 768 维）
  proj     : encode_image 输出，对齐文本空间后的特征（1024 维）

★ 自比用法 ★
  本项目在 CC3M 2M 量级下自比，绝对值不与大规模预训练模型（DINOv3 等）对照。
  关注的是同源变体间的相对差异：gt_base / pcm_w* / gemma_dense 谁的图像塔更好。

  关键对照：gemma_dense 的 zero-shot 只有 1.85%（文本塔崩溃），
  但若其 k-NN 不差，则说明纯 dense 训练的**图像塔是好的，只是文本塔坏了**——
  这是 zero-shot 永远回答不了的问题。

用法:
  python scripts/eval/eval_knn_probe.py --ckpt logs/.../epoch_10.pt --tag pcm_w0.3
  python scripts/eval/eval_knn_probe.py --ckpt ... --tag gt_base --per-class 20 --num-classes 200
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

IMAGENET_VAL = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-val")


def load_model(ckpt_path, device):
    from open_clip import create_model_and_transforms
    from open_clip.model import CLIPLeJEPA

    base, _, val_tr = create_model_and_transforms(
        "PE-Core-B-16-dinov3", "", precision="fp32", device="cpu",
        output_dict=True, force_context_length=256)
    model = CLIPLeJEPA(clip_model=base, sigreg_target="cls", output_dict=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  加载 {Path(ckpt_path).name}: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    return model, val_tr


@torch.no_grad()
def extract_feats(model, val_tr, device, num_classes, per_class, batch=50):
    """返回 (backbone_cls, proj_cls, labels)。backbone = 投影前，proj = 投影后。"""
    dirs = sorted([d for d in IMAGENET_VAL.iterdir() if d.is_dir()])[:num_classes]
    paths, labels = [], []
    for i, d in enumerate(dirs):
        for img in sorted(d.iterdir())[:per_class]:
            paths.append(img)
            labels.append(i)

    visual = model.visual
    has_trunk = hasattr(visual, "trunk") and hasattr(visual.trunk, "forward_features")
    dt = torch.float16 if device == "cuda" else torch.float32

    bb_list, proj_list = [], []
    for i in range(0, len(paths), batch):
        imgs = torch.stack([val_tr(Image.open(p).convert("RGB")) for p in paths[i:i + batch]])
        imgs = imgs.to(device=device, dtype=dt)
        if has_trunk:
            bb_list.append(visual.trunk.forward_features(imgs)[:, 0, :].float().cpu())
        proj_list.append(model.encode_image(imgs, normalize=True).float().cpu())
    bb = torch.cat(bb_list) if bb_list else None
    proj = torch.cat(proj_list)
    return bb, proj, torch.tensor(labels)


def knn_accuracy(feats, labels, k=20, temp=0.07, num_classes=None):
    """留一法 k-NN：每个样本用其余全部样本做近邻投票（DINO 的做法）。

    余弦相似度 + softmax(sim/temp) 加权投票。排除自身（对角置 -inf）。
    """
    n = feats.shape[0]
    nc = num_classes or int(labels.max().item()) + 1
    f = F.normalize(feats.float(), dim=-1)
    correct = 0
    # 分块避免 n×n 矩阵驻留
    for s in range(0, n, 500):
        e = min(s + 500, n)
        sim = f[s:e] @ f.T                              # [B, n]
        # 排除自身
        for r, idx in enumerate(range(s, e)):
            sim[r, idx] = -float("inf")
        topv, topi = sim.topk(k, dim=1)                 # [B, k]
        w = (topv / temp).softmax(dim=1)                # 相似度加权
        votes = torch.zeros(e - s, nc)
        votes.scatter_add_(1, labels[topi], w)
        correct += (votes.argmax(1) == labels[s:e]).sum().item()
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-classes", type=int, default=100)
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--k", type=int, default=20, help="k-NN 的 k（DINO 默认 20）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.tag}] IN-1k k-NN probe | device={device} "
          f"{args.num_classes} 类 × {args.per_class} 图, k={args.k}", flush=True)
    print("  （纯图像口径：无文本参与，冻结骨干，零训练）", flush=True)

    model, val_tr = load_model(args.ckpt, device)
    bb, proj, labels = extract_feats(model, val_tr, device, args.num_classes, args.per_class)
    n, nc = len(labels), args.num_classes
    print(f"  特征: backbone={tuple(bb.shape) if bb is not None else None} "
          f"proj={tuple(proj.shape)}  样本={n}  随机基线={1/nc:.2%}", flush=True)

    if bb is not None:
        acc_bb = knn_accuracy(bb, labels, k=args.k, num_classes=nc)
        print(f"  k-NN backbone (投影前, {bb.shape[1]}d): {acc_bb:.4f}", flush=True)
    acc_proj = knn_accuracy(proj, labels, k=args.k, num_classes=nc)
    print(f"  k-NN proj     (投影后, {proj.shape[1]}d): {acc_proj:.4f}", flush=True)


if __name__ == "__main__":
    main()
