#!/usr/bin/env python3
"""IN-1k k-NN probe：不依赖文本的纯图像特征质量评测。

⚠️⚠️ 口径声明：这是**项目自定义口径，不能与 DINO 系论文的 k-NN 数字对标** ⚠️⚠️

  与 DINO / DINOv2 / DINOv3 官方 k-NN 协议的差异（三处）：

  | 项      | DINO 系论文              | 本脚本                      |
  |---------|--------------------------|-----------------------------|
  | 数据    | 全量 50000 图 / 1000 类  | 同（默认已改为全量）        |
  | 特征    | 最后一层**投影后** + LN  | trunk CLS（**投影前**, 768d）|
  | 距离    | 欧氏距离                 | 余弦相似度                  |
  | 投票    | 距离加权 1/d             | softmax(sim/0.07) 温度加权  |

  ⚠️ 2026-08-25 口径整改：默认从 `100 类 × 20 图 = 2000 张` 改为
  `1000 类 × 50 图 = 50000 张`（IN-1k val 全量）。

  起因：`in1k_fullscope_retest.md` 证明 100 类子集会让 IN-1k zero-shot
  **组间排序都翻转**（E_firstbox 子集 25.50 → 全量 27.15，方向不一致），
  因为 `sorted()[:100]` 只取前 100 个 wnid，类别难度分布有偏。
  本脚本第 80 行用的是**同一批前 100 个 wnid**，所以同样的偏置风险适用于
  此前所有 k-NN 结论。旧的 2000 图数字一律作废，输出行带 `⚠️子集` 标记。

  无随机：类目录与文件名都按 sorted() 取前 N 个，完全可复现，但**不是随机抽样**。


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

  关键对照：gemma_dense 的 zero-shot 只有 0.96%（文本塔崩溃），
  但若其 k-NN 不差，则说明纯 dense 训练的**图像塔是好的，只是文本塔坏了**——
  这是 zero-shot 永远回答不了的问题。

用法:
  python scripts/eval/eval_knn_probe.py --ckpt logs/.../epoch_10.pt --tag pcm_w0.2
  # 全量已是默认值，不要再传 --num-classes / --per-class
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
    from open_clip.factory import context_length_from_checkpoint
    from open_clip.model import CLIPLeJEPA

    ctx = context_length_from_checkpoint(ckpt_path)
    base, _, val_tr = create_model_and_transforms(
        "PE-Core-B-16-dinov3", "", precision="fp32", device="cpu",
        output_dict=True, force_context_length=ctx)
    model = CLIPLeJEPA(clip_model=base, sigreg_target="cls", output_dict=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  加载 {Path(ckpt_path).name}: missing={len(missing)} unexpected={len(unexpected)} "
          f"context_length={ctx}（探测值；本口径纯图像，不受它影响，只为杜绝静默 resize）", flush=True)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    return model, val_tr


@torch.no_grad()
def extract_feats(model, val_tr, device, num_classes, per_class, batch=100,
                  num_workers=12):
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

    # 多进程解码：单线程 ~90 img/s，全量 5 万张要 9 分钟且 GPU 空转。
    # DataLoader(shuffle=False) 顺序与 transform 不变，逐样本结果等价。
    class _ValSet(torch.utils.data.Dataset):
        def __len__(self): return len(paths)
        def __getitem__(self, i):
            return val_tr(Image.open(paths[i]).convert("RGB")), labels[i]

    loader = torch.utils.data.DataLoader(
        _ValSet(), batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    bb_list, proj_list, lab_list = [], [], []
    done = 0
    for ts, lab in loader:
        imgs = ts.to(device=device, dtype=dt)
        if has_trunk:
            bb_list.append(visual.trunk.forward_features(imgs)[:, 0, :].float().cpu())
        proj_list.append(model.encode_image(imgs, normalize=True).float().cpu())
        lab_list.append(lab)
        done += len(lab)
        if done % 10000 == 0:
            print(f"    ... {done}/{len(paths)}", flush=True)
    bb = torch.cat(bb_list) if bb_list else None
    proj = torch.cat(proj_list)
    return bb, proj, torch.cat(lab_list)


def knn_accuracy(feats, labels, k=20, temp=0.07, num_classes=None, device="cpu"):
    """留一法 k-NN：每个样本用其余全部样本做近邻投票（DINO 的做法）。

    余弦相似度 + softmax(sim/temp) 加权投票。排除自身（对角置 -inf）。
    全量 5 万样本时 n×n 相似度在 CPU 上要几分钟，故搬到 GPU 分块算
    （峰值显存 = feats 154MB + 单块 sim 100MB，可与训练共存）。
    """
    n = feats.shape[0]
    nc = num_classes or int(labels.max().item()) + 1
    f = F.normalize(feats.float().to(device), dim=-1)
    lab = labels.to(device)
    correct = 0
    # 分块避免 n×n 矩阵驻留
    for s in range(0, n, 500):
        e = min(s + 500, n)
        sim = f[s:e] @ f.T                              # [B, n]
        # 排除自身
        sim[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -float("inf")
        topv, topi = sim.topk(k, dim=1)                 # [B, k]
        w = (topv / temp).softmax(dim=1)                # 相似度加权
        votes = torch.zeros(e - s, nc, device=device)
        votes.scatter_add_(1, lab[topi], w)
        correct += (votes.argmax(1) == lab[s:e]).sum().item()
    return correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--per-class", type=int, default=50)
    ap.add_argument("--k", type=int, default=20, help="k-NN 的 k（DINO 默认 20）")
    ap.add_argument("--num-workers", type=int, default=12, help="图像解码进程数")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    scope = ("★全量★" if (args.num_classes >= 1000 and args.per_class >= 50)
             else "⚠️子集(不可与全量混比)")
    print(f"[{args.tag}] IN-1k k-NN probe | device={device} "
          f"{args.num_classes} 类 × {args.per_class} 图, k={args.k} {scope}", flush=True)
    print("  （纯图像口径：无文本参与，冻结骨干，零训练）", flush=True)

    model, val_tr = load_model(args.ckpt, device)
    bb, proj, labels = extract_feats(model, val_tr, device,
                                     args.num_classes, args.per_class,
                                     num_workers=args.num_workers)
    n, nc = len(labels), args.num_classes
    print(f"  特征: backbone={tuple(bb.shape) if bb is not None else None} "
          f"proj={tuple(proj.shape)}  样本={n}  随机基线={1/nc:.2%}", flush=True)

    if bb is not None:
        acc_bb = knn_accuracy(bb, labels, k=args.k, num_classes=nc, device=device)
        print(f"  k-NN backbone (投影前, {bb.shape[1]}d): {acc_bb:.4f} {scope}", flush=True)
    acc_proj = knn_accuracy(proj, labels, k=args.k, num_classes=nc, device=device)
    print(f"  k-NN proj     (投影后, {proj.shape[1]}d): {acc_proj:.4f} {scope}", flush=True)


if __name__ == "__main__":
    main()
