#!/usr/bin/env python3
"""gemma dense 长文本 checkpoint 的真实价值评估。

动机：gemma dense（长文本训练）在短模板 IN-1k 上只有 0.93%——但这是假象，
zero-shot 用短模板（"a photo of X"），长文本塔对短句不匹配。

本脚本用两种方式评估：
1. 长描述模板 IN-1k：描述式模板（非短句），测长文本塔的分类能力
2. 长文本检索 COCO：用 dense caption 作为 query 测检索质量（对比 gt caption）

用法:
  python scripts/eval/eval_gemma_dense.py \
      --ckpt logs/visreg_gemma_dense_256_E_0806_2011/checkpoints/epoch_9.pt \
      --tag gemma_dense
  python scripts/eval/eval_gemma_dense.py \
      --ckpt logs/visreg_gemma_gt_gt_base_0806_2011/checkpoints/epoch_9.pt \
      --tag gemma_gt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

IMAGENET_VAL = "/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-val"
COCO = "/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
COCO_GT = f"{COCO}/karpathy_5cap.tsv"
COCO_DENSE = "/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations/clip_train_dense_256.tsv"

# 长描述模板（替代短句模板）：描述式，与 gemma dense 长文本匹配
LONG_TEMPLATES = (
    lambda c: f'an image showing a {c}, with rich visual detail and clear composition.',
    lambda c: f'a detailed photograph of a {c}, capturing its texture and surroundings.',
)

# 标准短模板（对照）
SHORT_TEMPLATES = (
    lambda c: f'a photo of a {c}.',
)


def load_classnames():
    """ImageNet classnames（classname.txt 顺序对应排序后的目录）。"""
    names = []
    with open(f"{IMAGENET_VAL}/classname.txt") as f:
        for line in f:
            parts = line.strip().split(",", 1)
            names.append(parts[1].strip() if len(parts) == 2 else parts[0].strip())
    return names


class ImageNetVal(Dataset):
    def __init__(self, root, transform, per_class=50, num_classes=None):
        self.paths, self.labels = [], []
        dirs = [d for d in sorted(Path(root).iterdir()) if d.is_dir()]
        if num_classes:
            dirs = dirs[:num_classes]  # 子集加速（CPU 评估）
        for i, d in enumerate(dirs):
            for img in sorted(d.iterdir())[:per_class]:  # 抽 per_class/类
                self.paths.append(img)
                self.labels.append(i)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.transform(Image.open(self.paths[i]).convert("RGB")), self.labels[i]


def evaluate_imagenet(model, tokenizer, templates, preprocess, device, per_class=50, num_classes=None):
    """长模板/短模板 IN-1k top1。"""
    ds = ImageNetVal(IMAGENET_VAL, preprocess, per_class, num_classes)
    dl = DataLoader(ds, batch_size=64, num_workers=0)  # 单进程 + torch 多线程前向
    classnames = load_classnames()[:num_classes] if num_classes else load_classnames()  # 与目录对齐
    # 构建文本 classifier
    with torch.no_grad():
        texts = [t(c) for c in classnames for t in templates]
        toks = tokenizer(texts).to(device)
        cls_emb = model.encode_text(toks, normalize=True)
        cls_emb = cls_emb.reshape(len(classnames), len(templates), -1).mean(1)
        cls_emb = torch.nn.functional.normalize(cls_emb, dim=-1).T
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in dl:
            _dt = next(model.parameters()).dtype
            feat = model.encode_image(imgs.to(device=device, dtype=_dt), normalize=True)
            logits = feat @ cls_emb
            correct += (logits.argmax(1) == labels.to(device)).sum().item()
            total += len(labels)
    return correct / total


def load_tsv_captions(path, n=1000):
    """从 TSV 加载前 n 条 (path, caption)。"""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            p, c = line.rstrip("\n").split("\t", 1)
            rows.append((p, c))
            if len(rows) >= n:
                break
    return rows


def evaluate_retrieval(model, tokenizer, preprocess, device, dense_path, gt_path, n=1000):
    """长文本检索：用 dense caption 作 query 测 COCO 检索 R@1。"""
    img_tr = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    # 用 COCO 图片（karpathy val）作为图集
    gt = load_tsv_captions(gt_path, n)
    # dense caption 版：从 cc3m dense 按路径匹配（同图）
    dense_map = {p: c for p, c in load_tsv_captions(dense_path, 200000)}
    imgs = []
    for p, _ in gt:
        imgs.append(img_tr(Image.open(p).convert("RGB")))
    imgs = torch.stack(imgs)
    # query: 用 gt caption（短）作为对照
    with torch.no_grad():
        _dt = next(model.parameters()).dtype
        img_feat = model.encode_image(imgs.to(device=device, dtype=_dt), normalize=True)
        # gt caption 检索
        gt_toks = tokenizer([c for _, c in gt]).to(device)
        gt_txt = model.encode_text(gt_toks, normalize=True)
        # dense caption 检索（匹配到的）
        dense_caps = [dense_map.get(p, "") for p, _ in gt]
        dense_pairs = [(i, c) for i, c in enumerate(dense_caps) if c]
        dense_toks = tokenizer([c for _, c in dense_pairs]).to(device)
        dense_txt = model.encode_text(dense_toks, normalize=True)

    sim_gt = img_feat @ gt_txt.T
    r1_gt = (sim_gt.argmax(1) == torch.arange(n, device=device)).float().mean().item()
    if dense_pairs:
        dense_ids = [i for i, _ in dense_pairs]
        sim_d = img_feat[dense_ids] @ dense_txt.T
        r1_d = (sim_d.argmax(1) == torch.arange(len(dense_ids), device=device)).float().mean().item()
    else:
        r1_d = 0
    return r1_gt, r1_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--per-class", type=int, default=50, help="ImageNet 每类抽样数")
    ap.add_argument("--num-classes", type=int, default=None, help="ImageNet 类子集（CPU 评估加速）")
    ap.add_argument("--retrieval-n", type=int, default=1000, help="COCO 检索样本数")
    args = ap.parse_args()

    import torch
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.model import CLIPLeJEPA

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(16)  # CPU 多线程前向加速
    if device == "cuda":
        torch.cuda.set_device(0)  # 显式设可见 GPU（CUDA_VISIBLE_DEVICES 已映射）

    # 与训练一致：先建 CLIPLeJEPA（CPU fp32），加载后转 cuda half 省显存
    base, _, preprocess = create_model_and_transforms(
        "PE-Core-B-16-dinov3", "", precision="fp32", device="cpu",
        output_dict=True, force_context_length=256,
    )
    model = CLIPLeJEPA(clip_model=base, sigreg_target="cls", output_dict=True)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    # 剥 module. 前缀（DDP 保存）
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"加载 {args.ckpt}: missing={len(missing)} unexpected={len(unexpected)}")
    model = model.to(device)
    if device == "cuda":
        model = model.half()  # GPU 半精度省显存（受限 GPU）
    model.eval()
    tok = get_tokenizer("PE-Core-B-16-dinov3", context_length=256)

    # 1. IN-1k 短模板 vs 长模板
    acc_short = evaluate_imagenet(model, tok, SHORT_TEMPLATES, preprocess, device,
                                  args.per_class, args.num_classes)
    acc_long = evaluate_imagenet(model, tok, LONG_TEMPLATES, preprocess, device,
                                 args.per_class, args.num_classes)
    print(f"[{args.tag}] IN-1k top1({args.per_class}/类, {args.num_classes or 1000}类): "
          f"短模板={acc_short:.3f} 长模板={acc_long:.3f}")

    # 2. COCO 检索：gt caption vs dense caption 作 query
    r1_gt, r1_dense = evaluate_retrieval(model, tok, preprocess, device,
                                         COCO_DENSE, COCO_GT, args.retrieval_n)
    print(f"[{args.tag}] COCO R@1({args.retrieval_n}): gt-query={r1_gt:.4f} dense-query={r1_dense:.4f}")


if __name__ == "__main__":
    main()
