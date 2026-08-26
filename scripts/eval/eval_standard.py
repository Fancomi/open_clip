#!/usr/bin/env python3
"""标准评估：业界统一协议 + 训练配方对齐的 neg-mode 口径。

★ 关键约束（踩过坑，务必遵守）★
  本仓库 CHAMPION/E 配方带 `--neg-mode projective`，其训练目标是 |cos| → 1，
  正样本会收敛到 cos → −1 分支（实测 99% 为负）。因此：
    - projective 训练的模型：必须用 |cos| 排序（--neg-mode projective）
    - standard 训练的模型：必须用 cos 排序（--neg-mode standard）
  口径与训练配方不匹配时，指标会系统性归零（不是随机，是精确 0）——
  这不是模型坏，是评估错。详见 analysis/research/eval_protocol.md。

评估内容：
1. COCO karpathy 5cap 检索（全量 5000 图）：i2t/t2i R@1/5/10
2. IN-1k zero-shot（80 官方模板，可选长描述模板对照）：top1/top5

用法:
  # projective 配方模型（E 配方 / CHAMPION，默认）
  python scripts/eval/eval_standard.py --ckpt logs/.../epoch_9.pt --tag gt_base --retrieval
  # standard 配方模型
  python scripts/eval/eval_standard.py --ckpt ... --tag foo --neg-mode standard --retrieval
  # 长描述模板对照（测长文本塔对长模板的响应）
  python scripts/eval/eval_standard.py --ckpt ... --tag gemma_dense --long-template
"""
import argparse
import sys
import time
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

COCO_GT = "/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations/karpathy_5cap.tsv"
IMAGENET_VAL = "/root/paddlejob/workspace/env_run/penghaotian/datas/imagenet-val"

# CC3M 系训练 TSV（gt / dense / dual / mix / concat 共享同一批图，互为训练集）
CC3M_ANN = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations")

# 长描述模板（对照 OPENAI_IMAGENET_TEMPLATES 的短句），测长文本塔的模板响应
LONG_TEMPLATES = (
    lambda c: f'an image showing a {c}, with rich visual detail and clear composition.',
    lambda c: f'a detailed photograph of a {c}, capturing its texture and surroundings.',
)


def assert_no_train_overlap(eval_paths, train_tsvs=None, sample=2000):
    """零重叠检查：评测图与训练集有交集就直接拒跑（eval_protocol.md 铁律 2）。

    事故背景：曾直接读 clip_train_dense_256.tsv 前 1000 行当评测集，
    与训练集重叠 1000/1000，产出的指标全部作废。
    """
    if train_tsvs is None:
        train_tsvs = sorted(CC3M_ANN.glob("clip_train*.tsv"))
    probe = set(str(p) for p in eval_paths[:sample])
    for tsv in train_tsvs:
        if not Path(tsv).exists():
            continue
        hit = 0
        with open(tsv) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                if line.split("\t", 1)[0] in probe:
                    hit += 1
                    if hit >= 1:
                        break
        if hit:
            raise SystemExit(
                f"\n!!!! 拒绝评测：评测样本与训练集重叠 !!!!\n"
                f"  训练 TSV : {tsv}\n"
                f"  评测样本 : 前 {len(probe)} 条中至少 {hit} 条命中\n"
                f"  CC3M 系数据（gt/dense/dual/mix/concat）互为训练集，不能互当评测集。\n"
                f"  干净评测目前只有 COCO karpathy 5cap 与 IN-1k val。\n"
                f"  详见 analysis/research/eval_protocol.md 第 3 节。\n")
    print(f"  [零重叠检查] 通过：{len(probe)} 条评测样本与 {len(train_tsvs)} 个训练 TSV 无交集", flush=True)


def apply_neg_mode(sim, neg_mode, neg_alpha=1.0):
    """把原始 cos 相似度转成与训练配方一致的排序分数。

    与 src/open_clip_train/train.py:get_clip_metrics 及 zero_shot.py:run 完全一致。
    """
    if neg_alpha < 1.0:
        return neg_alpha * sim + (1.0 - neg_alpha) * sim.abs()
    if neg_mode == 'projective':
        return sim.abs()
    if neg_mode == 'antipodal':
        return -sim
    return sim


def load_model(ckpt_path, device):
    from open_clip import create_model_and_transforms, get_tokenizer
    from open_clip.model import CLIPLeJEPA

    base, train_tr, val_tr = create_model_and_transforms(
        "PE-Core-B-16-dinov3", "", precision="fp32", device="cpu",
        output_dict=True, force_context_length=256)
    model = CLIPLeJEPA(clip_model=base, sigreg_target="cls", output_dict=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    tok = get_tokenizer("PE-Core-B-16-dinov3", context_length=256)
    return model, tok, val_tr


def eval_coco_retrieval(model, tok, val_tr, device, neg_mode="projective", neg_alpha=1.0):
    """全量 5000 图 × 5 cap，i2t/t2i R@1/5/10。neg_mode 必须与训练配方一致。"""
    imgs, caps_by_img = [], []
    cur = None
    with open(COCO_GT) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            p, c = line.rstrip("\n").split("\t", 1)
            if cur is None or p != cur[0]:
                cur = [p, []]
                imgs.append(cur[0])
                caps_by_img.append(cur[1])
            cur[1].append(c)
    n_img = len(imgs)
    assert_no_train_overlap(imgs)
    n_caps = len(caps_by_img[0])
    all_caps = [c for cs in caps_by_img for c in cs]
    # 每图的正确 caption 列范围
    pos_cols = [slice(i * n_caps, (i + 1) * n_caps) for i in range(n_img)]

    dt = torch.float16 if device == "cuda" else torch.float32
    with torch.no_grad():
        # 图像分块编码（逐块，和训练一致，不驻留全量）
        img_feats = []
        for i in range(0, n_img, 25):
            ts = torch.stack([val_tr(Image.open(p).convert("RGB")) for p in imgs[i:i + 25]])
            img_feats.append(model.encode_image(ts.to(device=device, dtype=dt), normalize=True).float().cpu())
        img_feat = torch.cat(img_feats)  # [n_img, d]
        # 文本分块编码（每块 100 条，避免 attention 峰值爆炸）
        txt_feats = []
        toks = tok(all_caps)
        for j in range(0, len(all_caps), 100):
            txt_feats.append(model.encode_text(toks[j:j + 100].to(device), normalize=True).float().cpu())
        gt_txt = torch.cat(txt_feats)  # [n_img*n_caps, d]

        # 分块计算 i2t sim（避免整矩阵驻留显存）
        top1_all, top5_all, top10_all = [], [], []
        for i in range(0, n_img, 200):
            block = img_feat[i:i + 200]  # [200, d]
            sim_b = apply_neg_mode(block @ gt_txt.T, neg_mode, neg_alpha)  # [200, n_txt] on CPU
            top1_all.append(sim_b.argmax(1))
            top5_all.append(sim_b.topk(5, dim=1).indices)
            top10_all.append(sim_b.topk(10, dim=1).indices)
        top1 = torch.cat(top1_all)
        top5 = torch.cat(top5_all)
        top10 = torch.cat(top10_all)
        r1 = sum(1 for i in range(n_img) if top1[i] in range(i * n_caps, (i + 1) * n_caps)) / n_img
        r5 = sum(1 for i in range(n_img) if any(t in range(i * n_caps, (i + 1) * n_caps) for t in top5[i])) / n_img
        r10 = sum(1 for i in range(n_img) if any(t in range(i * n_caps, (i + 1) * n_caps) for t in top10[i])) / n_img
        print(f"  i2t R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f}", flush=True)

        # t2i 分块
        cap_img = [i for i, cs in enumerate(caps_by_img) for _ in cs]  # [n_txt]
        top1_t, top5_t, top10_t = [], [], []
        for j in range(0, len(cap_img), 1000):
            block = gt_txt[j:j + 1000]  # [1000, d]
            sim_b = apply_neg_mode(block @ img_feat.T, neg_mode, neg_alpha)  # [1000, n_img]
            top1_t.append(sim_b.argmax(1))
            top5_t.append(sim_b.topk(5, dim=1).indices)
            top10_t.append(sim_b.topk(10, dim=1).indices)
        top1_t = torch.cat(top1_t)
        top5_t = torch.cat(top5_t)
        top10_t = torch.cat(top10_t)
        tr1 = sum(1 for j in range(len(cap_img)) if top1_t[j] == cap_img[j]) / len(cap_img)
        tr5 = sum(1 for j in range(len(cap_img)) if cap_img[j] in top5_t[j]) / len(cap_img)
        tr10 = sum(1 for j in range(len(cap_img)) if cap_img[j] in top10_t[j]) / len(cap_img)
        print(f"  t2i R@1={tr1:.4f} R@5={tr5:.4f} R@10={tr10:.4f}", flush=True)
    return {"i2t_R1": r1, "i2t_R5": r5, "i2t_R10": r10, "t2i_R1": tr1, "t2i_R5": tr5, "t2i_R10": tr10}


def eval_imagenet(model, tok, val_tr, device, num_classes=1000, per_class=50,
                  neg_mode="projective", neg_alpha=1.0, long_template=False,
                  num_workers=12):
    """IN-1k zero-shot。默认 80 官方模板；long_template=True 时用长描述模板对照。"""
    from open_clip.zero_shot_classifier import build_zero_shot_classifier
    from open_clip.zero_shot_metadata import OPENAI_IMAGENET_TEMPLATES, IMAGENET_CLASSNAMES

    templates = LONG_TEMPLATES if long_template else OPENAI_IMAGENET_TEMPLATES
    classnames = IMAGENET_CLASSNAMES[:num_classes]
    with torch.no_grad():
        classifier = build_zero_shot_classifier(
            model, tok, classnames, templates,
            num_classes_per_batch=10, device=device)
        classifier = classifier.float().cpu()

        val_root = Path(IMAGENET_VAL)
        dirs = sorted([d for d in val_root.iterdir() if d.is_dir()])[:num_classes]
        correct1 = correct5 = total = 0
        dt = torch.float16 if device == "cuda" else torch.float32
        # 批量前向（全量 5 万张时逐张会慢一个数量级）
        paths, labels = [], []
        for i, d in enumerate(dirs):
            for img in sorted(d.iterdir())[:per_class]:
                paths.append(img)
                labels.append(i)
        # 多进程解码：单线程 ~90 img/s，全量 5 万张要 9 分钟且 GPU 空转；
        # DataLoader(shuffle=False) 顺序与 transform 不变，逐样本结果等价。
        class _ValSet(torch.utils.data.Dataset):
            def __len__(self): return len(paths)
            def __getitem__(self, i):
                return val_tr(Image.open(paths[i]).convert("RGB")), labels[i]

        BS = 100
        loader = torch.utils.data.DataLoader(
            _ValSet(), batch_size=BS, shuffle=False,
            num_workers=num_workers, pin_memory=True)
        for ts, lab in loader:
            f = model.encode_image(ts.to(device=device, dtype=dt), normalize=True).float().cpu()
            raw = apply_neg_mode(100. * (f @ classifier), neg_mode, neg_alpha)
            top5i = raw.topk(5, dim=1).indices
            correct1 += (top5i[:, 0] == lab).sum().item()
            correct5 += (top5i == lab.unsqueeze(1)).any(1).sum().item()
            total += len(lab)
            if total % 10000 == 0:
                print(f"    ... {total}/{len(paths)}", flush=True)
        top1 = correct1 / total
        top5 = correct5 / total
        tname = "长模板" if long_template else "80官方模板"
        scope = "★全量★" if (num_classes >= 1000 and per_class >= 50) else "⚠️子集(不可与全量混比)"
        print(f"  IN-1k top1={top1:.4f} top5={top5:.4f} "
              f"({total} 图, {num_classes} 类, {tname}, {scope})", flush=True)
    return {"in1k_top1": top1, "in1k_top5": top5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--neg-mode", default="projective",
                    choices=["standard", "projective", "antipodal", "orthogonal"],
                    help="必须与训练配方的 --neg-mode 一致（E 配方=projective）")
    ap.add_argument("--neg-alpha", type=float, default=1.0, help="与训练一致；<1.0 时覆盖 neg-mode")
    ap.add_argument("--in1k-classes", type=int, default=1000,
                    help="IN-1k 类数。★默认全量 1000（open_clip 标准协议）★；"
                         "改小只用于快速冒烟，子集数字不可与全量混比")
    ap.add_argument("--in1k-per-class", type=int, default=50,
                    help="IN-1k 每类图数。★默认全量 50（= val 全部）★")
    ap.add_argument("--retrieval", action="store_true", help="是否跑 COCO 检索")
    ap.add_argument("--long-template", action="store_true", help="IN-1k 额外跑长描述模板对照")
    ap.add_argument("--num-workers", type=int, default=12, help="IN-1k 图像解码进程数")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.tag}] device={device} neg_mode={args.neg_mode} (须与训练配方一致)", flush=True)
    model, tok, val_tr = load_model(args.ckpt, device)
    print(f"[{args.tag}] 模型加载完成", flush=True)

    if args.retrieval:
        print(f"[{args.tag}] COCO 检索 (neg_mode={args.neg_mode}):", flush=True)
        eval_coco_retrieval(model, tok, val_tr, device, args.neg_mode, args.neg_alpha)

    print(f"[{args.tag}] IN-1k zero-shot (neg_mode={args.neg_mode}):", flush=True)
    eval_imagenet(model, tok, val_tr, device, args.in1k_classes, args.in1k_per_class,
                  args.neg_mode, args.neg_alpha, long_template=False,
                  num_workers=args.num_workers)
    if args.long_template:
        print(f"[{args.tag}] IN-1k zero-shot 长模板对照:", flush=True)
        eval_imagenet(model, tok, val_tr, device, args.in1k_classes, args.in1k_per_class,
                      args.neg_mode, args.neg_alpha, long_template=True,
                      num_workers=args.num_workers)


if __name__ == "__main__":
    main()
