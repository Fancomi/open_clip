#!/usr/bin/env python3
"""DOCCI 长文本检索评测 —— Urban-1k 之外的第二个长文本 benchmark。

为什么必须接（不是"补充"而是"必要"，见 eval_protocol.md §4.6.2）：
  1. Urban-1k 只有 1000 个 query，实标出来的 2σ 地板是 **2.36**（i2t），
     而且 n=4 时 σ̂ 自己的 95% CI 宽达 ×[0.567, 3.73] → 本项目**大量长文本
     结论卡在 1×~2× 地板之间，只能写「未决」**。地板随 query 数缩小
     （参照 COCO i2t 5000 图 → 0.75 vs t2i 25000 条 → 0.13），
     DOCCI test 有 **5000** 个 query，地板应当明显更小。
  2. 那个地板只度量**训练随机性**，不含 **query 抽样噪声**
     （R@1 在 n=1000、p≈0.19 下的二项 2σ 约 2.5 点）→ 即便某个差过了线，
     证明的也只是"在这 1000 张图上更好"。换一个数据集才能谈泛化。
  3. **Urban-1k 的长描述是 GPT-4V 生成的，DOCCI 是人写的**（平均 123 词、
     中位 114 词，标注流程含多轮人工校验）。两者的失效模式不相关 ——
     如果只用 GPT 生成的评测集，很可能测的是"对生成式文风的亲和度"，
     而我们的训练 caption 也正是模型（gemma4）生成的。**人写文本是关键的
     独立证人。**

★ 干净性 ★
  DOCCI 图像与 CC3M / COCO / IN-1k / Urban-1k 均无关（Google 自采），
  训练从未见过。1:1 配对（每图恰好一条长描述），与 Urban-1k 同一套指标定义，
  可以并排放进同一张表。

协议与 Urban-1k **逐位相同**：模型加载、neg-mode 排序口径、R@1/5/10 定义
全部直接 import `eval_urban1k`，避免两条长文本口径之间出现管线漂移。
neg-mode 必须与训练配方一致（铁律 1，查 logs/<run>/params.txt）。

用法:
  python scripts/eval/eval_docci.py --ckpt logs/.../epoch_10.pt \
      --tag gt_base --neg-mode projective
  # 冒烟：--limit 200
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from eval_urban1k import apply_neg_mode, load_model  # 同一套加载，禁止另写一份

DOCCI = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/docci")


def load_docci(split, limit=0):
    """返回 (img_paths, captions)，按 example_id 排序后严格对齐。"""
    desc = DOCCI / "docci_descriptions.jsonlines"
    if not desc.exists():
        raise SystemExit(f"缺 {desc} —— 先下 docci_descriptions.jsonlines")
    # ★ 只认 images/ ★ DOCCI-AAR 是**另一个数据集**（train 4932 vs DOCCI 9647），
    # 不是同一批图的缩放版，descriptions 也不覆盖它。按 index 硬配 = 随机配对，
    # 会得到「R@1 恰好等于随机基线」的假结果（08-27 实际踩过，见 memory）。
    img_root = DOCCI / "images"
    if not img_root.exists():
        raise SystemExit(f"缺图像目录 {img_root} —— 需要 docci_images.tar.gz（7.59 GB），"
                         f"**不是 docci_images_aar.tar.gz**")

    rows = []
    with open(desc, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["split"] != split:
                continue
            rows.append((d["example_id"], d["image_file"], d["description"].strip()))
    if not rows:
        raise SystemExit(f"split={split} 一条都没有")
    rows.sort(key=lambda r: r[0])

    paths, caps, missing = [], [], 0
    for _, fn, cap in rows:
        p = img_root / fn
        if not p.exists():
            missing += 1
            continue
        paths.append(str(p))
        caps.append(cap)
    if missing:
        raise SystemExit(f"{missing}/{len(rows)} 张图缺失 —— 图像包与 descriptions 对不上，"
                         f"很可能下错了包（AAR ≠ DOCCI）。**不允许跳过后继续跑**："
                         f"缺图会改变 query 数，地板跟着变，且掩盖配错的可能。")
    if limit > 0:
        paths, caps = paths[:limit], caps[:limit]
    return paths, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--split", default="test",
                    choices=["test", "train", "qual_dev", "qual_test"],
                    help="默认 test（5000 图）—— query 数越大地板越小，别随便换")
    ap.add_argument("--neg-mode", default="projective",
                    choices=["standard", "projective", "antipodal", "orthogonal"],
                    help="必须与训练配方一致（查 logs/<run>/params.txt 的 neg_mode）")
    ap.add_argument("--neg-alpha", type=float, default=1.0)
    ap.add_argument("--tok-context-length", type=int, default=None,
                    help="只改分词窗口、不动模型（默认跟 ckpt 探测值）。把 320 训练的 ckpt 按 256 分词，用来把\"模型变好\"与\"评测文本少截断\"分开")
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 对（冒烟用，★数字不可与全量混比★）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.tag}] DOCCI 长文本检索 | device={device} "
          f"split={args.split} neg_mode={args.neg_mode}", flush=True)

    paths, caps = load_docci(args.split, args.limit)
    n = len(paths)
    scope = f"★limit={args.limit}（非全量，不可与全量混比）★" if args.limit else "★全量★"
    print(f"  数据: {n} 图 × {n} 条人写长描述（1:1 配对，训练集外）{scope}", flush=True)

    model, tok, val_tr = load_model(args.ckpt, device, tok_ctx=args.tok_context_length)
    dt = torch.float16 if device == "cuda" else torch.float32

    toks = tok(caps)
    ctx = toks.shape[1]
    # 截断率：DOCCI 最长的描述有 518 词，256 窗口未必装得下。装不下就不是"完整长文本"，
    # 必须报出来，否则会把"窗口不够"读成"长文本能力不足"。
    nonzero = (toks != 0).sum(1)
    n_trunc = int((nonzero >= ctx).sum())
    print(f"  BPE token: 均值 {nonzero.float().mean():.1f} / 中位 {int(nonzero.median())} / "
          f"最大 {int(nonzero.max())}，窗口 {ctx} → 触顶 {n_trunc} 条 ({100*n_trunc/n:.1f}%)",
          flush=True)

    with torch.no_grad():
        img_feats = []
        for i in range(0, n, args.batch):
            ts = torch.stack([val_tr(Image.open(p).convert("RGB")) for p in paths[i:i + args.batch]])
            img_feats.append(model.encode_image(ts.to(device=device, dtype=dt), normalize=True).float().cpu())
        img_feat = torch.cat(img_feats)

        txt_feats = []
        for j in range(0, n, args.batch):
            txt_feats.append(model.encode_text(toks[j:j + args.batch].to(device), normalize=True).float().cpu())
        txt_feat = torch.cat(txt_feats)

    sim = apply_neg_mode(img_feat @ txt_feat.T, args.neg_mode, args.neg_alpha)  # [n, n]
    gt = torch.arange(n)

    top10 = sim.topk(10, dim=1).indices
    r1 = (top10[:, 0] == gt).float().mean().item()
    r5 = (top10[:, :5] == gt.unsqueeze(1)).any(1).float().mean().item()
    r10 = (top10 == gt.unsqueeze(1)).any(1).float().mean().item()
    print(f"  i2t R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f}  {scope}", flush=True)

    top10_t = sim.T.topk(10, dim=1).indices
    tr1 = (top10_t[:, 0] == gt).float().mean().item()
    tr5 = (top10_t[:, :5] == gt.unsqueeze(1)).any(1).float().mean().item()
    tr10 = (top10_t == gt.unsqueeze(1)).any(1).float().mean().item()
    print(f"  t2i R@1={tr1:.4f} R@5={tr5:.4f} R@10={tr10:.4f}  {scope}", flush=True)

    # 提醒：难度与 Urban-1k 不可直接比 —— query 池大 5 倍，随机基线低 5 倍。
    print(f"  ⚠️ 口径提示：{n} 个 query 的随机基线 R@1 = {100.0/n:.2f}%"
          f"（Urban-1k 是 0.10%），**DOCCI 与 Urban 的绝对值不可直接比大小**，"
          f"只能各自与自己的基线组比。", flush=True)


if __name__ == "__main__":
    main()
