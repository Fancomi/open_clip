#!/usr/bin/env python3
"""Urban-1k 长文本检索评测（Long-CLIP 官方配套 benchmark，业界可对标）。

为什么需要它：COCO karpathy 5cap 与 IN-1k 的 query 都是**短文本**，
在设计上就测不出长文本（dense caption）的价值。Urban-1k 是 Long-CLIP
(ECCV 2024) 配套发布的长文本检索集：1000 图 × 1000 条长描述
（平均 132 BPE token，256 窗口零截断）。

★ 干净性 ★
  Urban-1k 与 CC3M / COCO 均无关，训练从未见过 —— 这是当前唯一干净的
  长文本评测集。（对比：直接读 clip_train_dense_256.tsv 做评测会造成
  100% 训练集重叠，见 analysis/research/eval_protocol.md 第 3 节）

指标：i2t / t2i R@1/5/10，1:1 配对（每图恰好一条长描述）。
neg-mode 必须与训练配方一致（铁律 1）。

用法:
  python scripts/eval/eval_urban1k.py --ckpt logs/.../epoch_9.pt \
      --tag pcm_proj --neg-mode projective
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

URBAN1K = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/urban1k/Urban1k")


def apply_neg_mode(sim, neg_mode, neg_alpha=1.0):
    """排序分数口径，与 eval_standard.py / train.py:get_clip_metrics 一致。"""
    if neg_alpha < 1.0:
        return neg_alpha * sim + (1.0 - neg_alpha) * sim.abs()
    if neg_mode == 'projective':
        return sim.abs()
    if neg_mode == 'antipodal':
        return -sim
    return sim


def load_model(ckpt_path, device, tok_ctx=None):
    from open_clip import create_model_and_transforms, get_tokenizer
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
    # tok_ctx 只改**分词窗口**，不动模型 —— `CustomTextCLIP.encode_text` 是按 seq_len
    # 切位置编码的（`transformer.py:1289`），所以 320 的模型喂 256 长的 token 是**精确等价**于
    # 一个 256 模型看同一段被截到 256 的文本，无插值、无损。用途：把"模型变好了"与
    # "评测文本少被截了"分开（DOCCI 在 256 窗口下有 3.2% 的 query 触顶）。
    tok_ctx = ctx if tok_ctx is None else int(tok_ctx)
    print(f"  加载 {Path(ckpt_path).name}: missing={len(missing)} unexpected={len(unexpected)} "
          f"模型 context_length={ctx}（ckpt 探测）分词窗口={tok_ctx}", flush=True)
    model = model.to(device).eval()
    if device == "cuda":
        model = model.half()
    tok = get_tokenizer("PE-Core-B-16-dinov3", context_length=tok_ctx)
    return model, tok, val_tr


def load_urban1k():
    """返回 (img_paths, captions)，按图 id 严格对齐。"""
    img_dir, cap_dir = URBAN1K / "image", URBAN1K / "caption"
    if not img_dir.exists():
        raise SystemExit(f"缺 {img_dir} —— 先下载解压 Urban1k.zip")
    ids = sorted(p.stem for p in img_dir.glob("*.jpg"))
    paths, caps = [], []
    for i in ids:
        cap_f = cap_dir / f"{i}.txt"
        if not cap_f.exists():
            continue
        paths.append(str(img_dir / f"{i}.jpg"))
        caps.append(cap_f.read_text(encoding="utf-8").strip())
    return paths, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--neg-mode", default="projective",
                    choices=["standard", "projective", "antipodal", "orthogonal"],
                    help="必须与训练配方一致（查 logs/<run>/params.txt 的 neg_mode）")
    ap.add_argument("--neg-alpha", type=float, default=1.0)
    ap.add_argument("--tok-context-length", type=int, default=None,
                    help="只改分词窗口、不动模型（默认跟 ckpt 探测值）。把 320 训练的 ckpt 按 256 分词，用来把\"模型变好\"与\"评测文本少截断\"分开")
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[{args.tag}] Urban-1k 长文本检索 | device={device} neg_mode={args.neg_mode}", flush=True)

    paths, caps = load_urban1k()
    n = len(paths)
    print(f"  数据: {n} 图 × {n} 长描述（1:1 配对，训练集外）", flush=True)

    model, tok, val_tr = load_model(args.ckpt, device, tok_ctx=args.tok_context_length)
    dt = torch.float16 if device == "cuda" else torch.float32

    with torch.no_grad():
        img_feats = []
        for i in range(0, n, args.batch):
            ts = torch.stack([val_tr(Image.open(p).convert("RGB")) for p in paths[i:i + args.batch]])
            img_feats.append(model.encode_image(ts.to(device=device, dtype=dt), normalize=True).float().cpu())
        img_feat = torch.cat(img_feats)

        txt_feats = []
        toks = tok(caps)
        for j in range(0, n, args.batch):
            txt_feats.append(model.encode_text(toks[j:j + args.batch].to(device), normalize=True).float().cpu())
        txt_feat = torch.cat(txt_feats)

    sim = apply_neg_mode(img_feat @ txt_feat.T, args.neg_mode, args.neg_alpha)  # [n, n]
    gt = torch.arange(n)

    # i2t：每图找回自己的长描述
    top10 = sim.topk(10, dim=1).indices
    r1 = (top10[:, 0] == gt).float().mean().item()
    r5 = (top10[:, :5] == gt.unsqueeze(1)).any(1).float().mean().item()
    r10 = (top10 == gt.unsqueeze(1)).any(1).float().mean().item()
    print(f"  i2t R@1={r1:.4f} R@5={r5:.4f} R@10={r10:.4f}", flush=True)

    # t2i：每条长描述找回自己的图
    top10_t = sim.T.topk(10, dim=1).indices
    tr1 = (top10_t[:, 0] == gt).float().mean().item()
    tr5 = (top10_t[:, :5] == gt.unsqueeze(1)).any(1).float().mean().item()
    tr10 = (top10_t == gt.unsqueeze(1)).any(1).float().mean().item()
    print(f"  t2i R@1={tr1:.4f} R@5={tr5:.4f} R@10={tr10:.4f}", flush=True)


if __name__ == "__main__":
    main()
