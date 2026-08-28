#!/usr/bin/env python3
"""统计 caption 的 CLIP-BPE token 长度分布。

为什么需要它：`build_gemma_tsv.py` 生成 `clip_train_dense_256.tsv` 时，
对超过 256 token 的行**整行丢弃**（`:110-116`，不是截断，尽管 docstring 写的是"截断到"），
2894191 → 2006804，丢掉 30.66%。要把长文本塔拉到 320 token 并改成
"截断输入而不是丢图"，先得知道：320 这个窗口到底覆盖多少、被截掉的是尾巴的几成。

口径与训练/建表完全一致：`get_tokenizer('PE-Core-B-16')` 的 CLIP BPE，
计数 = `len(tok.encode(text)) + 2`（SOT/EOT），与 `build_gemma_tsv.py:111` 同一行公式。

用法：
    python scripts/data/stat_caption_tokens.py \
        --tsv .../annotations/clip_train_dense.tsv --col 1 --nproc 12
    # 抽样先看形状（几秒）：加 --limit 50000
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

_tok = None


def _get_tokenizer():
    global _tok
    if _tok is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
        from open_clip import get_tokenizer
        _tok = get_tokenizer('PE-Core-B-16')
    return _tok


def _count_chunk(args):
    lines, col = args
    tok = _get_tokenizer()
    out = np.empty(len(lines), dtype=np.int32)
    n = 0
    for ln in lines:
        parts = ln.rstrip("\n").split("\t")
        if len(parts) <= col:
            continue
        out[n] = len(tok.encode(parts[col])) + 2
        n += 1
    return out[:n]


def _iter_chunks(path, col, chunk, limit):
    buf, total = [], 0
    with open(path) as f:
        for ln in f:
            buf.append(ln)
            total += 1
            if len(buf) >= chunk:
                yield (buf, col)
                buf = []
            if limit and total >= limit:
                break
    if buf:
        yield (buf, col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", required=True)
    ap.add_argument("--col", type=int, default=1, help="caption 所在列（0-based）")
    ap.add_argument("--nproc", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--limit", type=int, default=0, help="只看前 N 行（0=全量）")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[77, 128, 192, 256, 320, 384, 448, 512])
    ap.add_argument("--save-npy", default="", help="把逐行 token 数存下来，便于后续复用")
    args = ap.parse_args()

    with Pool(args.nproc) as pool:
        parts = pool.imap(_count_chunk, _iter_chunks(args.tsv, args.col, args.chunk, args.limit))
        lens = np.concatenate(list(parts))

    n = len(lens)
    print(f"文件 {args.tsv}  列 {args.col}  行数 {n}")
    print(f"mean {lens.mean():.2f}  std {lens.std():.2f}  min {lens.min()}  max {lens.max()}")
    ps = [1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9]
    qs = np.percentile(lens, ps)
    print("分位数： " + "  ".join(f"p{p:g}={q:.0f}" for p, q in zip(ps, qs)))
    print()
    print("两种策略的对比（truncate = 现在要改成的做法；drop = 现状 dense_256 的做法）")
    print(f"{'窗口':>6} {'≤窗口行占比':>12} {'trunc保留token':>15} {'drop保留token':>14} "
          f"{'drop保留行':>11} {'超窗行均长':>11}")
    total_tok = lens.sum()
    for w in args.windows:
        over = lens > w
        keep_tr = np.minimum(lens, w).sum()
        keep_dr = lens[~over].sum()
        avg_over = lens[over].mean() if over.any() else 0.0
        print(f"{w:>6} {100 * (~over).mean():>11.2f}% {100 * keep_tr / total_tok:>14.2f}% "
              f"{100 * keep_dr / total_tok:>13.2f}% {100 * (~over).mean():>10.2f}% {avg_over:>11.1f}")
    if args.save_npy:
        np.save(args.save_npy, lens)
        print(f"\n逐行 token 数已存 {args.save_npy}")


if __name__ == "__main__":
    main()
