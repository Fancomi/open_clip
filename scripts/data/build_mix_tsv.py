#!/usr/bin/env python3
"""Build mixed-caption TSV: per-row random pick of short gt or long gemma dense.

Task 2 数据混合训练：单塔 + 双列混合。从 clip_train_dual.tsv（每行都有
caption_short + caption_dense 双版本）按比例随机选取一列，输出单列 caption
TSV，完全复用现有 `--csv-caption-key caption` 训练管线（零代码改动）。

用途：验证"gt + gemma dense 混合能否修复 dense 塔对齐、并保留下限"。
  --dense-ratio 0.5  → 一半样本用 dense 长文本（默认，50/50）
  --dense-ratio 0.7  → 70% dense（接近 dense_256 对 gt 的 69.3% 覆盖率）

用法:
  python scripts/data/build_mix_tsv.py --dense-ratio 0.5 --out clip_train_mix50.tsv
  python scripts/data/build_mix_tsv.py --dense-ratio 0.7 --out clip_train_mix70.tsv
"""
import argparse
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ANN = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual", default=str(ANN / "clip_train_dual.tsv"),
                    help="双列 TSV（filepath\tcaption_short\tcaption_dense）")
    ap.add_argument("--dense-ratio", type=float, default=0.5, help="选 dense 长文本的比例（0-1）")
    ap.add_argument("--out", default="clip_train_mix.tsv")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = ANN / args.out

    n_dense = n_short = 0
    with open(args.dual, encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
        fo.write("filepath\tcaption\n")
        for line in fi:
            line = line.rstrip("\n")
            if not line or line.startswith("filepath\t"):
                continue
            p, cs, cd = line.split("\t", 2)
            cap = cd if rng.random() < args.dense_ratio else cs
            if cap is cd:
                n_dense += 1
            else:
                n_short += 1
            fo.write(f"{p}\t{cap}\n")
    log.info(f"DONE {out}: dense={n_dense:,} ({n_dense/(n_dense+n_short):.1%}) short={n_short:,}")


if __name__ == "__main__":
    main()
