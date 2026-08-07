#!/usr/bin/env python3
"""Build dual-caption TSV for DualTextCLIP training (short gt + long gemma dense).

输入: clip_train_gt.tsv (filepath\tcaption) + clip_train_dense_256.tsv
输出: clip_train_dual.tsv (filepath\tcaption_short\tcaption_dense)

按 filepath 对齐（与 cc3m-tsv 图片一一对应）。dense 缺失的样本跳过
（两列都必须非空），确保每行都有短+长双文本。
"""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


def _load(fp: Path):
    """filepath -> caption (跳过 header)。"""
    d = {}
    for line in open(fp, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("filepath\t"):
            continue
        p, c = line.split("\t", 1)
        d[p] = c
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann-dir", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations")
    ap.add_argument("--short", default="clip_train_gt.tsv", help="短 caption TSV（默认 gt）")
    ap.add_argument("--long", default="clip_train_dense_256.tsv", help="长 caption TSV（默认 dense_256）")
    ap.add_argument("--out", default="clip_train_dual.tsv")
    args = ap.parse_args()

    ann = Path(args.ann_dir)
    short = _load(ann / args.short)
    long = _load(ann / args.long)
    log.info(f"short={len(short)} long={len(long)}")

    out = ann / args.out
    n = 0
    with out.open("w", encoding="utf-8") as fo:
        fo.write("filepath\tcaption_short\tcaption_dense\n")
        for p, cs in short.items():
            cd = long.get(p)
            if cd is None:
                continue
            fo.write(f"{p}\t{cs}\t{cd}\n")
            n += 1
    log.info(f"DONE {n:,} dual rows -> {out}")


if __name__ == "__main__":
    main()
