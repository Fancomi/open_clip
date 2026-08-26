#!/usr/bin/env python3
"""合并 PCM + 区域监督的训练 TSV（G 组：把本周两条最优线叠加）。

输入:
  clip_train_dual.tsv   —— filepath / caption_short(gt) / caption_dense(gemma4 长文)
  clip_train_region.tsv —— filepath / caption(gt) / regions(JSON, 归一化坐标)

输出:
  clip_train_pcmregion.tsv —— filepath / caption_dense / caption_short / regions
    caption_dense : 主分支（长文本，PCM 的长分支）
    caption_short : PCM 短分支（PCA_k(img) × gt 短文）
    regions       : 区域分支（RoIAlign(patch_map) × 短语）

行数 = 两者按 filepath 取交集。dual 覆盖 200 万（dense_256 的覆盖率），
region 覆盖 287 万，交集约 200 万 —— 比单独任一组少，报指标时要注意
样本量差异（PCM 组也是 200 万，所以与 pcm_w0.2 可比；与 C3 的 287 万不完全可比）。

用法:
  python scripts/data/build_pcm_region_tsv.py
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ANN = Path("/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dual", default=str(ANN / "clip_train_dual.tsv"))
    ap.add_argument("--region", default=str(ANN / "clip_train_region.tsv"))
    ap.add_argument("--out", default=str(ANN / "clip_train_pcmregion.tsv"))
    args = ap.parse_args()

    log.info("① 载入 region（filepath -> regions JSON 字符串）...")
    reg = {}
    with open(args.region, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            reg[parts[0]] = parts[2]
    log.info(f"   region: {len(reg):,}")

    log.info("② 逐行读 dual，取交集输出 ...")
    n_out = n_miss = 0
    with open(args.dual, encoding="utf-8") as fi, open(args.out, "w", encoding="utf-8") as fo:
        fo.write("filepath\tcaption_dense\tcaption_short\tregions\n")
        for i, line in enumerate(fi):
            if i == 0:
                continue
            parts = line.rstrip("\n").split("\t", 2)
            if len(parts) < 3:
                continue
            path, cs, cd = parts
            r = reg.get(path)
            if r is None:
                n_miss += 1
                continue
            fo.write(f"{path}\t{cd}\t{cs}\t{r}\n")
            n_out += 1
            if n_out % 500000 == 0:
                log.info(f"   {n_out:,} 行 ...")

    log.info(f"DONE {args.out}")
    log.info(f"  输出 {n_out:,} 行 | dual 里无区域标注而跳过 {n_miss:,}")


if __name__ == "__main__":
    main()
