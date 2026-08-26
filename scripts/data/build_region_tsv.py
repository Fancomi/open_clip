#!/usr/bin/env python3
"""把 clean_rec jsonl 转成训练用 TSV（区域-短语监督，FG-CLIP 式）。

输入:
  clean_rec/clean_shard{0-7}.jsonl  —— {path, img_wh, grounding:{phrase:[[x1,y1,x2,y2],...]}}
  clip_train_gt.tsv                 —— filepath \t caption（gt 短文，主分支用）

输出:
  clip_train_region.tsv  —— filepath \t caption \t regions
    regions = JSON [[phrase, x1, y1, x2, y2], ...]，坐标已按 img_wh **归一化到 [0,1]**

★ 坐标约定 ★
  clean_rec 里是原图像素，这里除以 img_wh 归一化。训练时用 resize-only transform
  （不做 RandomResizedCrop），所以归一化坐标可直接乘 feature map 边长喂 roi_align。
  若改用随机裁剪，必须让 dataloader 返回裁剪参数并重算框 —— 本脚本不支持。

过滤（clean_rec 已清洗过，这里只做训练相关的裁剪）:
  - n_phrase == 0 的图跳过（3.07%）
  - 每图最多保留 MAX_REGION 个 (phrase, box) 对（按面积降序，大框语义更可靠）
  - 退化框跳过：宽或高 < 2 像素

用法:
  python scripts/data/build_region_tsv.py
  python scripts/data/build_region_tsv.py --max-region 12 --out clip_train_region.tsv
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path("/root/paddlejob/workspace/env_run/penghaotian/datas")
CLEAN_REC = ROOT / "cc3m_region/clean_rec"
ANN = ROOT / "cc3m-tsv/annotations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", default=str(ANN / "clip_train_gt.tsv"))
    ap.add_argument("--out", default=str(ANN / "clip_train_region.tsv"))
    ap.add_argument("--max-region", type=int, default=12, help="每图最多保留的区域数（p90=11）")
    args = ap.parse_args()

    log.info("加载 gt caption ...")
    gt = {}
    with open(args.gt, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            p, c = line.rstrip("\n").split("\t", 1)
            gt[p] = c
    log.info(f"gt caption: {len(gt):,}")

    n_in = n_out = n_empty = n_nogt = 0
    n_region = 0
    with open(args.out, "w", encoding="utf-8") as fo:
        fo.write("filepath\tcaption\tregions\n")
        for shard in sorted(CLEAN_REC.glob("clean_shard*.jsonl")):
            for line in open(shard, encoding="utf-8"):
                n_in += 1
                r = json.loads(line)
                if r["n_phrase"] == 0:
                    n_empty += 1
                    continue
                path = r["path"]
                cap = gt.get(path)
                if cap is None:
                    n_nogt += 1
                    continue
                W, H = r["img_wh"]
                if W <= 0 or H <= 0:
                    continue
                # 展开 (phrase, box) 对，记面积用于排序
                items = []
                for ph, boxes in r["grounding"].items():
                    ph = ph.strip()
                    if not ph:
                        continue
                    for b in boxes:
                        x1, y1, x2, y2 = b
                        if x2 - x1 < 2 or y2 - y1 < 2:      # 退化框
                            continue
                        area = (x2 - x1) * (y2 - y1) / (W * H)
                        items.append((area, ph,
                                      round(max(0.0, x1 / W), 4), round(max(0.0, y1 / H), 4),
                                      round(min(1.0, x2 / W), 4), round(min(1.0, y2 / H), 4)))
                if not items:
                    n_empty += 1
                    continue
                items.sort(key=lambda t: -t[0])              # 大框优先
                items = items[:args.max_region]
                regions = [[it[1], it[2], it[3], it[4], it[5]] for it in items]
                n_region += len(regions)
                fo.write(f"{path}\t{cap}\t{json.dumps(regions, ensure_ascii=False)}\n")
                n_out += 1
                if n_out % 500000 == 0:
                    log.info(f"  {n_out:,} 行 ...")

    log.info(f"DONE {args.out}")
    log.info(f"  读入 {n_in:,} | 输出 {n_out:,} | 空 grounding {n_empty:,} | 无 gt {n_nogt:,}")
    log.info(f"  区域总数 {n_region:,}（{n_region/max(n_out,1):.2f}/图）")


if __name__ == "__main__":
    main()
