#!/usr/bin/env python3
"""用 verdict 边车给 region TSV 挂上审核信息（E 组数据）。

★ 与旧版 attach_verdict.py 的区别 ★
  旧版按 (path, phrase, box) 三元组 join `verify_full.jsonl`，匹配率只有 ~65%，
  且分不清「判 NO」与「没审过」。边车 `verdict/` 已按框位展开、未审显式写 null，
  且与 `clean_rec` **严格逐行对齐**（已自检 36 万行 path 零不一致），直接 zip 即可。

★ MANIFEST 的三条硬约束（本脚本已遵守）★
  1. verdict 只覆盖**首框**（40.79% 的框位）。多框短语的非首框永远是 null。
     → `--first-box-only`（默认开）：只保留每个短语的首框，与 verdict 口径对齐。
        否则 E 组与 C1 会混「框选择策略」和「审核」两个变量。
  2. `null` 不是 `NO`。未审的框位按 --miss-weight 处理（默认 1.0，视为未审）。
  3. verdict 不能用于同一短语的多框取舍 —— 本脚本不做这种取舍。

模式：
  soft : 输出 weight 列（YES→1.0 / NO→--no-weight / null→--miss-weight）
         MANIFEST 推荐：verdict 当置信度分层，别当硬过滤
  hard : 只丢明确 NO（null 保留！丢 null 会砍掉六成数据，混淆变量）
  none : 不用 verdict，仅做 first-box 过滤 —— 这是 E 组的**对照基线**
         （C1-firstbox），用来把「框选择策略」的影响单独隔离出来

用法:
  python scripts/data/attach_verdict_sidecar.py --mode none  # C1-firstbox 对照
  python scripts/data/attach_verdict_sidecar.py --mode soft --no-weight 0.3
  python scripts/data/attach_verdict_sidecar.py --mode hard
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path("/root/paddlejob/workspace/env_run/penghaotian/datas")
CLEAN_REC = ROOT / "cc3m_region/clean_rec"
VERDICT = ROOT / "cc3m_region/verdict"
ANN = ROOT / "cc3m-tsv/annotations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "soft", "hard"], required=True)
    ap.add_argument("--no-weight", type=float, default=0.3, help="soft: NO 的权重")
    ap.add_argument("--miss-weight", type=float, default=1.0, help="soft: null(未审) 的权重")
    ap.add_argument("--first-box-only", action="store_true", default=True,
                    help="只保留每短语首框（与 verdict 口径对齐）")
    ap.add_argument("--all-boxes", dest="first_box_only", action="store_false")
    ap.add_argument("--gt", default=str(ANN / "clip_train_gt.tsv"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-region", type=int, default=12)
    args = ap.parse_args()
    suffix = {"none": "firstbox", "soft": "soft", "hard": "hard"}[args.mode]
    out = args.out or str(ANN / f"clip_train_region_{suffix}.tsv")

    log.info("① 载入 gt caption ...")
    gt = {}
    with open(args.gt, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            p, c = line.rstrip("\n").split("\t", 1)
            gt[p] = c
    log.info(f"   {len(gt):,}")

    cs = sorted(CLEAN_REC.glob("clean_shard*.jsonl"))
    vs = sorted(VERDICT.glob("verdict_shard*.jsonl"))
    assert len(cs) == len(vs), (len(cs), len(vs))

    log.info(f"② 逐行 zip + 输出（mode={args.mode}, first_box_only={args.first_box_only}）...")
    n_out = n_yes = n_no = n_null = n_drop_img = n_region = 0
    n_align_err = 0
    with open(out, "w", encoding="utf-8") as fo:
        fo.write("filepath\tcaption\tregions\n")
        for cf, vf in zip(cs, vs):
            with open(cf, encoding="utf-8") as fc, open(vf, encoding="utf-8") as fv:
                for lc, lv in zip(fc, fv):
                    rc = json.loads(lc)
                    rv = json.loads(lv)
                    if rc["path"] != rv["path"]:
                        n_align_err += 1
                        continue
                    if rc["n_phrase"] == 0:
                        continue
                    path = rc["path"]
                    cap = gt.get(path)
                    if cap is None:
                        continue
                    W, H = rc["img_wh"]
                    if W <= 0 or H <= 0:
                        continue
                    vmap = rv.get("verdict", {})
                    items = []
                    for ph, boxes in rc["grounding"].items():
                        ph_s = ph.strip()
                        if not ph_s:
                            continue
                        vlist = vmap.get(ph, vmap.get(ph_s, []))
                        use = boxes[:1] if args.first_box_only else boxes
                        for bi, b in enumerate(use):
                            x1, y1, x2, y2 = b
                            if x2 - x1 < 2 or y2 - y1 < 2:
                                continue
                            v = vlist[bi] if bi < len(vlist) else None
                            if v == "YES":
                                n_yes += 1
                                w = 1.0
                            elif v == "NO":
                                n_no += 1
                                if args.mode == "hard":
                                    continue            # hard: 只丢明确 NO
                                w = args.no_weight if args.mode == "soft" else 1.0
                            else:                        # null / 缺失 = 未审
                                n_null += 1
                                w = args.miss_weight if args.mode == "soft" else 1.0
                            area = (x2 - x1) * (y2 - y1) / (W * H)
                            items.append((area, ph_s,
                                          round(max(0.0, x1 / W), 4), round(max(0.0, y1 / H), 4),
                                          round(min(1.0, x2 / W), 4), round(min(1.0, y2 / H), 4), w))
                    if not items:
                        n_drop_img += 1
                        continue
                    items.sort(key=lambda t: -t[0])
                    items = items[:args.max_region]
                    if args.mode == "soft":
                        regions = [[it[1], it[2], it[3], it[4], it[5], it[6]] for it in items]
                    else:
                        regions = [[it[1], it[2], it[3], it[4], it[5]] for it in items]
                    n_region += len(regions)
                    fo.write(f"{path}\t{cap}\t{json.dumps(regions, ensure_ascii=False)}\n")
                    n_out += 1
                    if n_out % 500000 == 0:
                        log.info(f"   {n_out:,} 行 ...")

    log.info(f"DONE {out}")
    log.info(f"  图 {n_out:,}（区域全丢 {n_drop_img:,} | 对齐异常 {n_align_err:,}）")
    log.info(f"  区域 {n_region:,}（{n_region/max(n_out,1):.2f}/图）")
    tot = n_yes + n_no + n_null
    log.info(f"  verdict: YES {n_yes:,} ({n_yes/max(tot,1):.1%}) | "
             f"NO {n_no:,} ({n_no/max(tot,1):.1%}) | null {n_null:,} ({n_null/max(tot,1):.1%})")


if __name__ == "__main__":
    main()
