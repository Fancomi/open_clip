#!/usr/bin/env python3
"""Build multi-version CC3M TSV training data from gemma4 annotated captions.

数据源: cc3m_annotate/out/gemma_cap/shard*.jsonl
  每行 {id(tsv内行号), shard(tsv编号), path, gt_caption, gemma_short, gemma_dense, dt_s}
  —— 与 cc3m-tsv/_shards/cc3m-train-<n>.tsv 逐行对齐（id = 该 tsv 内行号）。

图片复用 cc3m-tsv/images/<shard>/<key>.jpg（已由 build_cc3m_tsv_from_wds.py 抽出），
本脚本不重新抽图、不重编码，仅把 gemma caption 写成 open_clip 训练用的 TSV。

每个文本版本各出一个 TSV，所有版本共享同一 filepath 列（图像零重复，唯一变量是文本）：
  <out-root>/annotations/clip_train_gt.tsv        (filepath\tcaption, 原 CC3M alt-text)
  <out-root>/annotations/clip_train_short.tsv     (gemma_short, 一句话描述)
  <out-root>/annotations/clip_train_dense.tsv     (gemma_dense, 一段密集描述)
  <out-root>/annotations/clip_train_dense_256.tsv (dense 中 ≤256 token 的行，**超窗行整行丢掉**)

⚠️⚠️ `dense_256` 的命名与旧注释都写成"截断到 256"，**那是错的** ——
第 20 行才是实际行为：**丢整行，不是截断**。代价实测（`stat_caption_tokens.py`，
全量 289.4 万行）：只保留 **69.34% 的图 + 61.89% 的 BPE token 质量**，
即 30.66% 的图从未参与训练。若要"截断而非丢行"，直接用 `clip_train_dense.tsv`
（本脚本已产出的全量版）—— 运行时截断由 `tokenizer.py:263-267` 负责，
保头、末位强写 EOT、绝不丢行。

过滤规则:
  - gemma_dense 缺失（error 条目, 无 dense）→ 该版本跳过该行
  - 默认只保留 gemma_dense 非空的行（完整 2,655,317 条）；--keep-error 时保留但该版本跳过
  - dense_256: 用 open_clip tokenizer 数 BPE token, 超 256 的行**跳过整行**（保证 256 窗口零截断）

用法:
  python scripts/data/build_gemma_tsv.py --cap-dir .../out/gemma_cap --out-root .../cc3m-tsv
  python scripts/data/build_gemma_tsv.py --shard-limit 2   # 冒烟: 只处理前 2 个 jsonl shard
"""
import argparse
import json
import logging
import os
import re
import sys
from multiprocessing import Pool
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# CLIP-BPE tokenizer（PE-Core 文本塔同款）; 惰性加载, 供 dense_256 的**丢行**判定用
_tok = None


def _get_tokenizer(context_length: int = 256):
    global _tok
    if _tok is None:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
        from open_clip import get_tokenizer
        _tok = get_tokenizer('PE-Core-B-16', context_length=context_length)
    return _tok


def _clean(text: str) -> str:
    """清洗: 去换行/tab/回车, 压空格。gemma 文本已是自然语言, 不做 lower。"""
    return re.sub(r"\s+", " ", (text or "").strip().replace("\t", " ")).strip()


def _load_shard_records(jsonl_path):
    """按 (shard,id) 索引该 jsonl 的记录。缺失文件返回空 dict。"""
    recs = {}
    if not os.path.exists(jsonl_path):
        return recs
    for line in open(jsonl_path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        recs[(r["shard"], r["id"])] = r
    return recs


def _worker(task):
    """处理单个 jsonl shard: 产出该 shard 的 3+1 个版本 part TSV + done marker。

    注意: shardN.jsonl 的命名是"生成时按 id 取模 8 分片", 记录里的 shard 字段才是真实
    tsv 编号（shard0.jsonl 覆盖 72 个 tsv: 0,8,...,568）。因此按 shard 字段分目录写
    part 文件, 每个 tsv 的 part 可能由多个 jsonl 共同产出 → 落盘用 append 模式,
    done marker 也按 tsv 编号记。
    """
    jsonl_path, images_root, ann_shard_dir, keep_error = task
    images_root = Path(images_root)
    ann_shard_dir = Path(ann_shard_dir)
    recs = _load_shard_records(jsonl_path)
    tok = _get_tokenizer(256) if "dense_256" in VERSIONS else None

    # 本 jsonl 覆盖的 tsv 编号集合
    shard_nos = sorted({s for (s, _) in recs.keys()})
    touched = {}   # shard_no -> {version: count}
    for (shard_no, iid), r in sorted(recs.items()):
        out_dir = images_root / f"cc3m-train-{shard_no:04d}"
        img_path = out_dir / Path(r["path"]).name
        if not img_path.exists():
            continue
        dense = r.get("gemma_dense")
        vals = {
            "gt": _clean(r.get("gt_caption", "")),
            "short": _clean(r.get("gemma_short", "")),
            "dense": _clean(dense),
        }
        # error 条目: 无 dense。keep_error=False → 整行跳过; True → 保留 gt/short
        if not vals["dense"]:
            if not keep_error:
                continue
        st = touched.setdefault(shard_no, {v: 0 for v in VERSIONS})
        for v, cap in vals.items():
            if not cap:
                continue
            part = ann_shard_dir / f"cc3m-train-{shard_no:04d}.gemma.{v}.tsv"
            with part.open("a") as fo:
                fo.write(f"{img_path}\t{cap}\n")
            st[v] += 1
        if "dense_256" in VERSIONS and vals["dense"]:
            n_tok = len(tok.encode(vals["dense"])) + 2  # +SOT/EOT
            if n_tok <= 256:
                part = ann_shard_dir / f"cc3m-train-{shard_no:04d}.gemma.dense_256.tsv"
                with part.open("a") as fo:
                    fo.write(f"{img_path}\t{vals['dense']}\n")
                st["dense_256"] += 1

    rows_total = 0
    for shard_no in shard_nos:
        done_file = ann_shard_dir / f"cc3m-train-{shard_no:04d}.gemma.done"
        done_file.write_text(json.dumps({
            "versions": touched.get(shard_no, {v: 0 for v in VERSIONS}),
        }))
        rows_total += sum(touched.get(shard_no, {}).values())
    log.info(f"[{Path(jsonl_path).stem}] {len(shard_nos)} tsvs, "
             f"{sum(len(v) for v in touched.values())} rows")
    return Path(jsonl_path).stem, len(shard_nos), rows_total, touched, False


# 版本名 -> 字段
VERSIONS = {"gt": None, "short": None, "dense": None, "dense_256": None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cap-dir", required=True,
                   default="/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/cc3m_annotate/out/gemma_cap")
    p.add_argument("--images-root", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/images")
    p.add_argument("--out-root", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--shard-limit", type=int, default=0, help=">0 时只处理前 N 个 jsonl（冒烟）")
    p.add_argument("--keep-error", action="store_true",
                   help="保留 dense 缺失(error) 的记录（各版本自动跳过, 用于统计）")
    args = p.parse_args()

    cap_dir = Path(args.cap_dir)
    out_root = Path(args.out_root)
    images_root = Path(args.images_root)
    ann_dir = out_root / "annotations"
    ann_shard_dir = out_root / "_shards"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_shard_dir.mkdir(parents=True, exist_ok=True)

    jsonls = sorted(cap_dir.glob("shard*.jsonl"))
    if args.shard_limit:
        jsonls = jsonls[:args.shard_limit]
    if not jsonls:
        raise FileNotFoundError(f"No shard*.jsonl matched {cap_dir}")

    log.info(f"jsonl shards={len(jsonls)} versions={list(VERSIONS)} "
             f"images={images_root} out={out_root}")
    tasks = [(str(j), str(images_root), str(ann_shard_dir), args.keep_error) for j in jsonls]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))

    # 按 shard 顺序拼接每个版本的完整 TSV（确定性）
    # 注意: 8 个 jsonl 各自 append 了各自的 part, 同一 tsv 的 part 由多个 jsonl 共同产出,
    # 需先按 tsv 编号去重(每条记录只在一个 jsonl 里)再拼接。
    stats_versions = {}
    for v in VERSIONS:
        out_tsv = ann_dir / f"clip_train_{v}.tsv"
        n = 0
        seen = set()
        with out_tsv.open("w") as out:
            out.write("filepath\tcaption\n")
            for shard_no in range(576):
                part = ann_shard_dir / f"cc3m-train-{shard_no:04d}.gemma.{v}.tsv"
                if not part.exists():
                    continue
                for line in part.open(encoding="utf-8"):
                    if line in seen:      # 跨 jsonl 可能重复写同一行（同一条记录）
                        continue
                    seen.add(line)
                    out.write(line)
                    n += 1
        stats_versions[v] = n
        log.info(f"[{v}] {n:,} rows -> {out_tsv}")

    errors = 0
    stats = {
        "num_jsonl_shards": len(jsonls),
        "versions": stats_versions,
        "errors": errors,
        "images_root": str(images_root),
        "note": "dense_256 = dense 中 BPE<=256 的**行子集**（超窗行整行丢弃，不是截断）；保留 69.34% 图 / 61.89% token",
    }
    (ann_dir / "_gemma_stats.json").write_text(json.dumps(stats, indent=2))
    log.info(f"DONE shards={len(jsonls)} versions={stats_versions}")


if __name__ == "__main__":
    main()
