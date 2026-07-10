#!/usr/bin/env python3
"""Build multi-version CC3M TSV training data from rewritten captions.

图片只抽一份到 images/<shard>/<key>.jpg; 每个 caption 版本各出一个 TSV,
所有版本共享同一 filepath 列 (图像零重复, 唯一变量是文本)。

版本来源: caption_rewrite/outputs/rewritten/shards/<shard>.jsonl
  每行 {key, original, rewritten, changed}。
VERSIONS 字典把"版本名 -> 取哪个字段"解耦, 之后加新版本(过滤/融合等)只加一行。

输出:
  <out-root>/images/<shard>/<key>.jpg
  <out-root>/annotations/clip_train_orig.tsv        (filepath\tcaption, 原文)
  <out-root>/annotations/clip_train_rewritten.tsv   (filepath\tcaption, 改写)

多进程按 shard 并行; 每 shard 落 done marker, 可断点续跑。

用法:
  python -m scripts.data.build_cc3m_rewrite_tsv --workers 96
  SHARD_LIMIT=2 python scripts/data/build_cc3m_rewrite_tsv.py   # 冒烟
"""
import argparse
import io
import json
import logging
import os
import tarfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")

# 版本名 -> 从 jsonl 记录取哪个字段作 caption。加新版本只在此加一行。
VERSIONS = {
    "orig": lambda r: r["original"],
    "rewritten": lambda r: r["rewritten"],
}


def _clean(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def _load_shard_captions(jsonl_path):
    """key -> record dict。缺失分片返回空 (该 shard 跳过, 不报错)。"""
    caps = {}
    if not os.path.exists(jsonl_path):
        return caps
    for line in open(jsonl_path, encoding="utf-8"):
        r = json.loads(line)
        caps[r["key"]] = r
    return caps


def _worker(task):
    """抽一个 tar 的图片 + 为每个版本写该 shard 的 part TSV。"""
    tar_path, images_root, ann_shard_dir, jsonl_dir = map(Path, task)
    shard = tar_path.stem
    out_dir = images_root / shard
    done_file = ann_shard_dir / f"{shard}.done"

    # 每版本的 part TSV 路径
    part_paths = {v: ann_shard_dir / f"{shard}.{v}.tsv" for v in VERSIONS}

    if done_file.exists() and all(p.exists() for p in part_paths.values()):
        meta = json.loads(done_file.read_text())
        return shard, meta["rows"], meta.get("errors", 0), True

    caps = _load_shard_captions(jsonl_dir / f"{shard}.jsonl")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 读 tar 内图片字节 (只需图片; caption 来自 jsonl)
    img_bytes = {}
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name)
            key, ext = name.stem, name.suffix.lower()
            if ext not in IMG_EXTS:
                continue
            f = tf.extractfile(member)
            if f is not None:
                img_bytes[key] = (ext, f.read())

    errors = 0
    version_rows = {v: [] for v in VERSIONS}
    for key, (ext, data) in img_bytes.items():
        rec = caps.get(key)
        if rec is None:                      # jsonl 无此 key: caption 缺失, 跳过
            errors += 1
            continue
        img_path = out_dir / f"{key}{ext}"
        if not img_path.exists():
            try:
                Image.open(io.BytesIO(data)).verify()  # 校验不重编码
                img_path.write_bytes(data)
            except Exception:
                errors += 1
                continue
        for v, getter in VERSIONS.items():
            cap = _clean(str(getter(rec)))
            version_rows[v].append(f"{img_path}\t{cap}")

    rows = 0
    for v, lines in version_rows.items():
        part_paths[v].write_text("\n".join(lines) + ("\n" if lines else ""))
        rows = len(lines)                    # 各版本行数一致
    done_file.write_text(json.dumps({"rows": rows, "errors": errors}))
    log.info(f"[{shard}] {rows:>6,} rows x{len(VERSIONS)}ver, {errors} errors")
    return shard, rows, errors, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wds-dir", default="/root/paddlejob/gpfsspace/cc3m-wds")
    p.add_argument("--jsonl-dir", default="caption_rewrite/outputs/rewritten/shards")
    p.add_argument("--out-root", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv")
    p.add_argument("--workers", type=int, default=96)
    p.add_argument("--pattern", default="cc3m-*.tar")
    args = p.parse_args()

    wds_dir = Path(args.wds_dir)
    out_root = Path(args.out_root)
    images_root = out_root / "images"
    ann_dir = out_root / "annotations"
    ann_shard_dir = out_root / "_shards"
    ann_dir.mkdir(parents=True, exist_ok=True)
    ann_shard_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(wds_dir.glob(args.pattern))
    limit = int(os.environ.get("SHARD_LIMIT", 0))
    if limit:
        shards = shards[:limit]
    if not shards:
        raise FileNotFoundError(f"No shards matched {wds_dir / args.pattern}")

    log.info(f"shards={len(shards)} versions={list(VERSIONS)} "
             f"out={out_root} workers={args.workers}")
    tasks = [(str(s), str(images_root), str(ann_shard_dir), str(args.jsonl_dir))
             for s in shards]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))

    # 按 shard 顺序拼接每个版本的完整 TSV (确定性)
    total_rows = 0
    for v in VERSIONS:
        out_tsv = ann_dir / f"clip_train_{v}.tsv"
        n = 0
        with out_tsv.open("w") as out:
            out.write("filepath\tcaption\n")
            for shard in shards:
                part = ann_shard_dir / f"{shard.stem}.{v}.tsv"
                if part.exists():
                    text = part.read_text()
                    out.write(text)
                    n += text.count("\n")
        total_rows = n
        log.info(f"[{v}] {n:,} rows -> {out_tsv}")

    errors = sum(r[2] for r in results)
    stats = {
        "num_shards": len(shards),
        "num_rows": total_rows,
        "versions": list(VERSIONS),
        "errors": errors,
        "images_root": str(images_root),
    }
    (ann_dir / "_stats.json").write_text(json.dumps(stats, indent=2))
    log.info(f"DONE shards={len(shards)} rows={total_rows:,} errors={errors:,} "
             f"versions={list(VERSIONS)} -> {ann_dir}")


if __name__ == "__main__":
    main()
