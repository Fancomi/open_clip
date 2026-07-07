#!/usr/bin/env python3
"""Extract CC3M WebDataset shards into image files and build COCO-style TSV.

Input WDS sample layout:
  <key>.jpg + <key>.txt (+ optional <key>.json)

Output:
  <out-root>/images/cc3m-train-0000/<key>.jpg
  <out-root>/annotations/clip_train.tsv  (filepath\tcaption)

Why extract images instead of only writing tar paths:
  CsvDataset uses PIL.Image.open(filepath), so filepath must be a normal file.
"""
import argparse
import io
import json
import logging
import tarfile
from multiprocessing import Pool
from pathlib import Path

from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def _clean(text: str) -> str:
    return text.strip().replace("\n", " ").replace("\r", " ").replace("\t", " ")


def _worker(task):
    tar_path, images_root, shard_done = map(Path, task)
    shard_name = tar_path.stem
    out_dir = images_root / shard_name
    done_file = shard_done / f"{shard_name}.json"

    if done_file.exists():
        meta = json.loads(done_file.read_text())
        return shard_name, meta["rows"], meta.get("errors", 0), True

    out_dir.mkdir(parents=True, exist_ok=True)
    samples = {}
    errors = 0

    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            name = Path(member.name)
            key, ext = name.stem, name.suffix.lower()
            if ext not in IMG_EXTS and ext != ".txt":
                continue
            f = tf.extractfile(member)
            if f is None:
                continue
            samples.setdefault(key, {})[ext] = f.read()

    rows = []
    for key, parts in samples.items():
        txt = parts.get(".txt")
        img_ext = next((e for e in IMG_EXTS if e in parts), None)
        if txt is None or img_ext is None:
            errors += 1
            continue
        img_path = out_dir / f"{key}{img_ext}"
        if not img_path.exists():
            try:
                # Validate image bytes before writing. Keep original bytes; no re-encode.
                Image.open(io.BytesIO(parts[img_ext])).verify()
                img_path.write_bytes(parts[img_ext])
            except Exception:
                errors += 1
                continue
        caption = _clean(txt.decode("utf-8", errors="replace"))
        rows.append(f"{img_path}\t{caption}")

    part_path = shard_done / f"{shard_name}.tsv"
    part_path.write_text("\n".join(rows) + ("\n" if rows else ""))
    done_file.write_text(json.dumps({"rows": len(rows), "errors": errors}, indent=2))
    log.info(f"[{shard_name}] {len(rows):>6,} rows, {errors} errors")
    return shard_name, len(rows), errors, False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wds-dir", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds")
    p.add_argument("--out-root", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--pattern", default="cc3m-train-*.tar")
    args = p.parse_args()

    wds_dir = Path(args.wds_dir)
    out_root = Path(args.out_root)
    images_root = out_root / "images"
    ann_dir = out_root / "annotations"
    shard_done = out_root / "_shards"
    ann_dir.mkdir(parents=True, exist_ok=True)
    shard_done.mkdir(parents=True, exist_ok=True)

    shards = sorted(wds_dir.glob(args.pattern))
    if not shards:
        raise FileNotFoundError(f"No shards matched {wds_dir / args.pattern}")

    log.info(f"shards={len(shards)} | out={out_root} | workers={args.workers}")
    tasks = [(str(s), str(images_root), str(shard_done)) for s in shards]
    with Pool(args.workers) as pool:
        results = list(pool.imap_unordered(_worker, tasks))

    # Concatenate in shard order for deterministic TSV.
    rows = 0
    out_tsv = ann_dir / "clip_train.tsv"
    with out_tsv.open("w") as out:
        out.write("filepath\tcaption\n")
        for shard in shards:
            part = shard_done / f"{shard.stem}.tsv"
            if part.exists():
                text = part.read_text()
                out.write(text)
                rows += text.count("\n")

    errors = sum(r[2] for r in results)
    stats = {
        "num_shards": len(shards),
        "num_rows": rows,
        "errors": errors,
        "tsv": str(out_tsv),
        "images_root": str(images_root),
    }
    (ann_dir / "_stats.json").write_text(json.dumps(stats, indent=2))
    log.info(f"DONE rows={rows:,}, errors={errors:,} -> {out_tsv}")


if __name__ == "__main__":
    main()
