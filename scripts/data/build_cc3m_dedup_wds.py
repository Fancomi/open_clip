#!/usr/bin/env python3
"""Build a text-deduplicated copy of cc3m-wds as a new webdataset.

Dedup rule: normalize the .txt caption (lowercase, collapse whitespace, strip);
keep the FIRST sample for each unique normalized caption (global, deterministic
by shard/sample order); drop every later sample whose caption already appeared.

All member files of a kept sample (.jpg/.json/.txt) are copied byte-for-byte
into new shards. Shards are written with a fixed maxcount so every shard is the
same size (except the last) — this keeps `--dataset-resampled` well behaved.

Output:
  <out-dir>/cc3m-train-{00000..NNNNN}.tar
  <out-dir>/_dedup_stats.json
"""
import argparse
import glob
import io
import json
import os
import re
import tarfile
import time

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", s.strip()).lower()


def group_members(tf):
    """Yield (key, {ext: bytes}) grouped by sample key, in tar order."""
    cur_key = None
    parts = {}
    for m in tf:
        if not m.isfile():
            continue
        name = m.name
        dot = name.rfind(".")
        key, ext = name[:dot], name[dot:]
        data = tf.extractfile(m).read()
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            yield cur_key, parts
            parts = {}
            cur_key = key
        parts[ext] = data
    if cur_key is not None and parts:
        yield cur_key, parts


class ShardWriter:
    """Minimal fixed-size tar shard writer (no re-encode)."""

    def __init__(self, out_dir, prefix, maxcount):
        self.out_dir = out_dir
        self.prefix = prefix
        self.maxcount = maxcount
        self.shard_idx = 0
        self.count = 0
        self.tf = None
        self.shards = []
        self._open()

    def _open(self):
        path = os.path.join(self.out_dir, f"{self.prefix}-{self.shard_idx:05d}.tar")
        self.tmp = path + ".tmp"
        self.tf = tarfile.open(self.tmp, "w")
        self.cur_path = path
        self.count = 0

    def _close(self):
        self.tf.close()
        os.rename(self.tmp, self.cur_path)
        self.shards.append((os.path.basename(self.cur_path), self.count))

    def write(self, key, parts):
        for ext, data in parts.items():
            info = tarfile.TarInfo(key + ext)
            info.size = len(data)
            info.mtime = 0
            self.tf.addfile(info, io.BytesIO(data))
        self.count += 1
        if self.count >= self.maxcount:
            self._close()
            self.shard_idx += 1
            self._open()

    def finalize(self):
        if self.count > 0:
            self._close()
        else:
            # nothing written to the open shard; discard it
            self.tf.close()
            if os.path.exists(self.tmp):
                os.remove(self.tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds")
    ap.add_argument("--pattern", default="cc3m-train-*.tar")
    ap.add_argument("--out", default="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-dedup-wds")
    ap.add_argument("--prefix", default="cc3m-train")
    ap.add_argument("--maxcount", type=int, default=5000, help="samples per output shard")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tars = sorted(glob.glob(os.path.join(args.src, args.pattern)))
    print(f"[dedup] {len(tars)} source tars -> {args.out}", flush=True)

    seen = set()
    writer = ShardWriter(args.out, args.prefix, args.maxcount)
    total = kept = dropped = no_txt = 0
    t0 = time.time()

    for ti, tar in enumerate(tars):
        with tarfile.open(tar, "r") as tf:
            for key, parts in group_members(tf):
                total += 1
                txt = parts.get(".txt")
                if txt is None:
                    no_txt += 1
                    continue
                cap = norm(txt.decode("utf-8", "ignore"))
                if not cap or cap in seen:
                    dropped += 1
                    continue
                seen.add(cap)
                writer.write(key, parts)
                kept += 1
        if (ti + 1) % 25 == 0 or ti + 1 == len(tars):
            dt = time.time() - t0
            print(f"[dedup] {ti+1}/{len(tars)} tars | total={total} kept={kept} "
                  f"dropped={dropped} no_txt={no_txt} | {dt:.0f}s", flush=True)

    writer.finalize()

    stats = {
        "src": args.src,
        "out": args.out,
        "num_source_tars": len(tars),
        "total_samples": total,
        "kept_samples": kept,
        "dropped_dup": dropped,
        "no_txt": no_txt,
        "num_output_shards": len(writer.shards),
        "maxcount": args.maxcount,
        "last_shard": writer.shards[-1][0] if writer.shards else None,
    }
    with open(os.path.join(args.out, "_dedup_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== dedup done ===")
    print(json.dumps(stats, indent=2))
    print(f"\nCC3M_DEDUP_TRAIN pattern: {args.prefix}-{{00000..{len(writer.shards)-1:05d}}}.tar")
    print(f"CC3M_DEDUP_N_TRAIN={kept}")


if __name__ == "__main__":
    main()
