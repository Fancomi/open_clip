#!/usr/bin/env python3
"""Measure INTERNAL caption duplication within each dataset separately.

For each dataset: count total captions (one per sample) vs unique normalized
captions -> duplication rate. Also report the top repeated captions.
"""
import argparse
import glob
import os
import re
import sys
import tarfile
from collections import Counter
from multiprocessing import Pool

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", s.strip()).lower()


def count_tar(path: str) -> Counter:
    """Return Counter of normalized captions in one webdataset tar."""
    c = Counter()
    try:
        with tarfile.open(path, "r") as tf:
            for m in tf:
                if not m.name.endswith(".txt"):
                    continue
                f = tf.extractfile(m)
                if f is None:
                    continue
                cap = norm(f.read().decode("utf-8", "ignore"))
                if cap:
                    c[cap] += 1
    except Exception as e:  # noqa
        sys.stderr.write(f"[warn] {path}: {e}\n")
    return c


def count_wds(data_dir: str, pattern: str, procs: int) -> Counter:
    tars = sorted(glob.glob(os.path.join(data_dir, pattern)))
    print(f"[wds] {data_dir}: {len(tars)} tars", flush=True)
    total = Counter()
    done = 0
    with Pool(procs) as p:
        for c in p.imap_unordered(count_tar, tars, chunksize=1):
            total.update(c)
            done += 1
            if done % 100 == 0 or done == len(tars):
                print(f"[wds]   {done}/{len(tars)} tars, {len(total)} unique caps", flush=True)
    return total


def count_tsv(paths, caption_col="caption") -> Counter:
    c = Counter()
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            try:
                ci = header.index(caption_col)
            except ValueError:
                ci = len(header) - 1
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= ci:
                    continue
                cap = norm(parts[ci])
                if cap:
                    c[cap] += 1
    return c


def report(name: str, c: Counter):
    total = sum(c.values())
    uniq = len(c)
    dup_instances = total - uniq  # redundant copies
    dup_keys = sum(1 for v in c.values() if v > 1)  # captions appearing >1 time
    print(f"\n=== {name} internal duplication ===")
    print(f"  total captions (samples): {total}")
    print(f"  unique captions:          {uniq}")
    print(f"  duplicate captions (keys appearing >1x): {dup_keys}")
    print(f"  redundant copies (total-unique):         {dup_instances}")
    if total:
        print(f"  redundancy rate (redundant/total):       {dup_instances/total:.4%}")
        print(f"  unique rate (unique/total):              {uniq/total:.4%}")
    print(f"  top 15 most-repeated captions:")
    for cap, n in c.most_common(15):
        print(f"    {n:>7}x  {cap[:110]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/paddlejob/workspace/env_run/penghaotian/datas")
    ap.add_argument("--procs", type=int, default=48)
    args = ap.parse_args()

    cc3m = count_wds(os.path.join(args.base, "cc3m-wds"), "cc3m-train-*.tar", args.procs)
    report("CC3M", cc3m)
    del cc3m

    cc12m = count_wds(os.path.join(args.base, "cc12m-wds"), "cc12m-train-*.tar", args.procs)
    report("CC12M", cc12m)
    del cc12m

    coco_ann = os.path.join(args.base, "coco", "annotations")
    coco = count_tsv([
        os.path.join(coco_ann, "karpathy_5cap.tsv"),
        os.path.join(coco_ann, "clip_train_dedup.tsv"),
    ])
    report("COCO (karpathy_5cap + clip_train_dedup)", coco)


if __name__ == "__main__":
    main()
