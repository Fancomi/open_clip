#!/usr/bin/env python3
"""Check exact-text (caption) overlap between cc3m-wds, cc12m-wds, coco.

Phase 1: extract normalized captions from each dataset -> dedup set -> file.
Phase 2: compute pairwise intersections between the three caption sets.

Normalization: lowercase, collapse internal whitespace, strip. This makes the
match robust to trivial spacing/case differences while still being an *exact
caption* match (not fuzzy / substring).
"""
import argparse
import glob
import os
import re
import sys
import tarfile
from multiprocessing import Pool

WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", s.strip()).lower()


def extract_tar(path: str) -> set:
    """Return the set of normalized captions in one webdataset tar (.txt members)."""
    out = set()
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
                    out.add(cap)
    except Exception as e:  # noqa
        sys.stderr.write(f"[warn] {path}: {e}\n")
    return out


def extract_wds(data_dir: str, pattern: str, procs: int) -> set:
    tars = sorted(glob.glob(os.path.join(data_dir, pattern)))
    print(f"[wds] {data_dir}: {len(tars)} tars", flush=True)
    result = set()
    done = 0
    with Pool(procs) as p:
        for s in p.imap_unordered(extract_tar, tars, chunksize=1):
            result |= s
            done += 1
            if done % 50 == 0 or done == len(tars):
                print(f"[wds]   {done}/{len(tars)} tars, {len(result)} unique caps", flush=True)
    return result


def extract_tsv(paths, caption_col="caption") -> set:
    out = set()
    for path in paths:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            header = fh.readline().rstrip("\n").split("\t")
            try:
                ci = header.index(caption_col)
            except ValueError:
                ci = len(header) - 1  # last column fallback
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= ci:
                    continue
                cap = norm(parts[ci])
                if cap:
                    out.add(cap)
        print(f"[tsv] {path}: total unique so far {len(out)}", flush=True)
    return out


def dump(s: set, path: str):
    with open(path, "w", encoding="utf-8") as fh:
        for x in sorted(s):
            fh.write(x + "\n")


def report(name_a, a, name_b, b):
    inter = a & b
    print(f"\n=== {name_a} ∩ {name_b} ===")
    print(f"  {name_a}: {len(a)} unique | {name_b}: {len(b)} unique | overlap: {len(inter)}")
    if a:
        print(f"  overlap / {name_a} = {len(inter)/len(a):.4%}")
    if b:
        print(f"  overlap / {name_b} = {len(inter)/len(b):.4%}")
    for i, ex in enumerate(sorted(inter)[:10]):
        print(f"    ex{i}: {ex[:120]}")
    return inter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/paddlejob/workspace/env_run/penghaotian/datas")
    ap.add_argument("--out", default="/tmp/text_overlap")
    ap.add_argument("--procs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    cc3m = extract_wds(os.path.join(args.base, "cc3m-wds"), "cc3m-train-*.tar", args.procs)
    dump(cc3m, os.path.join(args.out, "cc3m_caps.txt"))

    cc12m = extract_wds(os.path.join(args.base, "cc12m-wds"), "cc12m-train-*.tar", args.procs)
    dump(cc12m, os.path.join(args.out, "cc12m_caps.txt"))

    coco_ann = os.path.join(args.base, "coco", "annotations")
    coco = extract_tsv([
        os.path.join(coco_ann, "karpathy_5cap.tsv"),
        os.path.join(coco_ann, "clip_train_dedup.tsv"),
    ])
    dump(coco, os.path.join(args.out, "coco_caps.txt"))

    print("\n############ OVERLAP REPORT ############")
    print(f"CC3M unique={len(cc3m)}  CC12M unique={len(cc12m)}  COCO unique={len(coco)}")
    report("CC3M", cc3m, "CC12M", cc12m)
    report("CC3M", cc3m, "COCO", coco)
    report("CC12M", cc12m, "COCO", coco)
    triple = cc3m & cc12m & coco
    print(f"\n=== CC3M ∩ CC12M ∩ COCO === {len(triple)}")
    for i, ex in enumerate(sorted(triple)[:10]):
        print(f"    ex{i}: {ex[:120]}")
    print("\nDone. Caption files written to", args.out)


if __name__ == "__main__":
    main()
