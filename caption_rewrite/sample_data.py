"""从 CC3M wds 采样 caption, 只保留含稀有 BPE token 的句子, 切 train/val jsonl。

只有含稀有 token 的 caption 才需重写, 因此数据集只收这类句子。
"""
import argparse
import glob
import json
import logging
import os
import random
import sys
import tarfile

sys.path.insert(0, os.path.dirname(__file__))
import bpe_freq

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def iter_captions(tars, limit_scan):
    seen = 0
    for tar_path in tars:
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if not member.name.endswith('.txt'):
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                yield fobj.read().decode('utf-8', errors='ignore').strip()
                seen += 1
                if limit_scan and seen >= limit_scan:
                    return


def main():
    p = argparse.ArgumentParser(description='采样含稀有 token 的 caption')
    p.add_argument('--tars', required=True)
    p.add_argument('--freq', default='caption_rewrite/outputs/bpe_freq.json')
    p.add_argument('--config', default='caption_rewrite/outputs/config.json')
    p.add_argument('--out-dir', default='caption_rewrite/data')
    p.add_argument('--n-train', type=int, default=40)
    p.add_argument('--n-val', type=int, default=20)
    p.add_argument('--limit-scan', type=int, default=50000, help='最多扫描多少条 caption')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    n = json.load(open(args.config))['rare_threshold_n']
    rare = bpe_freq.rare_ids(bpe_freq.load_freq(args.freq), n)
    log.info(f'[sample] N={n}, rare token 数={len(rare)}')

    tars = sorted(glob.glob(args.tars))
    pool = []
    for cap in iter_captions(tars, args.limit_scan):
        if not cap:
            continue
        nr = bpe_freq.count_rare(cap, rare)
        if nr > 0:
            pool.append({'caption': cap, 'n_rare': nr})
    log.info(f'[sample] 含稀有 token 的句子 {len(pool)} 条')

    random.Random(args.seed).shuffle(pool)
    need = args.n_train + args.n_val
    if len(pool) < need:
        raise SystemExit(f'含稀有词句子不足: {len(pool)} < {need}, 调大 --limit-scan')
    train, val = pool[:args.n_train], pool[args.n_train:need]

    os.makedirs(args.out_dir, exist_ok=True)
    for name, rows in (('train', train), ('val', val)):
        path = os.path.join(args.out_dir, f'{name}.jsonl')
        with open(path, 'w', encoding='utf-8') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        log.info(f'[sample] wrote {path} ({len(rows)})')


if __name__ == '__main__':
    main()
