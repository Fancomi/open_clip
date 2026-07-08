"""CLIP 文本塔 BPE token 频次统计 + 稀有词判定。

冻结的 token 频次表是评分锚: token 频次 < N 视为稀有。
用 open_clip 的 SimpleTokenizer.encode() 取纯 BPE id (不含 SOT/EOT/pad),
与 CLIP 训练时真正见到的单位一致。
"""
import argparse
import glob
import json
import logging
import os
import sys
import tarfile
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import open_clip

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)

_TOK = None


def get_tokenizer():
    global _TOK
    if _TOK is None:
        _TOK = open_clip.get_tokenizer('PE-Core-B-16-dinov3')
    return _TOK


def encode_ids(caption):
    """纯 BPE id 列表 (SimpleTokenizer.encode, 不含特殊符)。"""
    return list(get_tokenizer().encode(caption))


def count_tar_tokens(tars):
    """流式遍历 wds tar 的 .txt caption, 累计 BPE token-id 频次。"""
    freq = Counter()
    n_cap = n_tok = 0
    for ti, tar_path in enumerate(tars):
        with tarfile.open(tar_path, 'r') as tar:
            for member in tar:
                if not member.name.endswith('.txt'):
                    continue
                fobj = tar.extractfile(member)
                if fobj is None:
                    continue
                caption = fobj.read().decode('utf-8', errors='ignore')
                ids = encode_ids(caption)
                freq.update(ids)
                n_cap += 1
                n_tok += len(ids)
        log.info(f'[bpe_freq] tar {ti + 1}/{len(tars)} captions={n_cap} tokens={n_tok} vocab={len(freq)}')
    return freq, dict(captions=n_cap, tokens=n_tok, vocab=len(freq))


def load_freq(path):
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(k): int(v['count'] if isinstance(v, dict) else v) for k, v in raw.items()}


def rare_ids(freq, n):
    return {int(tid) for tid, c in freq.items() if int(c) < int(n)}


def count_rare(caption, rare_set):
    return sum(1 for i in encode_ids(caption) if i in rare_set)


def _save(freq, out_dir):
    tok = get_tokenizer()
    os.makedirs(out_dir, exist_ok=True)
    ordered = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    obj = {str(tid): {'count': c, 'subword': tok.decode([tid])} for tid, c in ordered}
    path = os.path.join(out_dir, 'bpe_freq.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    log.info(f'[bpe_freq] wrote {path}')
    return path


def _plot_dist(freq, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    counts = np.array(sorted(freq.values(), reverse=True), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].loglog(np.arange(1, len(counts) + 1), counts, color='#4C78A8')
    axes[0].set_xlabel('Token rank'); axes[0].set_ylabel('Frequency')
    axes[0].set_title('BPE token frequency (rank-freq, log-log)')
    axes[0].grid(alpha=0.3)
    for thr in (10, 50, 100, 500, 1000):
        n_below = int((counts < thr).sum())
        axes[1].bar(str(thr), n_below, color='#E45756')
        axes[1].text(str(thr), n_below, f' {n_below}', ha='center', va='bottom', fontsize=8)
    axes[1].set_xlabel('Threshold N'); axes[1].set_ylabel('# tokens with freq < N')
    axes[1].set_title('Rare-token count vs threshold')
    fig.tight_layout()
    path = os.path.join(out_dir, 'bpe_freq_dist.png')
    fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)
    log.info(f'[bpe_freq] wrote {path}')


def main():
    p = argparse.ArgumentParser(description='CC3M CLIP-BPE token frequency')
    p.add_argument('--tars', required=True, help='tar glob')
    p.add_argument('--out-dir', default='caption_rewrite/outputs')
    args = p.parse_args()
    tars = sorted(glob.glob(args.tars))
    if not tars:
        raise SystemExit(f'no tar matched: {args.tars}')
    freq, stats = count_tar_tokens(tars)
    _save(freq, args.out_dir)
    _plot_dist(freq, args.out_dir)
    with open(os.path.join(args.out_dir, 'bpe_freq_summary.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    log.info(f'[bpe_freq] {stats}')


if __name__ == '__main__':
    main()
