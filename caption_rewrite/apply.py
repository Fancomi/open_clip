"""全量 CC3M caption 改写落盘: 用 v5 优胜 gemma 提示词, 只改含稀有词的句子。

闭集口径(见 metric.py): 稀有具体词向上抽象到图片仍满足的更常见类别, 增强数据互连。
含稀有词的 caption 才送 gemma 改写; 其余原样直通。按 tar 分片落盘, 可断点续跑。

输出: out-dir/shards/<tar_stem>.jsonl, 每行 {key, original, rewritten, changed}。
  key = "<tar_stem>/<member_id>" (与 wds 样本对齐, 下游可 join/重建)。
跑完可 --merge 合并成单个 all.jsonl。

用法:
  # 前置: gemma 端点存活(serve), 已有 optimized_prompt_gemma.txt / bpe_freq.json / config.json
  python -m caption_rewrite.apply --student gemma
  python -m caption_rewrite.apply --merge          # 合并分片
  SHARD_LIMIT=2 python -m caption_rewrite.apply --student gemma   # 冒烟(前2个tar)
"""
import argparse
import glob
import json
import logging
import os
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import dspy

sys.path.insert(0, os.path.dirname(__file__))
import bpe_freq
import lab_lm
from rewrite_program import Rewriter

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


def load_prompt(path):
    with open(path, encoding='utf-8') as f:
        return f.read().strip()


def iter_tar_captions(tar_path):
    """yield (member_id, caption)。member_id 为去扩展名的样本键 (如 000000000)。"""
    with tarfile.open(tar_path, 'r') as tar:
        for member in tar:
            if not member.name.endswith('.txt'):
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            cap = fobj.read().decode('utf-8', errors='ignore').strip()
            yield member.name[:-4], cap


def shard_done(path, expected):
    """分片已完整落盘? 行数达标即视为 done (断点续跑跳过)。"""
    if not os.path.exists(path):
        return False
    with open(path, encoding='utf-8') as f:
        return sum(1 for _ in f) >= expected


def make_rewriter(prompt, students):
    """每线程绑定一个端点的 program, 轮转分摊到多端口。"""
    progs = []
    for st in students:
        p = Rewriter()
        p.predict.signature.instructions = prompt
        progs.append((p, st))
    return progs


def rewrite_shard(tar_path, out_path, rare, prompt, ports, which, num_threads):
    """改写单个 tar 的所有 caption, 原子落盘到 out_path。

    含稀有 token 的 caption 送 gemma 改写(多端口并发); 其余 changed=False 原样直通。
    """
    students = [lab_lm.make_student(which, port=p) for p in ports]
    progs = make_rewriter(prompt, students)

    rows = list(iter_tar_captions(tar_path))
    results = [None] * len(rows)

    def work(idx):
        key, cap = rows[idx]
        if bpe_freq.count_rare(cap, rare) == 0:
            return idx, {'key': key, 'original': cap, 'rewritten': cap, 'changed': False}
        prog, st = progs[idx % len(progs)]
        try:
            with dspy.context(lm=st):
                new = (prog(caption=cap).rewritten_caption or '').strip()
        except Exception as e:
            log.warning(f'[apply] rewrite failed key={key}: {e}')
            new = cap
        return idx, {'key': key, 'original': cap, 'rewritten': new,
                     'changed': new != cap}

    with ThreadPoolExecutor(max_workers=num_threads) as ex:
        for idx, row in ex.map(work, range(len(rows))):
            results[idx] = row

    tmp = out_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(tmp, out_path)
    n_changed = sum(1 for r in results if r['changed'])
    return len(results), n_changed


def merge_shards(shard_dir, out_path):
    """合并所有分片为单个 all.jsonl。"""
    shards = sorted(glob.glob(os.path.join(shard_dir, '*.jsonl')))
    n = n_changed = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for sp in shards:
            for line in open(sp, encoding='utf-8'):
                out.write(line)
                n += 1
                if json.loads(line)['changed']:
                    n_changed += 1
    log.info(f'[apply] merged {len(shards)} shards -> {out_path} '
             f'({n} rows, {n_changed} changed, {n_changed/max(n,1):.1%})')
    return n, n_changed


def main():
    p = argparse.ArgumentParser(description='全量 CC3M caption 改写落盘')
    p.add_argument('--tars', default='/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds/*.tar')
    p.add_argument('--student', choices=['qwen', 'gemma'], default='gemma')
    p.add_argument('--prompt', default='caption_rewrite/outputs/optimized_prompt_gemma.txt')
    p.add_argument('--freq', default='caption_rewrite/outputs/bpe_freq.json')
    p.add_argument('--config', default='caption_rewrite/outputs/config.json')
    p.add_argument('--out-dir', default='caption_rewrite/outputs/rewritten')
    p.add_argument('--num-threads', type=int, default=8)
    p.add_argument('--ports', default='', help='逗号分隔端口, 覆盖默认 (多卡并发时用全部端口)')
    p.add_argument('--merge', action='store_true', help='合并已有分片为 all.jsonl 后退出')
    args = p.parse_args()

    shard_dir = os.path.join(args.out_dir, 'shards')
    os.makedirs(shard_dir, exist_ok=True)

    if args.merge:
        merge_shards(shard_dir, os.path.join(args.out_dir, 'all.jsonl'))
        return

    cfg = json.load(open(args.config))
    rare = bpe_freq.rare_ids(bpe_freq.load_freq(args.freq),
                             cfg['rare_threshold_n'], mode=cfg.get('freq_mode', 'word'))
    prompt = load_prompt(args.prompt)
    _, ports = lab_lm.MODELS[args.student]
    if args.ports:
        ports = [int(x) for x in args.ports.split(',') if x.strip()]

    tars = sorted(glob.glob(args.tars))
    limit = int(os.environ.get('SHARD_LIMIT', 0))
    if limit:
        tars = tars[:limit]
    log.info(f'[apply] tars={len(tars)} rare_ids={len(rare)} student={args.student} '
             f'ports={ports} threads={args.num_threads}')

    tot_rows = tot_changed = done = 0
    for ti, tar_path in enumerate(tars):
        stem = os.path.splitext(os.path.basename(tar_path))[0]
        out_path = os.path.join(shard_dir, f'{stem}.jsonl')
        n_expected = sum(1 for m in tarfile.open(tar_path).getmembers()
                         if m.name.endswith('.txt'))
        if shard_done(out_path, n_expected):
            done += 1
            log.info(f'[apply] {ti+1}/{len(tars)} {stem} 已完成, 跳过')
            continue
        n, nc = rewrite_shard(tar_path, out_path, rare, prompt, ports,
                              args.student, args.num_threads)
        tot_rows += n
        tot_changed += nc
        log.info(f'[apply] {ti+1}/{len(tars)} {stem}: {n} caps, {nc} changed '
                 f'({nc/max(n,1):.1%})')

    log.info(f'[apply] 完成 (跳过{done}个已存在分片)。本次新写 {tot_rows} caps, '
             f'{tot_changed} changed。--merge 可合并为 all.jsonl')


if __name__ == '__main__':
    main()
