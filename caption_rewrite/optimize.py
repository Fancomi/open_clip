"""GEPA 优化 caption 改写 prompt。student(gemma/qwen) 改写, teacher(Opus) 反思+保真裁判。

用法:
  # 前置: bash serve_models.sh; python -m caption_rewrite.bpe_freq ...; python sample_data.py ...
  python -m caption_rewrite.optimize --student qwen --max-metric-calls 30
  SMOKE=1 python -m caption_rewrite.optimize --student qwen   # 冒烟(小数据+少调用)
  python -m caption_rewrite.optimize --student qwen --apply   # (预留)全量重写, 本阶段不实现
"""
import argparse
import json
import os
import sys

import dspy

sys.path.insert(0, os.path.dirname(__file__))
import bpe_freq
import lab_lm
import metric as metric_mod
from rewrite_program import Rewriter


def load_examples(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    return [dspy.Example(caption=r['caption']).with_inputs('caption') for r in rows]


def avg_report(prog, dset, m):
    """在验证集上跑, 汇总三项指标。"""
    scores, reds, edits, faith = [], [], [], 0
    for ex in dset:
        pred = prog(**ex.inputs())
        out = m(ex, pred)
        scores.append(out.score)
        o_rare = metric_mod._count_rare(ex.caption)
        n_rare = metric_mod._count_rare(getattr(pred, 'rewritten_caption', '') or '')
        reds.append(metric_mod.rare_reduction_rate(o_rare, n_rare))
        edits.append(metric_mod.norm_levenshtein(ex.caption, getattr(pred, 'rewritten_caption', '') or ''))
        faith += int(not out.feedback.startswith('语义被改变'))
    n = len(dset)
    return dict(score=sum(scores) / n, rare_reduction=sum(reds) / n,
                edit=sum(edits) / n, faithful_rate=faith / n)


def main():
    p = argparse.ArgumentParser(description='GEPA 优化 caption 改写')
    p.add_argument('--student', choices=['qwen', 'gemma'], default='qwen')
    p.add_argument('--data-dir', default='caption_rewrite/data')
    p.add_argument('--freq', default='caption_rewrite/outputs/bpe_freq.json')
    p.add_argument('--config', default='caption_rewrite/outputs/config.json')
    p.add_argument('--out-dir', default='caption_rewrite/outputs')
    p.add_argument('--max-metric-calls', type=int, default=30)
    p.add_argument('--lam', type=float, default=0.3)
    p.add_argument('--num-threads', type=int, default=4)
    p.add_argument('--apply', action='store_true', help='(预留) 全量重写, 本阶段未实现')
    args = p.parse_args()

    if args.apply:
        raise SystemExit('[optimize] --apply 全量重写为后续阶段, 本模块未实现。')

    smoke = os.environ.get('SMOKE') == '1'
    cfg = json.load(open(args.config))
    n = cfg['rare_threshold_n']
    rare = bpe_freq.rare_ids(bpe_freq.load_freq(args.freq), n, mode=cfg.get('freq_mode', 'word'))

    trainset = load_examples(os.path.join(args.data_dir, 'train.jsonl'))
    valset = load_examples(os.path.join(args.data_dir, 'val.jsonl'))
    if smoke:
        trainset, valset = trainset[:4], valset[:3]
        args.max_metric_calls = min(args.max_metric_calls, 10)

    student = lab_lm.make_student(args.student, cache=False)
    teacher = lab_lm.make_teacher()
    dspy.configure(lm=student)

    m = metric_mod.make_metric(teacher=teacher, rare_set=rare, lam=args.lam)
    program = Rewriter()

    print(f"== 优化前 (student={args.student}) ==")
    print("  ", avg_report(program, valset, m))

    gepa = dspy.GEPA(metric=m, reflection_lm=teacher,
                     max_metric_calls=args.max_metric_calls,
                     num_threads=args.num_threads, track_stats=True)
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    print("== 优化后 ==")
    rep = avg_report(optimized, valset, m)
    print("  ", rep)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = 'smoke' if smoke else args.student
    prompt_path = os.path.join(args.out_dir, f'optimized_prompt_{tag}.txt')
    with open(prompt_path, 'w', encoding='utf-8') as f:
        f.write(optimized.predict.signature.instructions)
    with open(os.path.join(args.out_dir, f'report_{tag}.json'), 'w') as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print(f"  wrote {prompt_path}")
    print("== 学到的新指令 ==")
    print(optimized.predict.signature.instructions)


if __name__ == '__main__':
    main()
