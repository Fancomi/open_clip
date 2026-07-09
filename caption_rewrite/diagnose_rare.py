"""诊断稀有词构成: 我们要求模型消除的稀有 token, 到底有多少是"可替换的"。

不跑任何模型/网络。以整词为单位归类 (BPE 子词碎片单独统计),
判断任务天花板是否由数据构成(专有名词/碎片/无同义词具体词)决定。
"""
import argparse
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
import bpe_freq

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def load_captions(data_dir):
    caps = []
    for fn in ('train.jsonl', 'val.jsonl'):
        p = os.path.join(data_dir, fn)
        if os.path.exists(p):
            for l in open(p, encoding='utf-8'):
                caps.append(json.loads(l)['caption'])
    return caps


def load_slot_categories(slot_path):
    """slot 词->主类别 (nouns/verbs/.../proper_nouns)。"""
    from collections import defaultdict
    catcnt = defaultdict(Counter)
    if not os.path.exists(slot_path):
        return {}
    for l in open(slot_path, encoding='utf-8'):
        d = json.loads(l)
        for cat, ws in d.get('slots', {}).items():
            for w in ws:
                catcnt[w.strip().lower()][cat] += 1
    return {w: c.most_common(1)[0][0] for w, c in catcnt.items()}


def word_is_rare(word, rare_set):
    """整词是否触发稀有: 其任一 BPE token 落在 rare_set。"""
    return any(t in rare_set for t in bpe_freq.encode_ids(word))


def classify_word(word, slot_cat, freq_word_set):
    """给触发稀有的整词归类。返回 (类别, 是否可替换)。"""
    lw = word.lower()
    # 专有名词: slot 标注 proper_nouns, 或首字母大写且非句首常见
    cat = slot_cat.get(lw)
    if cat == 'proper_nouns':
        return 'proper_noun', False
    if word[0].isupper() and lw not in freq_word_set:
        return 'proper_noun_guess', False
    if cat in ('nouns', 'verbs', 'adjectives', 'adverbs'):
        return cat, True          # 有词性且是实词, 通常可找同义词
    if cat == 'numbers':
        return 'number', False
    if cat == 'spatial_relations':
        return 'spatial', True
    return 'uncategorized', True   # 未归类词, 保守认为可尝试


def main():
    p = argparse.ArgumentParser(description='诊断稀有词构成 (无模型)')
    p.add_argument('--data-dir', default='caption_rewrite/data')
    p.add_argument('--freq', default='caption_rewrite/outputs/bpe_freq.json')
    p.add_argument('--config', default='caption_rewrite/outputs/config.json')
    p.add_argument('--slots', default='analysis/outputs/slots/cc3m_50000/slots.jsonl')
    p.add_argument('--out', default='caption_rewrite/outputs/rare_diagnosis.json')
    args = p.parse_args()

    cfg = json.load(open(args.config))
    N, mode = cfg['rare_threshold_n'], cfg.get('freq_mode', 'word')
    freq = bpe_freq.load_freq(args.freq)
    rare = bpe_freq.rare_ids(freq, N, mode=mode)
    tok = bpe_freq.get_tokenizer()

    # 常见整词集合(词面总频次>=N), 用于判专有名词猜测
    freq_word_set = set()
    from collections import defaultdict
    wf = defaultdict(int)
    for tid, c in freq.items():
        wf[tok.decode([int(tid)]).strip().lower()] += int(c)
    freq_word_set = {w for w, c in wf.items() if c >= N and w}

    slot_cat = load_slot_categories(args.slots)
    caps = load_captions(args.data_dir)

    # 整词级统计: 遍历 caption 里每个整词, 若触发稀有则归类
    word_occ = Counter()          # 触发稀有的整词 -> 出现次数
    cat_occ = Counter()           # 类别 -> 触发稀有整词的出现次数
    replaceable_occ = 0
    total_trigger_occ = 0
    fragment_occ = 0              # 整词由多 token 组成且触发 (子词碎片场景)

    for cap in caps:
        for m in WORD_RE.finditer(cap):
            w = m.group(0)
            if not word_is_rare(w, rare):
                continue
            total_trigger_occ += 1
            word_occ[w.lower()] += 1
            ntok = len(bpe_freq.encode_ids(w))
            if ntok > 1:
                fragment_occ += 1
            cat, replaceable = classify_word(w, slot_cat, freq_word_set)
            cat_occ[cat] += 1
            if replaceable:
                replaceable_occ += 1

    n_cap = len(caps)
    summary = {
        'captions': n_cap,
        'rare_token_ids': len(rare),
        'threshold_N': N, 'mode': mode,
        'unique_trigger_words': len(word_occ),
        'total_trigger_occ': total_trigger_occ,
        'multitoken_word_occ': fragment_occ,
        'replaceable_occ': replaceable_occ,
        'replaceable_frac': round(replaceable_occ / max(total_trigger_occ, 1), 4),
        'category_occ': dict(cat_occ.most_common()),
        'top_trigger_words': word_occ.most_common(60),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"captions={n_cap}  触发稀有的整词: unique={len(word_occ)} occ={total_trigger_occ}")
    print(f"多-token 整词(含子词碎片) occ={fragment_occ} ({fragment_occ/max(total_trigger_occ,1):.1%})")
    print(f"估计可替换 occ={replaceable_occ} ({summary['replaceable_frac']:.1%})  "
          f"不可替换(专名/数字){total_trigger_occ-replaceable_occ}")
    print("\n=== 类别分布 (触发稀有整词出现次数) ===")
    for cat, c in cat_occ.most_common():
        print(f"  {c:5d}  {cat}")
    print("\n=== TOP 40 触发稀有的整词 ===")
    for w, c in word_occ.most_common(40):
        cat, repl = classify_word(w, slot_cat, freq_word_set)
        print(f"  {c:4d}  {w:20s} [{cat}{'' if repl else ' *不可换'}]")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
