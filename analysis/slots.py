"""Slot extraction data utilities for caption datasets."""
import csv
import json
import os
import re
import string
from collections import Counter, defaultdict

SLOT_TYPES = (
    'nouns', 'verbs', 'adjectives', 'adverbs', 'numbers',
    'spatial_relations', 'proper_nouns', 'others',
)

_REQUIRED_FIELDS = ('id', 'dataset', 'row_index', 'filepath', 'caption', 'slots')
_STRIP_CHARS = string.whitespace + string.punctuation + '“”‘’—–…'
_SPACE_RE = re.compile(r'\s+')
_FUNCTION_WORDS = {
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'its', 'it', 'they',
    'of', 'with', 'and', 'or', 'to', 'as', 'for', 'than', 'then', 'there',
}
_WEAK_VERBS = {
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'do', 'does', 'did',
}
_NON_SPATIAL_RELATIONS = {'of', 'with', 'as', 'to', 'for', 'away'}
_BAD_ADVERBS = {'many', 'right', 'away'}
_BAD_OTHERS = {'up', 'visible'}


def _normalize_sep(sep):
    if sep in ('\\t', 'tab'):
        return '\t'
    if len(sep) != 1:
        raise ValueError(f'CSV separator must be one character, got {sep!r}')
    return sep


def _iter_caption_tsv(tsv_path, img_key='filepath', caption_key='caption', sep='\t'):
    with open(tsv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=_normalize_sep(sep))
        columns = reader.fieldnames or []
        missing = [c for c in (img_key, caption_key) if c not in columns]
        if missing:
            raise ValueError(f'{tsv_path} missing columns: {missing}; columns={columns}')
        for row_index, row in enumerate(reader):
            yield row_index, row


def read_caption_tsv(tsv_path, img_key='filepath', caption_key='caption', sep='\t'):
    """Read a caption TSV and return normalized sample dicts."""
    records, total, empty = [], 0, 0
    for row_index, row in _iter_caption_tsv(tsv_path, img_key, caption_key, sep):
        total += 1
        caption = str(row.get(caption_key) or '').strip()
        if not caption:
            empty += 1
            continue
        filepath = str(row.get(img_key) or '').strip()
        records.append(dict(row_index=int(row_index), filepath=filepath, caption=caption))
    return records, dict(total_rows=total, valid_rows=len(records), empty_captions=empty)


def write_slot_request_jsonl(tsv_path, out_path, dataset, img_key='filepath',
                             caption_key='caption', sep='\t', limit=None):
    """Create JSONL requests for a later LLM slot-extraction runner."""
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    limit = None if limit is None else max(0, int(limit))
    total = empty = written = 0
    with open(out_path, 'w', encoding='utf-8') as f:
        for row_index, row in _iter_caption_tsv(tsv_path, img_key, caption_key, sep):
            if limit is not None and written >= limit:
                break
            total += 1
            caption = str(row.get(caption_key) or '').strip()
            if not caption:
                empty += 1
                continue
            filepath = str(row.get(img_key) or '').strip()
            item = dict(
                id=f'{dataset}_{row_index:09d}',
                dataset=dataset,
                row_index=int(row_index),
                filepath=filepath,
                caption=caption,
            )
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            written += 1
    return dict(total_rows=total, valid_rows=total - empty, empty_captions=empty,
                written=written, out_path=out_path)


def normalize_word(word):
    """Normalize a slot word/phrase without stemming or semantic edits."""
    if word is None:
        return ''
    text = _SPACE_RE.sub(' ', str(word).lower()).strip(_STRIP_CHARS)
    return _SPACE_RE.sub(' ', text).strip()


def _normalize_slots(slots):
    if not isinstance(slots, dict):
        raise ValueError('slots must be a dict')
    out = {}
    for slot in SLOT_TYPES:
        vals = slots.get(slot, [])
        if vals is None:
            vals = []
        if isinstance(vals, str):
            vals = [vals]
        if not isinstance(vals, (list, tuple)):
            raise ValueError(f'slot {slot} must be a list')
        seen, clean = set(), []
        for val in vals:
            norm = normalize_word(val)
            if slot == 'others' and norm in _FUNCTION_WORDS:
                continue
            if slot == 'verbs' and norm in _WEAK_VERBS:
                continue
            if slot == 'spatial_relations' and norm in _NON_SPATIAL_RELATIONS:
                continue
            if slot == 'adverbs' and norm in _BAD_ADVERBS:
                continue
            if slot == 'others' and norm in _BAD_OTHERS:
                continue
            if norm and norm not in seen:
                seen.add(norm)
                clean.append(norm)
        out[slot] = clean
    return out


def _validate_record(obj, line_no):
    missing = [k for k in _REQUIRED_FIELDS if k not in obj]
    if missing:
        raise ValueError(f'line {line_no}: missing fields {missing}')
    rec = {k: obj[k] for k in _REQUIRED_FIELDS}
    rec['id'] = str(rec['id'])
    rec['dataset'] = str(rec['dataset'])
    rec['row_index'] = int(rec['row_index'])
    rec['filepath'] = str(rec['filepath'])
    rec['caption'] = str(rec['caption'])
    rec['slots'] = _normalize_slots(rec['slots'])
    return rec


def read_slot_jsonl(path, strict=True):
    """Read normalized LLM slot JSONL records."""
    records, bad = [], 0
    with open(path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    raise ValueError('line is not a JSON object')
                records.append(_validate_record(obj, line_no))
            except Exception as e:
                if strict:
                    raise ValueError(f'{path}:{line_no}: {e}') from e
                bad += 1
    return records, dict(records=len(records), bad_lines=bad, path=path)


def collect_slot_frequencies(records, slot_types=SLOT_TYPES):
    """Count word frequencies by sample occurrence for each slot type."""
    freqs = {slot: Counter() for slot in slot_types}
    for rec in records:
        slots = rec.get('slots', {})
        for slot in slot_types:
            for word in set(slots.get(slot, [])):
                freqs[slot][word] += 1
    return {slot: dict(counter.most_common()) for slot, counter in freqs.items()}


def select_frequency_words(freq, top_k=5, bottom_k=5, min_count=1):
    """Select high- and low-frequency words from one slot frequency dict."""
    items = [(w, int(c)) for w, c in freq.items() if int(c) >= min_count]
    high = [w for w, _ in sorted(items, key=lambda x: (-x[1], x[0]))[:top_k]]
    low_pool = [(w, c) for w, c in items if w not in set(high)]
    low = [w for w, _ in sorted(low_pool, key=lambda x: (x[1], x[0]))[:bottom_k]]
    return high, low


def parse_slot_types(value):
    if not value:
        return list(SLOT_TYPES)
    slots = [s.strip() for s in str(value).split(',') if s.strip()]
    unknown = [s for s in slots if s not in SLOT_TYPES]
    if unknown:
        raise ValueError(f'unknown slot types {unknown}; valid={SLOT_TYPES}')
    return slots


def _as_str_list(paths):
    return [str(p) for p in list(paths)]


def build_word_to_indices(records, paths=None, slot_type='nouns', match_by='filepath',
                          n_features=None, min_match_rate=0.8, allow_low_match=False):
    """Map each slot word to feature row indices with strict alignment checks."""
    if slot_type not in SLOT_TYPES:
        raise ValueError(f'unknown slot_type={slot_type}; valid={SLOT_TYPES}')
    if match_by not in ('filepath', 'row_index'):
        raise ValueError('match_by must be filepath or row_index')

    if paths is not None:
        paths = _as_str_list(paths)
        if n_features is not None and len(paths) != int(n_features):
            raise ValueError(f'len(paths)={len(paths)} != n_features={n_features}')

    if match_by == 'filepath':
        if paths is None:
            raise ValueError('filepath matching requires paths from the probe npz')
        key_to_indices = defaultdict(list)
        for i, path in enumerate(paths):
            key_to_indices[path].append(i)
    else:
        if n_features is None:
            n_features = len(paths) if paths is not None else None
        if n_features is None:
            raise ValueError('row_index matching requires n_features')
        key_to_indices = {i: [i] for i in range(int(n_features))}

    word_to_indices = defaultdict(set)
    matched = unmatched = with_words = 0
    for rec in records:
        words = rec.get('slots', {}).get(slot_type, [])
        if not words:
            continue
        with_words += 1
        key = rec['filepath'] if match_by == 'filepath' else int(rec['row_index'])
        idxs = key_to_indices.get(key)
        if not idxs:
            unmatched += 1
            continue
        matched += 1
        for word in words:
            word_to_indices[word].update(idxs)

    denom = max(1, with_words)
    match_rate = matched / denom
    stats = dict(
        slot_type=slot_type,
        match_by=match_by,
        total_records=len(records),
        records_with_slot_words=with_words,
        matched_records=matched,
        unmatched_records=unmatched,
        match_rate=match_rate,
        total_feature_rows=int(n_features if n_features is not None else len(paths)),
        unique_feature_paths=len(set(paths)) if paths is not None else None,
        unique_words=len(word_to_indices),
    )
    if with_words and match_rate < min_match_rate and not allow_low_match:
        raise ValueError(
            f'low match rate for {slot_type}: {match_rate:.3f} '
            f'({matched}/{with_words}); use --allow-low-match only after checking alignment')

    return {w: sorted(map(int, idxs)) for w, idxs in word_to_indices.items()}, stats


def save_frequencies(freqs, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(freqs, f, ensure_ascii=False, indent=2)


def save_json(obj, out_path):
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

