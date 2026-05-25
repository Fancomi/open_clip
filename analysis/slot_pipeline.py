"""CLI pipeline for caption slot statistics and feature overlays."""
import csv
import logging
import os

import numpy as np

from .metrics import compute_knn_curvature, compute_knn_density
from .slot_viz import (
    fit_umap_embedding,
    plot_slot_feature_overlay,
    plot_slot_frequency_bars,
    plot_slot_frequency_hist,
)
from .slots import (
    build_word_to_indices,
    collect_slot_frequencies,
    parse_slot_types,
    read_slot_jsonl,
    save_frequencies,
    save_json,
    select_frequency_words,
    write_slot_request_jsonl,
)


def run_make_slot_input(args):
    stats = write_slot_request_jsonl(
        args.data,
        args.slot_out,
        args.dataset,
        img_key=args.img_key,
        caption_key=args.caption_key,
        sep=args.csv_separator,
        limit=args.limit,
    )
    logging.info(f'[slots] request jsonl stats: {stats}')


def run_collect_slots(args):
    os.makedirs(args.out_dir, exist_ok=True)
    records, read_stats = read_slot_jsonl(args.slots, strict=not args.non_strict_slots)
    slot_types = parse_slot_types(args.slot_types)
    freqs = collect_slot_frequencies(records, slot_types)
    freq_path = os.path.join(args.out_dir, 'slot_frequencies.json')
    save_frequencies(freqs, freq_path)
    plot_slot_frequency_bars(freqs, args.out_dir, top_n=args.top_n)
    plot_slot_frequency_hist(freqs, args.out_dir, bins=args.hist_bins)

    rows = []
    for slot, freq in freqs.items():
        total = sum(freq.values())
        rows.append({
            'slot_type': slot,
            'unique_words': len(freq),
            'total_occurrences': total,
            'top_words': list(freq.items())[:10],
        })
    save_json({'read_stats': read_stats, 'summary': rows}, os.path.join(args.out_dir, 'slot_summary.json'))
    logging.info(f'[slots] wrote frequencies to {freq_path}')


def _load_probe_npz(path, feature_key='auto'):
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)
    if feature_key == 'auto':
        key = 'proj_features' if 'proj_features' in keys else 'features'
    else:
        key = feature_key
    if key not in keys:
        raise ValueError(f'feature key {key} not found in {path}; keys={sorted(keys)}')
    feats = np.asarray(data[key])
    paths = data['paths'] if 'paths' in keys else None
    if feats.ndim != 2:
        raise ValueError(f'{path}:{key} must be 2D, got shape={feats.shape}')
    if paths is not None and len(paths) != feats.shape[0]:
        raise ValueError(f'{path}: len(paths)={len(paths)} != features rows={feats.shape[0]}')
    return feats, paths, key


def _metrics(metric):
    return ('density', 'curvature') if metric == 'both' else (metric,)


def _metric_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        'mean': float(values.mean()),
        'median': float(np.median(values)),
        'p25': float(np.percentile(values, 25)),
        'p75': float(np.percentile(values, 75)),
    }


def _save_geometry_summary(feats, records, paths, selected, out_dir, args):
    logging.info('[slots] computing geometry summary density/curvature ...')
    density = compute_knn_density(feats, K=args.k)
    curvature = compute_knn_curvature(feats, K=args.k)
    rows, summary = [], {}
    for slot, info in selected['slots'].items():
        word_to_indices, align_stats = build_word_to_indices(
            records,
            paths=paths,
            slot_type=slot,
            match_by=args.match_by,
            n_features=feats.shape[0],
            min_match_rate=args.min_match_rate,
            allow_low_match=args.allow_low_match,
        )
        summary[slot] = {'alignment': align_stats, 'words': {}}
        groups = {w: 'high' for w in info['high_words']}
        groups.update({w: 'low' for w in info['low_words']})
        for word in info['high_words'] + info['low_words']:
            idx = np.asarray(word_to_indices.get(word, []), dtype=int)
            if len(idx) == 0:
                continue
            ds = _metric_summary(density[idx])
            cs = _metric_summary(curvature[idx])
            row = {
                'slot_type': slot,
                'group': groups[word],
                'word': word,
                'n': int(len(idx)),
                'density_mean': ds['mean'],
                'density_median': ds['median'],
                'density_p25': ds['p25'],
                'density_p75': ds['p75'],
                'curvature_mean': cs['mean'],
                'curvature_median': cs['median'],
                'curvature_p25': cs['p25'],
                'curvature_p75': cs['p75'],
            }
            rows.append(row)
            summary[slot]['words'][word] = row

    if not rows:
        logging.warning('[slots] skip geometry summary: no selected rows')
        return
    csv_path = os.path.join(out_dir, 'slot_geometry_summary.csv')
    json_path = os.path.join(out_dir, 'slot_geometry_summary.json')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_json(summary, json_path)
    logging.info(f'[slots] wrote {csv_path}')
    logging.info(f'[slots] wrote {json_path}')


def run_overlay_slots(args):
    os.makedirs(args.out_dir, exist_ok=True)
    records, read_stats = read_slot_jsonl(args.slots, strict=not args.non_strict_slots)
    slot_types = parse_slot_types(args.slot_types)
    freqs = collect_slot_frequencies(records, slot_types)
    feats, paths, feature_key = _load_probe_npz(args.probe, args.feature_key)
    model_name = args.model_name or f'{os.path.basename(args.probe)}:{feature_key}'
    logging.info(f'[slots] probe={args.probe} feature_key={feature_key} shape={feats.shape}')

    # ── Fit UMAP once on all features, cache to disk ──────────────────────────
    umap_cache = os.path.join(args.out_dir, 'umap_embedding.npy')
    embedding_2d, reducer_tag = fit_umap_embedding(
        feats,
        n_neighbors=getattr(args, 'umap_n_neighbors', 15),
        min_dist=getattr(args, 'umap_min_dist', 0.1),
        seed=args.seed,
        cache_path=umap_cache,
    )

    selected = {'read_stats': read_stats, 'probe': args.probe, 'feature_key': feature_key,
                'reducer': reducer_tag, 'slots': {}}
    for slot in slot_types:
        high, low = select_frequency_words(
            freqs.get(slot, {}), top_k=args.top_k, bottom_k=args.bottom_k,
            min_count=args.min_count,
        )
        words = high + low
        if not words:
            logging.warning(f'[slots] no selected words for {slot}')
            continue
        word_to_indices, align_stats = build_word_to_indices(
            records,
            paths=paths,
            slot_type=slot,
            match_by=args.match_by,
            n_features=feats.shape[0],
            min_match_rate=args.min_match_rate,
            allow_low_match=args.allow_low_match,
        )
        selected['slots'][slot] = {
            'high_words': high,
            'low_words': low,
            'alignment': align_stats,
            'plots': {},
        }
        logging.info(f'[slots] {slot} selected high={high} low={low}')
        logging.info(f'[slots] {slot} alignment={align_stats}')

        for metric in _metrics(args.metric):
            out = os.path.join(args.out_dir, f'slot_overlay_{slot}_{metric}.png')
            plot_stats = plot_slot_feature_overlay(
                feats,
                word_to_indices,
                words,
                out,
                slot_type=slot,
                model_name=model_name,
                metric=metric,
                k=args.k,
                max_points_per_word=args.max_points_per_word,
                metric_max_points=args.metric_max_points,
                background_max_points=args.background_max_points,
                seed=args.seed,
                embedding_2d=embedding_2d,
                reducer_tag=reducer_tag,
                high_words=high,
                low_words=low,
            )
            selected['slots'][slot]['plots'][metric] = plot_stats

    save_json(selected, os.path.join(args.out_dir, 'slot_selected_words.json'))
    if args.save_geometry_summary:
        if paths is None:
            raise ValueError('geometry summary requires paths in probe npz')
        _save_geometry_summary(feats, records, paths, selected, args.out_dir, args)
