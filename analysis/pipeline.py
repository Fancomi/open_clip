"""Analysis pipeline modes: pretrained (COCO/CC3M), overlap, anisotropy, epochs."""
import csv, glob, json, logging, os, re
import numpy as np
import pandas as pd
import torch

from .models   import CKPT, load_pe_core, load_siglip2, load_dinov3, \
                      load_radio, load_eupe, load_tips
from .extractors import (load_from_cache,
                          extract_clip_img, extract_clip_txt,
                          extract_pe_core_img_raw,
                          extract_dinov3_img, extract_radio_img,
                          extract_eupe_img, extract_tips_img, extract_tips_txt,
                          extract_wds_features,
                          make_dino_crops,
                          extract_dinov3_pil, extract_clip_pil,
                          extract_pe_core_pil_raw,
                          extract_radio_pil, extract_eupe_pil, extract_tips_pil)
from .metrics      import fps_sample, compute_anisotropy
from .viz          import plot_scatter, plot_overlap, plot_anisotropy, plot_aniso_evolution, plot_evolution, plot_crop_probe
from .pc_alignment import _plot_final_pc_pairs

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_DATA = dict(
    data     = f'{_BASE}/datas/coco/annotations/karpathy_1cap.tsv',
    out_dir  = f'{_BASE}/datas/coco/feature_probe',
    coco_dir = f'{_BASE}/datas/coco/feature_probe/pretrained',
    cc3m_wds = f'{_BASE}/datas/LLaVA-ReCap-CC3M/wds/{{00000..00280}}.tar',
    cc3m_out = f'{_BASE}/datas/LLaVA-ReCap-CC3M/feature_probe',
    cc3m_dir = f'{_BASE}/datas/LLaVA-ReCap-CC3M/feature_probe/pretrained',
)

# ── Anisotropy summary table helper ──────────────────────────────────────────

def _log_aniso_table(aniso: dict):
    hdr = (f"{'Model':<12} {'EffRank':>8} {'PR':>7} {'StableR':>8} {'NumRank':>8}"
           f" {'AvgCos':>7} {'StdCos':>7} {'top4%':>6} {'top10%':>7} {'top50%':>7}")
    logging.info('\n=== Anisotropy summary ===\n' + hdr)
    for name, m in aniso.items():
        logging.info(
            f"{name:<12} {m['effective_rank']:8.1f} {m['participation_ratio']:7.4f}"
            f" {m['stable_rank']:8.1f} {m['numerical_rank']:8d}"
            f" {m['avg_cos_sim']:7.4f} {m['std_cos_sim']:7.4f}"
            f" {m['pct_var_top4']:6.1f} {m['pct_var_top10']:7.1f} {m['pct_var_top50']:7.1f}")


# ── Mode: pretrained (COCO tsv or CC3M wds) ──────────────────────────────────

def run_pretrained(args):
    out = os.path.join(args.out_dir, 'pretrained')
    os.makedirs(out, exist_ok=True)

    # ── Try cache-first (skip all model loading if hit) ────────────────────
    cached = load_from_cache(out, args.force)

    if args.data_type == 'wds':
        if cached is None:
            feats = extract_wds_features(
                *load_pe_core(args.pe_ckpt),
                *load_siglip2(args.sig2_ckpt),
                load_dinov3(args.dino_repo, args.dino_ckpt),
                *load_radio(args.radio),
                load_eupe(args.eupe_repo, args.eupe_ckpt),
                *load_tips(args.tips),
                args.data, out, max_samples=args.max_samples, force=args.force,
            )
        else:
            feats = cached
        pe_img,   pe_txt   = feats['pe_img'],   feats['pe_txt']
        sig2_img, sig2_txt = feats['sig2_img'], feats['sig2_txt']
        dino_img  = feats['dino_img']
        radio_img = feats.get('radio_img')
        eupe_img  = feats.get('eupe_img')
        tips_img, tips_txt = feats.get('tips_img'), feats.get('tips_txt')

    else:   # tsv (COCO)
        if cached is not None:
            pe_img,   pe_txt   = cached['pe_img'],   cached['pe_txt']
            sig2_img, sig2_txt = cached['sig2_img'], cached['sig2_txt']
            dino_img  = cached['dino_img']
            radio_img = cached.get('radio_img')
            eupe_img  = cached.get('eupe_img')
            tips_img, tips_txt = cached['tips_img'], cached['tips_txt']
            # Raw backbone CLS cache (may not exist on first run after this change)
            _raw_p = os.path.join(out, 'pe_core_img_raw.npz')
            pe_img_raw = np.load(_raw_p)['features'] if os.path.exists(_raw_p) else None
        else:
            df = pd.read_csv(args.data, sep='\t')
            paths, caps = df['filepath'].tolist(), df['caption'].tolist()

            pe_m, pe_p, pe_t = load_pe_core(args.pe_ckpt)
            # projected [B,1024]: kept for modality_gap (CLIP-aligned text-image space)
            pe_img = extract_clip_img(pe_m, paths, pe_p, os.path.join(out,'pe_core_img.npz'), args.force)
            # raw backbone CLS [B,768]: used for image_allmodels / anisotropy / crop_probe
            pe_img_raw = extract_pe_core_img_raw(pe_m, paths, pe_p, os.path.join(out,'pe_core_img_raw.npz'), args.force)
            pe_txt = extract_clip_txt(pe_m, pe_t, caps, os.path.join(out,'pe_core_txt.npz'), args.force)
            del pe_m; torch.cuda.empty_cache()

            s2_m, s2_p, s2_t = load_siglip2(args.sig2_ckpt)
            sig2_img = extract_clip_img(s2_m, paths, s2_p, os.path.join(out,'siglip2_img.npz'), args.force)
            sig2_txt = extract_clip_txt(s2_m, s2_t, caps, os.path.join(out,'siglip2_txt.npz'), args.force)
            del s2_m; torch.cuda.empty_cache()

            dn = load_dinov3(args.dino_repo, args.dino_ckpt)
            dino_img = extract_dinov3_img(dn, paths, os.path.join(out,'dinov3_img.npz'), args.force)
            del dn; torch.cuda.empty_cache()

            ra, ra_c = load_radio(args.radio)
            radio_img = extract_radio_img(ra, ra_c, paths, os.path.join(out,'radio_img.npz'), args.force)
            del ra; torch.cuda.empty_cache()

            eu = load_eupe(args.eupe_repo, args.eupe_ckpt)
            eupe_img = None
            if eu is not None:
                eupe_img = extract_eupe_img(eu, paths, os.path.join(out,'eupe_img.npz'), args.force)
                del eu; torch.cuda.empty_cache()

            ti_m, ti_t = load_tips(args.tips)
            tips_img = extract_tips_img(ti_m, paths, os.path.join(out,'tips_img.npz'), args.force)
            tips_txt = extract_tips_txt(ti_m, ti_t, caps, os.path.join(out,'tips_txt.npz'), args.force)
            del ti_m; torch.cuda.empty_cache()

    # ── Modality gap plots (models with text towers) ────────────────────────
    _MOD_COLORS = ['#0055FF', '#FF2200']   # Image=blue, Text=red

    def _modality_gap(img, txt, model_name, out_path):
        """Full scatter (all PC pairs + T-SNE) with blue/red high-contrast colors.

        FPS is computed in image space; same indices shown in text space too
        so you can see where a specific image's paired caption lands.
        """
        fps_img = fps_sample(img, k=5).tolist()
        plot_scatter(
            {f'{model_name} Image': img, f'{model_name} Text': txt},
            f'{model_name}: Image vs Text',
            out_path,
            n_pca=args.n_pca,
            fps_indices=fps_img,
            colors=_MOD_COLORS,
            fps_pair_link=True,
        )

    _modality_gap(pe_img,   pe_txt,   'PE-Core',  os.path.join(out, 'pe_core_modality_gap.png'))
    _modality_gap(sig2_img, sig2_txt, 'SigLIP2',  os.path.join(out, 'siglip2_modality_gap.png'))
    if tips_img is not None and tips_txt is not None:
        _modality_gap(tips_img, tips_txt, 'TIPSv2', os.path.join(out, 'tips_modality_gap.png'))

    # ── All-model image comparison + FPS tracking ───────────────────────────
    # All entries use backbone raw CLS / summary token (no CLIP projection head).
    # PE-Core: pe_img_raw [B,768] trunk CLS  (pe_img [B,1024] is projection, modality_gap only)
    # SigLIP2: sig2_img [B,768] — head=Sequential() empty, already backbone CLS
    # DINOv3/EUPE: x_norm_clstoken [B,768]   RADIO: summary token
    img_feats = {k: v for k, v in [
        ('DINOv3', dino_img), ('RADIO', radio_img), ('EUPE', eupe_img),
        ('TIPSv2', tips_img),
        ('PE-Core', pe_img_raw if pe_img_raw is not None else pe_img),
        ('SigLIP2', sig2_img),
    ] if v is not None}

    # ── FPS anchors (source model selectable via --fps-model) ──────────────
    fps_model = getattr(args, 'fps_model', 'DINOv3')
    fps_source = img_feats.get(fps_model)
    if fps_source is None:
        fps_model  = list(img_feats.keys())[0]
        fps_source = img_feats[fps_model]
        logging.warning(f'--fps-model not found, falling back to {fps_model}')
    fps_idx = fps_sample(fps_source, k=5)
    logging.info(f'FPS anchor indices ({fps_model} space): {fps_idx.tolist()}')

    plot_scatter(img_feats,
                 f'Vision Encoder Image Features  (* = FPS anchors from {fps_model} space)',
                 os.path.join(out, 'image_allmodels.png'),
                 n_pca=args.n_pca, fps_indices=fps_idx)

    # ── Anisotropy (includes rank + multimodality) ──────────────────────────
    aniso = {name: compute_anisotropy(feat) for name, feat in img_feats.items()}
    plot_anisotropy(aniso, os.path.join(out, 'anisotropy.png'))
    _log_aniso_table(aniso)


# ── Mode: overlap ─────────────────────────────────────────────────────────────

def run_overlap(args):
    npz_pairs = {
        'PE-Core':  ('pe_core_img.npz',  'pe_core_img.npz'),
        'SigLIP2':  ('siglip2_img.npz',  'siglip2_img.npz'),
        'DINOv3':   ('dinov3_img.npz',   'dinov3_img.npz'),
        'RADIO':    ('radio_img.npz',    'radio_img.npz'),
        'EUPE':     ('eupe_img.npz',     'eupe_img.npz'),
        'TIPSv2':   ('tips_img.npz',     'tips_img.npz'),
    }
    out = os.path.join(os.path.dirname(args.cc3m_dir.rstrip('/')), 'overlap')
    os.makedirs(out, exist_ok=True)

    available = {}
    for model, (cf, mf) in npz_pairs.items():
        cp = os.path.join(args.coco_dir, cf)
        mp = os.path.join(args.cc3m_dir, mf)
        if os.path.exists(cp) and os.path.exists(mp):
            available[model] = (cp, mp)
        else:
            logging.info(f'  skip {model}: cache missing')
    assert available, 'No cached npz pairs found — run coco and cc3m modes first.'

    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    # ── per-model: two separate plots (COCO-on-top / CC3M-on-top) ─────────
    model_data = {}
    for model, (cp, mp) in available.items():
        fa = np.load(cp)['features']
        fb = np.load(mp)['features']
        combined = np.concatenate([fa, fb])
        pca = PCA(n_components=2).fit(combined)
        pa  = pca.transform(fa)   # COCO
        pb  = pca.transform(fb)   # CC3M
        d   = float(np.linalg.norm(pa.mean(0) - pb.mean(0)))
        model_data[model] = (pa, pb, d)

        plot_overlap(pa, pb, 'COCO', 'CC3M', model,
                     os.path.join(out, f'overlap_{model.lower()}_coco_top.png'),
                     a_on_top=True,  centroid_dist=d)
        plot_overlap(pa, pb, 'COCO', 'CC3M', model,
                     os.path.join(out, f'overlap_{model.lower()}_cc3m_top.png'),
                     a_on_top=False, centroid_dist=d)

    # ── summary grid: n_models rows × 2 cols ──────────────────────────────
    n = len(available)
    fig, axes = plt.subplots(n, 2, figsize=(10, 5 * n))
    axes = np.array(axes).reshape(n, 2)
    col_titles = ['COCO on top', 'CC3M on top']
    for row, (model, (pa, pb, d)) in enumerate(model_data.items()):
        for col, a_on_top in enumerate([True, False]):
            ax = axes[row, col]
            if a_on_top:
                ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.3, color='coral',
                           label='CC3M', rasterized=True)
                ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color='steelblue',
                           label='COCO', rasterized=True)
            else:
                ax.scatter(pa[:, 0], pa[:, 1], s=2, alpha=0.3, color='steelblue',
                           label='COCO', rasterized=True)
                ax.scatter(pb[:, 0], pb[:, 1], s=2, alpha=0.3, color='coral',
                           label='CC3M', rasterized=True)
            ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
            ax.legend(markerscale=4, fontsize=8)
            title = f'{model} — {col_titles[col]}'
            if col == 1:
                title += f'\ncentroid dist={d:.3f}'
            ax.set_title(title, fontsize=9)

    fig.suptitle('COCO vs CC3M Feature Distribution Overlap', fontsize=11)
    plt.tight_layout()
    grid = os.path.join(out, 'overlap_grid.png')
    plt.savefig(grid, dpi=150, bbox_inches='tight'); plt.close()
    print(f'[viz] {grid}')


# ── Mode: anisotropy ──────────────────────────────────────────────────────────

def run_anisotropy(args):
    npz_map = {
        'PE-Core': 'pe_core_img.npz', 'SigLIP2': 'siglip2_img.npz',
        'DINOv3':  'dinov3_img.npz',  'RADIO':   'radio_img.npz',
        'EUPE':    'eupe_img.npz',    'TIPSv2':  'tips_img.npz',
    }
    metrics = {}
    for name, fname in npz_map.items():
        p = os.path.join(args.aniso_dir, fname)
        if not os.path.exists(p):
            logging.info(f'  skip {name}: {fname} not found')
            continue
        f = np.load(p)['features']
        logging.info(f'  {name}  shape={f.shape}')
        metrics[name] = compute_anisotropy(f)
    assert metrics, f'No npz found in {args.aniso_dir}'
    plot_anisotropy(metrics, os.path.join(args.aniso_dir, 'anisotropy.png'))
    _log_aniso_table(metrics)


# ── Mode: epochs / steps ─────────────────────────────────────────────────────

def run_epochs(args):
    """Load probe npz files (step_*.npz preferred, epoch_*.npz as fallback)
    and render GIF evolution + static trajectory plot."""
    import re
    probe_dir = args.probe_dir

    # Prefer step-based files; fall back to epoch-based
    step_files = sorted(glob.glob(os.path.join(probe_dir, 'step_*.npz')))
    epoch_files = sorted(glob.glob(os.path.join(probe_dir, 'epoch_*.npz')))

    if step_files:
        files = step_files
        id_label = 'Step'
        def _parse_id(fname):
            return int(re.search(r'step_(\d+)', os.path.basename(fname)).group(1))
    elif epoch_files:
        files = epoch_files
        id_label = 'Epoch'
        def _parse_id(fname):
            return int(os.path.splitext(os.path.basename(fname))[0].split('_')[1])
    else:
        assert False, f'No step_*.npz or epoch_*.npz found in {probe_dir}'

    ids, feats, txt_feats_list, proj_feats_list = [], [], [], []
    has_txt  = None
    has_proj = None
    for f in files:
        ids.append(_parse_id(f))
        data = np.load(f)
        # features = backbone CLS (primary; used for aniso + geometry)
        feats.append(data['features'])
        if 'txt_features' in data:
            txt_feats_list.append(data['txt_features'])
            has_txt = True
        else:
            has_txt = False
        # proj_features = projected CLIP space (PE-Core only; for step_evolution modality gap)
        if 'proj_features' in data:
            proj_feats_list.append(data['proj_features'])
            has_proj = True
        else:
            has_proj = False
        logging.info(f'  {id_label} {ids[-1]:>6d}: bb_cls={feats[-1].shape}'
                     + (f'  proj={proj_feats_list[-1].shape}' if has_proj and proj_feats_list else '')
                     + (f'  txt={txt_feats_list[-1].shape}'   if has_txt  and txt_feats_list  else ''))

    txt_feats  = txt_feats_list  if (has_txt  and len(txt_feats_list)  == len(feats)) else None
    proj_feats = proj_feats_list if (has_proj and len(proj_feats_list) == len(feats)) else None
    if txt_feats is None:
        logging.info('[epochs] no txt_features in npz — image-only step_evolution')
    if proj_feats is None:
        logging.info('[epochs] no proj_features in npz — step_evolution uses backbone CLS')

    # Place plots at <log_root>/probe/plots  (sibling of checkpoints/)
    out = os.path.join(probe_dir, '..', '..', 'probe', 'plots')
    os.makedirs(out, exist_ok=True)
    # step_evolution GIF: prefer projected CLIP space (modality gap visible there)
    # fall back to backbone CLS when proj_features not present
    evo_feats = proj_feats if proj_feats is not None else feats
    plot_evolution(evo_feats, ids, out, n_traj=args.n_traj, id_label=id_label,
                   txt_feats=txt_feats)

    # ── Anisotropy evolution (backbone CLS — geometry of the VLM-usable features) ──
    logging.info(f'[epochs] computing anisotropy for {len(feats)} checkpoints...')
    aniso_list = [compute_anisotropy(f) for f in feats]
    plot_aniso_evolution(ids, aniso_list,
                         os.path.join(out, 'aniso_evolution.png'),
                         id_label=id_label)
    m0, m1 = aniso_list[0], aniso_list[-1]
    logging.info(f'[epochs] backbone CLS  EffRank {m0["effective_rank"]:.1f} → {m1["effective_rank"]:.1f}'
                 f'  top4% {m0["pct_var_top4"]:.1f} → {m1["pct_var_top4"]:.1f}'
                 f'  AvgCos {m0["avg_cos_sim"]:.4f} → {m1["avg_cos_sim"]:.4f}')

    # ── Export CSV for downstream analysis ───────────────────────────────────
    csv_path = os.path.join(out, 'aniso_evolution.csv')
    _scalar_keys = [k for k, v in aniso_list[0].items() if isinstance(v, (int, float))]
    fieldnames   = [id_label.lower()] + _scalar_keys
    with open(csv_path, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for sid, row in zip(ids, aniso_list):
            w.writerow({id_label.lower(): sid,
                        **{k: row[k] for k in _scalar_keys}})
    logging.info(f'[epochs] data → {csv_path}')

    # ── PC pairs scatter (final checkpoint, image + text if available) ────────
    _plot_final_pc_pairs(
        evo_feats[-1], n_pcs=20,
        step_id=ids[-1], id_label=id_label, plots_dir=out,
        txt_feats=txt_feats[-1] if txt_feats is not None else None,
    )


# ── Mode: crop_probe ──────────────────────────────────────────────────────────

def run_crop_probe(args):
    """Generate & visualise DINOv3-style global/local crop features for FPS samples.

    Reads cached full-distribution npzs (produced by pretrained mode) to build
    the background cloud, then reloads each model to extract features for only
    5×3=15 small crops.  Results are cached in crop_probe.npz so subsequent
    runs skip model loading.

    Output: <out_dir>/pretrained/crop_probe.png
    """
    out = os.path.join(args.out_dir, 'pretrained')

    # ── Load full distributions from cache ────────────────────────────────────
    dino_npz_path = os.path.join(out, 'dinov3_img.npz')
    assert os.path.exists(dino_npz_path), \
        f'DINOv3 cache missing: {dino_npz_path}\nRun pretrained mode first.'
    dino_npz = np.load(dino_npz_path)
    dino_img = dino_npz['features']

    # Resolve image paths (saved in npz by tsv-mode extractor; fall back to TSV)
    if 'paths' in dino_npz:
        all_paths = dino_npz['paths'].tolist()
    else:
        df = pd.read_csv(args.data, sep='\t')
        all_paths = df['filepath'].tolist()
        if len(all_paths) != len(dino_img):
            raise RuntimeError(
                f'crop_probe requires image paths on disk (tsv/COCO mode).\n'
                f'  Feature count : {len(dino_img)}\n'
                f'  TSV row count : {len(all_paths)}\n'
                f'This cache was built from a wds/tar dataset whose images are '
                f'not individually accessible. Run crop_probe against the COCO '
                f'feature cache instead:\n'
                f'  bash analysis/probe.sh crop_probe'
            )
        logging.warning('[crop_probe] paths not in npz — using TSV order (assumes same)')

    # Recompute FPS indices (deterministic — same seed as run_pretrained)
    fps_idx = fps_sample(dino_img, k=5)
    fps_paths = [all_paths[i] for i in fps_idx]
    logging.info(f'[crop_probe] FPS indices: {fps_idx.tolist()}')

    # Load all available full-distribution features
    npz_map = {
        'DINOv3':  'dinov3_img.npz',  'PE-Core':  'pe_core_img.npz',
        'SigLIP2': 'siglip2_img.npz', 'RADIO':    'radio_img.npz',
        'EUPE':    'eupe_img.npz',     'TIPSv2':   'tips_img.npz',
    }
    img_feats = {}
    for name, fname in npz_map.items():
        p = os.path.join(out, fname)
        if os.path.exists(p):
            img_feats[name] = np.load(p)['features']
        else:
            logging.info(f'[crop_probe] skip {name}: cache not found')

    # ── Generate DINOv3-style crops ───────────────────────────────────────────
    logging.info('[crop_probe] generating global + local crops ...')
    orig_imgs, gc_imgs, lc_imgs = make_dino_crops(fps_paths, seed=42)

    # ── Crop feature cache ────────────────────────────────────────────────────
    # Key scheme: <model_slug>_orig / <model_slug>_gc / <model_slug>_lc
    _SLUG = {'DINOv3': 'dinov3', 'PE-Core': 'pecore', 'SigLIP2': 'siglip2',
             'RADIO': 'radio',  'EUPE': 'eupe',       'TIPSv2': 'tips'}
    crop_cache_path = os.path.join(out, 'crop_probe.npz')

    crops_feats = {}
    save_kwargs = {}

    if not args.force and os.path.exists(crop_cache_path):
        logging.info('[crop_probe] loading crop features from cache')
        cd = np.load(crop_cache_path)
        for name in img_feats:
            slug = _SLUG[name]
            keys = (f'{slug}_orig', f'{slug}_gc', f'{slug}_lc')
            if all(k in cd for k in keys):
                crops_feats[name] = {'orig': cd[f'{slug}_orig'],
                                     'global': cd[f'{slug}_gc'],
                                     'local':  cd[f'{slug}_lc']}
            else:
                logging.info(f'[crop_probe] {name} missing from cache — will recompute')

    missing = [n for n in img_feats if n not in crops_feats]
    if missing:
        logging.info(f'[crop_probe] extracting crops for: {missing}')

        def _store(name, orig_f, gc_f, lc_f):
            crops_feats[name] = {'orig': orig_f, 'global': gc_f, 'local': lc_f}
            slug = _SLUG[name]
            save_kwargs.update({f'{slug}_orig': orig_f,
                                 f'{slug}_gc':   gc_f,
                                 f'{slug}_lc':   lc_f})

        if 'DINOv3' in missing:
            logging.info('[crop_probe]   DINOv3 ...')
            dn = load_dinov3(args.dino_repo, args.dino_ckpt)
            _store('DINOv3',
                   extract_dinov3_pil(dn, orig_imgs),
                   extract_dinov3_pil(dn, gc_imgs),
                   extract_dinov3_pil(dn, lc_imgs))
            del dn; torch.cuda.empty_cache()

        if 'PE-Core' in missing:
            logging.info('[crop_probe]   PE-Core ...')
            pe_m, pe_p, _ = load_pe_core(args.pe_ckpt)
            _store('PE-Core',
                   extract_pe_core_pil_raw(pe_m, pe_p, orig_imgs),
                   extract_pe_core_pil_raw(pe_m, pe_p, gc_imgs),
                   extract_pe_core_pil_raw(pe_m, pe_p, lc_imgs))
            del pe_m; torch.cuda.empty_cache()

        if 'SigLIP2' in missing:
            logging.info('[crop_probe]   SigLIP2 ...')
            s2_m, s2_p, _ = load_siglip2(args.sig2_ckpt)
            _store('SigLIP2',
                   extract_clip_pil(s2_m, s2_p, orig_imgs),
                   extract_clip_pil(s2_m, s2_p, gc_imgs),
                   extract_clip_pil(s2_m, s2_p, lc_imgs))
            del s2_m; torch.cuda.empty_cache()

        if 'RADIO' in missing:
            logging.info('[crop_probe]   RADIO ...')
            ra, ra_c = load_radio(args.radio)
            _store('RADIO',
                   extract_radio_pil(ra, ra_c, orig_imgs),
                   extract_radio_pil(ra, ra_c, gc_imgs),
                   extract_radio_pil(ra, ra_c, lc_imgs))
            del ra; torch.cuda.empty_cache()

        if 'EUPE' in missing:
            logging.info('[crop_probe]   EUPE ...')
            eu = load_eupe(args.eupe_repo, args.eupe_ckpt)
            if eu is not None:
                _store('EUPE',
                       extract_eupe_pil(eu, orig_imgs),
                       extract_eupe_pil(eu, gc_imgs),
                       extract_eupe_pil(eu, lc_imgs))
                del eu; torch.cuda.empty_cache()

        if 'TIPSv2' in missing:
            logging.info('[crop_probe]   TIPSv2 ...')
            ti_m, _ = load_tips(args.tips)
            _store('TIPSv2',
                   extract_tips_pil(ti_m, orig_imgs),
                   extract_tips_pil(ti_m, gc_imgs),
                   extract_tips_pil(ti_m, lc_imgs))
            del ti_m; torch.cuda.empty_cache()

        # Merge with any previously cached entries and save
        if os.path.exists(crop_cache_path):
            existing = dict(np.load(crop_cache_path))
            existing.update(save_kwargs)
            save_kwargs = existing
        np.savez_compressed(crop_cache_path, **save_kwargs)
        logging.info(f'[crop_probe] saved → {crop_cache_path}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    out_fig = os.path.join(out, 'crop_probe.png')
    plot_crop_probe(img_feats, crops_feats, fps_idx, out_fig)
