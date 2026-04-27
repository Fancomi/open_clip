"""Analysis pipeline modes: pretrained (COCO/CC3M), overlap, anisotropy, epochs."""
import glob, logging, os, re
import numpy as np
import pandas as pd
import torch

from .models   import CKPT, load_pe_core, load_siglip2, load_dinov3, \
                      load_radio, load_eupe, load_tips
from .extractors import (load_from_cache,
                          extract_clip_img, extract_clip_txt,
                          extract_dinov3_img, extract_radio_img,
                          extract_eupe_img, extract_tips_img, extract_tips_txt,
                          extract_wds_features)
from .metrics  import fps_sample, compute_anisotropy
from .viz      import plot_scatter, plot_overlap, plot_anisotropy, plot_aniso_evolution, plot_evolution

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
        else:
            df = pd.read_csv(args.data, sep='\t')
            paths, caps = df['filepath'].tolist(), df['caption'].tolist()

            pe_m, pe_p, pe_t = load_pe_core(args.pe_ckpt)
            pe_img = extract_clip_img(pe_m, paths, pe_p, os.path.join(out,'pe_core_img.npz'), args.force)
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
    def _modality_gap(img, txt, model_name, out_path):
        """PCA shared on img+txt, then plot_overlap for high-contrast two-color view."""
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2).fit(np.concatenate([img, txt]))
        pa  = pca.transform(img)
        pb  = pca.transform(txt)
        dist = float(np.linalg.norm(pa.mean(0) - pb.mean(0)))
        plot_overlap(pa, pb, f'{model_name} Image', f'{model_name} Text',
                     model_name, out_path, a_on_top=True, centroid_dist=dist)

    _modality_gap(pe_img,   pe_txt,   'PE-Core',  os.path.join(out, 'pe_core_modality_gap.png'))
    _modality_gap(sig2_img, sig2_txt, 'SigLIP2',  os.path.join(out, 'siglip2_modality_gap.png'))
    if tips_img is not None and tips_txt is not None:
        _modality_gap(tips_img, tips_txt, 'TIPSv2', os.path.join(out, 'tips_modality_gap.png'))

    # ── All-model image comparison + FPS tracking ───────────────────────────
    img_feats = {k: v for k, v in [
        ('DINOv3', dino_img), ('RADIO', radio_img), ('EUPE', eupe_img),
        ('TIPSv2', tips_img), ('PE-Core', pe_img),  ('SigLIP2', sig2_img),
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

    ids, feats = [], []
    for f in files:
        ids.append(_parse_id(f))
        feats.append(np.load(f)['features'])
        logging.info(f'  {id_label} {ids[-1]:>6d}: {feats[-1].shape}')

    out = os.path.join(probe_dir, 'plots')
    os.makedirs(out, exist_ok=True)
    plot_evolution(feats, ids, out, n_traj=args.n_traj, id_label=id_label)

    # ── Anisotropy evolution ────────────────────────────────────────────────
    logging.info(f'[epochs] computing anisotropy for {len(feats)} checkpoints...')
    aniso_list = [compute_anisotropy(f) for f in feats]
    plot_aniso_evolution(ids, aniso_list,
                         os.path.join(out, 'aniso_evolution.png'),
                         id_label=id_label)
    # Log final vs initial delta
    m0, m1 = aniso_list[0], aniso_list[-1]
    logging.info(f'[epochs] EffRank  {m0["effective_rank"]:.1f} → {m1["effective_rank"]:.1f}')
    logging.info(f'[epochs] top4%    {m0["pct_var_top4"]:.1f} → {m1["pct_var_top4"]:.1f}')
    logging.info(f'[epochs] AvgCos   {m0["avg_cos_sim"]:.4f} → {m1["avg_cos_sim"]:.4f}')
