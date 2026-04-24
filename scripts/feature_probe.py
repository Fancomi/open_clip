#!/usr/bin/env python3
"""
Feature space analysis for pretrained models (Exp1) and training dynamics (Exp2).

Exp1 - pretrained comparison (COCO):
    python feature_probe.py --mode pretrained

Exp1 - pretrained comparison (CC3M, wds streaming):
    python feature_probe.py --mode pretrained --data-type wds \
        --data '/path/to/wds/{00000..00280}.tar' \
        --out-dir /path/to/LLaVA-ReCap-CC3M/feature_probe \
        --max-samples 100000

Exp2 - epoch evolution (after training with --probe-data):
    python feature_probe.py --mode epochs --probe-dir LOGS/RUN/checkpoints/probe
"""
import argparse, glob, logging, os, sys
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from itertools import combinations
from PIL import Image
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_DEFAULTS = dict(
    data    = '/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations/karpathy_1cap.tsv',
    pe_ckpt = '/root/paddlejob/workspace/env_run/penghaotian/models/timm/PE-Core-B-16/open_clip_model.safetensors',
    sig2_ckpt = '/root/paddlejob/workspace/env_run/penghaotian/models/timm/ViT-B-16-SigLIP2/open_clip_model.safetensors',
    dino    = '/root/paddlejob/workspace/env_run/penghaotian/models/dino/dinov3-vitb16-pretrain-lvd1689m',
    out_dir = '/root/paddlejob/workspace/env_run/penghaotian/datas/coco/feature_probe',
    cc3m_data   = '/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds/{00000..00280}.tar',
    cc3m_out    = '/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/feature_probe',
)


# ─── Model loading ────────────────────────────────────────────────────────────

def load_pe_core(ckpt_path):
    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms('PE-Core-B-16', pretrained=ckpt_path)
    tok = open_clip.get_tokenizer('PE-Core-B-16')
    return model.eval().to(DEVICE), preproc, tok


def load_siglip2(ckpt_path):
    import open_clip
    model, _, preproc = open_clip.create_model_and_transforms('ViT-B-16-SigLIP2', pretrained=ckpt_path)
    tok = open_clip.get_tokenizer('ViT-B-16-SigLIP2')
    return model.eval().to(DEVICE), preproc, tok


def load_dinov3(model_path):
    from transformers import AutoImageProcessor, AutoModel
    proc  = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).eval().to(DEVICE)
    return model, proc


# ─── Feature extraction ───────────────────────────────────────────────────────

def _npz_cache(path, force):
    """Return cached features if path exists and not force, else None."""
    if not force and os.path.exists(path):
        logging.info(f'[cache] Loading {path}')
        return np.load(path)['features']
    return None


def extract_clip_img(model, paths, preproc, out_path, force=False, batch_size=256):
    """Extract image features from any open_clip model."""
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    from open_clip_train.probe_hook import extract_image_features
    feat = extract_image_features(model, paths, preproc, DEVICE, batch_size)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_clip_txt(model, tok, captions, out_path, force=False, batch_size=512):
    """Extract text features from any open_clip model."""
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(captions), batch_size):
        t = tok(captions[i:i+batch_size]).to(DEVICE)
        feats.append(model.encode_text(t, normalize=True).cpu().float().numpy())
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat)
    return feat


# keep old names as aliases for backward compat
extract_pe_img = extract_clip_img
extract_pe_txt = extract_clip_txt


@torch.no_grad()
def extract_dino_img(model, proc, paths, out_path, force=False, batch_size=64):
    feat = _npz_cache(out_path, force)
    if feat is not None:
        return feat
    feats = []
    for i in range(0, len(paths), batch_size):
        imgs = [Image.open(p).convert('RGB') for p in paths[i:i+batch_size]]
        inputs = {k: v.to(DEVICE) for k, v in proc(images=imgs, return_tensors='pt').items()}
        cls = F.normalize(model(**inputs).last_hidden_state[:, 0, :], dim=-1)
        feats.append(cls.cpu().float().numpy())
    feat = np.concatenate(feats, 0)
    np.savez_compressed(out_path, features=feat, paths=np.array(paths))
    return feat


@torch.no_grad()
def extract_wds_features(pe_model, pe_preproc, pe_tok,
                         sig2_model, sig2_preproc, sig2_tok,
                         dino_model, dino_proc,
                         pattern, out_dir, max_samples=100000, force=False,
                         batch_size=128):
    """Stream wds, extract PE-Core / SigLIP2 (img+txt) + DINOv3 (img) features."""
    import webdataset as wds

    paths = {
        'pe_img':   os.path.join(out_dir, 'pe_core_img.npz'),
        'pe_txt':   os.path.join(out_dir, 'pe_core_txt.npz'),
        'sig2_img': os.path.join(out_dir, 'siglip2_img.npz'),
        'sig2_txt': os.path.join(out_dir, 'siglip2_txt.npz'),
        'dino_img': os.path.join(out_dir, 'dinov3_img.npz'),
    }
    if not force and all(os.path.exists(p) for p in paths.values()):
        logging.info('[cache] Loading all wds features from cache ...')
        return {k: np.load(p)['features'] for k, p in paths.items()}

    bufs = {'pe': [], 'sig2': [], 'dino': [], 'cap': []}
    acc  = {k: [] for k in paths}
    count = 0

    ds = (wds.WebDataset(pattern, shardshuffle=False)
          .decode('pil')
          .to_tuple('jpg', 'txt'))

    def _flush():
        with torch.no_grad():
            # PE-Core image
            pb = torch.stack([pe_preproc(img) for img in bufs['pe']]).to(DEVICE)
            acc['pe_img'].append(pe_model.encode_image(pb, normalize=True).cpu().float().numpy())
            # SigLIP2 image
            sb = torch.stack([sig2_preproc(img) for img in bufs['sig2']]).to(DEVICE)
            acc['sig2_img'].append(sig2_model.encode_image(sb, normalize=True).cpu().float().numpy())
            # DINOv3 image
            db = dino_proc(images=bufs['dino'], return_tensors='pt')
            db = {k: v.to(DEVICE) for k, v in db.items()}
            cls = F.normalize(dino_model(**db).last_hidden_state[:, 0, :], dim=-1)
            acc['dino_img'].append(cls.cpu().float().numpy())
            # PE-Core text
            pt = pe_tok(bufs['cap']).to(DEVICE)
            acc['pe_txt'].append(pe_model.encode_text(pt, normalize=True).cpu().float().numpy())
            # SigLIP2 text
            st = sig2_tok(bufs['cap']).to(DEVICE)
            acc['sig2_txt'].append(sig2_model.encode_text(st, normalize=True).cpu().float().numpy())

    for img, cap in ds:
        bufs['pe'].append(img); bufs['sig2'].append(img); bufs['dino'].append(img)
        bufs['cap'].append(cap)
        count += 1
        if len(bufs['pe']) == batch_size:
            _flush()
            bufs = {k: [] for k in bufs}
            logging.info(f'  wds extracted {count}/{max_samples} ...')
        if count >= max_samples:
            break
    if bufs['pe']:
        _flush()

    result = {k: np.concatenate(v, 0) for k, v in acc.items()}
    for k, p in paths.items():
        np.savez_compressed(p, features=result[k])
    logging.info('  wds done: ' + ', '.join(f'{k} {v.shape}' for k, v in result.items()))
    return result


# ─── Visualization ────────────────────────────────────────────────────────────

def _fit_pca(feats_list, n):
    """Shared PCA if all dims equal, else independent per-model PCAs."""
    dims = [f.shape[1] for f in feats_list]
    if len(set(dims)) == 1:
        pca = PCA(n_components=n).fit(np.concatenate(feats_list, 0))
        return [pca] * len(feats_list), pca.explained_variance_ratio_
    else:
        pcas = [PCA(n_components=n).fit(f) for f in feats_list]
        return pcas, None   # independent → no shared variance to report


def plot_scatter(feats_dict, title, save_path, n_pca=4):
    """
    Multi-axis PCA grid: all PC pairs + explained variance bar.
    Same-dim  → shared PCA space (can compare positions).
    Diff-dim  → independent PCA per model, side-by-side rows.
    """
    labels = list(feats_dict.keys())
    feats  = list(feats_dict.values())
    colors = cm.tab10(np.linspace(0, 0.45, len(labels)))
    pairs  = list(combinations(range(n_pca), 2))          # all (i,j) pairs
    pcas, shared_var = _fit_pca(feats, n_pca)
    projs  = [pca.transform(f) for pca, f in zip(pcas, feats)]
    shared = shared_var is not None

    ncols = len(pairs) + 1   # PC-pair cols + variance col

    if shared:
        # One row: all models overlaid per subplot
        fig, axes = plt.subplots(1, ncols, figsize=(4.5 * ncols, 4.5))
        for col, (pi, pj) in enumerate(pairs):
            ax = axes[col]
            for label, proj, c in zip(labels, projs, colors):
                ax.scatter(proj[:, pi], proj[:, pj], s=3, alpha=0.35,
                           color=c, rasterized=True,
                           label=label if col == 0 else '')
            ax.set_xlabel(f'PC{pi+1}'); ax.set_ylabel(f'PC{pj+1}')
            ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
            if col == 0:
                ax.legend(markerscale=4, fontsize=8, loc='best')
        # Variance bar
        ax = axes[-1]
        ax.bar(range(1, n_pca + 1), shared_var * 100, color='steelblue')
        ax.set_xlabel('Component'); ax.set_ylabel('Variance explained (%)')
        ax.set_title('Explained variance', fontsize=9)
        fig.suptitle(title, fontsize=12, y=1.01)
    else:
        # One row per model
        nrows = len(labels)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4.5 * nrows))
        axes = np.array(axes).reshape(nrows, ncols)
        for row, (label, pca, proj, feat, c) in enumerate(
                zip(labels, pcas, projs, feats, colors)):
            var = pca.explained_variance_ratio_
            for col, (pi, pj) in enumerate(pairs):
                ax = axes[row, col]
                ax.scatter(proj[:, pi], proj[:, pj], s=3, alpha=0.35,
                           color=c, rasterized=True)
                ax.set_xlabel(f'PC{pi+1}'); ax.set_ylabel(f'PC{pj+1}')
                ax.set_title(f'PC{pi+1} vs PC{pj+1}', fontsize=9)
                if col == 0:
                    ax.set_ylabel(f'{label}  (dim={feat.shape[1]})\nPC{pj+1}', fontsize=8)
            ax = axes[row, -1]
            ax.bar(range(1, n_pca + 1), var * 100, color=c)
            ax.set_xlabel('Component'); ax.set_ylabel('Var. explained (%)')
            ax.set_title(f'{label}  explained var.', fontsize=9)
        fig.suptitle(title + '  [independent PCA axes]', fontsize=12, y=1.01)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f'Saved: {save_path}')


def plot_evolution(epoch_feats, epoch_ids, save_dir, n_traj=100, seed=42):
    """Evolution grid + sample trajectory plot in shared PCA space.

    PCA is fit on the FINAL epoch only — the final representation serves as the
    stable reference; earlier epochs are projected onto it to show convergence.
    Both epoch_evolution and trajectory share the same PCA axes and axis limits.
    """
    # Fit PCA on final epoch → stable reference space
    pca   = PCA(n_components=2).fit(epoch_feats[-1])
    projs = [pca.transform(f) for f in epoch_feats]
    n     = len(epoch_ids)

    # Compute unified axis limits across all epochs
    all_pts = np.concatenate(projs, 0)
    pad = 0.05
    x0, x1 = all_pts[:, 0].min(), all_pts[:, 0].max()
    y0, y1 = all_pts[:, 1].min(), all_pts[:, 1].max()
    xp, yp = (x1 - x0) * pad, (y1 - y0) * pad
    xlim = (x0 - xp, x1 + xp)
    ylim = (y0 - yp, y1 + yp)

    # ── Distribution grid ────────────────────────────────────────────────────
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (eid, proj) in enumerate(zip(epoch_ids, projs)):
        axes[i].scatter(proj[:, 0], proj[:, 1], s=3, alpha=0.4,
                        c=np.arange(len(proj)), cmap='viridis', rasterized=True)
        axes[i].set_title(f'Epoch {eid}', fontsize=9)
        axes[i].set_xlim(xlim); axes[i].set_ylim(ylim)
        axes[i].axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Image Feature Distribution per Epoch  [PCA fitted on final epoch]',
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'epoch_evolution.png'), dpi=150)
    plt.close()
    logging.info(f'Saved: {os.path.join(save_dir, "epoch_evolution.png")}')

    # ── Trajectory ───────────────────────────────────────────────────────────
    # Same PCA / same axis limits as epoch_evolution.
    # Each trajectory line goes light → dark along the epoch axis so that
    # later (more meaningful) epochs are visually dominant.
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(len(epoch_feats[0]), size=min(n_traj, len(epoch_feats[0])), replace=False)
    colors = cm.tab20(np.linspace(0, 1, len(idx)))
    alphas = np.linspace(0.10, 1.00, n)   # early epochs = faint
    lws    = np.linspace(0.3,  1.8,  n)   # early epochs = thin

    fig, ax = plt.subplots(figsize=(8, 7))
    for si, color in zip(idx, colors):
        pts = np.array([p[si] for p in projs])
        for t in range(len(pts) - 1):
            ax.plot(pts[t:t+2, 0], pts[t:t+2, 1], '-',
                    color=color, alpha=float(alphas[t + 1]), lw=float(lws[t + 1]))
        # start: small faint dot; end: large bright star
        ax.scatter(pts[0, 0],  pts[0, 1],  color=color, s=12, zorder=3,
                   marker='o', alpha=float(alphas[0]))
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, s=40, zorder=4,
                   marker='*', alpha=1.0)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_title(
        f'Sample Trajectories  N={len(idx)}\n'
        f'o=start  ★=end  light→dark = early→late epoch',
        fontsize=10,
    )
    ax.set_xlabel('PC1  (final epoch)'); ax.set_ylabel('PC2  (final epoch)')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'trajectory.png'), dpi=150)
    plt.close()
    logging.info(f'Saved: {os.path.join(save_dir, "trajectory.png")}')


# ─── Modes ────────────────────────────────────────────────────────────────────

def run_pretrained(args):
    out = os.path.join(args.out_dir, 'pretrained')
    os.makedirs(out, exist_ok=True)

    if args.data_type == 'wds':
        logging.info('Loading PE-Core ...')
        pe_model, pe_preproc, pe_tok = load_pe_core(args.pe_ckpt)
        logging.info('Loading SigLIP2 ...')
        sig2_model, sig2_preproc, sig2_tok = load_siglip2(args.sig2_ckpt)
        logging.info('Loading DINOv3 ...')
        dino_model, dino_proc = load_dinov3(args.dino)
        feats = extract_wds_features(
            pe_model, pe_preproc, pe_tok,
            sig2_model, sig2_preproc, sig2_tok,
            dino_model, dino_proc,
            args.data, out, max_samples=args.max_samples, force=args.force,
        )
        pe_img, pe_txt     = feats['pe_img'],   feats['pe_txt']
        sig2_img, sig2_txt = feats['sig2_img'], feats['sig2_txt']
        dino_img           = feats['dino_img']
    else:
        # TSV mode (COCO)
        df = pd.read_csv(args.data, sep='\t')
        paths, captions = df['filepath'].tolist(), df['caption'].tolist()

        logging.info('Loading PE-Core ...')
        pe_model, pe_preproc, pe_tok = load_pe_core(args.pe_ckpt)
        logging.info(f'Extracting PE-Core image features  N={len(paths)} ...')
        pe_img = extract_clip_img(pe_model, paths, pe_preproc,
                                  os.path.join(out, 'pe_core_img.npz'), args.force)
        logging.info('Extracting PE-Core text features ...')
        pe_txt = extract_clip_txt(pe_model, pe_tok, captions,
                                  os.path.join(out, 'pe_core_txt.npz'), args.force)
        del pe_model; torch.cuda.empty_cache()

        logging.info('Loading SigLIP2 ...')
        sig2_model, sig2_preproc, sig2_tok = load_siglip2(args.sig2_ckpt)
        logging.info('Extracting SigLIP2 image features ...')
        sig2_img = extract_clip_img(sig2_model, paths, sig2_preproc,
                                    os.path.join(out, 'siglip2_img.npz'), args.force)
        logging.info('Extracting SigLIP2 text features ...')
        sig2_txt = extract_clip_txt(sig2_model, sig2_tok, captions,
                                    os.path.join(out, 'siglip2_txt.npz'), args.force)
        del sig2_model; torch.cuda.empty_cache()

        logging.info('Loading DINOv3 ...')
        dino_model, dino_proc = load_dinov3(args.dino)
        logging.info('Extracting DINOv3 image features ...')
        dino_img = extract_dino_img(dino_model, dino_proc, paths,
                                    os.path.join(out, 'dinov3_img.npz'), args.force)

    # Plot 1: PE-Core modality gap (image vs text)
    plot_scatter(
        {'PE-Core Image': pe_img, 'PE-Core Text': pe_txt},
        'PE-Core: Image vs Text Feature Distribution',
        os.path.join(out, 'pe_core_modality_gap.png'),
        n_pca=args.n_pca,
    )
    # Plot 2: SigLIP2 modality gap (image vs text)
    plot_scatter(
        {'SigLIP2 Image': sig2_img, 'SigLIP2 Text': sig2_txt},
        'SigLIP2: Image vs Text Feature Distribution',
        os.path.join(out, 'siglip2_modality_gap.png'),
        n_pca=args.n_pca,
    )
    # Plot 3: three-way image space comparison
    plot_scatter(
        {'DINOv3 Image': dino_img, 'PE-Core Image': pe_img, 'SigLIP2 Image': sig2_img},
        'DINOv3 vs PE-Core vs SigLIP2: Image Feature Distribution',
        os.path.join(out, 'image_3way.png'),
        n_pca=args.n_pca,
    )


def run_epochs(args):
    files = sorted(glob.glob(os.path.join(args.probe_dir, 'epoch_*.npz')))
    assert files, f'No epoch_*.npz found in {args.probe_dir}'
    epoch_ids, epoch_feats = [], []
    for f in files:
        eid = int(os.path.splitext(os.path.basename(f))[0].split('_')[1])
        epoch_ids.append(eid)
        epoch_feats.append(np.load(f)['features'])
        logging.info(f'  epoch {eid:02d}: {epoch_feats[-1].shape}')
    out = os.path.join(args.probe_dir, 'plots')
    os.makedirs(out, exist_ok=True)
    plot_evolution(epoch_feats, epoch_ids, out, n_traj=args.n_traj)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',        choices=['pretrained', 'epochs'], required=True)
    p.add_argument('--data',        default=_DEFAULTS['data'])
    p.add_argument('--data-type',   choices=['tsv', 'wds'], default='tsv')
    p.add_argument('--out-dir',     default=_DEFAULTS['out_dir'])
    p.add_argument('--pe-ckpt',     default=_DEFAULTS['pe_ckpt'])
    p.add_argument('--sig2-ckpt',   default=_DEFAULTS['sig2_ckpt'])
    p.add_argument('--dino',        default=_DEFAULTS['dino'])
    p.add_argument('--max-samples', type=int, default=100000, help='wds mode: subsample size')
    p.add_argument('--n-pca',       type=int, default=4,      help='number of PCA components')
    p.add_argument('--force',       action='store_true',       help='re-extract even if npz cached')
    p.add_argument('--probe-dir',   default=None,              help='epochs mode: dir with epoch_*.npz')
    p.add_argument('--n-traj',      type=int, default=100,     help='epochs mode: trajectory sample count')
    args = p.parse_args()

    if args.mode == 'pretrained':
        run_pretrained(args)
    else:
        assert args.probe_dir, '--probe-dir required for epochs mode'
        run_epochs(args)


if __name__ == '__main__':
    main()
