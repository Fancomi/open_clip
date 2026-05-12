"""Single-model GPU worker — spawned by pipeline.py for parallel extraction.

Called as:
  CUDA_VISIBLE_DEVICES=<id> python -m analysis._worker \
      --model <name> --out-dir <dir> --data-type <tsv|wds> \
      [--data <path>] [--force] [--max-samples N] \
      [--pe-ckpt ...] [--sig2-ckpt ...] [--dino-ckpt ...] \
      [--radio ...] [--eupe-ckpt ...] [--tips ...]

Supported model names:
  pe_img, pe_txt, sig2_img, sig2_txt,
  dino_img, radio_img, eupe_img, tips_img, tips_txt

Each worker loads exactly one model, extracts its npz, then exits.
"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

# NOTE: models.DEVICE is evaluated at import time.
# Since CUDA_VISIBLE_DEVICES is set BEFORE this process starts, DEVICE=cuda:0
# refers to the physical GPU assigned to this worker.
from .models import (CKPT, DEVICE,
                     load_pe_core, load_siglip2, load_dinov3,
                     load_radio, load_eupe, load_tips)
from .extractors import (
    extract_clip_img, extract_clip_txt,
    extract_pe_core_img_raw,
    extract_dinov3_img, extract_radio_img,
    extract_eupe_img, extract_tips_img, extract_tips_txt,
    _DINO_TF, _RADIO_TF, _EUPE_TF, _TIPS_TF, _TIPS_SZ, _TIPS_CHUNK,
)

_WDS_BS = 64   # wds streaming batch size per worker


def _do_tsv(model_name, args):
    df    = pd.read_csv(args.data, sep='\t')
    paths = df['filepath'].tolist()
    caps  = df['caption'].tolist() if 'caption' in df.columns else []
    out   = os.path.join(args.out_dir, 'pretrained')
    os.makedirs(out, exist_ok=True)
    force = args.force

    if model_name == 'pe_img':
        pe_m, pe_p, pe_t = load_pe_core(args.pe_ckpt)
        extract_clip_img(pe_m, paths, pe_p,
                         os.path.join(out, 'pe_core_img.npz'), force)
        extract_pe_core_img_raw(pe_m, paths, pe_p,
                                os.path.join(out, 'pe_core_img_raw.npz'), force)
        if caps:
            extract_clip_txt(pe_m, pe_t, caps,
                             os.path.join(out, 'pe_core_txt.npz'), force)
    elif model_name == 'sig2_img':
        s2_m, s2_p, s2_t = load_siglip2(args.sig2_ckpt)
        extract_clip_img(s2_m, paths, s2_p,
                         os.path.join(out, 'siglip2_img.npz'), force)
        extract_clip_txt(s2_m, s2_t, caps,
                         os.path.join(out, 'siglip2_txt.npz'), force)
    elif model_name == 'dino_img':
        dn = load_dinov3(args.dino_repo, args.dino_ckpt)
        extract_dinov3_img(dn, paths, os.path.join(out, 'dinov3_img.npz'), force)
    elif model_name == 'radio_img':
        ra, ra_c = load_radio(args.radio)
        extract_radio_img(ra, ra_c, paths,
                          os.path.join(out, 'radio_img.npz'), force)
    elif model_name == 'eupe_img':
        eu = load_eupe(args.eupe_repo, args.eupe_ckpt)
        if eu is not None:
            extract_eupe_img(eu, paths, os.path.join(out, 'eupe_img.npz'), force)
    elif model_name == 'tips_img':
        ti_m, ti_t = load_tips(args.tips)
        extract_tips_img(ti_m, paths, os.path.join(out, 'tips_img.npz'), force)
        extract_tips_txt(ti_m, ti_t, caps,
                         os.path.join(out, 'tips_txt.npz'), force)
    else:
        raise ValueError(f'Unknown model: {model_name}')


def _do_wds(model_name, args):
    import webdataset as wds
    from PIL import Image

    out = os.path.join(args.out_dir, 'pretrained')
    os.makedirs(out, exist_ok=True)
    force = args.force

    # npz paths for this model
    npz_map = {
        'pe_img':    ('pe_core_img.npz', 'pe_core_txt.npz'),
        'sig2_img':  ('siglip2_img.npz', 'siglip2_txt.npz'),
        'dino_img':  ('dinov3_img.npz',  None),
        'radio_img': ('radio_img.npz',   None),
        'eupe_img':  ('eupe_img.npz',    None),
        'tips_img':  ('tips_img.npz',    'tips_txt.npz'),
    }
    img_fname, txt_fname = npz_map[model_name]
    img_path = os.path.join(out, img_fname)
    txt_path = os.path.join(out, txt_fname) if txt_fname else None

    # Cache hit check
    if not force:
        hit = os.path.exists(img_path) and (txt_path is None or os.path.exists(txt_path))
        if hit:
            logging.info(f'[worker/{model_name}] cache hit — skip')
            return

    # Load model
    if model_name == 'pe_img':
        pe_m, pe_p, pe_t = load_pe_core(args.pe_ckpt)
    elif model_name == 'sig2_img':
        s2_m, s2_p, s2_t = load_siglip2(args.sig2_ckpt)
    elif model_name == 'dino_img':
        dn = load_dinov3(args.dino_repo, args.dino_ckpt)
    elif model_name == 'radio_img':
        ra, ra_c = load_radio(args.radio)
    elif model_name == 'eupe_img':
        eu = load_eupe(args.eupe_repo, args.eupe_ckpt)
        if eu is None:
            logging.warning('[worker/eupe_img] EUPE unavailable — skip')
            return
    elif model_name == 'tips_img':
        ti_m, ti_t = load_tips(args.tips)
    else:
        raise ValueError(f'Unknown wds model: {model_name}')

    img_acc, txt_acc = [], []
    count = 0

    @torch.no_grad()
    def _flush(imgs, caps):
        if model_name == 'pe_img':
            pb = torch.stack([pe_p(im) for im in imgs]).to(DEVICE)
            img_acc.append(
                pe_m.encode_image(pb, normalize=True).detach().cpu().float().numpy())
            txt_acc.append(
                pe_m.encode_text(pe_t(caps).to(DEVICE), normalize=True)
                .detach().cpu().float().numpy())
        elif model_name == 'sig2_img':
            sb = torch.stack([s2_p(im) for im in imgs]).to(DEVICE)
            img_acc.append(
                s2_m.encode_image(sb, normalize=True).detach().cpu().float().numpy())
            txt_acc.append(
                s2_m.encode_text(s2_t(caps).to(DEVICE), normalize=True)
                .detach().cpu().float().numpy())
        elif model_name == 'dino_img':
            dx = torch.stack([_DINO_TF(im) for im in imgs]).to(DEVICE)
            img_acc.append(
                dn.forward_features(dx)['x_norm_clstoken'].cpu().float().numpy())
        elif model_name == 'radio_img':
            rx = torch.stack([_RADIO_TF(im) for im in imgs]).to(DEVICE)
            if ra_c is not None:
                rx = ra_c(rx)
            out_r = ra(rx)
            s = (out_r[0] if isinstance(out_r, (tuple, list))
                 else getattr(out_r, 'summary', out_r[0]))
            img_acc.append(s.cpu().float().numpy())
        elif model_name == 'eupe_img':
            ex = torch.stack([_EUPE_TF(im) for im in imgs]).to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                enabled=(DEVICE.type != 'cpu')):
                eout = eu.forward_features(ex)
            img_acc.append(eout['x_norm_clstoken'].cpu().float().numpy())
        elif model_name == 'tips_img':
            tiles = [_TIPS_TF(im.resize((_TIPS_SZ, _TIPS_SZ), Image.BICUBIC))
                     for im in imgs]
            chunks = []
            for ci in range(0, len(tiles), _TIPS_CHUNK):
                tx = torch.stack(tiles[ci:ci + _TIPS_CHUNK]).to(DEVICE)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.bfloat16,
                                    enabled=(DEVICE.type != 'cpu')):
                    tout = ti_m.encode_image(tx)
                chunks.append(tout.cls_token.squeeze(1).float().cpu().numpy())
            img_acc.append(np.concatenate(chunks))
            ids, pads = ti_t.tokenize(caps, max_len=ti_m.config.max_len)
            ids  = torch.from_numpy(ids).to(DEVICE)
            pads = torch.from_numpy(pads).to(DEVICE)
            txt_acc.append(
                F.normalize(ti_m.encode_text(ids, pads), dim=-1)
                .cpu().float().numpy())

    ds = (wds.WebDataset(args.data, shardshuffle=False)
          .decode('pil').to_tuple('jpg', 'txt'))
    buf_imgs, buf_caps = [], []
    for img, cap in ds:
        buf_imgs.append(img); buf_caps.append(cap); count += 1
        if len(buf_imgs) == _WDS_BS:
            _flush(buf_imgs, buf_caps)
            buf_imgs, buf_caps = [], []
            logging.info(f'  [worker/{model_name}] {count}/{args.max_samples}')
        if count >= args.max_samples:
            break
    if buf_imgs:
        _flush(buf_imgs, buf_caps)

    if img_acc:
        img_feat = np.concatenate(img_acc)
        np.savez_compressed(img_path, features=img_feat)
        logging.info(f'[worker/{model_name}] saved {img_path}  shape={img_feat.shape}')
    if txt_acc:
        txt_feat = np.concatenate(txt_acc)
        np.savez_compressed(txt_path, features=txt_feat)
        logging.info(f'[worker/{model_name}] saved {txt_path}  shape={txt_feat.shape}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model',        required=True)
    p.add_argument('--data-type',    choices=['tsv', 'wds'], default='tsv')
    p.add_argument('--data',         required=True)
    p.add_argument('--out-dir',      required=True)
    p.add_argument('--force',        action='store_true')
    p.add_argument('--max-samples',  type=int, default=100_000)
    p.add_argument('--pe-ckpt',      default=CKPT['pe_core'])
    p.add_argument('--sig2-ckpt',    default=CKPT['siglip2'])
    p.add_argument('--dino-repo',    default=CKPT['dino_repo'])
    p.add_argument('--dino-ckpt',    default=CKPT['dino_ckpt'])
    p.add_argument('--radio',        default=CKPT['radio'])
    p.add_argument('--eupe-repo',    default=CKPT['eupe_repo'])
    p.add_argument('--eupe-ckpt',    default=CKPT['eupe_ckpt'])
    p.add_argument('--tips',         default=CKPT['tips'])
    args = p.parse_args()

    if args.data_type == 'tsv':
        _do_tsv(args.model, args)
    else:
        _do_wds(args.model, args)


if __name__ == '__main__':
    main()
