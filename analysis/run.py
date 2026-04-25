"""CLI entry point for feature-space analysis.

Usage (from repo root):
  python -m analysis.run --mode coco
  python -m analysis.run --mode cc3m
  python -m analysis.run --mode overlap
  python -m analysis.run --mode anisotropy [--aniso-dir <dir>]
  python -m analysis.run --mode epochs --probe-dir <dir>
"""
import argparse, logging, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')

from .models   import CKPT
from .pipeline import _DATA, run_pretrained, run_overlap, run_anisotropy, run_epochs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['pretrained', 'epochs', 'overlap', 'anisotropy'])
    # Data
    p.add_argument('--data',         default=_DATA['data'])
    p.add_argument('--data-type',    choices=['tsv', 'wds'], default='tsv')
    p.add_argument('--out-dir',      default=_DATA['out_dir'])
    p.add_argument('--max-samples',  type=int, default=100_000)
    p.add_argument('--n-pca',        type=int, default=4)
    p.add_argument('--force',        action='store_true')
    # Model checkpoints (all default to CKPT values)
    p.add_argument('--pe-ckpt',      default=CKPT['pe_core'])
    p.add_argument('--sig2-ckpt',    default=CKPT['siglip2'])
    p.add_argument('--dino-repo',    default=CKPT['dino_repo'])
    p.add_argument('--dino-ckpt',    default=CKPT['dino_ckpt'])
    p.add_argument('--radio',        default=CKPT['radio'])
    p.add_argument('--eupe-repo',    default=CKPT['eupe_repo'])
    p.add_argument('--eupe-ckpt',    default=CKPT['eupe_ckpt'])
    p.add_argument('--tips',         default=CKPT['tips'])
    p.add_argument('--fps-model',     default='DINOv3',
                   help='Which model space to use for FPS anchor selection '
                        '(default: DINOv3)')
    # Mode-specific
    p.add_argument('--probe-dir',    default=None)
    p.add_argument('--n-traj',       type=int, default=100)
    p.add_argument('--coco-dir',     default=_DATA['coco_dir'])
    p.add_argument('--cc3m-dir',     default=_DATA['cc3m_dir'])
    p.add_argument('--aniso-dir',    default=_DATA['coco_dir'])
    args = p.parse_args()

    if   args.mode == 'pretrained': run_pretrained(args)
    elif args.mode == 'epochs':
        assert args.probe_dir, '--probe-dir required for epochs mode'
        run_epochs(args)
    elif args.mode == 'overlap':    run_overlap(args)
    elif args.mode == 'anisotropy': run_anisotropy(args)


if __name__ == '__main__':
    main()
