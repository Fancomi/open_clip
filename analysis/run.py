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

_BASE = '/root/paddlejob/workspace/env_run/penghaotian'
_DATA = dict(
    data=f'{_BASE}/datas/coco/annotations/karpathy_1cap.tsv',
    out_dir=f'{_BASE}/datas/coco/feature_probe',
    coco_dir=f'{_BASE}/datas/coco/feature_probe/pretrained',
    cc3m_dir=f'{_BASE}/datas/LLaVA-ReCap-CC3M/feature_probe/pretrained',
)
_CKPT = dict(
    pe_core=f'{_BASE}/models/timm/PE-Core-B-16/open_clip_model.safetensors',
    siglip2=f'{_BASE}/models/timm/ViT-B-16-SigLIP2/open_clip_model.safetensors',
    dino_repo=f'{_BASE}/vision_encoder/dinov3',
    dino_ckpt=f'{_BASE}/models/dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    radio=f'{_BASE}/models/C-RADIOv4-SO400M',
    eupe_repo=f'{_BASE}/vision_encoder/EUPE',
    eupe_ckpt=f'{_BASE}/models/EUPE-ViT-B/EUPE-ViT-B.pt',
    tips=f'{_BASE}/models/tipsv2-b14',
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', required=True,
                   choices=['pretrained', 'epochs', 'overlap', 'anisotropy',
                            'pc_alignment', 'crop_probe', 'eval_pretrained',
                            'make_slot_input', 'collect_slots', 'overlay_slots'])
    # Data
    p.add_argument('--data',         default=_DATA['data'])
    p.add_argument('--data-type',    choices=['tsv', 'wds'], default='tsv')
    p.add_argument('--out-dir',      default=_DATA['out_dir'])
    p.add_argument('--max-samples',  type=int, default=100_000)
    p.add_argument('--n-pca',        type=int, default=4)
    p.add_argument('--force',        action='store_true')
    p.add_argument('--csv-separator', default='\t')
    p.add_argument('--img-key',      default='filepath')
    p.add_argument('--caption-key',  default='caption')
    # Model checkpoints (all default to _CKPT values)
    p.add_argument('--pe-ckpt',      default=_CKPT['pe_core'])
    p.add_argument('--sig2-ckpt',    default=_CKPT['siglip2'])
    p.add_argument('--dino-repo',    default=_CKPT['dino_repo'])
    p.add_argument('--dino-ckpt',    default=_CKPT['dino_ckpt'])
    p.add_argument('--radio',        default=_CKPT['radio'])
    p.add_argument('--eupe-repo',    default=_CKPT['eupe_repo'])
    p.add_argument('--eupe-ckpt',    default=_CKPT['eupe_ckpt'])
    p.add_argument('--tips',         default=_CKPT['tips'])
    p.add_argument('--fps-model',     default='DINOv3',
                   help='Which model space to use for FPS anchor selection '
                        '(default: DINOv3)')
    # Mode-specific
    p.add_argument('--probe-dir',    default=None)
    p.add_argument('--n-traj',       type=int, default=100)
    p.add_argument('--coco-dir',     default=_DATA['coco_dir'])
    p.add_argument('--cc3m-dir',     default=_DATA['cc3m_dir'])
    p.add_argument('--aniso-dir',    default=_DATA['coco_dir'])
    p.add_argument('--n-pcs',        type=int, default=20,
                   help='PCs to compare in pc_alignment mode (default: 20)')
    p.add_argument('--eval-model',   default=None,
                   choices=['pe_core', 'siglip2'],
                   help='Pretrained model for eval_pretrained mode')
    # Slot analysis
    p.add_argument('--dataset',      default='coco')
    p.add_argument('--slot-out',     default=None)
    p.add_argument('--slots',        default=None)
    p.add_argument('--probe',        default=None)
    p.add_argument('--slot-types',   default=None)
    p.add_argument('--top-n',        type=int, default=50)
    p.add_argument('--hist-bins',    type=int, default=50)
    p.add_argument('--top-k',        type=int, default=5)
    p.add_argument('--bottom-k',     type=int, default=5)
    p.add_argument('--min-count',    type=int, default=1)
    p.add_argument('--metric',       choices=['density', 'curvature', 'both'], default='density')
    p.add_argument('--k',            type=int, default=50)
    p.add_argument('--feature-key',  default='auto')
    p.add_argument('--match-by',     choices=['filepath', 'row_index'], default='filepath')
    p.add_argument('--max-points-per-word', type=int, default=200)
    p.add_argument('--metric-max-points', type=int, default=0)
    p.add_argument('--background-max-points', type=int, default=0)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--limit',        type=int, default=None)
    p.add_argument('--min-match-rate', type=float, default=0.8)
    p.add_argument('--allow-low-match', action='store_true')
    p.add_argument('--non-strict-slots', action='store_true')
    p.add_argument('--model-name',   default=None)
    p.add_argument('--save-geometry-summary', action='store_true')
    args = p.parse_args()

    if args.mode == 'make_slot_input':
        assert args.slot_out, '--slot-out required for make_slot_input mode'
        from .slot_pipeline import run_make_slot_input
        run_make_slot_input(args)
    elif args.mode == 'collect_slots':
        assert args.slots, '--slots required for collect_slots mode'
        from .slot_pipeline import run_collect_slots
        run_collect_slots(args)
    elif args.mode == 'overlay_slots':
        assert args.slots, '--slots required for overlay_slots mode'
        assert args.probe, '--probe required for overlay_slots mode'
        from .slot_pipeline import run_overlay_slots
        run_overlay_slots(args)
    else:
        from .pipeline import (run_pretrained, run_overlap, run_anisotropy,
                               run_epochs, run_crop_probe, run_eval_pretrained)
        from .pc_alignment import run_pc_alignment
        if   args.mode == 'pretrained': run_pretrained(args)
        elif args.mode == 'epochs':
            assert args.probe_dir, '--probe-dir required for epochs mode'
            run_epochs(args)
        elif args.mode == 'overlap':    run_overlap(args)
        elif args.mode == 'anisotropy': run_anisotropy(args)
        elif args.mode == 'pc_alignment':
            assert args.probe_dir, '--probe-dir required for pc_alignment mode'
            run_pc_alignment(args)
        elif args.mode == 'crop_probe': run_crop_probe(args)
        elif args.mode == 'eval_pretrained':
            assert args.eval_model, '--eval-model required for eval_pretrained mode'
            run_eval_pretrained(args)


if __name__ == '__main__':
    main()
