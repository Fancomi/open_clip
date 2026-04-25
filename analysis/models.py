"""Model loaders for all 6 vision encoders.

Each loader uses CKPT defaults; any argument overrides the default.
Return types:
  CLIP-type  (PE, SigLIP2) : (model, preproc, tokenizer)
  Vision-only (DINOv3, EUPE): model
  With aux    (RADIO)       : (model, conditioner)
  With text   (TIPSv2)      : (model, tokenizer)
"""
import os, sys, json, shutil, logging
import torch

DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_HF_CACHE = os.path.expanduser('~/.cache/huggingface/modules/transformers_modules')
_BASE     = '/root/paddlejob/workspace/env_run/penghaotian'

CKPT = dict(
    pe_core   = f'{_BASE}/models/timm/PE-Core-B-16/open_clip_model.safetensors',
    siglip2   = f'{_BASE}/models/timm/ViT-B-16-SigLIP2/open_clip_model.safetensors',
    dino_repo = f'{_BASE}/vision_encoder/dinov3',
    dino_ckpt = f'{_BASE}/models/dino/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
    radio     = f'{_BASE}/models/C-RADIOv4-SO400M',
    eupe_repo = f'{_BASE}/vision_encoder/EUPE',
    eupe_ckpt = f'{_BASE}/models/EUPE-ViT-B/EUPE-ViT-B.pt',
    tips      = f'{_BASE}/models/tipsv2-b14',
)


def load_pe_core(ckpt=None):
    import open_clip
    m, _, p = open_clip.create_model_and_transforms(
        'PE-Core-B-16', pretrained=ckpt or CKPT['pe_core'])
    return m.eval().to(DEVICE), p, open_clip.get_tokenizer('PE-Core-B-16')


def load_siglip2(ckpt=None):
    import open_clip
    m, _, p = open_clip.create_model_and_transforms(
        'ViT-B-16-SigLIP2', pretrained=ckpt or CKPT['siglip2'])
    return m.eval().to(DEVICE), p, open_clip.get_tokenizer('ViT-B-16-SigLIP2')


def load_dinov3(repo=None, ckpt=None):
    repo = repo or CKPT['dino_repo']
    ckpt = ckpt or CKPT['dino_ckpt']
    logging.info('Loading DINOv3 ...')
    m = torch.hub.load(repo, 'dinov3_vitb16', source='local', pretrained=False)
    m.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    return m.eval().to(DEVICE)


def load_radio(path=None):
    from transformers import AutoModel
    path = path or CKPT['radio']
    logging.info('Loading C-RADIOv4 ...')
    m = AutoModel.from_pretrained(path, trust_remote_code=True).eval().to(DEVICE)
    return m, getattr(m, 'input_conditioner', None)


def load_eupe(repo=None, ckpt=None):
    repo = repo or CKPT['eupe_repo']
    ckpt = ckpt or CKPT['eupe_ckpt']
    if not os.path.exists(os.path.join(repo, 'hubconf.py')):
        logging.warning(f'EUPE hubconf.py not found at {repo}')
        return None
    try:
        m = torch.hub.load(repo, 'eupe_vitb16', source='local',
                           weights=ckpt, trust_repo=True)
        return m.eval().to(DEVICE)
    except Exception as e:
        logging.warning(f'EUPE load failed: {e}')
        return None


def load_tips(path=None):
    """Returns (model, tokenizer). Tokenizer not needed for image-only tasks."""
    from safetensors.torch import load_file as sf_load
    path  = path or CKPT['tips']
    cache = os.path.join(_HF_CACHE, 'tipsv2_hyphen_b14')
    for f in ('image_encoder.py', 'text_encoder.py'):
        dst, src = os.path.join(cache, f), os.path.join(path, f)
        if not os.path.exists(dst) and os.path.exists(src):
            shutil.copy(src, dst)
    if _HF_CACHE not in sys.path:
        sys.path.insert(0, _HF_CACHE)
    from transformers_modules.tipsv2_hyphen_b14.configuration_tips import TIPSv2Config
    from transformers_modules.tipsv2_hyphen_b14.modeling_tips     import TIPSv2Model
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'tips_te', os.path.join(cache, 'text_encoder.py'))
    te = importlib.util.module_from_spec(spec); spec.loader.exec_module(te)
    tok   = te.Tokenizer(os.path.join(path, 'tokenizer.model'))
    raw   = json.load(open(os.path.join(path, 'config.json')))
    skip  = {'_name_or_path', 'transformers_version', 'auto_map',
             'architectures', 'model_type', 'torch_dtype'}
    cfg   = TIPSv2Config(**{k: v for k, v in raw.items() if k not in skip})
    logging.info('Loading TIPSv2 ...')
    m = TIPSv2Model(cfg)
    m.load_state_dict(sf_load(os.path.join(path, 'model.safetensors')), strict=True)
    return m.eval().to(DEVICE), tok
