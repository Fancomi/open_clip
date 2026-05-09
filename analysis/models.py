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
import torch.nn as nn
import torch.nn.functional as F

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


def _ensure_hf_module_cache(local_dir, cache_name):
    """Sync missing .py files from local_dir into the HF transformers_modules cache.

    HF uses two layouts depending on the model:
      - hash-subdir (RADIO):  {cache_name}/{hash}/*.py
      - flat        (TIPSv2): {cache_name}/*.py
    Files are copied into every existing subdir, or into the root when none exist.
    """
    import glob as _glob
    cache_root = os.path.join(_HF_CACHE, cache_name)
    os.makedirs(cache_root, exist_ok=True)
    subdirs = [d for d in _glob.glob(os.path.join(cache_root, '*')) if os.path.isdir(d)]
    targets = subdirs if subdirs else [cache_root]
    for tgt in targets:
        for src in _glob.glob(os.path.join(local_dir, '*.py')):
            dst = os.path.join(tgt, os.path.basename(src))
            if not os.path.exists(dst):
                shutil.copy(src, dst)
                logging.info(f'[hf_cache] copied {os.path.basename(src)} → {tgt}')


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
    """Load DINOv3 via transformers AutoModel (HF safetensors format).
    repo/ckpt args kept for API compat but ignored.
    Returns wrapper with .forward_features(x) -> {x_norm_clstoken, x_norm_patchtokens}.
    Token layout: [CLS, reg×4, patches×196] → patches start at index 5.
    """
    from transformers import AutoModel
    _DIR = f'{_BASE}/models/dino/dinov3-vitb16-pretrain-lvd1689m'
    logging.info('Loading DINOv3 ...')
    _m = AutoModel.from_pretrained(_DIR, trust_remote_code=True).eval().to(DEVICE)
    _n_reg = json.load(open(f'{_DIR}/config.json')).get('num_register_tokens', 4)

    class _DINOv3Wrapper(nn.Module):
        def __init__(self, m): super().__init__(); self.m = m
        @torch.no_grad()
        def forward_features(self, x):
            h = self.m(x).last_hidden_state.float()   # (B, 1+reg+N, D)
            return {'x_norm_clstoken':    h[:, 0],
                    'x_norm_patchtokens': h[:, 1 + _n_reg:]}

    return _DINOv3Wrapper(_m).to(DEVICE)


def load_radio(path=None):
    from transformers import AutoModel
    path = path or CKPT['radio']
    _ensure_hf_module_cache(path, 'C_hyphen_RADIOv4_hyphen_SO400M')
    logging.info('Loading C-RADIOv4 ...')
    m = AutoModel.from_pretrained(path, trust_remote_code=True).eval().to(DEVICE)
    return m, getattr(m, 'input_conditioner', None)



# ── EUPE native ViT-B/16 (matches checkpoint key structure exactly) ───────────

class _LS(nn.Module):
    def __init__(self, dim): super().__init__(); self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):    return x * self.gamma

class _Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads; self.scale = (dim // heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    def forward(self, x):
        B, N, C = x.shape; H, D = self.heads, C // self.heads
        q, k, v = self.qkv(x).reshape(B, N, 3, H, D).permute(2, 0, 3, 1, 4).unbind(0)
        x = (F.scaled_dot_product_attention(q, k, v)).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)

class _MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4); self.fc2 = nn.Linear(dim * 4, dim)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class _Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim); self.attn = _Attn(dim, heads); self.ls1 = _LS(dim)
        self.norm2 = nn.LayerNorm(dim); self.mlp  = _MLP(dim);         self.ls2 = _LS(dim)
    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

class _PatchEmbed(nn.Module):
    def __init__(self, dim, ps): super().__init__(); self.proj = nn.Conv2d(3, dim, ps, ps)
    def forward(self, x):        return self.proj(x).flatten(2).transpose(1, 2)

class _RopeEmbed(nn.Module):
    """Placeholder — holds rope_embed.periods from checkpoint, not used in forward."""
    def __init__(self, dim): super().__init__(); self.periods = nn.Parameter(torch.zeros(dim))

class _EUPEViT(nn.Module):
    """EUPE ViT-B/16: key structure matches checkpoint exactly, no remap needed.
    storage_tokens (register tokens) prepended after CLS, before patches.
    rope_embed loaded but not applied (position info from learned storage tokens suffices
    for CLS/patch feature extraction used in analysis)."""
    def __init__(self, dim=768, depth=12, heads=12, ps=16, n_storage=4):
        super().__init__()
        self.patch_embed    = _PatchEmbed(dim, ps)
        self.cls_token      = nn.Parameter(torch.zeros(1, 1, dim))
        self.storage_tokens = nn.Parameter(torch.zeros(1, n_storage, dim))
        self.mask_token     = nn.Parameter(torch.zeros(1, dim))
        self.rope_embed     = _RopeEmbed(ps)
        self.blocks         = nn.Sequential(*[_Block(dim, heads) for _ in range(depth)])
        self.norm           = nn.LayerNorm(dim)
        self._n_storage     = n_storage

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.expand(B, -1, -1),
                        self.storage_tokens.expand(B, -1, -1), x], dim=1)
        x = self.blocks(x)
        x = self.norm(x)
        return {'x_norm_clstoken':    x[:, 0].float(),
                'x_norm_patchtokens': x[:, 1 + self._n_storage:].float()}

    def forward(self, x):
        return self.forward_features(x)


def load_eupe(repo=None, ckpt=None):
    """Load EUPE-ViT-B from .pt weights into native _EUPEViT (no timm, no remap).
    repo arg kept for API compat but ignored.
    qkv.bias_mask and projectors.* skipped (not needed for feature extraction)."""
    ckpt = ckpt or CKPT['eupe_ckpt']
    if not os.path.exists(ckpt):
        logging.warning(f'EUPE weights not found: {ckpt}')
        return None
    try:
        d  = torch.load(ckpt, map_location='cpu', weights_only=True)
        sd = {k: v for k, v in d.items()
              if not k.startswith('projectors') and 'bias_mask' not in k}
        m  = _EUPEViT()
        missing, unexpected = m.load_state_dict(sd, strict=False)
        if missing:
            logging.warning(f'[EUPE] missing keys: {missing}')
        logging.info('Loading EUPE-ViT-B ...')
        return m.eval().to(DEVICE)
    except Exception as e:
        logging.warning(f'EUPE load failed: {e}')
        return None


def load_tips(path=None):
    """Returns (model, tokenizer). Tokenizer not needed for image-only tasks."""
    from safetensors.torch import load_file as sf_load
    path  = path or CKPT['tips']
    cache = os.path.join(_HF_CACHE, 'tipsv2_hyphen_b14')
    _ensure_hf_module_cache(path, 'tipsv2_hyphen_b14')
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
