"""
probe_pos_embed.py — 探测 OpenCLIP 中 APE 和 RoPE 的数值特性

用法：
  source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
  cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
  PYTHONPATH=./src python scripts/probe_pos_embed.py
"""

import sys
sys.path.insert(0, './src')

import torch
import numpy as np
import open_clip

torch.manual_seed(42)
DEVICE = 'cpu'

# ───────────────────────────────────────────────
# 0. 公共工具
# ───────────────────────────────────────────────
def stat(name, t):
    t = t.float().detach()
    print(f"  {name:40s}  shape={list(t.shape)}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}  "
          f"min={t.min():.4f}  max={t.max():.4f}")

def per_token_norm(t):
    """t: [N, D] -> [N] L2 norm per token"""
    return t.float().norm(dim=-1)

def isotropy_score(t):
    """
    各向同性得分：用 participation ratio (PR) 衡量
    PR = (sum λ_i)^2 / sum(λ_i^2)，越接近 D 表示越各向同性
    t: [N, D]
    """
    t = t.float() - t.float().mean(0)
    _, s, _ = torch.linalg.svd(t, full_matrices=False)
    lam = s ** 2
    pr = lam.sum() ** 2 / (lam ** 2).sum()
    return pr.item(), t.shape[1]  # (PR, D)

# ───────────────────────────────────────────────
# 1. VisionTransformer（标准 ViT）的 APE
# ───────────────────────────────────────────────
print("=" * 60)
print("1. VisionTransformer APE (ViT-B-16-exp)")
print("=" * 60)

model_vit, _, _ = open_clip.create_model_and_transforms('ViT-B-16-exp', pretrained=None)
model_vit.eval()
vit = model_vit.visual  # VisionTransformer

ape = vit.positional_embedding.data  # [N+1, D]
print(f"\n  pos_embed shape = {list(ape.shape)}")
stat("APE全部token", ape)
stat("APE CLS token", ape[0:1])
stat("APE patch token(均值)", ape[1:])

# 每个 token 的 L2 范数
norms = per_token_norm(ape)
print(f"\n  每token L2范数: mean={norms.mean():.4f}  std={norms.std():.4f}")
print(f"  CLS norm={norms[0]:.4f}, patch norm range=[{norms[1:].min():.4f}, {norms[1:].max():.4f}]")

# 各向同性得分
pr, D = isotropy_score(ape[1:])  # 仅 patch 部分
print(f"\n  Participation Ratio (patch): {pr:.2f} / {D}  "
      f"({'各向同性' if pr/D > 0.5 else '各向异性'}，PR/D={pr/D:.3f})")

# 相邻 patch 的余弦相似度（空间连续性）
patch_pe = ape[1:].view(14, 14, -1)  # ViT-B-16: 224/16=14
right_sim  = torch.nn.functional.cosine_similarity(patch_pe[:, :-1], patch_pe[:, 1:],  dim=-1)
down_sim   = torch.nn.functional.cosine_similarity(patch_pe[:-1, :], patch_pe[1:,  :], dim=-1)
diag_sim   = torch.nn.functional.cosine_similarity(patch_pe[:-1, :-1], patch_pe[1:, 1:], dim=-1)
print(f"\n  空间连续性 cos_sim:")
print(f"    水平相邻: mean={right_sim.mean():.4f}  std={right_sim.std():.4f}")
print(f"    垂直相邻: mean={down_sim.mean():.4f}   std={down_sim.std():.4f}")
print(f"    对角相邻: mean={diag_sim.mean():.4f}   std={diag_sim.std():.4f}")
print(f"  → 值越高说明 APE 空间上越连续（有结构）")

# ───────────────────────────────────────────────
# 2. EVA 模型的 APE + RoPE
# ───────────────────────────────────────────────
print()
print("=" * 60)
print("2. EVA / PE-Core-B-16 的 APE + RoPE")
print("=" * 60)

try:
    model_eva, _, _ = open_clip.create_model_and_transforms('PE-Core-B-16', pretrained=None)
    model_eva.eval()
    eva = model_eva.visual  # Eva 模块

    # 2a. APE（如果存在）
    if hasattr(eva, 'pos_embed') and eva.pos_embed is not None:
        ape_eva = eva.pos_embed.data  # [1, N+1, D] 或 [N+1, D]
        print(f"\n  EVA pos_embed shape = {list(ape_eva.shape)}")
        stat("EVA APE", ape_eva.squeeze())
        norms_eva = per_token_norm(ape_eva.squeeze())
        print(f"  每token L2范数: mean={norms_eva.mean():.4f}  std={norms_eva.std():.4f}")
        pr_eva, D_eva = isotropy_score(ape_eva.squeeze())
        print(f"  Participation Ratio: {pr_eva:.2f} / {D_eva}  (PR/D={pr_eva/D_eva:.3f})")
    else:
        print("  EVA no absolute pos_embed")

    # 2b. RoPE
    if hasattr(eva, 'rope') and eva.rope is not None:
        rope = eva.rope
        print(f"\n  RoPE type: {type(rope).__name__}")
        re = rope.get_embed()  # [N, head_dim] or [N, 2*head_dim]
        print(f"  RoPE embed shape = {list(re.shape)}")

        # split sin / cos
        sin_e, cos_e = re.chunk(2, dim=-1)
        stat("RoPE sin部分", sin_e)
        stat("RoPE cos部分", cos_e)

        # 验证 sin^2 + cos^2 = 1 (每个频率维度)
        unity = (sin_e ** 2 + cos_e ** 2)
        print(f"\n  sin²+cos²: mean={unity.mean():.6f}  std={unity.std():.8f}  (应=1.0)")

        # 每个 token 的旋转角度（低频维度 #0）
        angles = torch.atan2(sin_e[:, 0], cos_e[:, 0])
        print(f"\n  维度0的旋转角度 [rad]:")
        print(f"    range=[{angles.min():.4f}, {angles.max():.4f}]  "
              f"mean={angles.mean():.4f}  std={angles.std():.4f}")

        # 模长不变性验证：对随机向量施加 RoPE，检查 L2 norm
        from timm.layers.pos_embed_sincos import apply_rot_embed_cat
        N, half = re.shape[0], re.shape[1]
        q_rand = torch.randn(1, 1, N, half * 2)  # [B, nH, N, D]
        re_4d = re.unsqueeze(0).unsqueeze(0)      # [1, 1, N, 2*half]
        q_rot = apply_rot_embed_cat(q_rand, re_4d)
        norm_before = q_rand.norm(dim=-1)
        norm_after  = q_rot.norm(dim=-1)
        print(f"\n  RoPE模长不变性验证:")
        print(f"    norm before: mean={norm_before.mean():.6f}")
        print(f"    norm after:  mean={norm_after.mean():.6f}")
        print(f"    max diff:    {(norm_before - norm_after).abs().max():.2e}  (应≈0)")
    else:
        print("  EVA no RoPE")
except Exception as e:
    print(f"  [跳过 PE-Core-B-16: {e}]")

# ───────────────────────────────────────────────
# 3. 对 latent 数值影响：hook 采样
# ───────────────────────────────────────────────
print()
print("=" * 60)
print("3. APE 对 latent 数值的影响（VisionTransformer）")
print("=" * 60)

records = {}

def hook_after_conv1(module, inp, out):
    records['after_conv1'] = out.detach().clone()  # [B, N, D] after permute? no, before

def hook_after_ape(module, inp, out):
    records['after_ape'] = out.detach().clone()

def hook_after_lnpre(module, inp, out):
    records['after_ln_pre'] = out.detach().clone()

# 注册 hook
h1 = vit.conv1.register_forward_hook(hook_after_conv1)
h2 = vit.ln_pre.register_forward_hook(hook_after_lnpre)

# 手动 patch _embeds 来捕获 APE 之后
_orig_embeds = vit._embeds.__func__

def _patched_embeds(self, x):
    x = self.conv1(x)
    x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
    x_pre_ape = x.clone()
    records['before_ape'] = x_pre_ape.detach()
    x = torch.cat([
        x.new_zeros(x.shape[0], 1, x.shape[2]) if not hasattr(self, 'class_embedding')
        else open_clip.transformer._expand_token(self.class_embedding, x.shape[0]).to(x.dtype),
        x
    ], dim=1)
    x = x + self.positional_embedding.to(x.dtype)
    records['after_ape'] = x.detach().clone()
    x = self.patch_dropout(x)
    x = self.ln_pre(x)
    return x

import types
vit._embeds = types.MethodType(_patched_embeds, vit)

dummy_img = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    _ = vit(dummy_img)

h1.remove()
h2.remove()

if 'before_ape' in records and 'after_ape' in records:
    before = records['before_ape']     # [B, N, D] patch only
    after  = records['after_ape'][:, 1:, :]  # 去掉 CLS
    ape_vals = ape[1:].unsqueeze(0)    # [1, N, D]

    print(f"\n  patch latent (before APE):")
    stat("patch_before", before[0])
    print(f"  patch latent (after APE):")
    stat("patch_after ", after[0])
    print(f"  APE 本身:")
    stat("ape_values  ", ape_vals[0])

    delta = after[0] - before[0]
    print(f"\n  delta = after - before (应 ≈ APE):")
    stat("delta       ", delta)
    print(f"  max(|delta - APE|) = {(delta - ape_vals[0]).abs().max():.2e}  (应≈0)")

    # APE 相对于 latent 的信噪比
    snr = before[0].norm(dim=-1).mean() / ape[1:].norm(dim=-1).mean()
    print(f"\n  APE / latent 幅度比 = {1/snr:.4f}  (APE幅度 / patch embedding幅度)")
    print(f"  → 值越大说明 APE 对 latent 影响越强")

print()
print("Done.")
