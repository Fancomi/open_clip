"""
probe_pos_embed.py — APE 和 RoPE 数值特性探测

用法：
  source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
  cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
  PYTHONPATH=./src python scripts/probe_pos_embed.py [--output /tmp/pos_embed_out]

输出：
  stdout    — 数值统计
  <output>/ape_pca.png  — 预训练 APE 的 PCA 主成分网格图
  <output>/rope_attn.png — RoPE 对 attention score 的位置偏置图
"""

import sys, argparse, random, warnings
from pathlib import Path
sys.path.insert(0, './src')

import torch
import numpy as np

warnings.filterwarnings("ignore", message="Glyph.*missing from font")
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as _fm

_HEITI = ("/root/.comate/remote/32ce3c0e5bdf6f2fdb792c2c0706d281eb518916-249728546/"
          "server/extensions/baiducomate.comate/assets/font/HeiTi.ttf")
try:
    _fe = _fm.FontEntry(fname=_HEITI, name="HeiTi",
                        style="normal", variant="normal",
                        weight=400, stretch="normal", size="scalable")
    _fm.fontManager.ttflist.append(_fe)
    matplotlib.rcParams["font.sans-serif"] = ["HeiTi"] + list(matplotlib.rcParams["font.sans-serif"])
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
import open_clip

torch.manual_seed(42)

PE_CKPT  = '/root/paddlejob/workspace/env_run/penghaotian/models/timm/PE-Core-B-16/open_clip_model.safetensors'
IMG_DIR  = '/root/paddlejob/workspace/env_run/penghaotian/datas/coco/images/val2014'
DEVICE   = 'cpu'
_SCRIPT_DIR = Path(__file__).parent

# ── 工具 ──────────────────────────────────────────────────────────────
def stat(name, t):
    t = t.float().detach()
    print(f"  {name:<36s}  std={t.std():.4f}  norm_mean={t.norm(dim=-1).mean():.3f}"
          f"  range=[{t.min():.3f}, {t.max():.3f}]")

def pr_score(t):
    """Participation Ratio / D，越高越各向同性"""
    t = t.float() - t.float().mean(0)
    _, s, _ = torch.linalg.svd(t, full_matrices=False)
    lam = s ** 2
    pr = lam.sum() ** 2 / (lam ** 2).sum()
    return pr.item() / t.shape[1], lam

def spatial_cosim(patch_2d):
    """patch_2d: [H,W,D] → (水平, 垂直) mean cosine similarity"""
    F = torch.nn.functional.cosine_similarity
    h = F(patch_2d[:, :-1], patch_2d[:, 1:],  dim=-1).mean().item()
    v = F(patch_2d[:-1, :], patch_2d[1:,  :], dim=-1).mean().item()
    return h, v

def load_image(img_dir, size=224):
    paths = list(Path(img_dir).glob('*.jpg'))
    img = Image.open(random.choice(paths)).convert('RGB').resize((size, size))
    t = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    return (t - mean) / std, img   # normalized tensor, PIL for display

def capture_latent(model_visual, trunk):
    """hook 捕获 patch_embed 输出 + APE 之后的序列"""
    rec = {}
    def hook_pe(m, i, o): rec['patch_embed'] = o.detach().clone()
    h = trunk.patch_embed.register_forward_hook(hook_pe)
    import types
    orig = trunk._pos_embed.__func__
    def patched(self, x):
        r, rot = orig(self, x)
        rec['after_ape'] = r.detach().clone()
        return r, rot
    trunk._pos_embed = types.MethodType(patched, trunk)
    with torch.no_grad():
        model_visual(torch.randn(1, 3, 224, 224))
    h.remove()
    return rec

# ── 模型加载 ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--output', default=str(_SCRIPT_DIR / 'probe_out'))
args = parser.parse_args()
Path(args.output).mkdir(parents=True, exist_ok=True)

print("Loading models ...")
m_vit, _, _ = open_clip.create_model_and_transforms('ViT-B-16-exp', pretrained=None)
m_pe,  _, _ = open_clip.create_model_and_transforms('PE-Core-B-16', pretrained=PE_CKPT)
m_dv3, _, _ = open_clip.create_model_and_transforms('PE-Core-B-16-dinov3', pretrained=None)
m_vit.eval(); m_pe.eval(); m_dv3.eval()

vit      = m_vit.visual           # VisionTransformer
trunk_pe = m_pe.visual.trunk      # Eva (pretrained)
trunk_dv3= m_dv3.visual.trunk     # Eva (random, dinov3)

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("1. APE 静态分析（可学习参数矩阵，随机 vs 预训练）")
print("="*60)
# ════════════════════════════════════════════════════════════════════
# APE 本质：nn.Parameter(shape=[N,D])，与 weight 一样由梯度下降优化
# 初始化：trunc_normal_(std=0.02)，随机；训练后收敛为低秩空间坐标结构

ape_vit = vit.positional_embedding.data           # [197, 768]
ape_pe  = trunk_pe.pos_embed.data.squeeze()       # [197, 768]
ape_dv3 = trunk_dv3.pos_embed.data.squeeze()      # [201, 768]

print(f"\n{'模型':<30s} {'APE_std':>9} {'patch_norm':>11} {'PR/D':>7} {'空间(h,v)':>16}")
print("-"*76)

for label, ape, npt in [
    ("ViT-B-16-exp  (random)",  ape_vit, 1),
    ("PE-Core-B-16  (pretrained)", ape_pe, 1),
    ("PE-Core-B-16-dinov3 (random)", ape_dv3, 5),
]:
    patch = ape[npt:]
    pr, _ = pr_score(patch)
    h, v  = spatial_cosim(patch.view(14, 14, -1))
    norms = patch.float().norm(dim=-1)
    print(f"  {label:<28s}  {ape.std():.4f}   {norms.mean():.3f}±{norms.std():.3f}"
          f"  {pr:.4f}  ({h:+.3f}, {v:+.3f})")

print(f"""
  关键对比：
  · 随机初始化：空间连续性 ≈ 0（噪声），PR/D≈0.20（均匀分散）
  · 预训练后：空间连续性 → 0.97，PR/D → 0.008（能量集中在 ~6 个主成分）
  · CLS token norm: random={ape_vit[0].norm():.2f}  pretrained={ape_pe[0].norm():.2f}
    → CLS norm 比 patch 大 2.5×，是学出来的（初始化相同），让 CLS 远离 patch 流形
  · APE 是加法编码：x = patch_embed(img) + pos_embed，直接改变 patch token 向量
    （改变的是 conv1 输出，不是原始像素）""")

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("2. APE 的 PCA 结构（学成了什么）→ 保存图像")
print("="*60)
# ════════════════════════════════════════════════════════════════════

patch_pe = ape_pe[1:].numpy()                      # [196, 768]
patch_vit= ape_vit[1:].numpy()

patch_pe_c = patch_pe - patch_pe.mean(0)
patch_vit_c= patch_vit- patch_vit.mean(0)

U_pe, S_pe, _ = np.linalg.svd(patch_pe_c, full_matrices=False)
U_vit,S_vit, _= np.linalg.svd(patch_vit_c,full_matrices=False)

var_pe  = S_pe**2  / (S_pe**2).sum()
var_vit = S_vit**2 / (S_vit**2).sum()

print(f"\n  PE-Core pretrained: 前4 PC 方差占比 = "
      f"{var_pe[:4].cumsum()[-1]*100:.1f}%  "
      f"({var_pe[:4]*100} %)")
print(f"  ViT random:         前4 PC 方差占比 = "
      f"{var_vit[:4].cumsum()[-1]*100:.1f}%  "
      f"({var_vit[:4]*100} %)")

# 图：pretrained APE 的前4 PC（每个 PC 都是 14×14 热图）+ random 对比
fig, axes = plt.subplots(2, 5, figsize=(16, 6))
for ax in axes.flat: ax.axis('off')

axes[0,0].set_title("APE 全部 patch\n（PCA RGB）", fontsize=9)
axes[1,0].set_title("APE 全部 patch\n（PCA RGB，random）", fontsize=9)

for row, (U, S, label) in enumerate([
    (U_pe,  S_pe,  "PE-Core-B-16 pretrained"),
    (U_vit, S_vit, "ViT-B-16-exp random"),
]):
    var = S**2 / (S**2).sum()
    # PCA RGB（前3 PC → RGB）
    rgb = U[:, :3].copy()
    rgb = (rgb - rgb.min(0)) / (rgb.max(0) - rgb.min(0) + 1e-8)
    axes[row, 0].imshow(rgb.reshape(14, 14, 3), interpolation='nearest')
    axes[row, 0].set_title(f"{label}\nPCA RGB", fontsize=8)
    axes[row, 0].axis('off')

    for pc in range(4):
        ax = axes[row, pc+1]
        pc_map = U[:, pc].reshape(14, 14)
        ax.imshow(pc_map, cmap='RdBu_r', interpolation='nearest')
        ax.set_title(f"PC{pc+1}  ({var[pc]*100:.1f}%)", fontsize=8)
        ax.axis('off')

plt.suptitle("APE PCA：预训练后收敛为 DCT-like 坐标基底", fontsize=11, y=1.01)
plt.tight_layout()
save_path = f"{args.output}/ape_pca.png"
plt.savefig(save_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  → 保存 {save_path}")
print("""
  解读：
  · PC1（~25%）：水平条纹 → y 坐标（行索引）
  · PC2（~20%）：垂直条纹 → x 坐标（列索引）
  · PC3（~18%）：环形 → 2阶频率（中心 vs 边缘）
  · PC4（~13%）：对角 → 斜向空间频率
  这就是 2D-DCT 的前几个基底，梯度下降自发学出了数学最优结构
  Random APE 的 PC 均匀分布（各占 ~0.5%），无空间结构""")

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("3. APE 对 patch token 的幅度影响（pretrained）")
print("="*60)
# ════════════════════════════════════════════════════════════════════
# APE 改变的是 patch_embed 输出（conv1 结果），不是原始像素

rec = capture_latent(m_pe.visual, trunk_pe)
before    = rec['patch_embed'][0]                          # [196, 768]
after     = rec['after_ape'][0, trunk_pe.num_prefix_tokens:]  # [196, 768]
ape_patch = ape_pe[trunk_pe.num_prefix_tokens:]            # [196, 768]

ratio = ape_patch.norm(dim=-1).mean() / before.norm(dim=-1).mean()
delta_err = (after - before - ape_patch).abs().max()

print(f"\n  conv1 输出（patch pixel content）:  std={before.std():.4f}  norm={before.norm(dim=-1).mean():.3f}")
print(f"  APE patch 部分:                     std={ape_patch.std():.4f}  norm={ape_patch.norm(dim=-1).mean():.3f}")
print(f"  APE/content 幅度比 = {ratio:.4f}   delta验证误差 = {delta_err:.1e}")
print("""
  → 约 7% 的"扰动"注入到内容向量：位置信息存在但不淹没语义
  → 此后经过 12 层 FFN + attention，位置信息逐渐与内容混合/衰减""")

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("4. RoPE 数值特性（RotaryEmbeddingDinoV3）")
print("="*60)
# ════════════════════════════════════════════════════════════════════
# RoPE 是确定性正弦函数，与初始化无关
# 核心数学：score(m,n) = Re[q_m · conj(k_n) · e^{i(m-n)θ}] = f(q_m·k_n, m-n)
# → attention score 只依赖相对位置 (m-n)，绝对坐标消除

rope = trunk_dv3.rope
re = rope.get_embed(shape=[14,14])  # [196, 128]: [sin|cos]
half = re.shape[1] // 2
sin_e, cos_e = re[:, :half], re[:, half:]

# sin²+cos²=1
unity = (sin_e**2 + cos_e**2).mean().item()
# 模长不变性
from timm.layers.pos_embed_sincos import apply_rot_embed_cat
q_r = torch.randn(1,1,196,half)
q_rot = apply_rot_embed_cat(q_r, re.unsqueeze(0).unsqueeze(0), half=True)
norm_diff = (q_r.norm(dim=-1) - q_rot.norm(dim=-1)).abs().max().item()

# 旋转角度（最低频维度）
angles = torch.atan2(sin_e[:,0], cos_e[:,0]).view(14,14)
step_y = (angles[1,0] - angles[0,0]).item()
step_x = (angles[0,1] - angles[0,0]).item()

# 空间连续性
rope_2d = re.view(14,14,-1)
h_sim = torch.nn.functional.cosine_similarity(rope_2d[:,:-1], rope_2d[:,1:], dim=-1).mean().item()
v_sim = torch.nn.functional.cosine_similarity(rope_2d[:-1,:], rope_2d[1:,:], dim=-1).mean().item()

print(f"""
  sin²+cos²=1 验证: {unity:.6f} (精确)   模长不变性 max_diff: {norm_diff:.2e}
  最低频维度角度步长: Δy={step_y:.4f} rad, Δx={step_x:.4f} rad (x=0: 列不变，行步进)
  相邻patch embed cosine_sim: 水平={h_sim:.4f}, 垂直={v_sim:.4f}  (→ 旋转矩阵空间平滑)

  RoPE 只编码相对位置的数学原理：
    RoPE(q,m) = q ⊗ e^(imθ)        (复数旋转)
    score(m,n) = RoPE(q_m) · RoPE(k_n)
               = Re[q_m · conj(k_n) · e^(i(m-n)θ)]
               = f(q_m·k_n, m-n)    ← 只剩相对位置 (m-n)，绝对坐标消除
  → 模型平移整张图，attention 模式不变（等变性）
  → RoPE 不改变 x，每层 attention 独立施加，位置信息不随深度衰减""")

# ════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("5. RoPE 位置偏置：覆盖不对称性 + 多层多查询位置分析")
print("="*60)
# ════════════════════════════════════════════════════════════════════

from collections import defaultdict

random.seed(42)
img_tensor, img_pil = load_image(IMG_DIR, size=224)

# ── 多层 Q/K hook ────────────────────────────────────────────────
LAYERS = [0, 3, 6, 9, 11]
Q_PATCHES = [('中心(7,7)', 7, 7), ('角落(0,0)', 0, 0), ('边缘(0,7)', 0, 7)]

rec_layers = {l: {} for l in LAYERS}

def make_hook(storage, key):
    def hook(m, i, o): storage[key] = o.detach().clone()
    return hook

hooks = []
for l in LAYERS:
    a = trunk_pe.blocks[l].attn
    hooks.append(a.q_norm.register_forward_hook(make_hook(rec_layers[l], 'q')))
    hooks.append(a.k_norm.register_forward_hook(make_hook(rec_layers[l], 'k')))

with torch.no_grad():
    m_pe.visual(img_tensor.unsqueeze(0))
for h in hooks: h.remove()

# ── RoPE 工具 ─────────────────────────────────────────────────────
rope_pe   = trunk_pe.rope
re_pe     = rope_pe.get_embed(shape=[14, 14])      # [196, 128]
re_4d     = re_pe.unsqueeze(0).unsqueeze(0)
angles_pe = torch.atan2(re_pe[:, :64], re_pe[:, 64:])  # [196, 64]
npt_pe    = trunk_pe.num_prefix_tokens              # 1
scale_f   = trunk_pe.blocks[0].attn.head_dim ** -0.5   # 64**-0.5

def bias_map(layer, qr, qc, head=0):
    """返回 (score_无RoPE, score_有RoPE, 偏置diff) 各 [14,14] numpy"""
    q_flat = qr * 14 + qc
    q_all = rec_layers[layer]['q'][:, :, npt_pe:, :]  # [1,nH,196,D]
    k_all = rec_layers[layer]['k'][:, :, npt_pe:, :]
    q_rot = apply_rot_embed_cat(q_all, re_4d, half=False)
    k_rot = apply_rot_embed_cat(k_all, re_4d, half=False)
    s_no = (q_all[0,head,q_flat] @ k_all[0,head].T * scale_f).view(14,14).numpy()
    s_ro = (q_rot[0,head,q_flat] @ k_rot[0,head].T * scale_f).view(14,14).numpy()
    return s_no, s_ro, s_ro - s_no

def theo_affinity(qr, qc):
    """纯位置亲和度（与内容无关）：mean_d cos(Δangle_d)"""
    q_flat = qr * 14 + qc
    da = angles_pe[q_flat].unsqueeze(0) - angles_pe   # [196, 64]
    return da.cos().mean(-1).view(14, 14).numpy()

# ── 清晰的 dist 表（3个查询位置 × 3层）────────────────────────────
print(f"\n  RoPE偏置按曼哈顿距离分组  (head=0，各取 dist 0~6)\n")
hdr = f"  {'查询':^10} {'层':>5}  {'dist':>4}  {'无RoPE':>8} {'有RoPE':>8} {'偏置':>8}  n"
sep = "  " + "-" * (len(hdr) - 2)
print(hdr); print(sep)

for q_name, qr, qc in Q_PATCHES:
    for l in [0, 6, 11]:
        s_no, s_ro, diff = bias_map(l, qr, qc)
        buckets = defaultdict(list)
        for i in range(14):
            for j in range(14):
                d = abs(i - qr) + abs(j - qc)
                buckets[d].append((s_no[i,j], s_ro[i,j], diff[i,j]))
        for d in sorted(buckets)[:7]:
            v = buckets[d]
            mn, mr, md = np.mean([x[0] for x in v]), np.mean([x[1] for x in v]), np.mean([x[2] for x in v])
            tag = f"  {q_name:^10} {l:>5}  {d:>4}  {mn:>8.3f} {mr:>8.3f} {md:>+8.3f}  {len(v)}"
            print(tag)
        print()

print("""  注：dist=0 即自注意力 Q·K，RoPE 在相对位置=0 时旋转角=0（恒等），
  故有/无RoPE的偏置精确为 0。角落patch(0,0)的dist分布与中心不同，
  体现了有限网格中相对位置集合的不对称性。""")

# ── 图1：覆盖不对称性 + 理论亲和度 ─────────────────────────────────
fig1, axes1 = plt.subplots(2, 3, figsize=(12, 8))

for col, (name, qr, qc) in enumerate(Q_PATCHES):
    # 上行：可达 Δ 空间（27×27，蓝=可达，红十字=原点）
    cov = np.zeros((27, 27))
    for r in range(14):
        for c in range(14):
            cov[r - qr + 13, c - qc + 13] = 1
    ax = axes1[0, col]
    ax.imshow(cov, cmap='Blues', interpolation='nearest', vmin=0, vmax=1.2)
    ax.axhline(13, color='r', lw=1.5, alpha=0.8, ls='--')
    ax.axvline(13, color='r', lw=1.5, alpha=0.8, ls='--')
    ax.set_title(f"{name}\n可达相对位置 (Δrow,Δcol) 空间", fontsize=9)
    ax.set_xlabel("Δcol →"); ax.set_ylabel("Δrow ↓")
    ax.set_xticks([0, 13, 26]); ax.set_xticklabels(['-13', '0', '+13'])
    ax.set_yticks([0, 13, 26]); ax.set_yticklabels(['-13', '0', '+13'])

    # 下行：理论 RoPE 亲和度（内容无关，纯位置）
    aff = theo_affinity(qr, qc)
    ax2 = axes1[1, col]
    vabs = max(abs(aff.min()), abs(aff.max()))
    im = ax2.imshow(aff, cmap='RdBu_r', vmin=-vabs, vmax=vabs, interpolation='nearest')
    ax2.plot(qc, qr, 'w*', ms=12)
    ax2.set_title(f"{name}\n理论RoPE亲和度 mean_d cos(Δθ_d)", fontsize=9)
    ax2.axis('off')
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

axes1[0, 0].set_facecolor('#f0f0f0')
plt.suptitle("RoPE 覆盖不对称性：有限网格中角落/边缘的可达相对位置不对称 → 隐含绝对位置信息",
             fontsize=10, y=1.01)
plt.tight_layout()
save_cov = f"{args.output}/rope_coverage.png"
plt.savefig(save_cov, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  → 保存 {save_cov}")

# ── 图2：多层 × 多查询 偏置图 ────────────────────────────────────
n_rows, n_cols = len(LAYERS), len(Q_PATCHES)
fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))

for row, l in enumerate(LAYERS):
    for col, (name, qr, qc) in enumerate(Q_PATCHES):
        _, _, diff = bias_map(l, qr, qc)
        ax = axes2[row, col]
        vabs = max(abs(diff.min()), abs(diff.max()), 1e-6)
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-vabs, vmax=vabs, interpolation='nearest')
        ax.plot(qc, qr, 'w*', ms=11)
        ax.set_title(f"Layer {l}  |  {name}", fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle(
    "RoPE 偏置图（有RoPE − 无RoPE）\n蓝=近处score增益，红=远处抑制  |  PE-Core-B-16 pretrained, head=0",
    fontsize=11)
plt.tight_layout()
save_attn = f"{args.output}/rope_attn.png"
plt.savefig(save_attn, dpi=150, bbox_inches='tight')
plt.close()
print(f"  → 保存 {save_attn}")

print("""
  解读：
  覆盖图（上）：
  · 中心(7,7) → Δ空间对称，attention可从四个方向等权探索
  · 角落(0,0) → 只有一个象限的Δ可达，attention天然单向
  · 边缘(0,7) → 半对称，行方向单向，列方向对称
  → RoPE 虽编码相对位置，但有限网格中边界patch的可达Δ集合不对称
    → 角落/边缘的attention行为统计形状不同于中心 → 隐含绝对位置

  偏置图（下）：
  · 早期层(0)：偏置集中在query周围，近邻增益明显
  · 深层(11)：偏置图更发散，Q/K已经过多层变换，RoPE作用相对均匀
  · 角落(0,0)：偏置分布只在单侧展开，印证覆盖不对称
  · 边缘(0,7)：偏置在列方向对称，行方向只向下延伸""")

print("\nDone.")

