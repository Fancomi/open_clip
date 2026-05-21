# Positive-Only Distillation: 无负样本的图文对齐

*最后更新: 2026-05-19 | 实验平台: COCO quick (wm_coco.sh)*

**系列**: `distill_01_reverse_lit.md` → `distill_02_multi_teacher.md` → **本文**

**前序关系**:
- `distill_01` (Reverse-LiT): frozen pretrained text + **对比 loss (SigLIP)**，在 CC3M 上 i2t=0.2560
- `distill_02` (Multi-Teacher): frozen multi-text + 对比 loss，多教师蒸馏
- **本文**: frozen pretrained text + **正样本 only (MSE/Sigmoid) + SIGReg**，探索去除负样本后能否保持性能

**本质**: 这是 reverse-LiT 的无负样本变体。从技术上说更接近**蒸馏**而非传统自监督——image encoder 从 frozen text encoder 的嵌入中学习，SIGReg 替代负样本提供防坍缩约束。

---

## 1. 背景

### 1.1 动机

前序 within-modal 实验（mgap_02）发现：仅用正样本对齐 + SIGReg 正则化也能达到接近 SigLIP 的性能，但需要极大的 lambda（~30）且 image 分布畸形。

NOVA (arXiv:2602.00653, 2026) 证明 MSE + SIGReg 可以超越 CLIP——完全不需要负样本。

### 1.2 核心问题

对比学习的负样本提供两个作用：
1. **排斥力**：防止所有特征坍缩到一个点
2. **区分力**：让不同语义的特征互相远离

SIGReg 可以替代作用 1（保证各向同性 → 防坍缩）。作用 2 则由数据多样性自然提供（不同图文对的正样本方向自然不同）。

### 1.3 实验设计

| # | 实验 | 正样本 Loss | SIGReg 模式 |
|---|------|------------|------------|
| 1 | posonly_sig_sep | -logsigmoid(scale*cos+bias) | 图/文分开 |
| 2 | posonly_sig_joint | -logsigmoid(scale*cos+bias) | 图+文联合 |
| 3 | posonly_mse_sep | (1-cos)² | 图/文分开 |
| 4 | posonly_mse_joint | (1-cos)² | 图+文联合 |

- 基础配置: PE-Core-B-16-dinov3, Muon, 20ep, BS=4096, COCO
- SIGReg weight: 1e-4
- sep 用 `--sigreg-target cls`，joint 用 `--sigreg-target clip`（保证同维度可 cat）

---

## 2. 方法

### 2.1 Positive Sigmoid

已有实现 `_cross_modal_positive_only`：

```python
pos_logits = logit_scale * (img * txt).sum(-1) + logit_bias  # [B]
loss = -F.logsigmoid(pos_logits).sum() / B
```

### 2.2 Positive MSE

```python
cos_sim = (img * txt).sum(-1)  # [B], 值域 [-1, 1]
loss = (1.0 - cos_sim).pow(2).mean()  # 目标: cos → 1
```

### 2.3 SIGReg Joint vs Separate

- **Separate**: `SIGReg(img_proj) + SIGReg(txt_proj)` — 分别约束各模态的各向同性
- **Joint**: `SIGReg(cat[img_proj, txt_proj])` — 约束联合分布的各向同性（隐式防止模态分离）

---

## 3. 结果

<!-- RESULTS_TABLE_START -->
| 实验 | best i2t R@1 | i2t Δ | best t2i R@1 | t2i Δ | best@ |
|------|-------------|-------|-------------|-------|-------|
| SigLIP baseline | 0.0172 | -- | 0.0140 | -- | 12 |
| posonly_sig_sep | -- | -- | -- | -- | -- |
| posonly_sig_joint | -- | -- | -- | -- | -- |
| posonly_mse_sep | -- | -- | -- | -- | -- |
| posonly_mse_joint | -- | -- | -- | -- | -- |
<!-- RESULTS_TABLE_END -->

*(实验进行中)*

---

## 4. 分析

*(待实验完成后补充)*

---

## 5. 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| pos_only 参数 | `src/open_clip/loss.py` | `SIGRegContrastiveLoss.__init__` |
| pos_only forward | `src/open_clip/loss.py` | `SIGRegContrastiveLoss.forward` pos_only 分支 |
| sigreg_joint | `src/open_clip/loss.py` | forward 中 `torch.cat(projs)` |
| CLI | `src/open_clip_train/params.py` | `--pos-only`, `--sigreg-joint` |

---

*文档版本: 2026-05-18 v1 | 实验进行中*
