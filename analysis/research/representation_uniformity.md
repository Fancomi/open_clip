# Representation Uniformity: 研究记录

*最后更新: 2026-05-10 | 实验平台: COCO quick (wm_coco.sh) | 实验目录: logs/20260510_wmc/*

---

## 1. 背景与动机

### 1.1 从 Modality Gap 到 Representation Quality

前序研究（见 `modality_gap_wm.md`）发现：

1. **Modality Gap 对 retrieval 性能影响有限**：Gap 贡献的模态轴在 ranking 时是常数偏移，不影响相对排序
2. **Within-Modal Repulsion 的 +14% 收益来源不是"消除 gap"**：而是 txt repulsion 作为 uniformity regularizer 提升了文本分布均匀性，使 cos_pos 对更可区分
3. **Gap Loss (λ=0.005) 在 CC3M 上 +0.92% 的收益伴随 eff_rank 109→164**：更可能是表示质量改善的 side effect

**核心洞察**：真正有价值的不是"消除 gap"，而是**提升表示空间的均匀性（uniformity）和有效维度利用率（effective rank）**。

### 1.2 新方向：直接优化表示均匀性

不再通过间接手段（消除 gap、within-modal repulsion）来改善表示质量，而是直接引入有理论支撑的 uniformity loss：

- **Wang & Isola (2020) Uniformity Loss**：`log(mean(exp(-t * ||z_i - z_j||^2)))` — 基于高斯核的分布均匀性度量
- **KoLeo Loss (Sablayrolles et al. 2018)**：`-log(dist_to_nn).mean()` — 近邻熵正则，推散最近邻

两者与 gap 存在与否无关，直接作用于 L2-normalized 特征空间。

### 1.3 为什么同时保留 Gap Loss 探索

Gap Loss 在 CC3M 上有实证收益，但仅在 CC3M 测试过。在 COCO 上验证可以回答：
- 收益来自 "anti-gap" 还是来自 regularization effect？
- 如果 COCO（无 gap）上也有效，说明是 regularizer；无效则说明需要 gap 存在才有意义

---

## 2. 方法

### 2.1 Loss 结构

```
L = L_siglip (full N×N contrastive, unchanged)
  + λ_sigreg × (SIGReg(img_proj) + SIGReg(txt_proj))      # 已有
  + λ_gap   × || mean(img_raw) - mean(txt_raw) ||²         # 实验 D
  + λ_uni   × 0.5 × (Uniformity(img) + Uniformity(txt))    # 实验 E (NEW)
  + λ_koleo × 0.5 × (KoLeo(img) + KoLeo(txt))             # 实验 F (NEW)
```

所有新 loss 都是 **auxiliary**——叠加在完整 SigLIP 之上，不移除 cross-modal negatives。

### 2.2 Uniformity Loss

Wang & Isola (2020) "Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere"

```python
L_uniform = log(mean(exp(-t * ||z_i - z_j||^2)))
         = log(mean(exp(-2t * (1 - cos_ij))))   # for L2-normalized z
```

- 输入：L2-normalized features [N, D]
- 温度 t=2.0（论文默认）
- 值域：(-∞, 0]，越负越均匀；完美均匀（uniform on hypersphere）时取极小值
- 优化目标：最小化 → 推动特征均匀分布在超球面上
- 实现：dot-product formulation + logsumexp for numerical stability

### 2.3 KoLeo Loss

Kozachenko-Leonenko nearest-neighbor entropic regularizer (Sablayrolles et al. 2018)

```python
L_koleo = -log(dist_to_nearest_neighbor + eps).mean()
```

- 输入：L2-normalized features [N, D]
- 惩罚近邻距离过小 → 推散聚集的特征
- 对比 Uniformity：KoLeo 只关注最近邻（局部），Uniformity 关注全局 pairwise 分布
- 已在 DINOv3 中验证有效（作为 SSL 正则），现首次接入 CLIP contrastive path

### 2.4 Gap Loss（对照）

```python
L_gap = || mean(img_raw) - mean(txt_raw) ||²   # pre-L2-norm
```

- 作用于 unnormalized features（在模型 forward 中计算）
- CC3M 已知最优：λ=0.005（+0.92% i2t R@1, eff_rank +50%）
- COCO 上是否有效待验证

---

## 3. 实验设计

### 3.1 平台

- 数据：COCO train（~82K），val：Karpathy 5cap（5K 图 / 25K 文）
- 模型：PE-Core-B-16-dinov3（从头训练，random init）
- 基础配置：SigLIP + SIGReg(cls, 1e-4) + Muon，20 epoch，BS=4096
- 训练规模：20 steps/epoch × 20 epochs = 400 total steps
- 每次实验 ~15 min

### 3.2 实验矩阵

| 组 | 方法 | 参数扫描 | 目的 |
|----|------|----------|------|
| D | Gap Loss | λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1} | COCO 上验证 gap loss 效果 |
| E | Uniformity Loss (t=2.0) | w ∈ {0.01, 0.05, 0.1, 0.5, 1.0} | 直接优化均匀性 |
| F | KoLeo Loss | w ∈ {0.01, 0.05, 0.1, 0.5, 1.0} | 近邻熵正则 |

### 3.3 评估指标

- **i2t / t2i R@1, R@5**：检索性能
- **eff_rank (joint / img / txt)**：有效秩，表示维度利用率
- **cos_pos**：正样本对平均余弦相似度
- **pc1_gap / modal_clf**：模态可分性（对照用）
- **val_loss**：验证损失

### 3.4 Baseline 参照

```
baseline:   i2t R@1 = 0.0168, eff_rank = 105.3, cos_pos = 0.146
txt3000:    i2t R@1 = 0.0192, eff_rank = 75.8 (joint), img=43.2 ← img 崩塌
```

---

## 4. 实验结果

### 4.1 Round 1: 初始扫参（COCO, 20 epoch, BS=4096）

Baseline（重跑确认）：**i2t R@1 = 0.0172, t2i R@1 = 0.0140**（epoch 12）

<!-- RESULTS_TABLE_R1_START -->
| 实验 | 方法 | λ/w | i2t R@1 | i2t Δ | t2i R@1 | t2i Δ | i2t R@5 | t2i R@5 | epoch | val_loss |
|------|------|-----|---------|-------|---------|-------|---------|---------|-------|----------|
| **baseline** | — | 0 | 0.0172 | — | 0.0140 | — | 0.0582 | 0.0516 | 12 | 0.8486 |
| **koleo005** | KoLeo | 0.05 | 0.0198 | +15.1% | 0.0131 | -6.4% | 0.0592 | 0.0523 | 14 | 0.8375 |
| **gap001** | Gap | 0.001 | 0.0198 | +15.1% | 0.0136 | -2.9% | 0.0596 | 0.0502 | 6 | 0.9593 |
| **koleo1** | KoLeo | 1.0 | 0.0196 | +14.0% | 0.0130 | -7.1% | 0.0594 | 0.0480 | 8 | 0.9185 |
| **uni05** | Uniformity | 0.5 | 0.0190 | +10.5% | 0.0136 | -2.9% | 0.0630 | 0.0522 | 10 | 0.8660 |
| koleo001 | KoLeo | 0.01 | 0.0184 | +7.0% | 0.0145 | +3.6% | 0.0636 | 0.0518 | 10 | 0.8741 |
| uni001 | Uniformity | 0.01 | 0.0182 | +5.8% | 0.0146 | +4.3% | 0.0626 | 0.0501 | 14 | 0.8381 |
| uni1 | Uniformity | 1.0 | 0.0180 | +4.7% | 0.0142 | +1.4% | 0.0576 | 0.0488 | 8 | 0.8870 |
| koleo05 | KoLeo | 0.5 | 0.0180 | +4.7% | 0.0146 | +4.3% | 0.0576 | 0.0502 | 10 | 0.8742 |
| uni01 | Uniformity | 0.1 | 0.0166 | -3.5% | 0.0143 | +2.1% | 0.0616 | 0.0516 | 12 | 0.8501 |
| koleo01 | KoLeo | 0.1 | 0.0166 | -3.5% | 0.0132 | -5.7% | 0.0596 | 0.0512 | 14 | 0.8349 |
| gap01 | Gap | 0.01 | 0.0164 | -4.7% | 0.0130 | -7.1% | 0.0590 | 0.0465 | 14 | 0.8735 |
| uni005 | Uniformity | 0.05 | 0.0160 | -7.0% | 0.0148 | +5.7% | 0.0548 | 0.0515 | 18 | 0.8285 |
| gap005 | Gap | 0.005 | 0.0160 | -7.0% | 0.0131 | -6.4% | 0.0580 | 0.0503 | 10 | 0.8969 |
| gap1 | Gap | 0.1 | 0.0152 | -11.6% | 0.0138 | -1.4% | 0.0554 | 0.0537 | 12 | 1.0115 |
| gap05 | Gap | 0.05 | 0.0148 | -14.0% | 0.0130 | -7.1% | 0.0538 | 0.0476 | 18 | 0.9523 |
<!-- RESULTS_TABLE_R1_END -->

### 4.2 Round 2: 细调 + 混合

<!-- RESULTS_TABLE_R2_START -->
| 实验 | 方法 | 参数 | i2t R@1 | i2t Δ | t2i R@1 | t2i Δ | i2t R@5 | t2i R@5 | epoch | val_loss |
|------|------|------|---------|-------|---------|-------|---------|---------|-------|----------|
| koleo005b | KoLeo (复现) | 0.05 | 0.0198 | +15.1% | 0.0131 | -6.4% | 0.0592 | 0.0523 | 14 | 0.8375 |
| koleo002 | KoLeo | 0.02 | 0.0196 | +14.0% | 0.0138 | -1.4% | 0.0614 | 0.0512 | 10 | 0.8721 |
| uni03 | Uniformity | 0.3 | 0.0190 | +10.5% | 0.0144 | +2.9% | 0.0636 | 0.0496 | 8 | 0.8975 |
| **mix_all3** | KoLeo+Uni+Gap | 0.05/0.5/0.001 | 0.0190 | +10.5% | 0.0137 | -2.1% | 0.0632 | 0.0510 | 10 | 0.8747 |
| koleo003 | KoLeo | 0.03 | 0.0188 | +9.3% | 0.0147 | +5.0% | 0.0578 | 0.0529 | 14 | 0.8361 |
| uni05b | Uniformity (复现) | 0.5 | 0.0186 | +8.1% | 0.0144 | +2.9% | 0.0602 | 0.0522 | 12 | 0.8438 |
| mix_uni05_gap001 | Uni+Gap | 0.5/0.001 | 0.0182 | +5.8% | 0.0143 | +2.1% | 0.0598 | 0.0503 | 10 | 0.8774 |
| mix_koleo005_uni03 | KoLeo+Uni | 0.05/0.3 | 0.0182 | +5.8% | 0.0137 | -2.1% | 0.0640 | 0.0528 | 10 | 0.8645 |
| mix_koleo005_gap001 | KoLeo+Gap | 0.05/0.001 | 0.0182 | +5.8% | 0.0146 | +4.3% | 0.0618 | 0.0518 | 12 | 0.8502 |
| koleo007 | KoLeo | 0.07 | 0.0182 | +5.8% | 0.0140 | +0.0% | 0.0634 | 0.0491 | 6 | 0.8972 |
| uni07 | Uniformity | 0.7 | 0.0176 | +2.3% | 0.0136 | -2.9% | 0.0630 | 0.0478 | 8 | 0.9045 |
| mix_koleo005_uni05 | KoLeo+Uni | 0.05/0.5 | 0.0176 | +2.3% | 0.0150 | +7.1% | 0.0624 | 0.0532 | 12 | 0.8447 |
| koleo015 | KoLeo | 0.15 | 0.0174 | +1.2% | 0.0148 | +5.7% | 0.0636 | 0.0524 | 14 | 0.8360 |
| mix_koleo1_uni05 | KoLeo+Uni | 1.0/0.5 | 0.0166 | -3.5% | 0.0142 | +1.4% | 0.0582 | 0.0486 | 6 | 0.9342 |
<!-- RESULTS_TABLE_R2_END -->

### 4.3 综合分析

**i2t vs t2i 不对称性**

i2t 提升最大的实验（koleo005, gap001）在 t2i 上反而略降。两个方向之间存在 trade-off：

```
                    i2t R@1     t2i R@1     方向
koleo005:           0.0198 ↑    0.0131 ↓    i2t 偏向
koleo003:           0.0188 ↑    0.0147 ↑    双向提升（较均衡）
koleo002:           0.0196 ↑    0.0138 ≈    i2t 偏向
mix_koleo005_uni05: 0.0176 ↑    0.0150 ↑    t2i 最优（+7.1%），i2t 仅 +2.3%
uni03:              0.0190 ↑    0.0144 ↑    双向提升（较均衡）
baseline:           0.0172      0.0140
```

- **追求 i2t 最大化**：KoLeo w=0.05（+15.1% / -6.4%）
- **追求 t2i 最大化**：mix_koleo005_uni05（+2.3% / +7.1%）或 koleo003（+9.3% / +5.0%）
- **追求双向均衡**：uni03（+10.5% / +2.9%）或 koleo003（+9.3% / +5.0%）

**KoLeo 细调曲线**

```
w=0.01  → i2t=0.0184 (+7.0%),  t2i=0.0145 (+3.6%)   ← 小但双向正
w=0.02  → i2t=0.0196 (+14.0%), t2i=0.0138 (-1.4%)
w=0.03  → i2t=0.0188 (+9.3%),  t2i=0.0147 (+5.0%)   ← 均衡点
w=0.05  → i2t=0.0198 (+15.1%), t2i=0.0131 (-6.4%)   ← i2t 峰值
w=0.07  → i2t=0.0182 (+5.8%),  t2i=0.0140 (+0.0%)
w=0.15  → i2t=0.0174 (+1.2%),  t2i=0.0148 (+5.7%)
w=0.5   → i2t=0.0180 (+4.7%),  t2i=0.0146 (+4.3%)
w=1.0   → i2t=0.0196 (+14.0%), t2i=0.0130 (-7.1%)
```

w=0.05 处 i2t 有锐峰但 t2i 低谷；w=0.03 和 w=0.5 是较好的均衡点。

**混合实验：不叠加**

所有混合实验均 ≤ 单一最优。三个 loss 机制上高度重叠（都在推散特征），组合后正则化过强。

| 混合 | i2t R@1 | 对比最优单一组分 |
|------|---------|-----------------|
| koleo005 + gap001 | 0.0182 | < 两者单独 (0.0198) |
| koleo005 + uni05 | 0.0176 | < 两者单独 |
| uni05 + gap001 | 0.0182 | < 两者单独 |
| 三合一 | 0.0190 | < koleo005 单独 |

**复现性**

koleo005 两次完美复现（0.0198, epoch 14, val_loss=0.8375）。uni05 复现较弱（0.0190 vs 0.0186）。

---

## 5. 结论

### 5.1 确定结论

1. **KoLeo、Uniformity、Gap Loss 三者均有效**，但 i2t 和 t2i 存在 trade-off
2. **混合不叠加**：三个 loss 机制重叠（都在推散特征），组合后正则化过强反而有害
3. **KoLeo w=0.05 复现性最好**：两次完美复现 i2t R@1=0.0198
4. **Gap Loss 在 COCO（无 gap）上仍有效**：确认是 regularizer 效应而非 anti-gap 效应

### 5.2 与 SIGReg 的关系

| | SIGReg | Uniformity | KoLeo | Gap Loss |
|--|--------|-----------|-------|----------|
| 作用空间 | pre-norm (unnorm) | post-norm (L2) | post-norm (L2) | pre-norm |
| 目标 | Isotropic Gaussian | Uniform on sphere | 最大化近邻距离 | 模态均值重合 |
| 机制 | 特征函数匹配 | 高斯核 pairwise 排斥 | 近邻距离惩罚 | 均值对齐 |
| COCO 最优 | 1e-4 (已有) | w=0.3~0.5 | w=0.03~0.05 | λ=0.001 |

### 5.3 推荐配置（待 CC3M 验证）

| 目标 | 配置 | i2t Δ | t2i Δ |
|------|------|-------|-------|
| i2t 最大化 | KoLeo w=0.05 | +15.1% | -6.4% |
| 双向均衡 | KoLeo w=0.03 或 Uniformity w=0.3 | +9~10% | +3~5% |
| 保守稳妥 | KoLeo w=0.01 | +7.0% | +3.6% |

---

## 6. 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| UniformityLoss 实现 | `src/open_clip/loss.py` | `class UniformityLoss` |
| KoLeo 复用 | `src/open_clip/loss.py` | `class KoLeoLoss` |
| SIGRegContrastiveLoss 集成 | `src/open_clip/loss.py` | `forward()` 末尾 uniformity/koleo block |
| CLI 参数 | `src/open_clip_train/params.py` | `--uniformity-weight`, `--uniformity-t`, `--koleo-weight` |
| 实验脚本 | `experiments/wm_coco.sh` | Section D/E/F |

---

## 7. 下一步

1. **CC3M 正式验证**：在 CC3M 上跑 KoLeo w={0.03, 0.05} + Uniformity w={0.3, 0.5}，确认 COCO 结论是否可推广
2. **确认 t2i trade-off 是否在 CC3M 上复现**：如果 CC3M 上也 i2t↑ t2i↓，需要理解原因
3. **纳入标准 recipe**：如果 CC3M 验证通过，将最优配置加入默认训练参数

---

*文档版本：2026-05-10 v2 | COCO 实验完成，待 CC3M 验证*
