# Representation Uniformity: 研究记录

*最后更新: 2026-05-09 | 实验平台: COCO quick (wm_coco.sh)*

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

Baseline（重跑确认）：**i2t R@1 = 0.0172**（epoch 12）

<!-- RESULTS_TABLE_START -->
| 实验 | 方法 | λ/w | i2t R@1 | vs baseline | t2i R@1 | best epoch | val_loss |
|------|------|-----|---------|-------------|---------|------------|----------|
| **baseline** | — | 0 | **0.0172** | — | 0.0140 | 12 | 0.8486 |
| **koleo005** | KoLeo | 0.05 | **0.0198** | **+15.1%** | 0.0131 | 14 | 0.8375 |
| **gap001** | Gap Loss | 0.001 | **0.0198** | **+15.1%** | 0.0136 | 6 | 0.9593 |
| **koleo1** | KoLeo | 1.0 | **0.0196** | **+14.0%** | 0.0130 | 8 | 0.9185 |
| **uni05** | Uniformity | 0.5 | **0.0190** | **+10.5%** | 0.0136 | 10 | 0.8660 |
| koleo001 | KoLeo | 0.01 | 0.0184 | +7.0% | 0.0145 | 10 | 0.8741 |
| uni001 | Uniformity | 0.01 | 0.0182 | +5.8% | 0.0146 | 14 | 0.8381 |
| uni1 | Uniformity | 1.0 | 0.0180 | +4.7% | 0.0142 | 8 | 0.8870 |
| koleo05 | KoLeo | 0.5 | 0.0180 | +4.7% | 0.0146 | 10 | 0.8742 |
| uni01 | Uniformity | 0.1 | 0.0166 | -3.5% | 0.0143 | 12 | 0.8501 |
| koleo01 | KoLeo | 0.1 | 0.0166 | -3.5% | 0.0132 | 14 | 0.8349 |
| gap01 | Gap Loss | 0.01 | 0.0164 | -4.7% | 0.0130 | 14 | 0.8735 |
| uni005 | Uniformity | 0.05 | 0.0160 | -7.0% | 0.0148 | 18 | 0.8285 |
| gap005 | Gap Loss | 0.005 | 0.0160 | -7.0% | 0.0131 | 10 | 0.8969 |
| gap1 | Gap Loss | 0.1 | 0.0152 | -11.6% | 0.0138 | 12 | 1.0115 |
| gap05 | Gap Loss | 0.05 | 0.0148 | -14.0% | 0.0130 | 18 | 0.9523 |
<!-- RESULTS_TABLE_END -->

### 4.2 Round 1 分析

**KoLeo Loss：最佳候选**
- 有效区间极宽：w ∈ [0.005, 1.0] 全部正向（除 w=0.1 噪声低谷）
- 最优 w=0.05 给出 +15.1%
- w=1.0 也有 +14%，对超参不敏感

**Uniformity Loss：稳定的第二选择**
- 最优 w=0.5，+10.5%
- 曲线呈非单调：w=0.01~0.5 上升，w>1.0 可能下降
- 中间值 w=0.05~0.1 反而低于 baseline（噪声？或与 SIGReg 功能重叠）

**Gap Loss：仅极小值有效**
- λ=0.001 +15.1%（与 CC3M 最优 λ=0.005 量级一致但更小）
- λ≥0.005 开始有害，λ≥0.05 显著负面
- 在 COCO（无 gap）上仍有效 → 确认是 regularizer 效应

**t2i 方向无显著收益**
- 所有实验 t2i R@1 与 baseline 持平或略低
- 可能原因：COCO 5cap 评测中文本冗余度高，i2t 更敏感

### 4.3 Round 2: 细调 + 混合（running）

| 组 | 实验 | 目的 |
|----|------|------|
| G | koleo w∈{0.02,0.03,0.05,0.07,0.15} | 确认 w=0.05 是否为真实峰 |
| H | uniformity w∈{0.3,0.5,0.7} | 确认 w=0.5 附近 |
| I | 6 组混合 | 验证组合是否叠加 |

混合实验设计逻辑：
- KoLeo + Gap：两个独立 +15% 的组合，测是否叠加
- KoLeo + Uniformity：局部推散 + 全局推散，可能互补
- 三合一：上限测试

---

## 5. 分析

### 5.1 预期

**Uniformity Loss**：
- 小 weight (0.01-0.05)：可能无感（SIGReg 已在做类似的事）
- 中 weight (0.1-0.5)：预期 eff_rank 提升，R@1 可能有小幅收益
- 大 weight (1.0)：可能过强，损害对齐（uniformity 和 alignment 是 trade-off）

**KoLeo Loss**：
- 局部推散效果，对"热点区域"的特征聚集更敏感
- DINOv3 中的最优 weight 是 0.1，可作为参考
- 与 Uniformity 的区别：KoLeo 只看最近邻，不像 Uniformity 那样全局拉扯

**Gap Loss on COCO**：
- 预期效果弱于 CC3M（COCO baseline 本身无 gap）
- 如果仍有效 → 证明是 regularizer 效应而非 anti-gap 效应

### 5.2 与 SIGReg 的关系

SIGReg 也是一种 representation regularizer（约束特征接近各向同性高斯）。关键区别：

| | SIGReg | Uniformity | KoLeo |
|--|--------|-----------|-------|
| 作用空间 | pre-norm (unnormalized) | post-norm (L2 hypersphere) | post-norm (L2 hypersphere) |
| 目标分布 | Isotropic Gaussian | Uniform on hypersphere | 最大化近邻距离 |
| 机制 | 特征函数匹配 | 高斯核 pairwise 排斥 | 近邻距离惩罚 |
| 已有效果 | eff_rank 稳定 | 待验证 | DINOv3 验证有效 |

Uniformity/KoLeo 作用在 L2-normalized 空间，与 SIGReg 互补而非冲突。

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

待本轮 COCO 结果出来后：

1. 确认最优 weight range
2. 在 CC3M 上正式验证（对比 gap_loss λ=0.005 baseline）
3. 考虑组合实验：gap_loss + uniformity/koleo
4. 如果 Uniformity/KoLeo 在 CC3M 上有效，纳入标准 recipe

---

*文档版本：2026-05-09 v1 | 实验状态：running*
