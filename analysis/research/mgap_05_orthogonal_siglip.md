# Orthogonal SigLIP: 研究记录

*最后更新: 2026-05-11 | 实验平台: COCO quick (wm_coco.sh) | 实验目录: logs/20260511_wmc/*

---

## 1. 背景与动机

### 1.1 前序研究总结

前序研究（`mgap_01_analysis.md` → `mgap_02_within_modal_repulsion.md` → `mgap_03_representation_uniformity.md` → `mgap_04_antipodal_siglip.md`）发现：

1. **模态鸿沟无法消除**：PC1 始终编码模态方向，各种直接对抗手段（Gap Loss、Within-Modal Repulsion、KoLeo、Uniformity）均未根本解决
2. **Antipodal SigLIP 可行但改善有限**：正样本推向 cos=-1 后，i2t +1.2%、t2i +5.0%，双向均衡。但对特征取负后可视化 PC1 发现结构与标准 SigLIP 完全相同——仅仅是翻转了符号，本质未变
3. **根本困局**：标准 SigLIP 的 N(N-1) 个跨模态负样本被推向 cos=-1，这使得"模态反方向"成为最低成本的逃逸路径，是模态鸿沟的结构性来源

### 1.2 核心洞察：负样本应正交而非对立

两个关键观察：

- **cos=+1（对齐）** 和 **cos=-1（对立）** 都编码了**关系**——只是方向相反
- 如果两个样本**无关**，最自然的几何关系是 **cos=0（正交）**——即互不相关

这启发了一个新方案：保持正样本 cos→+1 对齐不变，但限制负样本最多只能推到 cos=0（正交），而非 cos=-1（对立）。

### 1.3 三个预期效应

1. **模态鸿沟消失**：负样本无法利用"完全相反方向"作为逃逸路径。cos=0 意味着两个模态在 PC1 上没有系统性偏移
2. **有效秩提升**：正交约束迫使模型使用更多维度来区分负样本（cos=0 的空间远大于 cos=-1 的点），有效利用特征空间
3. **双向均衡**：正样本仍然对齐 (cos→+1)，负样本分布在正交超平面上，i2t/t2i 无结构性 trade-off

### 1.4 与 Antipodal 的对比

| 属性 | Standard SigLIP | Antipodal SigLIP | **Orthogonal SigLIP** |
|------|----------------|-----------------|---------------------|
| 正样本目标 | cos → +1 | cos → -1 | cos → +1 |
| 负样本目标 | cos → -1 | cos ≠ -1 | cos → 0 |
| 负样本逃逸 | 利用模态反方向 | 本质同标准(翻转) | **无反方向逃逸** |
| 模态鸿沟 | 必然产生 | 符号翻转后相同 | **预期消除** |
| 有效秩 | 受限 | 受限 | **预期提升** |
| eval sim_sign | +1 | -1 | +1 |

---

## 2. 方法

### 2.1 数学推导

标准 SigLIP:
```
logits = scale * cos(img, txt) + bias
loss = -log(σ(label * logit))     # label: +1=正, -1=负
```

对于负样本 (label=-1):
```
loss = -log(σ(-logit)) = softplus(logit) = softplus(scale * cos + bias)
```
当 cos → -1: softplus(-scale + bias) ≈ 0 → 无 loss（负样本"逃"到了对面）

**Orthogonal SigLIP 的唯一改动**——负样本使用 |cos| 代替 cos:
```
正样本: logits = scale * cos + bias          ← 不变
负样本: logits = scale * |cos| + bias        ← 唯一差别
```

展开分析：
- **正样本 (label=+1)**: 与标准 SigLIP 完全相同，cos → +1
- **负样本 (label=-1)**: loss = softplus(scale * |cos| + bias)
  - 当 cos = 0: softplus(bias) = softplus(-10) ≈ 0 ✓ （正交 = 满足）
  - 当 cos = +1: softplus(scale + bias) = softplus(0) = log(2) > 0 ✗ （受罚）
  - 当 cos = -1: softplus(scale + bias) = softplus(0) = log(2) > 0 ✗ （**也受罚！**）

**关键差异**：标准 SigLIP 中 cos=-1 是零 loss 的安全区；Orthogonal SigLIP 中 cos=-1 和 cos=+1 的 loss 相同——负样本只有在 cos=0 时才安全。

### 2.2 梯度分析

负样本对 cos 的梯度：
```
∂L/∂cos = scale * sign(cos) * σ(scale*|cos| + bias)
```

- cos > 0 时：梯度为正 → 推 cos 减小（朝 0）
- cos < 0 时：梯度为负 → 推 cos 增大（朝 0）
- cos = 0 时：梯度为 0（已到目标）

**对称收敛**：无论负样本余弦正负，都被推向 cos=0。

### 2.3 初始化行为

random init 时 cos ≈ 0, bias ≈ -10:
- 正样本 logit = scale*0 + bias = -10 → loss ≈ 10（需要学习）
- 负样本 logit = scale*|0| + bias = -10 → softplus(-10) ≈ 0（已满足）
- **与标准 SigLIP 初始化行为完全一致**

### 2.4 Loss 结构

```
L = L_orthogonal_siglip (cross-modal, neg pairs use |cos|)
  + lambda_sigreg * (SIGReg(img_proj) + SIGReg(txt_proj))   # unchanged
```

Optional combinations:
```
  + lambda_koleo * 0.5 * (KoLeo(img) + KoLeo(txt))          # 推散近邻
  + lambda_uni   * 0.5 * (Uniformity(img) + Uniformity(txt)) # 全局均匀
```

---

## 3. 实现

### 3.1 代码改动

统一重构 `antipodal: bool` → `neg_mode: str` ('standard' | 'antipodal' | 'orthogonal'):

| 文件 | 改动 |
|------|------|
| `src/open_clip/loss.py` | `SigLipLoss.__init__` 替换 antipodal 为 neg_mode |
| `src/open_clip/loss.py` | `SigLipLoss.get_logits` 添加 negative_only 参数，orthogonal 分支 |
| `src/open_clip/loss.py` | `SigLipLoss._loss` 传递 negative_only 到 get_logits |
| `src/open_clip/loss.py` | `SIGRegContrastiveLoss` 透传 neg_mode |
| `src/open_clip/loss.py` | `CLIPWithDINOLoss` 透传 neg_mode |
| `src/open_clip/factory.py` | `create_loss()` 路由 neg_mode |
| `src/open_clip_train/params.py` | `--neg-mode` CLI 参数替换 `--antipodal` |
| `src/open_clip_train/train.py` | `evaluate()` 和 `get_clip_metrics()` 基于 neg_mode |
| `src/open_clip_train/zero_shot.py` | `run()` 基于 neg_mode |

### 3.2 关键设计决策

1. **|cos| 而非 cos²**：|cos| 梯度在 cos≠0 处恒为 ±scale（强推力），cos² 在 cos→0 处梯度趋零（弱推力）。选择 |cos| 以保证收敛速度
2. **torch.abs 的可微性**：PyTorch 的 abs 在 x=0 处 grad=0（subgradient），恰好是我们的目标点——无梯度意味着"已到达"
3. **get_logits 添加 negative_only 参数**：分布式 SigLIP 的 cross-rank chunk 全为负样本，需要这个参数以正确处理
4. **eval 无需改动**：正样本仍在 cos=+1，检索排序与标准模式一致 (sim_sign=+1)

---

## 4. 实验设计

### 4.1 平台

- 数据：COCO train (~82K), val: Karpathy 5cap (5K 图 / 25K 文)
- 模型：PE-Core-B-16-dinov3 (random init)
- 基础配置：SigLIP + SIGReg(cls, 1e-4) + Muon, 20 epoch, BS=4096
- 每次实验 ~15 min

### 4.2 实验矩阵

| 实验 | 配置 | 目的 |
|------|------|------|
| orthogonal | SigLIP + SIGReg + Muon + `--neg-mode orthogonal` | 纯 orthogonal 基线 |
| ortho_koleo005 | + KoLeo w=0.05 | 验证 KoLeo 在 orthogonal 下的效果 |
| ortho_uni05 | + Uniformity w=0.5 | 验证 Uniformity 在 orthogonal 下的效果 |

参照:
- Standard SigLIP baseline: i2t R@1 = 0.0172, t2i R@1 = 0.0140
- Antipodal SigLIP: i2t R@1 = 0.0174, t2i R@1 = 0.0147

### 4.3 评估指标

- **i2t / t2i R@1, R@5**: 检索性能
- **sim_pos**: 正样本平均余弦 (预期从 ~0 趋向 +1，同标准模式)
- **eff_rank / cos_pos**: 表示质量
- **val_loss**: 验证损失

---

## 5. 实验结果

### 5.1 COCO 20 epoch

<!-- RESULTS_TABLE_START -->
| 实验 | 方法 | best i2t R@1 | i2t Delta | best t2i R@1 | t2i Delta | best epoch(i2t) | sim_pos (final) |
|------|------|-------------|-----------|-------------|-----------|-----------------|-----------------|
| **baseline** | Standard SigLIP | 0.0172 | -- | 0.0140 | -- | 12 | +0.97 |
| **antipodal** | Antipodal SigLIP | 0.0174 | +1.2% | 0.0147 | +5.0% | 10 | -0.97 |
| **orthogonal** | Orthogonal SigLIP | 0.0178 | **+3.5%** | 0.0150 | **+7.1%** | 10 | +0.97 |
| ortho_koleo005 | Ortho + KoLeo 0.05 | 0.0174 | +1.2% | 0.0146 | +4.3% | 12 | +0.97 |
| ortho_uni05 | Ortho + Uni 0.5 | 0.0192 | **+11.6%** | 0.0137 | -2.1% | 16 | +0.97 |
<!-- RESULTS_TABLE_END -->

### 5.2 Orthogonal 训练曲线

```
Epoch | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5
  0   | 0.0012  | 0.0012  | 0.0068  | 0.0061
  2   | 0.0062  | 0.0084  | 0.0284  | 0.0324
  4   | 0.0148  | 0.0135  | 0.0530  | 0.0513
  6   | 0.0168  | 0.0138  | 0.0590  | 0.0503
  8   | 0.0152  | 0.0133  | 0.0594  | 0.0501
 10   | 0.0178* | 0.0142  | 0.0682  | 0.0535   ← i2t best
 12   | 0.0164  | 0.0150* | 0.0638  | 0.0549   ← t2i best
 14   | 0.0176  | 0.0139  | 0.0650  | 0.0553
 16   | 0.0166  | 0.0137  | 0.0630  | 0.0541
 18   | 0.0162  | 0.0136  | 0.0626  | 0.0523
```

### 5.3 sim_pos 演化

```
Epoch 0 start:  sim_pos = +0.034  (random init)
Epoch 0 end:    sim_pos ≈ +0.20
Epoch 5:        sim_pos ≈ +0.60
Epoch 10:       sim_pos ≈ +0.92
Epoch 19:       sim_pos = +0.97   (与标准 SigLIP 相同)
```

**确认：正样本确实被推向 cos=+1（标准对齐），与标准 SigLIP 方向相同。**

---

## 6. 分析

### 6.1 核心发现

1. **Orthogonal SigLIP 双向超越 baseline**：i2t R@1 = 0.0178 (+3.5%)，t2i R@1 = 0.0150 (+7.1%)，均为所有方法中最佳双向组合
2. **超越 Antipodal**：orthogonal 在 i2t 和 t2i 上均优于 antipodal（0.0178 vs 0.0174, 0.0150 vs 0.0147）
3. **无 i2t/t2i trade-off**：纯 orthogonal 模式下双向同时提升，不存在 KoLeo/Uniformity 那样的此消彼长

### 6.2 全方法对比

```
方法                        | i2t R@1 | i2t Δ    | t2i R@1 | t2i Δ    | 双向?
Standard SigLIP (baseline)  | 0.0172  | --       | 0.0140  | --       | --
Antipodal SigLIP            | 0.0174  | +1.2%    | 0.0147  | +5.0%    | 双向提升 ✓
Orthogonal SigLIP           | 0.0178  | +3.5%    | 0.0150  | +7.1%    | 双向提升 ✓✓
Ortho + KoLeo w=0.05        | 0.0174  | +1.2%    | 0.0146  | +4.3%    | 较均衡
Ortho + Uni w=0.5           | 0.0192  | +11.6%   | 0.0137  | -2.1%    | i2t↑ t2i↓
KoLeo w=0.05 (standard)     | 0.0198  | +15.1%   | 0.0131  | -6.4%    | i2t↑ t2i↓ ✗
Anti + KoLeo w=0.05         | 0.0186  | +8.1%    | 0.0136  | -2.9%    | i2t↑ t2i↓
```

### 6.3 关键观察

**1. 负样本几何决定了双向平衡性**

纯 orthogonal 是唯一在两个方向上同时超越 baseline 最多的方法。标准/antipodal + KoLeo 虽然 i2t 更高（0.0198/0.0186），但 t2i 都下降了。这支持了核心假设：负样本正交（cos→0）比负样本对立（cos→-1）更有利于双向均衡。

**2. Uniformity 对 orthogonal 有强效应**

Ortho + Uni 达到 i2t 0.0192 (+11.6%)，这是仅次于标准 KoLeo 的最高 i2t 成绩，但 t2i trade-off 明显小于标准模式（-2.1% vs -6.4%）。说明 orthogonal 提供了更好的 t2i 基础。

**3. KoLeo 对 orthogonal 无益**

Ortho + KoLeo (0.0174/0.0146) 比纯 orthogonal (0.0178/0.0150) 差。KoLeo 的近邻推散可能与 orthogonal 的正交约束冲突——orthogonal 已经提供了一种"均匀性"，额外的近邻扰动反而干扰了优化。

**4. COCO 噪声考虑**

COCO 上 R@1 差异 ≤ 10 张图 (5K val set)。orthogonal 的 +3.5%/+7.1% 虽一致但需要 CC3M 验证统计显著性。

---

## 7. 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| SigLipLoss.neg_mode | `src/open_clip/loss.py` | `get_logits()` 中 orthogonal 分支 |
| CLI 参数 | `src/open_clip_train/params.py` | `--neg-mode` |
| Loss 路由 | `src/open_clip/factory.py` | `create_loss()` |
| Eval | `src/open_clip_train/train.py` | `evaluate()` + `get_clip_metrics()` |
| Zero-shot | `src/open_clip_train/zero_shot.py` | `run()` |
| 冒烟测试 | `scripts/smoke.sh` | A12-A13 |
| 实验脚本 | `experiments/wm_coco.sh` | Orthogonal SigLIP section |
| CC3M 实验 | `experiments/wds_cc3m.sh` | ortho_sigreg_muon |

---

## 8. CC3M 验证

### 8.1 实验配置

- 数据：CC3M train (2.9M), val: COCO Karpathy 5cap (5K 图 / 25K 文)
- 模型：PE-Core-B-16-dinov3 (random init)
- 配置：SigLIP + SIGReg(cls, 1e-4) + Muon + `--neg-mode orthogonal`, 10 epoch, BS=4096

CC3M 是 orthogonal 的关键验证场：
- CC3M 有稳定的模态 GAP（COCO 小数据集上 gap 会消退）
- 2.9M 样本 / 7000+ steps 提供统计显著性
- 如果 orthogonal 在 CC3M 上仍然双向提升，说明方法具有通用性

### 8.2 CC3M 结果

<!-- RESULTS_CC3M_START -->
| 实验 | 方法 | best i2t R@1 | i2t Delta | best t2i R@1 | t2i Delta |
|------|------|-------------|-----------|-------------|-----------|
| baseline | Standard SigLIP | -- | -- | -- | -- |
| ortho_sigreg_muon | Orthogonal SigLIP | -- | -- | -- | -- |
<!-- RESULTS_CC3M_END -->

*(CC3M 实验进行中，结果待填)*

---

## 9. 下一步

1. **等待 CC3M 实验完成**：分析 orthogonal 在大数据集上的表现
2. **Probe 分析**：利用已保存的 probe npz，分析 orthogonal 模型的 eff_rank、PC alignment、modality gap
3. **可视化**：PCA scatter 对比 orthogonal vs standard vs antipodal 的模态分布
4. **变体探索**（如果 CC3M 验证通过）：
   - alpha-blend: `cos_neg = alpha * cos + (1-alpha) * |cos|`，在标准和正交之间平滑过渡
   - orthogonal + 更小的 Uniformity (w=0.1-0.3): 寻找不牺牲 t2i 的 i2t 提升
5. **Zero-shot ImageNet**：验证 orthogonal 在分类上的表现

---

*文档版本: 2026-05-12 v2 | COCO 完成，CC3M 进行中*
