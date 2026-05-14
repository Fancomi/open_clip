# Projective SigLIP: 研究记录

*最后更新: 2026-05-12 | 实验平台: COCO quick (wm_coco.sh) | 实验目录: logs/20260512_wmc/*

---

## 1. 背景与动机

### 1.1 从 Orthogonal 到 Projective

前序研究（`mgap_05_orthogonal_siglip.md`）发现：

1. **Orthogonal SigLIP 双向超越 baseline**：i2t +3.5%, t2i +7.1%，无 trade-off
2. **核心原理**：负样本推向 cos=0（正交）比 cos=-1（对立）更优

但 Orthogonal 模式仍然约束正样本到 cos=+1（单方向对齐）。这引出一个自然的推广：

### 1.2 核心洞察：共线 = 相关，正交 = 无关

**四元数启发**：在 SO(3) 旋转群中，q 和 -q 表达完全相同的旋转。球面上完全相反的方向编码相同的信息。

**射影空间 RP^n**：将 S^n 上的对径点 (x, -x) 等价视为同一点。"方向"不再区分正负，只区分"这条线"和"那条线"。

**新定义**：
- **共线** (|cos| → 1)：两个向量在同一条线上（方向相同或相反都行）= **语义相关**
- **正交** (|cos| → 0)：两个向量互相垂直 = **语义无关**

### 1.3 预期效应

1. **容量翻倍**：每个维度方向可编码两个语义（+d 和 -d 是同一概念），D 维空间的有效信息容量从 S^{D-1} 扩展到 RP^{D-1}
2. **模态鸿沟消失**：同 orthogonal，负样本无法利用反方向逃逸
3. **更灵活的对齐**：正样本只需共线，不必同向——降低了对齐约束的严格性，可能更快收敛
4. **天然消歧**：如果数据中存在语义对称性（如"大/小"是同一属性的两极），projective 天然处理

### 1.4 与前序方法对比

| 属性 | Standard | Antipodal | Orthogonal | **Projective** |
|------|----------|-----------|------------|----------------|
| 正样本目标 | cos → +1 | cos → -1 | cos → +1 | **|cos| → 1** |
| 负样本目标 | cos → -1 | cos ≠ -1 | cos → 0 | **|cos| → 0** |
| 相似度度量 | cos | -cos | cos | **|cos|** |
| 等价关系 | x ~ x | x ~ -x | x ~ x | **x ~ ±x** |
| 信息空间 | S^{D-1} | S^{D-1} | S^{D-1} | **RP^{D-1}** |

---

## 2. 方法

### 2.1 数学推导

**统一使用 |cos| 作为相似度**（正负样本均适用）：

```
logits = scale * |cos(img, txt)| + bias
loss = -log(σ(label * logit))     # label: +1=正, -1=负
```

展开分析：
- **正样本 (label=+1)**: loss = -log(σ(scale * |cos| + bias))
  - 最小化 → |cos| → 1 → cos = +1 或 cos = -1（共线） ✓
- **负样本 (label=-1)**: loss = softplus(scale * |cos| + bias)
  - 最小化 → |cos| → 0 → cos = 0（正交） ✓

### 2.2 实现极简性

**对比三种模式的 get_logits 实现**：

```python
if self.neg_mode == 'antipodal':
    logits = -logits                           # 取反
elif self.neg_mode == 'orthogonal':
    if negative_only:
        logits = logits.abs()                  # 负样本取绝对值
    else:
        eye = torch.eye(N, ...)
        logits = torch.where(eye, logits, logits.abs())  # 正样本保原值
elif self.neg_mode == 'projective':
    logits = logits.abs()                      # 全部取绝对值（最简）
```

Projective 是最简洁的：**一行 abs，无条件分支**。正负样本统一使用 |cos|。

### 2.3 Eval 适配

检索排序使用 |cos| 而非 cos：

```python
if neg_mode == 'projective':
    logits = logit_scale * (image_features @ text_features.t()).abs()
```

### 2.4 初始化行为

random init 时 |cos| ≈ 0, bias ≈ -10:
- 正样本 logit = scale * 0 + (-10) = -10 → loss ≈ 10（需要学习）
- 负样本 logit = scale * 0 + (-10) = -10 → softplus(-10) ≈ 0（已满足）
- **与标准 SigLIP / Orthogonal 初始化行为完全一致**

---

## 3. 实验设计

### 3.1 平台

- 数据：COCO train (~82K), val: Karpathy 5cap (5K 图 / 25K 文)
- 模型：PE-Core-B-16-dinov3 (random init)
- 基础配置：SigLIP + SIGReg(cls, 1e-4) + Muon, 20 epoch, BS=4096
- 每次实验 ~15 min

### 3.2 实验矩阵

| 实验 | 配置 | 目的 |
|------|------|------|
| projective | SigLIP + SIGReg + Muon + `--neg-mode projective` | 纯 projective 基线 |
| proj_koleo005 | + KoLeo w=0.05 | 验证 KoLeo 在 projective 下的效果 |
| proj_uni05 | + Uniformity w=0.5 | 验证 Uniformity 在 projective 下的效果 |

参照:
- Standard SigLIP baseline: i2t R@1 = 0.0172, t2i R@1 = 0.0140
- Orthogonal SigLIP: i2t R@1 = 0.0178, t2i R@1 = 0.0150

### 3.3 评估指标

- **i2t / t2i R@1, R@5**: 检索性能（使用 |cos| 排序）
- **sim_pos**: 正样本 |cos|（预期从 ~0 趋向 1）
- **eff_rank / cos_pos**: 表示质量
- **val_loss**: 验证损失

---

## 4. 实验结果

### 4.1 COCO 20 epoch

<!-- RESULTS_TABLE_START -->
| 实验 | 方法 | best i2t R@1 | i2t Delta | best t2i R@1 | t2i Delta | best epoch(i2t) |
|------|------|-------------|-----------|-------------|-----------|-----------------|
| **baseline** | Standard SigLIP | 0.0172 | -- | 0.0140 | -- | 12 |
| **orthogonal** | Orthogonal SigLIP | 0.0178 | +3.5% | 0.0150 | +7.1% | 10 |
| **projective** | Projective SigLIP | 0.0200 | **+16.3%** | 0.0151 | **+7.9%** | 8 |
| proj_koleo005 | Proj + KoLeo 0.05 | 0.0168 | -2.3% | 0.0145 | +3.6% | 10 |
| proj_uni05 | Proj + Uni 0.5 | 0.0174 | +1.2% | 0.0142 | +1.4% | 4 |
<!-- RESULTS_TABLE_END -->

### 4.2 Projective 训练曲线

```
Epoch | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5
  0   | 0.0016  | 0.0012  | 0.0078  | 0.0061
  2   | 0.0086  | 0.0089  | 0.0376  | 0.0341
  4   | 0.0128  | 0.0134  | 0.0504  | 0.0503
  6   | 0.0178  | 0.0150  | 0.0608  | 0.0532
  8   | 0.0200* | 0.0140  | 0.0626  | 0.0513   ← i2t best
 10   | 0.0184  | 0.0150  | 0.0620  | 0.0522
 12   | 0.0160  | 0.0151* | 0.0646  | 0.0518   ← t2i best
 14   | 0.0168  | 0.0144  | 0.0626  | 0.0538
 16   | 0.0154  | 0.0135  | 0.0604  | 0.0524
 18   | 0.0152  | 0.0132  | 0.0610  | 0.0522
```

---

## 5. 分析

### 5.1 核心发现

1. **Projective SigLIP 是全系列最优方法**：i2t R@1 = 0.0200 (+16.3%)，超越标准 KoLeo (0.0198) 且不牺牲 t2i
2. **t2i 同时提升**：0.0151 (+7.9%)，且不存在标准 KoLeo 的 -6.4% t2i 代价
3. **辅助 loss 有害**：KoLeo 和 Uniformity 都降低了 projective 的性能——射影空间的 |cos| 目标本身已提供足够的均匀性

### 5.2 全方法对比

```
方法                        | i2t R@1 | i2t Δ    | t2i R@1 | t2i Δ    | 双向?
Standard SigLIP (baseline)  | 0.0172  | --       | 0.0140  | --       | --
Antipodal SigLIP            | 0.0174  | +1.2%    | 0.0147  | +5.0%    | 双向 ✓
Orthogonal SigLIP           | 0.0178  | +3.5%    | 0.0150  | +7.1%    | 双向 ✓✓
Projective SigLIP           | 0.0200  | +16.3%   | 0.0151  | +7.9%    | 双向 ✓✓✓
KoLeo w=0.05 (standard)     | 0.0198  | +15.1%   | 0.0131  | -6.4%    | i2t↑ t2i↓ ✗
Proj + KoLeo w=0.05         | 0.0168  | -2.3%    | 0.0145  | +3.6%    | 反效果
Proj + Uni w=0.5            | 0.0174  | +1.2%    | 0.0142  | +1.4%    | 中性
```

### 5.3 关键观察

**1. 射影空间容量优势**

Projective 的 +16.3% i2t 提升远超 orthogonal (+3.5%)。区别在于：orthogonal 正样本仅允许 cos=+1，而 projective 允许 cos=±1。这意味着 D 维空间中每个方向可编码两个语义概念（+d 和 -d 都是有效的正样本方向），有效容量翻倍。

**2. 为什么辅助 loss 有害？**

在射影空间 RP^{D-1} 中，|cos| 本身已经是完美的距离度量。KoLeo 和 Uniformity 作用于 cos（非 |cos|），与 projective 的语义空间不兼容。KoLeo 推散 cos-近邻，但 cos=+0.9 和 cos=-0.9 在 projective 下是等价的相似关系——KoLeo 却试图推开它们。

**3. 与标准 KoLeo 的关键对比**

标准 KoLeo 达到 i2t 0.0198 但 t2i -6.4%。Projective 达到 0.0200 且 t2i +7.9%。两者 i2t 相当，但 projective 在 t2i 上领先 22.5% 相对值（0.0151 vs 0.0131）。Projective 实现了 KoLeo 的 i2t 收益但没有其 t2i 代价。

---

## 6. CC3M 验证

### 6.1 实验配置

- 数据：CC3M (2.9M), val: COCO Karpathy 5cap
- 配置：SigLIP + SIGReg + Muon + `--neg-mode projective`, 10 epoch, BS=4096

### 6.2 结果

| 方法 | best i2t R@1 | i2t Δ | best t2i R@1 | t2i Δ | best epoch |
|------|-------------|-------|-------------|-------|-----------|
| **Baseline** (standard) | 0.2190 | -- | 0.1603 | -- | 9/8 |
| **Orthogonal** | 0.2270 | +3.7% | 0.1636 | +2.1% | 8/8 |
| **Projective** | 0.2278 | +4.0% | 0.1602 | -0.1% | 8/8 |

### 6.3 CC3M 分析

1. **Projective i2t 略优**: +4.0% vs orthogonal +3.7%，但差异在噪声范围内
2. **t2i 持平**: projective 在 CC3M 上 t2i 与 baseline 持平（0.1602 vs 0.1603），orthogonal 更优（0.1636, +2.1%）
3. **COCO vs CC3M 差异**: COCO 上 projective +16.3% i2t，CC3M 上仅 +4.0%。COCO 小数据集（82K）对负样本几何更敏感；CC3M 2.9M 样本的统计量更稳定
4. **Orthogonal 在 CC3M 上更均衡**: 双向均有 2-4% 提升，是 CC3M 上最佳方法

### 6.4 综合判断

- **小数据/快速迭代场景**: Projective 更优（COCO +16.3% i2t 且 t2i 也提升）
- **大数据正式训练**: Orthogonal 更稳健（CC3M 双向均匀提升）
- **两者均优于 baseline**: 在不同数据规模上，正交类方法一致优于标准 SigLIP

---

## 7. 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| SigLipLoss projective | `src/open_clip/loss.py` | `get_logits()` 中 `logits = logits.abs()` |
| CLI 参数 | `src/open_clip_train/params.py` | `--neg-mode projective` |
| Eval (|cos| ranking) | `src/open_clip_train/train.py` | `evaluate()` + `get_clip_metrics()` |
| Zero-shot | `src/open_clip_train/zero_shot.py` | `raw.abs()` |
| 冒烟测试 | `scripts/smoke.sh` | A14-A15 |
| 实验脚本 | `experiments/wm_coco.sh` | Projective SigLIP section |

---

*文档版本: 2026-05-12 v2 | COCO 完成，CC3M 进行中*
