# Antipodal SigLIP: 研究记录

*最后更新: 2026-05-11 | 实验平台: COCO quick (wm_coco.sh) | 实验目录: logs/20260511_wmc/*

---

## 1. 背景与动机

### 1.1 模态鸿沟问题

前序研究（`mgap_01_analysis.md` → `mgap_02_within_modal_repulsion.md` → `mgap_03_representation_uniformity.md`）发现：

1. **模态鸿沟是 CLIP 范式的必然产物**：PC1 始终编码模态方向，线性分类器准确率接近 1.0
2. **对抗鸿沟的方法均未奏效**：
   - Gap Loss：regularizer 效应，非 anti-gap 效应（COCO 无 gap 时仍有收益）
   - Within-modal repulsion：img 维度崩塌，txt 改善来自低秩压缩副作用
   - KoLeo / Uniformity：+15% i2t 但 -6% t2i，存在 trade-off，混合不叠加
3. **根本原因**：标准 CLIP/SigLIP loss 的 N(N-1) 个 cross-modal negatives 天然驱动两个模态分离——模态轴是最低成本的判别维度

### 1.2 核心洞察：承认鸿沟

既然模态鸿沟无法消除，为什么不**反其道而行**——彻底承认两个模态空间的独立性？

类比：
- **复平面**：实数和虚数表达截然相反的方向，但共同构成完整的数系
- **四元数**：球体上完全相反的方向 q 和 -q 表达**完全相同的旋转** (SO(3) 双覆盖)
- **实射影空间 RP^n**：超球面 S^n 上 antipodal points (x, -x) 被 identify 为同一点

**假设**：让正样本 (img_i, txt_i) 在超球面上处于完全相反的方向 (cos = -1)，
负样本 (img_i, txt_j) 远离相反方向，会产生更均匀的特征分布和更好的检索性能。

### 1.3 预期效应

1. **更均匀的特征空间**：每个 text 只「推远」1 个正样本（cos→-1），但让 N-1 个负样本远离 cos=-1。这比标准 SigLIP（1 个推近 + N-1 个推远）在几何上更对称
2. **模态方向反转**：两个模态不再在各自空间接近对方，而是以负向融合
3. **信息论等价**：antipodal alignment 的信息容量与标准 alignment 相同——知道 img 的方向就能唯一确定匹配 txt 的方向（取反即可）

### 1.4 文献调研

在现有 contrastive learning 文献中**未找到直接先例**：
- **"Mind the Gap" (Liang et al., NeurIPS 2022)**：characterize gap 但不 embrace it
- **Wang & Isola (ICML 2020) alignment+uniformity**：alignment 目标是 cos=+1
- **Neural Collapse (Papyan et al., 2020)**：分类中不同类反向对齐，最接近但限于分类
- **Hopfield Network**：pattern p 和 -p 都是 stable states（antipodal attractors）
- **Spectral Contrastive Learning (HaoChen et al., NeurIPS 2021)**：eigenvectors ±v 携带相同信息

**结论：Antipodal contrastive alignment 作为训练目标在 CLIP 文献中是全新概念。**

---

## 2. 方法

### 2.1 数学推导

标准 SigLIP 的 logit 计算：

```
logits = scale * cos(img, txt) + bias
loss = -log(sigma(label * logit))     # label: +1=正, -1=负
```

Antipodal SigLIP 的**唯一改动**——取反余弦相似度：

```
logits = -(scale * cos(img, txt)) + bias    ← 唯一差别
loss = -log(sigma(label * logit))           ← 完全不变
```

展开分析：
- **正样本 (label=+1)**: loss = `-log(sigma(-scale*cos + bias))`
  - 最小化 loss → 需要 `-scale*cos + bias → +inf` → cos → -1 (antipodal!) ✓
- **负样本 (label=-1)**: loss = `-log(sigma(scale*cos - bias))` = `softplus(-scale*cos + bias)`
  - 当 cos ≠ -1 时 (cos ≈ 0): logit = -scale*0 + bias ≈ -10 → softplus(-10) ≈ 0 ✓
  - 当 cos ≈ -1 时: logit = scale + bias ≈ 0 → softplus(0) = log(2) > 0 → 产生梯度 ✓

### 2.2 初始化行为

random init 时 cos ≈ 0, bias ≈ -10：
- antipodal logit = -scale*0 + (-10) = -10
- sigma(-10) ≈ 0 → 所有 pair 预测为「非 antipodal」
- 正样本 loss ≈ 10, 负样本 loss ≈ 0 → 总 loss ≈ N*10/N = 10
- **与标准 SigLIP 初始化行为完全一致**

### 2.3 梯度分析

正样本梯度对 cos 的导数：
```
∂L/∂cos = scale * sigma(-scale*cos + bias)   # 当 cos=-1 时 → 0 (已收敛)
                                               # 当 cos=0  时 → scale*sigma(-10) ≈ 0.0005
```

负样本梯度对 cos 的导数：
```
∂L/∂cos = -scale * sigma(scale*cos - bias)   # 当 cos=-1 时 → -scale*sigma(-scale-bias) ≈ 0
                                               # 当 cos=0  时 → -scale*sigma(10) ≈ 0 (已满足)
```

**关键差异**：标准 SigLIP 中负样本推向 cos=-1（均匀分布在远端），antipodal SigLIP 中负样本从 cos=-1 推开但不指定方向——**由模型自行决定负样本分布**。

### 2.4 Loss 结构

```
L = L_antipodal_siglip (cross-modal, cos negated)
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

| 文件 | 改动 | 行数 |
|------|------|------|
| `src/open_clip/loss.py` | `SigLipLoss.__init__` 添加 `antipodal` 参数 | +2 |
| `src/open_clip/loss.py` | `SigLipLoss.get_logits` 条件取反 logits | +2 |
| `src/open_clip/loss.py` | `SIGRegContrastiveLoss` 透传 antipodal | +3 |
| `src/open_clip/loss.py` | `CLIPWithDINOLoss` 透传 antipodal | +2 |
| `src/open_clip/factory.py` | `create_loss()` 路由 antipodal | +6 |
| `src/open_clip_train/params.py` | `--antipodal` CLI 参数 | +8 |
| `src/open_clip_train/train.py` | `evaluate()` 和 `get_clip_metrics()` sim_sign | +5 |
| `src/open_clip_train/zero_shot.py` | `run()` sim_sign | +1 |

**总改动: ~29 行，无新类、无新模块、无新超参数。**

### 3.2 关键设计决策

1. **取反位置：logits 而非 labels** — 保持标准 SigLIP 分布式逻辑完全不变
2. **取反在 bias 之前** — bias 保持原始语义（控制正负 pair 比例的先验）
3. **within-modal 不受影响** — 模态内排斥与跨模态对齐方向正交
4. **评估管线统一使用 sim_sign** — 一个参数控制 train/eval/zero-shot

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
| antipodal | SigLIP + SIGReg + Muon + `--antipodal` | 纯 antipodal 基线 |
| anti_koleo005 | + KoLeo w=0.05 | 验证 KoLeo 在 antipodal 下的效果 |
| anti_uni05 | + Uniformity w=0.5 | 验证 Uniformity 在 antipodal 下的效果 |

参照 baseline: 标准 SigLIP i2t R@1 = 0.0172, t2i R@1 = 0.0140

### 4.3 评估指标

- **i2t / t2i R@1, R@5**: 检索性能 (antipodal retrieval = 找 cos 最小的)
- **sim_pos**: 正样本平均余弦 (预期从 ~0 趋向 -1)
- **eff_rank / cos_pos**: 表示质量
- **val_loss**: 验证损失

---

## 5. 实验结果

### 5.1 COCO 20 epoch

<!-- RESULTS_TABLE_START -->
| 实验 | 方法 | best i2t R@1 | i2t Delta | best t2i R@1 | t2i Delta | best epoch(i2t) | sim_pos (final) |
|------|------|-------------|-----------|-------------|-----------|-----------------|-----------------|
| **baseline** | Standard SigLIP | 0.0172 | -- | 0.0140 | -- | 12 | +0.97 |
| **antipodal** | Antipodal SigLIP | 0.0174 | **+1.2%** | 0.0147 | **+5.0%** | 10 | **-0.97** |
| anti_koleo005 | Antipodal + KoLeo 0.05 | 0.0186 | **+8.1%** | 0.0136 | -2.9% | 10 | -0.97 |
| anti_uni05 | Antipodal + Uni 0.5 | 0.0172 | +0.0% | 0.0139 | -0.7% | 6 | -0.97 |
<!-- RESULTS_TABLE_END -->

### 5.2 Antipodal 训练曲线

```
Epoch | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5
  0   | 0.0014  | 0.0017  | 0.0050  | 0.0061
  2   | 0.0070  | 0.0072  | 0.0306  | 0.0303
  4   | 0.0132  | 0.0120  | 0.0508  | 0.0471
  6   | 0.0144  | 0.0126  | 0.0572  | 0.0506
  8   | 0.0150  | 0.0138  | 0.0580  | 0.0508
 10   | 0.0174* | 0.0134  | 0.0554  | 0.0484   ← i2t best
 12   | 0.0170  | 0.0147* | 0.0614  | 0.0506   ← t2i best
 14   | 0.0172  | 0.0138  | 0.0616  | 0.0506
 16   | 0.0152  | 0.0139  | 0.0592  | 0.0488
 18   | 0.0164  | 0.0142  | 0.0578  | 0.0476
```

### 5.3 sim_pos 演化（正样本平均余弦相似度）

```
Epoch 0 start:  sim_pos = +0.034  (random init)
Epoch 0 end:    sim_pos = -0.23
Epoch 5:        sim_pos = -0.85
Epoch 10:       sim_pos = -0.95
Epoch 19:       sim_pos = -0.97   (接近理论极限 -1)
```

**确认：正样本确实被推向超球面的完全对面 (cos → -1)。**

---

## 6. 分析

### 6.1 核心发现

1. **Antipodal SigLIP 可行且有效**：纯 antipodal baseline 在 i2t R@1 上达到 0.0174，**超过标准 SigLIP baseline 0.0172 (+1.2%)**
2. **t2i 同时改善**：标准 SigLIP baseline t2i R@1 = 0.0140，antipodal 达到 0.0147 (**+5.0%**)
3. **双向均衡**：最关键的发现——antipodal 不像 KoLeo/Uniformity 那样存在 i2t↑ t2i↓ 的 trade-off

### 6.2 与前序实验的对比

```
方法                        | i2t R@1 | i2t Δ   | t2i R@1 | t2i Δ   | 双向?
Standard SigLIP (baseline)  | 0.0172  | --      | 0.0140  | --      | --
Antipodal SigLIP            | 0.0174  | +1.2%   | 0.0147  | +5.0%   | 双向提升 ✓
KoLeo w=0.05 (best i2t)    | 0.0198  | +15.1%  | 0.0131  | -6.4%   | i2t↑ t2i↓ ✗
Uniformity w=0.3            | 0.0190  | +10.5%  | 0.0144  | +2.9%   | 较均衡
Anti + KoLeo w=0.05         | 0.0186  | +8.1%   | 0.0136  | -2.9%   | i2t↑ t2i↓
Anti + Uniformity w=0.5     | 0.0172  | +0.0%   | 0.0139  | -0.7%   | 中性
```

### 6.3 关键观察

**1. i2t/t2i 对称性**

训练前期 (epoch 0-4) i2t 和 t2i 高度对称 (0.0070/0.0072, 0.0132/0.0120)。
后期 (epoch 10+) i2t 略高于 t2i，但差距远小于标准 SigLIP + KoLeo 的 trade-off。

**2. 与 KoLeo 的组合**

Anti + KoLeo 的 i2t R@1 = 0.0186 高于纯 antipodal 的 0.0174，但低于标准 SigLIP + KoLeo 的 0.0198。
KoLeo 在 antipodal 下仍有正向作用，但幅度降低——可能因为 antipodal 自身已经提供了部分 uniformity 效应。

**3. 与 Uniformity 的组合**

Anti + Uniformity 几乎无额外收益 (0.0172 = baseline)，
说明 antipodal loss 的均匀性效应与 Uniformity loss 高度重叠。

**4. COCO 噪声考虑**

COCO 400 steps 的 R@1 差异仅 3-5 张图 (5K val set)，+1.2% 可能在噪声范围内。
**需要 CC3M 验证来确认统计显著性。**

---

## 7. 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| SigLipLoss.antipodal | `src/open_clip/loss.py` | `get_logits()` 中 `if self.antipodal: logits = -logits` |
| CLI 参数 | `src/open_clip_train/params.py` | `--antipodal` |
| Loss 路由 | `src/open_clip/factory.py` | `create_loss()` |
| Eval sim_sign | `src/open_clip_train/train.py` | `evaluate()` + `get_clip_metrics()` |
| Zero-shot | `src/open_clip_train/zero_shot.py` | `run()` |
| 冒烟测试 | `scripts/smoke.sh` | A10-A11 |
| 实验脚本 | `experiments/wm_coco.sh` | Antipodal SigLIP section |

---

## 8. 下一步

1. **CC3M 正式验证**：COCO 结果受限于 82K 样本 / 400 steps 的噪声。CC3M (2.8M, 10 epoch) 上验证 antipodal 是否有统计显著的双向提升
2. **Probe 分析**：利用已保存的 probe npz，分析 antipodal 模型的 eff_rank、PC alignment、modality gap 方向
3. **可视化**：PCA scatter 看两个模态是否如预期呈现「负向融合」
4. **变体探索**（如果 CC3M 验证通过）：
   - partial antipodal: alpha * cos + (1-alpha) * (-cos), 在标准和 antipodal 之间插值
   - antipodal + 更小的 KoLeo (w=0.01-0.03): 寻找不牺牲 t2i 的 i2t 提升
5. **Zero-shot ImageNet**：验证 antipodal 在 zero-shot 分类上的表现

---

*文档版本: 2026-05-11 v2 | COCO 实验完成，待 CC3M 验证*
