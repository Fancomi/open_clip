# Within-Modal Repulsion: 研究记录

*最后更新: 2026-05-06 | 实验平台: COCO quick (wm_coco.sh) + CC3M (modality_gap.sh)*

---

## 1. 背景

### 1.1 Modality Gap 现象

对任意 CLIP 范式模型（CLIP、OpenCLIP、SigLIP、PE-Core、TIPSv2）在 CC3M 或 COCO 50w 数据上提取图文特征并做联合 PCA，**PC1 始终是图文模态轴**：

- 一侧投影在 −0.4~−0.6，另一侧在 +0.4~+0.6
- 两族之间无重叠，PC1 threshold classifier accuracy ≈ 1.0
- 仅 PC1 如此，且无论模型/数据集组合均稳定复现

实测（CC3M 从头训练，各架构 final checkpoint）：

| 模型 | pc1_gap | pc1_clf | var_pc1 |
|------|---------|---------|---------|
| PE-dinov3 leproj muon | 0.651 | 1.000 | 11.0% |
| PE-dinov3 dinov3 muon | 0.616 | 1.000 | 10.1% |
| ViT (standard) | 0.703 | 1.000 | 12.9% |

### 1.2 成因

SigLIP / CLIP loss 结构天然驱动 gap 形成：

```
Full N×N similarity matrix:
  - 对角线（N 个正样本）：图文对齐 → 拉近匹配对
  - 非对角线（N(N-1) 个跨模态负样本）：图文排斥 → 推远非匹配对
```

N(N-1) 个跨模态负样本提供了强大的 "图文分离" 驱动力。在 L2-normalized 超球面上，模态轴是最廉价的判别维度——它可以同时降低所有负样本的余弦相似度，而 loss 不惩罚这一自由度。

**关键条件**：gap 的形成和维持需要**足够的数据多样性**。在 CC3M（2.8M 样本）上 gap 从 epoch 2 单调增长；在 COCO（82K 样本）上 gap 先形成（epoch 1-6）后消退（epoch 7-20），因为小数据导致负样本被"记忆化"，正样本对齐最终主导（见 §4）。

### 1.3 目标

把**模态轴占用的表示空间解放出来**，让内容语义承担更多权重。
不是强行让图文分布完全重叠（可能损伤语义对齐），而是消除 gap 形成的结构性驱动力。

---

## 2. 第一阶段：Modality Gap Loss（CC3M）

### 2.1 方法

```
L_gap = || mean(img_raw) - mean(txt_raw) ||²   (pre-L2-norm, batch level)
```

直接最小化两个模态 batch 均值之间的距离。

### 2.2 实验结论（CC3M，10 epoch）

- `λ=0.005`：i2t R@1 **+0.92%**（CC3M 实验历史最优）
- `λ=0.05`：过强，损害对齐
- Gap loss（λ=0.01）的 effective rank 显著高于 baseline（164 vs 109），但 gap 仍然形成（pc1_gap=0.48, clf=1.0）
- Gap loss 是正向有效的 **auxiliary regularizer**，但不从根本上改变 loss 结构——gap 仍会形成，只是幅度被压缩

---

## 3. 第二阶段：Within-Modal Repulsion

### 3.1 核心思想

用**同模态排斥**替换**跨模态排斥**，消除 gap 的结构性驱动力：

```
当前实现（within_modal_weight > 0 时自动切换）：
  L = cross_pos_only（仅对角线 N 个正样本）
    + λ × wm_loss（同模态排斥）

其中 wm_loss（取决于 --within-modal-sides）：
  'both': 0.5 × (wm_img + wm_txt)
  'img':  wm_img
  'txt':  wm_txt
```

**注意**：当前实现在启用 within-modal 时会**完全去掉跨模态负样本**（切换为 cross_pos_only）。这是一个重要的设计选择，后续分析表明这可能过于激进（见 §5）。

### 3.2 Scale/Bias 稳定性分析

**实验验证的结论**：

bias 同步传入 within-modal 是维持 scale 稳定的正确设计。不传 bias 会导致 scale 崩塌（10→5.5）。传入后所有实验 scale 稳定在 9.4~10.5，bias 稳定在 −10.02~−10.06。

**机制**：共享的 `logit_bias ≈ -10` 导致 within-modal loss 的 sigmoid 高度饱和：

```
within-modal pair: logit = s × cos_wm + b ≈ 10 × 0.2 + (-10) = -8
σ(-8) ≈ 3.4e-4  →  softplus(-8) ≈ 3.4e-4（极小）
```

这意味着 within-modal 对 scale/bias 的梯度贡献**微乎其微**，bias 实际上被正样本单向锁定。scale 不崩的原因不是"正负样本双向平衡"，而是 sigmoid 饱和导致 within-modal 对 scale 几乎无梯度——即**被动保护**，非主动平衡。

**推论**：需要极大的 λ（如 20~30）才能让 within-modal 产生有意义的梯度，因为每个 pair 的贡献被 sigmoid 压制了 ~3000 倍。

### 3.3 设计 bug 修复历程

| 版本 | 问题 | 现象 | 修复 |
|------|------|------|------|
| v1（no-bias 解耦） | within-modal 不传 logit_bias | scale 从 10→5.5 崩塌，R@1≈0.001 | 恢复传 logit_bias |
| v2（当前） | bias 同步传入，三者共享 scale/bias | scale 稳定在 9.4~10.5 | ✓ |

---

## 4. COCO 快速实验

### 4.1 实验设置

- 数据：COCO train（~82K），val：Karpathy 5cap（5K 图 / 25K 文）
- 模型：PE-Core-B-16-dinov3（**从头训练**，random init）
- 基础配置：SigLIP + SIGReg(cls, 1e-4) + Muon，20 epoch，BS=4096
- 训练规模：20 steps/epoch × 20 epochs = **400 total steps**

### 4.2 COCO 平台的根本局限

**COCO baseline 自身没有 modality gap**。Gap 先形成后消退：

| epoch | eff_rank | pc1_gap | modal_clf | cos_pos |
|-------|----------|---------|-----------|---------|
| 1 | 38.0 | 0.180 | 1.000 | 0.203 |
| 5 | 38.0 | 0.146 | 1.000 | 0.309 |
| 6 | 41.2 | 0.286 | 1.000 | 0.295 |
| 9 | 63.0 | 0.096 | 0.843 | 0.238 |
| 12 | 84.8 | 0.008 | 0.672 | 0.194 |
| 15 | 99.4 | 0.004 | 0.629 | 0.165 |
| 20 | 105.3 | 0.005 | 0.641 | 0.146 |

对比 CC3M baseline（同架构）：

| epoch | eff_rank | pc1_gap | modal_clf | cos_pos |
|-------|----------|---------|-----------|---------|
| 2 | 74.6 | 0.341 | 0.997 | 0.390 |
| 5 | 90.3 | 0.567 | 1.000 | 0.305 |
| 10 | 109.4 | 0.616 | 1.000 | 0.205 |

**原因**：COCO 仅 82K 样本，训练 20 epoch 后每个图文对被见 20 次。负样本对完全重复，跨模态排斥的"免费维度"优势消失，正样本对齐最终主导。CC3M 有 2.8M 样本，负样本组合空间远大于训练步数，gap 持续存在。

**结论：COCO 不适合验证 anti-gap 方法。WM 实验的 R@1 变化不能解释为 "消除 gap 的收益"，因为 baseline 本身就没有 gap。**

### 4.3 R@1 汇总结果

<!-- RESULTS_TABLE_START -->
| 实验 | sides | λ | i2t R@1 | t2i R@1 | i2t R@5 | t2i R@5 | val_loss | Scale | Epoch |
|------|-------|---|---------|---------|---------|---------|----------|-------|-------|
| baseline | — | 0 | 0.0168 | 0.0146 | 0.0568 | 0.0520 | 0.8295 | 10.5230 | 19 |
| ada002 | ? | 0 | 0.0000 | 0.0001 | 0.0008 | 0.0009 | 9.9804 | — | 18 |
| ada005 | ? | 0 | 0.0004 | 0.0005 | 0.0010 | 0.0012 | 9.9804 | — | 18 |
| ada01 | ? | 0 | 0.0000 | 0.0003 | 0.0012 | 0.0012 | 9.9804 | — | 18 |
| ada02 | ? | 0 | 0.0002 | 0.0003 | 0.0012 | 0.0012 | 9.9804 | — | 18 |
| ada05 | ? | 0 | 0.0004 | 0.0002 | 0.0018 | 0.0012 | 9.9804 | — | 18 |
| ada100 | ? | 0 | 0.0004 | 0.0002 | 0.0010 | 0.0011 | 9.9135 | — | 2 ★ |
| ada10 | ? | 0 | 0.0002 | 0.0003 | 0.0006 | 0.0011 | 9.9804 | — | 18 |
| ada1 | ? | 0 | 0.0008 | 0.0004 | 0.0026 | 0.0010 | 9.9804 | — | 18 |
| ada20 | ? | 0 | 0.0006 | 0.0004 | 0.0016 | 0.0010 | 9.9804 | — | 18 |
| ada2 | ? | 0 | 0.0000 | 0.0002 | 0.0010 | 0.0010 | 9.9804 | — | 18 |
| ada50 | ? | 0 | 0.0004 | 0.0003 | 0.0008 | 0.0010 | 9.9804 | — | 18 |
| ada5 | ? | 0 | 0.0000 | 0.0002 | 0.0012 | 0.0010 | 9.9804 | — | 18 |
<!-- RESULTS_TABLE_END -->

### 4.4 表示几何分析（epoch 20 probe）

| 实验 | eff_rank (joint) | eff_rank (img) | eff_rank (txt) | cos_pos | modal_clf |
|------|:---:|:---:|:---:|:---:|:---:|
| **baseline** | **105.3** | **105.2** | **101.9** | **0.146** | **0.641** |
| txt750 | 46.4 | 31.1 | 62.3 | 0.354 | — |
| txt1500 | 60.0 | 36.5 | 84.5 | 0.329 | — |
| txt2000 | 66.8 | 39.3 | 95.1 | 0.319 | — |
| txt2500 | 71.4 | 41.2 | 102.1 | 0.313 | 0.591 |
| txt3000 | 75.8 | 43.2 | 108.6 | 0.306 | 0.581 |
| img550 | 55.9 | 57.5 | 53.3 | 0.144 | 0.612 |
| img750 | 63.2 | 65.7 | 59.4 | 0.131 | — |
| wm2 (both) | 33.4 | 33.0 | 33.4 | 0.227 | 0.569 |
| wm075 (both) | 23.2 | 22.9 | 23.2 | 0.300 | 0.558 |

### 4.5 关键观察

**① txt-only: R@1 提升伴随严重的图像侧维度崩塌**

```
                 eff_rank_img    cos_pos    i2t R@1
baseline:          105.2          0.146      0.0168
txt3000:            43.2          0.306      0.0192 (+14%)
txt750:             31.1          0.354      0.0144 (-14%)
```

- cos_pos 翻倍（0.146→0.306~0.354），正样本对齐显著增强
- 但图像侧 eff_rank 崩塌到 baseline 的 30~41%
- λ 越大，img eff_rank 崩塌越轻（txt3000: 43 > txt750: 31）——反直觉，因为更大 λ 的 txt repulsion 推散了文本，给图像更多"空间"
- 文本侧 eff_rank 反而上升（txt3000: 109 > baseline: 102）——txt repulsion 确实在推散文本

**② img-only: 全面崩溃**

所有 λ 下 R@1 低于 baseline 2-5 倍，cos_pos 无改善（0.131~0.144 vs baseline 0.146）。

**③ both-sides: 最严重的维度崩塌**

eff_rank 20~33（baseline 105），两侧同时崩塌。

**④ scale/bias 稳定**

所有实验 scale 9.44~10.52，bias −10.02~−10.06。确认 bias-sharing 设计正确。

**⑤ cross_pos loss 升高反映了结构性变化**

当前实现去掉 cross-neg 后，cross_pos loss 就是 `-logsigmoid(s·cos_pos + b)`。
cos_pos 在 txt-only 中实际**提升**了（0.146→0.306），所以 cross_pos loss 应该**下降**。
如果 log 中 contrastive_loss 升高，需要确认是否是 within-modal_loss 的贡献被计入。

---

## 5. 机制分析

### 5.1 去掉 cross-neg 的代价：图像侧维度崩塌

当前实现在 `within_modal_weight > 0` 时**完全去掉跨模态负样本**。这导致：

**图像特征失去跨模态判别信号**：
- 标准 SigLIP 中，cross-neg `softplus(s·img_i⊤txt_j + b)` 给 img_i 的梯度方向是"远离 txt_j"
- 这让每个 img_i 不仅要与 txt_i 对齐，还要**与所有错误的 txt_j 可分**
- 去掉后，img_i 唯一的梯度来源是 cross_pos（拉向 txt_i）

**图像塌缩到文本子空间**：
- cross_pos 把每个 img_i 拉向对应的 txt_i
- 没有其他力量维持图像间的相互区分
- 图像特征丧失自主结构，塌缩为文本分布的低维映射
- 实测：img eff_rank 从 105 降到 31~43

**txt-only 为什么 R@1 能提升**：
- txt repulsion 推散文本特征 → txt eff_rank 上升（101→109）
- 图像追随文本分布排列（因为只有 cross_pos 梯度）
- cos_pos 大幅提升（0.146→0.306）说明正样本对齐更紧
- 在一个低维但结构化的子空间中，"近邻正确配对"的概率提升 → R@1 提升
- **但这是以牺牲表示容量为代价的**——图像侧丧失了大量维度

### 5.2 img-only 崩溃的原因

与 txt-only 的关键区别：

- **txt-only**：文本被推散提供了"锚点"分布，图像追随形成结构 → 低维但有序
- **img-only**：图像被推散但文本不受控（文本只有 cross_pos 拉向图像），txt 聚集为低 rank blob，img 试图追逐一个坍塌的目标 → 混乱

本质上是**谁提供稳定锚点**的问题。txt-only 中文本被 repulsion 固定在相对稳定的均匀分布上，图像可以有效追踪；img-only 中图像在快速移动，文本追不上。

### 5.3 为什么 COCO baseline gap 先形成后消退

| 因素 | COCO | CC3M |
|------|------|------|
| 数据量 | 82K | 2.8M |
| 训练 steps | 400 | 6830 |
| 每样本见次数 | ~20 次 | ~2.5 次 |
| 负样本多样性 | 低（重复对） | 高 |

Gap 的维持需要负样本提供**持续的、多样的**"模态可分"梯度。在 COCO 上：
- 前 6 epoch：每个 batch 的负样本对组合还有新意 → gap 正常形成
- 后 14 epoch：所有可能的负样本组合都已被充分覆盖 → 模型"记住"了所有配对关系 → 正样本对齐压过负样本排斥 → gap 消退

这类似于 "小数据集上 contrastive learning 过拟合" 的已知现象。

### 5.4 sigmoid 饱和问题

共享 `logit_bias ≈ -10` 的严重后果：within-modal 每个 pair 的有效梯度被压制 ~3000 倍。

```
标准 cross-neg pair (cos≈-0.2): softplus(10×(-0.2)+(-10)) = softplus(-12) ≈ 6e-6
within-modal pair  (cos≈+0.2): softplus(10×0.2+(-10))     = softplus(-8)  ≈ 3.4e-4
```

虽然 within-modal 比 cross-neg 梯度大 ~50 倍（因为 cos 更高），但两者都被 bias 压制到极小值。
这就是为什么需要 λ=20~30 才能看到效果——补偿 sigmoid 饱和。

---

## 6. 结论与下一步

### 6.1 确定结论

1. **COCO 不适合验证 anti-gap 方法**：baseline 无 gap，实验结果不可推广到 CC3M
2. **去掉 cross-neg 过于激进**：导致图像侧维度崩塌（eff_rank 105→31~43）
3. **Scale/bias 共享设计正确**：维持了数值稳定性
4. **Sigmoid 饱和迫使使用极大 λ**：共享 bias 的副作用
5. **txt-only R@1 提升是 "低维紧凑对齐" 的结果**，不是"消除 gap 的收益"

### 6.2 下一步实验计划

**Phase 1（最高优先级）：方向 C — 保留 full SigLIP + 叠加 txt-only WM**

核心改动：不去掉 cross-neg，只额外叠加 within-modal 作为 auxiliary loss。

```python
# 目标 loss 结构
L = standard_siglip(img, txt, scale, bias)        # 完整 N×N（保留 cross-neg）
  + λ_txt × within_modal(txt, scale, bias)        # 额外 txt repulsion
```

| 实验 | λ_txt | 平台 | 预期 |
|------|-------|------|------|
| P1.1 | 1.0 | CC3M | 保守起步，验证 img eff_rank 不崩 |
| P1.2 | 5.0 | CC3M | 中等强度 |
| P1.3 | 20.0 | CC3M | 对标当前 txt-only 的有效梯度量级 |

验收标准：
- img eff_rank ≥ baseline
- pc1_gap < baseline
- i2t R@1 ≥ baseline

**Phase 2：解耦 within-modal 的 bias**

sigmoid 饱和问题的根源是共享 bias=-10。探索 within-modal 使用独立参数化。

| 实验 | 改动 | 预期 |
|------|------|------|
| P2.1 | within-modal bias=0 (fixed) | 梯度不再被压死，λ 可以小很多 |
| P2.2 | within-modal 独立可学习 bias | 自适应 |
| P2.3 | 纯余弦 repulsion（scale=1, no bias） | 最简单形式 |

注意：Phase 2 需要在 Phase 1 的框架下做（保留 cross-neg）。

**Phase 3：替代 uniformity loss**

SigLIP-style softplus 在 sigmoid 饱和区不是有效的 uniformity loss。

| 实验 | loss 形式 | 参考 |
|------|----------|------|
| P3.1 | `log(mean(exp(-2*‖t_i-t_j‖²)))` | Wang & Isola 2020 |
| P3.2 | Eigenvalue entropy maximization | 直接优化 eff_rank |

### 6.3 实验优先级

```
Phase 1 (保留 cross-neg + txt-wm) >>> Phase 2 (解耦 bias) > Phase 3 (替代 loss)
```

Phase 1 是验证核心假说的最小改动方案。如果 Phase 1 在 CC3M 上成功缩小 gap 且不损害 R@1，再进入 Phase 2/3 做精细优化。

---

## 7. 参数接口（当前实现）

```bash
# 基础 within-modal 配置
--within-modal-weight FLOAT    # λ，默认 0.0（禁用）
--within-modal-sides  {both|img|txt}  # 默认 both

# 当前行为（within_modal_weight > 0 时）：
#   cross_modal 切换为 positive-only（仅对角线）  ← Phase 1 需要修改这里
#   within_modal 使用 logit_scale + logit_bias（同 SigLIP）
```

实验脚本：
- 快速验证（COCO, ~15min/run）：`experiments/wm_coco.sh`
- 正式实验（CC3M, ~6hr/run）：`experiments/modality_gap.sh`

---

## 8. 关键代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| within-modal loss 计算 | `src/open_clip/loss.py` | `SIGRegContrastiveLoss._within_modal_siglip` |
| cross-modal positive-only | `src/open_clip/loss.py` | `SIGRegContrastiveLoss._cross_modal_positive_only` |
| forward 分支逻辑 | `src/open_clip/loss.py` | `SIGRegContrastiveLoss.forward` (within_modal_weight > 0) |
| 参数注册 | `src/open_clip_train/params.py` | `--within-modal-weight`, `--within-modal-sides` |
| loss 工厂 | `src/open_clip/factory.py` | `create_loss` → `within_modal_sides` |

---

*文档版本：2026-05-06 v2 | 代码 commit: 37309b6*
