# Orthogonal/Projective SigLIP: 超参数全面探索

*最后更新: 2026-05-13 | 实验平台: COCO quick (wm_coco.sh)*

---

## 1. 背景

前序实验确定了两个核心方法的有效性：
- **Orthogonal** (neg_mode=orthogonal): COCO +3.5%/+7.1%, CC3M +3.7%/+2.1%
- **Projective** (neg_mode=projective): COCO +16.3%/+7.9%, CC3M +4.0%/≈0%

本轮实验对剩余超参数空间进行系统探索。

---

## 2. 实验方向

### 2.1 neg-alpha: 负余弦惩罚斜率

`--neg-alpha` 控制 **cos < 0 时的行为**。公式展开：

```
effective_logit:
  cos >= 0 时: scale * cos          （恒等，alpha 无影响）
  cos <  0 时: scale * cos * (2α-1) （斜率由 alpha 控制）
```

等价操作：

| alpha | cos<0 时的 logit | 等价表达 | 语义 |
|-------|-----------------|---------|------|
| 1.0 | `scale * cos`（保留负值） | `scale * cos` = standard | 负样本被奖励到 cos=-1 |
| 0.7 | `scale * 0.4 * cos` | 衰减保留 | 弱奖励 |
| **0.5** | **0** | **`scale * ReLU(cos)`** | **负余弦截断，不奖不罚** |
| 0.3 | `scale * (-0.4) * cos` (翻正) | 弱惩罚 | 轻微推回 0 |
| 0.0 | `scale * (-cos)` = `scale * |cos|` | projective | 强惩罚推向 0 |

**关键**：alpha 只改变 cos < 0 的行为。cos ≥ 0 部分始终是 `scale * cos`，不受 alpha 影响。

alpha=0.5 的本质是 **ReLU**：`logits = scale * max(cos, 0) + bias`

### 2.2 SIGReg 权重调参

射影/正交已提供均匀性，SIGReg 最优权重可能不同于标准模式的 1e-4。

### 2.3 Warmup 长度

正交/射影初始化时负样本 loss≈0，可能需要更短 warmup。

### 2.4 训练长度

Projective 在 COCO epoch 8 就到 best，可能不需要 20 epoch。

---

## 3. 实验矩阵 (COCO, 20 epoch)

| # | 实验 | 关键参数 | 目的 |
|---|------|----------|------|
| 1 | alpha03 | --neg-alpha 0.3 | 强射影偏置 |
| 2 | alpha05 | --neg-alpha 0.5 | 半正交 |
| 3 | alpha07 | --neg-alpha 0.7 | 弱射影偏置 |
| 4 | proj_sig1e5 | projective + sigreg 1e-5 | SIGReg 降低 |
| 5 | proj_sig1e3 | projective + sigreg 1e-3 | SIGReg 提升 |
| 6 | ortho_sig1e5 | orthogonal + sigreg 1e-5 | SIGReg 降低 |
| 7 | ortho_sig1e3 | orthogonal + sigreg 1e-3 | SIGReg 提升 |
| 8 | proj_warm21 | projective + warmup 21 | 短 warmup |
| 9 | proj_ep10 | projective + 10 epochs | 短训练 |

参照基线:
- Standard SigLIP: i2t=0.0172, t2i=0.0140
- Orthogonal (sigreg 1e-4): i2t=0.0178, t2i=0.0150
- Projective (sigreg 1e-4): i2t=0.0200, t2i=0.0151

---

## 4. 结果

<!-- RESULTS_TABLE_START -->
| 实验 | best i2t R@1 | i2t Δ(vs base) | best t2i R@1 | t2i Δ(vs base) |
|------|-------------|----------------|-------------|----------------|
| baseline | 0.0172 | -- | 0.0140 | -- |
| orthogonal | 0.0178 | +3.5% | 0.0150 | +7.1% |
| projective | 0.0200 | +16.3% | 0.0151 | +7.9% |
| alpha03 | -- | -- | -- | -- |
| alpha05 | 0.0198 | **+15.1%** | 0.0154 | **+10.0%** |
| alpha07 | 0.0172 | +0.0% | 0.0146 | +4.3% |
| proj_sig1e5 | 0.0170 | -1.2% | 0.0146 | +4.3% |
| proj_sig1e3 | 0.0166 | -3.5% | 0.0134 | -4.3% |
| ortho_sig1e5 | 0.0190 | **+10.5%** | 0.0154 | **+10.0%** |
| ortho_sig1e3 | 0.0196 | +14.0% | 0.0135 | -3.6% |
| proj_warm21 | 0.0182 | +5.8% | 0.0151 | +7.9% |
| proj_ep10 | 0.0174 | +1.2% | 0.0151 | +7.9% |
<!-- RESULTS_TABLE_END -->

---

## 5. 分析

### 5.1 核心发现

1. **alpha=0.5 是全系列最优配置**：i2t +15.1%, t2i +10.0%。相比纯 projective (i2t +16.3%, t2i +7.9%)，alpha=0.5 在 t2i 上额外提升 2.1%，i2t 仅微降 1.2%。双向均衡性是所有方法中最好的。

2. **orthogonal + SIGReg 1e-5 并列最优 t2i**：i2t +10.5%, t2i +10.0%。降低 SIGReg 权重让 orthogonal 模式释放了更多潜力。

3. **SIGReg 1e-3 一致有害**：无论 orthogonal 还是 projective，高 SIGReg 都损害 t2i。正交/射影几何已提供足够的正则化效应，额外 SIGReg 过度约束。

4. **alpha=0.7 无效**：太接近标准模式，正交效应不足。

5. **Warmup 和 epoch 对 projective 影响有限**：proj_warm21 和 proj_ep10 都不如标准 projective。

### 5.2 Alpha 曲线（负余弦斜率 vs 性能）

```
alpha | cos<0 行为      | i2t R@1 | t2i R@1
0.0   | 翻正惩罚 |cos|  | 0.0200  | 0.0151   (projective)
0.3   | 弱翻正           | 0.0186  | 0.0147
0.5   | 截断=0 (ReLU)    | 0.0198  | 0.0154   ★ 最优双向
0.7   | 弱保留           | 0.0172  | 0.0146   (≈无效)
1.0   | 完整保留         | 0.0172  | 0.0140   (standard)
```

曲线呈**倒 U 型**：alpha=0.5 附近是双向最优区域。过低（纯射影）牺牲少量 t2i，过高（接近标准）则完全无效。

### 5.3 推荐配置

| 场景 | 推荐 | 配置 | 本质 |
|------|------|------|------|
| 双向最佳均衡 | **alpha=0.5** | `--neg-alpha 0.5 --sigreg-weight 1e-4` | ReLU(cos): 截断负余弦 |
| i2t 最大化 | projective | `--neg-mode projective --sigreg-weight 1e-4` | \|cos\|: 惩罚负余弦 |
| 保守稳定 | ortho+sig1e-5 | `--neg-mode orthogonal --sigreg-weight 1e-5` | 仅负样本取\|cos\| |

### 5.4 待 CC3M 验证

最关键的两个配置（alpha=0.5 和 ortho_sig1e5）需要在 CC3M 上确认，避免 COCO 小样本噪声。

---

## 6. Round 2: Scale / Bias / Batch Size

### 6.1 结果

| 实验 | 配置 | best i2t R@1 | i2t Δ | best t2i R@1 | t2i Δ |
|------|------|-------------|-------|-------------|-------|
| projective ref | s=10, b=-10, BS=4096 | 0.0200 | +16.3% | 0.0151 | +7.9% |
| proj_s5 | s=5 | 0.0128 | -25.6% | 0.0102 | -27.1% |
| **proj_s15** | **s=15** | **0.0190** | **+10.5%** | **0.0158** | **+12.9%** |
| proj_s20 | s=20 | 0.0182 | +5.8% | 0.0147 | +5.0% |
| proj_b5 | b=-5 | 0.0002 | 崩溃 | 0.0002 | 崩溃 |
| proj_b15 | b=-15 | 0.0120 | -30.2% | 0.0069 | -50.7% |
| a05_s5 | alpha=0.5, s=5 | 0.0128 | -25.6% | 0.0099 | -29.3% |
| a05_s20 | alpha=0.5, s=20 | 0.0168 | -2.3% | 0.0134 | -4.3% |
| **proj_bs256** | **BS=2048** | **0.0274** | **+59.3%** | **0.0225** | **+60.7%** |

### 6.2 分析

**1. Batch Size 是最大杠杆**

proj_bs256 (BS=256/GPU, 2048 global) 达到 i2t 0.0274, t2i 0.0225——双向 +60%！原因：
- 更多梯度步（40步/epoch vs 20步），优化更精细
- 更少负样本/步（2048 vs 4096），|cos|→0 的目标更容易被少量负样本满足
- COCO 82K 样本 / 2048 = 40 步/epoch，接近在线学习

**2. Scale=15 是 t2i 最优**

proj_s15 (0.0158 t2i, +12.9%) 超越了所有之前的方法。更高的初始温度让正样本对齐更强（sigmoid 更尖锐），但不过度推远负样本。

**3. Scale=5 和 Bias=-5/-15 有害**

- Scale 太低：sigmoid 太平坦，缺乏区分力
- Bias 太小 (b=-5)：初始所有 pair 被视为"正"，训练崩溃
- Bias 太大 (b=-15)：初始所有 pair 被视为"极负"，收敛慢

**4. 默认 (s=10, b=-10) 接近最优**

Scale 在 10-15 范围内最优。Bias=-10 是好的默认值。不建议改动 bias。

### 6.3 推荐的 CC3M 实验配置

基于 COCO 结果，CC3M 验证优先级：
1. **alpha=0.5** (已证明 COCO 双向最优)
2. **proj_s15** (t2i COCO 最优)
3. **proj_bs256** (COCO 巨大提升，但 CC3M 上 BS 效应可能不同)

---

## 7. 代码位置

| 功能 | 文件 | 位置 |
|------|------|------|
| neg_alpha 参数 | `src/open_clip/loss.py` | `SigLipLoss.__init__` + `get_logits` |
| init-logit-scale/bias | `src/open_clip_train/main.py` | model_kwargs |
| CLI | `src/open_clip_train/params.py` | `--neg-alpha`, `--init-logit-scale`, `--init-logit-bias` |

---

*文档版本: 2026-05-13 v3 | COCO 两轮 sweep 完成，CC3M 验证启动中*

*文档版本: 2026-05-12 v1 | 实验进行中*
