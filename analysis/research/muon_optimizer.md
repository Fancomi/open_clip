# Muon Optimizer 集成记录

> 更新日期：2026-04-28  
> 文档范围：Muon 原理、与 Adam 的 lr 语义差异、集成实现、参数分组策略、调参指南、实验记录

---

## 1. 什么是 Muon

**Muon**（MomentUm Orthogonalized by Newton-schulz）来自 Keller Jordan 的工作：  
https://kellerjordan.github.io/posts/muon/

核心思路：在标准 SGD-momentum 基础上，对每步的 update 矩阵做**正交化后处理**，使得参数矩阵的更新方向始终接近一个正交矩阵，而非像 Adam 那样沿梯度方向的缩放步。

### Newton-Schulz 迭代（5 阶）

对动量矩阵 $G$，迭代：

$$X \leftarrow aX + (bA + cA^2)X, \quad A = XX^T$$

系数 $(a, b, c) = (3.4445, -4.7750, 2.0315)$，5 次迭代后 $X \approx UV^T$（$G = U S V^T$ 的 SVD），即**零幂/正交化**。实际输出接近 $US'V^T$，$S'_{ii} \sim \text{Uniform}(0.5, 1.5)$，对训练效果无负面影响。

正交化后乘以 $\max(m/n, 1)^{0.5}$（$m, n$ 为矩阵行列数）归一化幅度，再以 `lr` 步长更新参数。

---

## 2. Muon lr 语义与 Adam lr 的本质差异

### Adam lr

Adam 的更新规则（简化）：

$$\theta \leftarrow \theta - \text{lr} \cdot \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

`lr` 直接控制参数的**绝对步长**（参数值变化量），量纲是"参数空间的距离"。对于 768×768 的矩阵，Adam lr=3e-4 意味着每次更新中参数变化在 $\sim 10^{-4}$ 量级。

### Muon lr

Muon 的更新规则：

$$\theta \leftarrow \theta \cdot (1 - \text{lr} \cdot \text{wd}) - \text{lr} \cdot \text{orthogonalize}(\text{momentum})$$

`orthogonalize` 的输出是近正交矩阵，Frobenius norm $\approx \sqrt{\max(m, n)}$。因此 `lr` 控制的是**每步在 spectral norm 单位下的偏转幅度**，语义接近"最大奇异向量每步转多少弧度"。

**关键结论：**  
- Muon lr 与 Adam lr **不可直接比较**，数量级通常相差 2~3 个量级（Adam 1e-4 ≈ Muon 0.01~0.05）  
- Muon 官方典型值：`lr=0.02`（NanoGPT scale record）  
- 本项目实验使用：`--muon-lr 0.02`

---

## 3. 参数分组策略

Muon 的 README 明确说明：**Muon 只应用于 hidden weight layers**，以下参数必须走 AdamW：

- 输入 embedding（`embed` in name）
- 输出投影层 / logit（`logit_scale`, `logit_bias`）
- 所有 bias
- BatchNorm / LayerNorm 的 gain/bias（`bn`, `ln` in name）

### 本项目分组规则（`main.py`）

```
is_muon(n, p):  p.ndim >= 2
            AND "embed"       not in n
            AND "bn"          not in n
            AND "ln"          not in n
            AND "bias"        not in n
            AND "logit_scale" not in n
            AND "logit_bias"  not in n
```

| param_group | 参数类型 | optimizer | lr | weight_decay |
|---|---|---|---|---|
| Adam no-wd | bias, ln, logit_scale/bias, embed (1D) | AdamW | `--lr` | 0 |
| Adam wd | embed 矩阵等剩余不符合 Muon 的 2D 参数 | AdamW | `--lr` | `--wd` |
| **Muon** | hidden weight matrices（Attn QKV/proj, MLP…）| Muon | `--muon-lr` | `--wd` |

使用 `MuonWithAuxAdam`（来自 Muon 官方仓库），单 optimizer 对象统一 step，内部自动区分处理。

---

## 4. LR Scheduler 与多 param_group 的兼容

原始 `assign_learning_rate` 对所有 group 统一赋值，会破坏 Muon group 与 Adam group 的 lr 比值。

**修改策略（`scheduler.py`）：**

1. 创建 scheduler 时调用 `_record_initial_lr`，快照每个 group 的 `initial_lr` 和 `initial_lr_base`（= `base_lr` of scheduler）。
2. `assign_learning_rate` 改为按比例缩放：

```python
lr_group = initial_lr_group * (new_lr / initial_lr_base)
```

效果：Muon group 和 Adam group 的 lr 在 warmup / cosine decay 阶段**同比例变化**，各自保持独立量纲。例如：
- Adam lr: `LR_17_1 * cosine_factor`
- Muon lr: `0.02 * cosine_factor`

对已有 AdamW 单 group 训练**完全透明**（没有 `initial_lr` 时退化为原始行为）。

---

## 5. 使用方式

### CLI 新增参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--opt muon` | — | 启用 Muon optimizer |
| `--muon-lr` | 等于 `--lr` | Muon group 的学习率（spectral norm 单位） |
| `--muon-momentum` | `0.95` | Muon momentum（SGD-momentum beta） |

### 典型调用

```bash
# 最简：muon-lr 使用 0.02，Adam group 沿用 LR_17_1
run_cc3m "vit_muon" "ViT-B-16-exp" 29565 \
    "--siglip --epochs 10 --warmup 512 --lr ${LR_17_1} --opt muon --muon-lr 0.02"

# 带 lejepa + probe（当前跑的实验）
run_cc3m "vit_leproj_muon_lr002" "ViT-B-16-exp" 29557 \
    "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512 --lr ${LR_17_1} \
     --opt muon --muon-lr 0.02 --probe-data ${PROBE_TSV} --probe-freq-steps 176"
```

---

## 6. 超参调参指南

### muon-lr 选择

| 场景 | 建议 muon-lr |
|---|---|
| 从零训练（scratch） | 0.02 ~ 0.05 |
| 微调预训练权重 | 0.005 ~ 0.02（更保守） |
| 本项目 ViT-B scratch on CC3M | **0.02**（已验证有效） |

### muon-lr 与 Adam lr 的比值

以本项目 `LR_17_1 ≈ 3.4e-4`、`muon-lr=0.02` 为例：  
比值 ≈ 59x。这个比值在 NanoGPT 类实验中是合理范围（Muon lr 通常比 Adam lr 大 2~3 个量级）。

### weight_decay

Muon group 的 wd 使用与 AdamW 相同的 `--wd`（默认 0.2）。Muon 的 wd 实现为乘法衰减 `p *= (1 - lr * wd)`，与 AdamW decoupled wd 语义一致。

### momentum

固定 0.95，通常不需要调整。

---

## 7. 实验记录

| 实验名 | 模型 | optimizer | muon-lr | Adam lr | 附加 | 状态 |
|---|---|---|---|---|---|---|
| `cc3m_vit_leproj_muon_lr002_*` | ViT-B-16-exp | Muon+AdamW | 0.02 | LR_17_1 | lejepa-proj, probe | 已跑，效果好 |

> baseline 对比：`cc3m_vit_leproj_*`（同模型同数据，纯 AdamW）

---

## 8. 已知限制

1. **分布式要求**：`MuonWithAuxAdam` 使用 `dist.all_gather` 跨 rank 同步参数，**必须在 `torchrun` 多卡环境下运行**，单卡请换 `SingleDeviceMuonWithAuxAdam`（当前代码未对单卡路径做特殊处理，但本项目始终 8 卡训练，不影响）。

2. **梯度裁剪**：`--grad-clip-norm` 作用于梯度，在 Muon 的正交化之前生效，行为正确。

3. **checkpoint 兼容性**：optimizer state 结构与 AdamW 不同（`momentum_buffer` vs `exp_avg`/`exp_avg_sq`），AdamW 的 checkpoint **不能直接 resume** 为 Muon，反之亦然。

4. **AMP scaler**：Muon 内部用 bfloat16 做 Newton-Schulz 迭代，对 `amp_bf16` precision 友好；`amp`（fp16）时 scaler 对 Adam group 生效，Muon group 的 NS 迭代强制 bf16，两者互不干扰。
