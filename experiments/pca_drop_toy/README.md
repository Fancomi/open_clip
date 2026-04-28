# PCA Regularization Sandbox — Toy Experiments

> **隔离沙盒**：本目录与 OpenCLIP 主训练代码完全隔离。  
> 除非明确进入「OpenCLIP 接入阶段」，**不修改** `src/open_clip*` 下任何文件。

---

## 目录结构

```
experiments/pca_drop_toy/
├── configs/                   # 实验配置（YAML）
│   ├── toy_baseline.yaml              # 无 PCA，baseline
│   ├── toy_nuisance_high_variance.yaml  # Dataset B + attenuate_topk
│   ├── toy_signal_high_variance.yaml    # Dataset A + attenuate_topk（风险测试）
│   ├── toy_mixed.yaml                   # Dataset C + attenuate_topk（虚假相关）
│   ├── toy_drop_topk.yaml               # Dataset B + drop_topk
│   ├── toy_drop_all_weighted.yaml       # Dataset B + drop_all_pc_weighted
│   └── toy_momentum_ablation.yaml       # momentum 消融
├── scripts/
│   └── run_experiments.sh     # 一键运行全套实验
├── tests/
│   └── test_pca_regularizer.py  # 单元测试
├── datasets.py                # 合成数据集 A/B/C/D
├── models.py                  # MLP + PCARegularizer/PCADrop 插入
├── momentum_pca.py            # MomentumPCAStats（核心：EMA 协方差 + 特征分解）
├── pca_regularizer.py         # PCARegularizer（4 种模式）
├── pca_drop.py                # 旧版单 batch PCADrop（保留作对比）
├── train.py                   # 训练循环（输出 JSONL + summary.json）
├── metrics.py                 # 谱分析指标（effective rank、EVR、spurious alignment）
└── plot.py                    # 可视化（可选）
```

---

## 合成数据集设计

| 数据集 | 描述 | 预期结果 |
|--------|------|----------|
| **A** | top PCs = label signal；suppression 应**降低**精度 | 验证风险边界 |
| **B** | top PCs = nuisance；label 在低方差方向 | suppression 应**提升** OOD 精度 |
| **C** | 训练集有虚假高方差特征（spurious corr=0.9），测试集无 | suppression 应**提升**测试精度 |
| **D** | 各向同性高斯，无 spurious structure | 对照组，PCA-drop 与普通 dropout 效果相似 |

---

## 核心模块

### `MomentumPCAStats`

维护一个 EMA 协方差矩阵，周期性做特征分解，提供稳定的 PCA 基：

```
C_t = β · C_{t-1} + (1-β) · C_batch
C_t = V Λ V^T   (每 update_every 步更新一次)
```

- **不参与梯度反传**（buffer，不是 parameter）
- **float32 内部**（AMP 安全）
- **warmup_steps**：训练早期不应用正则化

### `PCARegularizer`

| mode | 公式 | 超参 |
|------|------|------|
| `none` | identity | — |
| `attenuate_topk` | `H' = H_c - α·H_c V_k V_k^T + μ` | `top_k`, `alpha` |
| `drop_topk` | 对前 k 个 PC 坐标做随机 mask | `top_k`, `drop_prob` |
| `drop_all_pc_weighted` | 按 λ 大小比例分配 drop prob | `max_drop_prob`, `min_drop_prob` |

- `train_only=True`（默认）：**eval 时自动退化为 identity**
- `detach_basis=True`（始终）：PCA basis 不反传
- NaN guard：数值失败时自动 fallback 到 identity

---

## 快速开始

```bash
cd experiments/pca_drop_toy

# 1. 单元测试（应全部通过）
python -m pytest tests/ -v

# 2. 最小 smoke test（5 epochs，验证代码能跑）
bash scripts/run_experiments.sh smoke

# 3. 运行 baseline + 核心对比实验
bash scripts/run_experiments.sh baseline

# 4. 运行全套实验（含 sweep）
bash scripts/run_experiments.sh all

# 5. 单独运行某个配置并覆盖超参
python train.py --config configs/toy_nuisance_high_variance.yaml \
    --seed 1 --pca_top_k 8 --pca_alpha 0.5 --pca_momentum 0.99
```

---

## 超参空间

```
pca_mode:        none / attenuate_topk / drop_topk / drop_all_pc_weighted
top_k:           1 / 2 / 4 / 8 / 16
alpha:           0.1 / 0.3 / 0.5 / 1.0
drop_prob:       0.05 / 0.1 / 0.2 / 0.5
momentum:        0.9 / 0.99 / 0.995
apply_location:  input (pca_insert_after=[0]) / penultimate (pca_insert_after=[-2])
batch_size:      64 / 128 / 256
```

---

## 输出格式

每次实验在 `outputs/<run_name>/seed<N>/` 下生成：

```
train_log.jsonl   # 逐 epoch 日志（含 train/val/ood acc、PCA 谱、effective rank）
summary.json      # 最终汇总（test acc、best val acc、特征指标）
best_model.pt     # 最优 val acc 对应的模型权重
```

`train_log.jsonl` 字段示例：
```json
{
  "epoch": 40,
  "train_loss": 0.312,   "train_acc": 0.881,
  "val_loss": 0.341,     "val_acc": 0.864,
  "ood_loss": 0.521,     "ood_acc": 0.712,
  "pca/effective_rank": 14.3,
  "pca/expl_var_ratio": 0.41,
  "pca/spectrum_top8": [12.1, 3.2, 1.1, 0.8, 0.6, 0.4, 0.3, 0.2],
  "input/effective_rank": 8.1,
  "input/explained_var_ratio": 0.72,
  "grad_norm": 0.043,
  "elapsed": 12.4
}
```

---

## 科学问题与实验对应关系

| 科学问题 | 关键实验 | 关键指标 |
|----------|----------|----------|
| 抑制高方差 PC 能否减少 spurious 依赖？ | Dataset B baseline vs attenuate | ood_acc ↑ |
| PCA suppression 的风险边界？ | Dataset A + attenuate_topk | val_acc ↓（预期） |
| random PC drop vs 普通 dropout？ | Dataset B drop_topk vs 标准 dropout | val/ood acc |
| 动量 PCA vs 单 batch PCA？ | momentum ablation + backend=batch | 训练稳定性、loss 曲线 |
| PC 处理能否提升 effective rank？ | 所有实验的 pca/effective_rank | er ↑ 表示 embedding 更均匀 |
| drop_all_pc_weighted 的效果？ | Dataset B drop_all_weighted | ood_acc + er |

---

## 进入 OpenCLIP 阶段的条件

满足以下**任一**正向条件，才进入 Stage 3（OpenCLIP 接入）：

- Dataset B/C 上 OOD accuracy 明显优于 baseline（>2%）
- 训练过程无明显不稳定（loss 无爆炸，无 NaN）
- effective rank 上升
- 计算开销可接受（wall-clock time 增量 < 20%）

若出现以下情况，先在 toy 阶段修正：

- 所有 setting 均显著欠拟合
- top PC 抑制导致 loss 爆炸
- momentum PCA 极不稳定
- 对 batch size 高度敏感

---

## Baseline 说明

OpenCLIP 主训练 baseline 为：

```bash
# scripts/quick.sh
run_cc3m "vit" "ViT-B-16-exp" 29562 \
    "--siglip --epochs 10 --warmup 512 --lr ${LR_17_1}"
```

Toy 实验验证通过后，将以**最小侵入方式**接入：
- 在 `encode_image` / `encode_text` 返回值前（normalization 前）插入 `PCARegularizer`
- 全部功能由 CLI flag 控制（`--pca-reg-enable` 等）
- 默认关闭，baseline 行为完全不变

---

## 分布式训练说明（当前状态）

当前实现：**单卡正确，每卡维护独立 PCA 统计**。

DDP 下每张卡的 `MomentumPCAStats` 使用本地 rank 的 batch 样本。  
由于 EMA 窗口很长，多卡统计会在若干步后自然收敛到相近值，实践中通常可接受。

如需精确跨卡同步，在 `MomentumPCAStats.update()` 中插入：
```python
dist.all_reduce(batch_cov)
batch_cov /= dist.get_world_size()
```
（代码中已有注释说明位置）
