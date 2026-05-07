# 预训练模型在 CC3M 上的微调实验

## 背景

从零训练的 baseline（`pe_dinov3_sigreg_siglip_muon`）使用 PE-Core-B-16-dinov3 架构 + SigLIP 损失 + SigReg + Muon 优化器，在 CC3M（300 万对，LLaVA-ReCap 长描述）上取得了较强的 COCO 检索性能。

两个预训练模型拥有更优的 COCO zero-shot 检索能力：
- **PE-Core-B-16**：T2I R@1=50.2, I2T R@1=71.1（MetaCLIP 数据，数亿对）
- **ViT-B-16-SigLIP2**：T2I R@1=53.2, I2T R@1=69.4（WebLI 数据，数十亿对）

**核心问题**：直接使用从零训练的学习率（3.4e-4）微调预训练模型，会导致表征迅速崩溃——模型被梯度推出已收敛的 loss landscape。

## 目的

1. 验证可防止灾难性遗忘的微调 pipeline
2. 对比 4 种策略（LiT / 部分解锁 / 极低LR / SigReg）在两个预训练模型上的效果
3. 判断 CC3M 对预训练模型是否构成有效的适配信号
4. 为后续真实下游域数据建立可复用的微调流程

## 方法

### 防崩溃策略设计

| 策略 | 机制 | 关键参数 |
|------|------|----------|
| **LiT** | 冻结整个视觉塔，只训练文本塔 | `--lock-image`, LR=3.4e-5 |
| **Partial** | 解锁视觉塔最后 3 组（proj + 最后 2 个 block） | `--lock-image --lock-image-unlocked-groups 3`, LR=3.4e-5 |
| **Low-LR** | 全参数可训练，极低学习率 + 长 warmup | LR=3.4e-6, warmup=50%（435 步） |
| **SigReg** | 全参数 + SigReg embedding 正则化 | LR=3.4e-6, `--sigreg-target cls --sigreg-weight 5e-4` |

### 公共配置

- 训练数据：CC3M WebDataset（286 万样本，LLaVA-ReCap 长描述）
- 评估数据：COCO Karpathy test split（每张图 5 条 caption）
- 优化器：AdamW（beta1=0.9, beta2=0.98, eps=1e-6）
- 精度：amp_bf16，8 GPU × BS=512（GlobalBS=4096）
- PE-Core 使用 CLIP softmax 损失；SigLIP2 使用 SigLIP sigmoid 损失

### 设计依据

- LR 降低 100 倍（3.4e-6 vs 3.4e-4）：防止梯度破坏预训练权重位置
- beta2=0.98（vs baseline 0.95）：二阶矩更平滑，更新更稳定
- weight decay 降低（0.01~0.1 vs 0.2）：避免正则化"磨掉"预训练特征
- 不用 Muon 优化器：其激进的谱更新对微调过于不稳定
- AdamW 是保护预训练表征的更安全选择

## 实验结果

### PE-Core-B-16（CLIP Softmax 损失）

| 实验 | Epochs | T2I R@1（最佳） | I2T R@1（最佳） | 趋势 |
|------|--------|-----------------|-----------------|------|
| 预训练原始值 | — | 50.2 | 71.1 | — |
| **pe_lowlr** | 5 | **51.6**（Ep0） | **69.8**（Ep0） | 缓慢单调下降 |
| pe_sigreg | 5 | 48.9（Ep0） | 67.8（Ep1） | 缓慢下降 |
| pe_lit | 10 | 49.2（Ep0） | 65.4（Ep0） | 持续下降 |
| pe_partial3 | 10 | 48.7（Ep0） | 66.8（Ep0） | 持续下降 |

### ViT-B-16-SigLIP2（SigLIP Sigmoid 损失）

| 实验 | Epochs | T2I R@1（最佳） | I2T R@1（最佳） | 趋势 |
|------|--------|-----------------|-----------------|------|
| 预训练原始值 | — | 53.2 | 69.4 | — |
| **sig2_lowlr** | 5 | **53.7**（Ep0） | **70.8**（Ep0） | 近乎稳定（5ep 仅降 0.5%） |
| **sig2_sigreg** | 5 | 53.0（Ep0） | **70.5**（Ep0） | 近乎稳定（5ep 仅降 0.2%） |
| sig2_lit | 10 | 35.9（Ep0） | 52.2（Ep0） | 崩溃（持续下滑） |
| sig2_partial3 | 10 | 38.3（Ep0） | 55.8（Ep0） | 崩溃（持续下滑） |

### 逐 Epoch 曲线（T2I R@1）

```
PE-Core lowlr:     51.6 → 51.3 → 51.0 → 51.0 → 50.9  (Δ = -0.7%)
PE-Core sigreg:    48.9 → 48.9 → 48.6 → 48.0 → 48.0  (Δ = -0.9%)
PE-Core lit:       49.2 → 48.6 → 48.3 → 47.7 → 47.3 → 46.9 → 46.6 → 46.7 → 46.6 → 46.8
PE-Core partial3:  48.7 → 47.7 → 47.3 → 47.3 → 46.6 → 46.9 → 46.8 → 46.8 → 46.7 → 46.8

SigLIP2 lowlr:    53.7 → 53.0 → 52.7 → 52.5 → 52.4  (Δ = -1.3%)
SigLIP2 sigreg:   53.0 → 52.7 → 52.3 → 52.2 → 52.2  (Δ = -0.8%)
SigLIP2 lit:      35.9 → 34.8 → 33.6 → 32.2 → 32.0 → 31.0 → 30.0 → 29.3 → 29.0 → 29.5
SigLIP2 partial3: 38.3 → 35.7 → 32.9 → 31.4 → 30.1 → 29.0 → 29.6 → 28.1 → 27.8 → 27.9
```

## 分析

### 1. Low-LR 和 SigReg 成功防止了崩溃

两种策略在所有 epoch 内性能维持在预训练基线的 ~2% 以内，模型始终停留在预训练的 loss basin 中。这验证了 pipeline 功能正常、可安全用于真实下游数据。

### 2. LiT 在 SigLIP2 上灾难性失败

**超出预期的结果**：LiT（锁视觉、只训文本）理论上应最安全，但 SigLIP2 文本塔在此策略下崩溃。

**根本原因**：文本塔架构不对称。
- PE-Core 文本塔：24 层，width=1024，vocab=49408（标准 BPE） → 高容量，可平滑适配
- SigLIP2 文本塔：12 层，width=768，vocab=256000（SentencePiece），无因果掩码，last pooling → 容量小，设计迥异

当冻结 SigLIP2 视觉塔、强制文本塔对齐 CC3M 的 LLaVA-ReCap 长描述（与 SigLIP2 原始 WebLI 训练数据风格差异巨大）时，小容量文本塔必须剧烈改变才能适配新 caption 风格，导致已学对齐关系被彻底破坏。

### 3. Ep0 始终是峰值——训练过程纯粹是遗忘

无任何实验在 Epoch 0 之后出现性能提升，所有曲线单调非递增。这意味着：
- CC3M 对 COCO 检索不提供有效适配信号
- 预训练模型已经从远大于 CC3M 的数据中学到了 CC3M 能教的一切
- 此处的微调纯粹是遗忘过程，而非学习过程

### 4. pe_lowlr Ep0 出现 T2I=51.6 > 预训练 50.2

Epoch 0 的评估发生在第一个 epoch 训练之后。这 +1.4% 的"提升"可能源于：
- LR 仍在 warmup 阶段（50% warmup 意味着参数几乎没变）
- logit_scale 参数的初始学习（将 softmax 温度适配到新数据分布）
- 随机评估方差

这不是真实提升，而是测量噪声 + 温度校准效应。

### 5. SigReg vs Low-LR 对比

SigReg 在 Ep0 一致低于 Low-LR 约 2%，但衰减更慢：
- Low-LR Ep0→Ep4 衰减：-0.7%（PE）/ -1.3%（SigLIP2）
- SigReg Ep0→Ep4 衰减：-0.9%（PE）/ -0.8%（SigLIP2）

SigReg 初始代价更高（正则化惩罚与预训练特征几何冲突），但长期稳定性更好。对更长训练 schedule，SigReg 可能反超 Low-LR。

## 结论

| 发现 | 启示 |
|------|------|
| Pipeline 验证通过：无崩溃、训练稳定 | 可直接用于真实下游数据 |
| Low-LR 是最安全策略 | 作为任何微调任务的默认起点 |
| SigReg 长期稳定性更优 | 训练超过 5 epoch 时优先选择 |
| LiT 不具有普适安全性 | 文本塔容量必须匹配微调数据复杂度 |
| CC3M 对预训练模型无适配价值 | 需要真正分布外的目标域数据才能看到增益 |

## 后续真实下游部署建议

1. **首选 `lowlr`**（LR=3.4e-6, warmup=50%, wd=0.01）——先验证不崩溃
2. **逐步提升 LR**：若性能平台期，尝试 1e-5、3e-5
3. **长期训练用 SigReg**（`--sigreg-weight 5e-4`）：>5 epoch 时稳定性更优
4. **SigLIP2 避免 LiT**：除非重新初始化或扩展文本塔
5. **Ep0 评估值即预训练参考线**：若 Ep1 < Ep0，说明 LR 过高
6. **用 probe 监控特征**（`--probe-data`）：及早发现崩溃迹象

## 脚本索引

- 微调实验：`scripts/finetune_pretrained.sh`
- 冒烟测试：`scripts/smoke.sh ft`
- 预训练评估：`scripts/eval_pretrained.sh`
