# Muon 优化器与 SigReg 正则化在预训练模型微调中的研究

## 背景

在 `book_20260508_clean` 数据（18,499 训练样本、2,055 验证样本）上微调 PE-Core-B-16，已确认 AdamW 最佳配置为 **lr=2e-4 nosig 100ep（T2I R@1=20.3%，峰值在 ep80）**。

初轮实验（`finetune_pretrained_book_clean.sh`，v1）测试了 Muon 和 SigReg，但结果均为负面：
- Muon mlr=1.5e-3：17.3%（-3.0%）
- Muon mlr=5e-3：15.9%（-4.4%）
- SigReg 未在 clean 数据上单独测试（v1 SigLIP2 side 有，PE side 无）

本研究目标：定位 Muon 和 SigReg 在 PE fine-tuning 上的可用区间。

## 为什么 v1 Muon 为负面

### Muon 的设计原理

`MuonWithAuxAdam` 将参数分为两组：
- **Muon 组**（隐层权重矩阵，ndim≥2，排除 embed/bn/ln/bias/logit）：Nesterov momentum + Zangel 正交化更新（`dist.all_gather`）
- **AdamW 组**（embed/norm/bias/logit 等）：标准 AdamW，lr = `--lr`

`muon_lr` 只作用于 Muon 组的矩阵权重。

### v1 系数的问题

| 参数 | v1 值 | 来源参考 | 问题 |
|------|-------|---------|------|
| adam_lr | 5e-5 | 微调安全 LR | 合理 |
| muon_lr | 1.5e-3 / 5e-3 | quick.sh from-scratch × 1/7 | 仍然过大 |
| warmup | 50 steps | 10% of 100ep≈450 steps | 偏短 |

quick.sh（从零训练）的比例是 muon_lr=0.01, adam_lr=3.4e-4，约 ×30。但这是**从零训练**的比例，微调时参数已在好的初始点，不需要大步长。且微调 total steps 仅 ~900 步（200ep × 4.5 steps/ep），warmup 完成时只完成 ~5% 训练。

### SigReg 在小数据上的负面效应

SigReg（Sketched Isotropic Gaussian Regularizer）作用于 **原始 CLS token**（backbone 输出 `feats[:,0]`，`[B, backbone_dim]`，归一化前），强制嵌入分布趋向各向同性高斯。

在大规模预训练中（CC3M, CC12M），SigReg 有助于防止特征坍缩。但在小数据域适配中：

- 训练数据仅 18K 样本，每轮 4.5 steps，SigReg 梯度在稀疏更新中占比更高
- 域适配需要让特征重新聚类（书籍封面 → 语义聚类），SigReg 抑制这种分布重组
- weight=5e-4（v1 SigLIP2 实验）在 Book 数据上 -5%；weight=1e-5 几乎中性（-0.3%）

## 目标（v2 实验）

1. **Muon 低 muon_lr**：找到比 AdamW（20.3%）更好的 muon_lr 区间
2. **AdamW 更高 LR**：探索 lr=5e-4/1e-3，是否能更快达到同等峰值
3. **SigReg 极小权重**：在 clean 数据上验证 weight=1e-6/5e-6/1e-5 的中性/正面点

## 方法

### 实验脚本

`scripts/finetune_pretrained_book_clean_v2.sh`

### 实验配置

| 组 | 配置 | 变量 | 理由 |
|----|------|------|------|
| A | AdamW high-LR | lr=5e-4/1e-3, 50/100ep | v1 peak 在 80ep，更高 LR 应使峰值前移 |
| B | Muon low muon_lr | mlr=1e-4/3e-4/5e-4; adam_lr=2e-4 (best) / 5e-5 | 在 AdamW 量级的 1/2~1/20 |
| C | SigReg tiny weight | weight=1e-6/5e-6/1e-5; target=cls/clip | 从旧数据中性点往下探 |

### 硬件

8×GPU，BS=512/GPU（GlobalBS=4096），PE-Core-B-16 full fine-tune。

### Muon 参数分析

PE-Core-B-16 中 Muon 组（`ndim≥2`, 非 embed/norm/bias/logit）：
- ViT blocks: qkv/out/fc1/fc2 weight matrices（12 blocks × 4 = 48 matrices）
- Text transformer: 类似结构
- 视觉投影头 `visual.head.proj.weight`

这些矩阵共同构成模型的"计算核心"，muon_lr 过大会导致这些关键矩阵偏离预训练初始点过远。

### SigReg 执行位置

`CLIPLeJEPA._get_image_raw()`（`src/open_clip/model.py`）：

```python
# sigreg_target = 'cls'
feats = visual.trunk.forward_features(image)  # [B, N, backbone_dim]
cls_raw = feats[:, 0]                          # [B, backbone_dim], 未归一化
# SigReg loss 作用于 cls_raw
```

`sigreg_target='clip'` 则作用于 CLIP embedding（经过 `forward_head` + `visual.head` 投影后的 `[B, embed_dim]`，归一化前）。两者的关键区别：
- `cls`：维度更高（768 vs 512），更接近视觉特征本身
- `clip`：维度更低，经过了语义压缩，可能对 downstream task 约束更直接

## 实验结果

> 待填写（实验完成后更新）

### v1 参考结果（clean book 数据）

| 配置 | T2I R@1 best | 峰值 epoch |
|------|:------------:|:----------:|
| PE lr=2e-4 nosig 100ep | **20.3%** | ep80 |
| PE lr=2e-4 nosig 200ep | 18.9% | ep185（过拟合）|
| PE lr=5e-5 nosig 200ep | 17.4% | ep190 |
| PE lr=5e-5 nosig 300ep | 17.2% | ep255 |
| PE lr=5e-6 nosig 300ep | 11.7% | ep85（未收敛）|
| PE muon mlr=1.5e-3 200ep | 17.3% | ep115 |
| PE muon mlr=5e-3 200ep | 15.9% | ep50 |

### v2 实验结果

> 待填写

## 分析

> 待填写（实验完成后更新）

### 预期

- **Muon**：预期 mlr=3e-4 附近存在正面区间（约为 AdamW adam_lr 的 1.5×）。若 mlr=1e-4 仍负面，则 Muon 在此小数据场景不适用。
- **SigReg**：预期 weight≤5e-6 在 clean 数据上接近中性。`clip` target 可能优于 `cls`，因为约束空间与 retrieval task 更对齐。
- **高 LR**：lr=5e-4 预期在 30-50ep 达峰，lr=1e-3 可能不稳定。

## 结论

> 待填写

## 参考文献

- Kosson et al., *Muon: Momentum-Orthogonal Gradient Descent*, 2024
- 内部实验：`analysis/research/finetune_pretrained.md`（三轮 AdamW 微调结论）
- SigReg：Stable Isotropic Gaussian regularizer，参见 model.py CLIPLeJEPA 实现
