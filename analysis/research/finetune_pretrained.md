# 预训练模型微调实验

## 目标

在下游域数据（运动/健身书籍）上微调 PE-Core-B-16 / ViT-B-16-SigLIP2，找到不崩溃的最佳配置。

## 实验轴

| 轴 | 值 |
|----|-----|
| 学习率 | 3e-6 / 5e-6 / 1e-5 / 5e-5 / 2e-4 |
| 冻结策略 | full / partial-1 / partial-2 / partial-3 |
| SigReg weight | 0 / 1e-5 / 5e-5 / 5e-4 |
| 训练长度 | 100 / 200 / 300 epochs |

## 数据

| | Book（目标域） | CC3M（代理验证） |
|--|---------------|-----------------|
| 规模 | 22K 图×1 caption（中文） | 286 万对（英文 LLaVA-ReCap） |
| Eval | book val 2637 样本，1cap | COCO karpathy 25K，5cap |
| 意义 | 真正的域适配 | 验证遗忘/对齐行为 |

## 结果

### Book 域适配（T2I R@1，zero-shot: PE=1.4%, SigLIP2=1.7%）

| 配置 | T2I best | 备注 |
|------|:--------:|------|
| **PE full lr=5e-5 nosig 300ep** | **24.7%** | 全场最佳，240ep达峰 |
| PE full lr=5e-5 nosig 200ep | 24.2% | 165ep达峰 |
| PE full lr=5e-5 nosig 100ep | 22.6% | 95ep仍上升 |
| PE full lr=5e-5 sig=1e-5 | 22.3% | sig=1e-5 几乎无害 |
| PE full lr=5e-5 sig=5e-5 | 22.0% | 轻微抑制 |
| PE full lr=5e-5 sig=5e-4 | 17.7% | sig过强，-5% |
| PE full lr=2e-4 sig=5e-4 | 20.1% | 70ep后过拟合 |
| PE partial-1 lr=2e-4 | 20.5% | 只 proj head |
| PE partial-2 lr=2e-4 | 20.8% | proj + last block |
| PE partial-3 lr=2e-4 | 20.4% | proj + 2 blocks |
| PE full lr=1e-5 sig=5e-4 | 14.2% | 太保守 |
| PE full lr=5e-6 sig=5e-4 | 11.6% | 100ep 不够 |
| PE full lr=3e-6 sig=5e-4 | 7.2% | 远未收敛 |
| | | |
| SigLIP2 full lr=5e-5 sig=5e-5 | 16.6% | SigLIP2 最佳 |
| SigLIP2 full lr=5e-5 sig=1e-5 | 16.3% | |
| SigLIP2 full lr=5e-5 nosig | 15.9% | |
| SigLIP2 full lr=5e-5 sig=5e-4 | 15.3% | |
| SigLIP2 full lr=1e-5 sig=5e-4 | 12.3% | |
| SigLIP2 full lr=5e-6 | 11.0% | |
| SigLIP2 full lr=2e-4 | 0.08% | **崩溃** |
| SigLIP2 partial-1/2/3 (any LR) | 3~7% | 全部无效 |

### CC3M 代理验证（T2I R@1，zero-shot: PE=48.7%, SigLIP2=35.0%）

| 配置 | T2I best | vs zero-shot |
|------|:--------:|:------------:|
| **SigLIP2 full lr=3e-6 sig=5e-4** | **53.0%** | **+18.0%** |
| SigLIP2 full lr=5e-6 sig=5e-4 | 52.7% | +17.7% |
| SigLIP2 full lr=1e-5 sig=1e-5 | 52.5% | +17.5% |
| SigLIP2 full lr=1e-5 sig=5e-4 | 52.4% | +17.4% |
| SigLIP2 full lr=1e-5 sig=5e-5 | 51.7% | +16.7% |
| PE full lr=1e-5 sig=1e-5 | **50.1%** | **+1.4%** |
| PE full lr=1e-5 sig=5e-5 | 49.8% | +1.1% |
| PE full lr=3e-6 sig=5e-4 | 48.8% | +0.1% |
| PE full lr=1e-5 sig=5e-4 | 49.1% | +0.4% |
| PE partial-1 lr=5e-5 | 48.5% | -0.2% |
| SigLIP2 partial-1/2 lr=1e-5 | 38.0% | +3.0% |
| SigLIP2 full lr=5e-5 | 45.5% | 遗忘 |

## 核心结论

### 1. 最佳配置

| 场景 | 推荐 |
|------|------|
| **小数据域适配** | PE-Core + full + lr=5e-5 + nosig + 长训练 |
| **大规模文本对齐** | SigLIP2 + full + lr=3e-6 + sig=5e-4 |

### 2. SigReg：weight 决定一切

| SigReg weight | Book 上的效果 | 解释 |
|:-------------:|:-------------:|------|
| 5e-4 | **有害** (-5%) | 过强，阻止语义聚类重组 |
| 5e-5 | 轻微抑制 (-0.6%) | |
| 1e-5 | **无害** (-0.3%) | 可安全使用 |
| 0 | 基准 | |

结论：小数据微调 SigReg weight ≤ 1e-5 或不用。从零训练时仍建议 5e-4。

### 3. Partial 冻结：PE 有效，SigLIP2 无效

**PE-Core**：partial-1（只 proj）配合 lr=2e-4 达 20.5%，与 partial-2/3 几乎一样（20.8%/20.4%）。说明 **PE 视觉特征已经够好，瓶颈在投影层**。但 full+nosig（22.6%）仍更优。

**SigLIP2**：所有 partial（unlock 1/2/3）均为 3-7%，而 full 可达 16%。说明 SigLIP2 需要全模型参与适配，局部解锁完全不足。

### 4. PE-Core vs SigLIP2

| 维度 | PE-Core | SigLIP2 |
|------|---------|---------|
| LR 容忍度 | 3e-6 ~ 2e-4 均可 | ≤ 5e-5（2e-4 崩溃） |
| Book 最佳 | **24.7%** | 16.6% |
| Loss 类型 | softmax（稳定） | sigmoid（高方差） |
| Partial | 有效 | 无效 |

PE-Core 在域适配中全面优于 SigLIP2：更鲁棒、天花板更高、更宽容。

### 5. SigLIP2 在 CC3M 上的"对齐效应"

SigLIP2 zero-shot 在 COCO eval 仅 35%（远低于 PE 的 48.7%），但 lr=3e-6 微调 5 epoch 后达 53%。原因：SigLIP2 视觉塔强（WebLI 训练），但文本塔与 COCO caption 风格不对齐。CC3M（LLaVA-ReCap 英文）修正了文本对齐。

### 6. 更长训练的收益

| epochs | PE Book T2I best |
|--------|:----------------:|
| 100 | 22.6% |
| 200 | 24.2% (+1.6%) |
| 300 | 24.7% (+0.5%) |

收益递减但未完全饱和。22K 数据 × 300ep = 仅 1800 步，模型远未过拟合。

## 脚本

| 文件 | 用途 |
|------|------|
| `scripts/finetune_pretrained.sh` | CC3M v2（基础实验矩阵） |
| `scripts/finetune_pretrained_v3.sh` | CC3M v3（低LR/partial层数/SigReg weight） |
| `scripts/finetune_pretrained_book.sh` | Book v2 |
| `scripts/finetune_pretrained_book_v3.sh` | Book v3 |
| `scripts/build_book_tsv.py` | book JSON→TSV |

## 实验日志

| 目录 | 内容 |
|------|------|
| `logs/20260508_0_ft_book/` | Book v2 + v3（32组） |
| `logs/20260508_0_ft_cc3m/` | CC3M v2 + v3（26组） |
| `logs/20260507_ft_cc3m/` | 第一轮探索 |
