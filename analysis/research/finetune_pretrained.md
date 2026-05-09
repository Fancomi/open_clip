# 预训练模型微调实验

## 背景

CLIP 范式的核心目标是获得更好的视觉塔。PE-Core-B-16 和 ViT-B-16-SigLIP2 在大规模 web 数据上训练，具备强通用表征。但直接在下游数据用标准 LR（3.4e-4）微调会导致表征崩溃——预训练权重被梯度推出良好收敛区域。

下游数据来自书籍（运动/健身类），包含插图、卡通、人体解剖图、照片等，与 WebLI/MetaCLIP 的 web 图文分布差异显著，是真正有意义的域适配场景。

## 目的

1. 在下游域数据上微调预训练模型，使其在该任务上超越 zero-shot 性能
2. 找到"学到新东西但不崩溃"的最佳 LR 临界点
3. 验证 Partial（部分解锁）策略是否能在保护底层特征的同时有效适配高层
4. 量化 SigReg 正则化在微调中的贡献

## 方法

### 实验轴（正交设计）

| 轴 | 值 | 作用 |
|----|-----|------|
| **学习率** | 1e-5 / 5e-5 / 2e-4 | 控制学习强度 |
| **冻结策略** | full / partial-3 | full=全参数；partial=锁视觉底层，解锁最后 3 组 |
| **SigReg** | +5e-4 / 无 | 正则化叠加项，防止 embedding 坍缩 |

### Partial 策略设计

第一轮实验中 Partial 效果差，原因是 LR 选择不当（太保守的 LR 配合 partial，既没学到新东西又破坏了对齐）。

正确用法：Partial 的优势在于**底层特征不变、高层可激进适配**。因此 Partial 应配合中高 LR（5e-5 / 2e-4），让解锁的顶层有足够学习能力，同时底层保持稳定。

TimmModel 的 unlock groups 对 ViT 的分组：
- Group 0: patch_embed + cls_token + pos_embed
- Group 1~N-2: transformer blocks（每 block 一组）
- Group N-1: last block + norm
- Group N: head/proj

`--lock-image-unlocked-groups 3` 解锁最后 3 组：proj + norm + 最后 2 个 block。

### 数据

| 属性 | 值 |
|------|-----|
| 来源 | 运动/健身书籍提取（插图/卡通/照片） |
| 训练集 | 22,687 图 × 1 caption |
| 验证集 | 2,520 图 × 1 caption |
| Caption 语言 | 中文，均值 446 字 |
| 实际 tokenize 容量 | PE-Core ~14 字 / SigLIP2 ~28 字 |
| TSV 截断 | 200 字（超出 tokenizer 容量但避免 pandas 溢出） |

### 训练配置

- Optimizer: AdamW (beta1=0.9, beta2=0.98, eps=1e-6, wd=0.05)
- Schedule: cosine, 100 epochs × 6 steps/ep = 600 steps, warmup=60
- Hardware: 8 GPU × BS=512 (GlobalBS=4096)
- Loss: PE-Core → CLIP softmax; SigLIP2 → SigLIP sigmoid
- Eval: 每 5 epoch 做一次 I2T/T2I R@1 on val

### 实验矩阵

| # | 模型 | 策略 | LR | SigReg | 对比目的 |
|---|------|------|-----|--------|----------|
| 0 | PE-Core | eval only | — | — | zero-shot 基线 |
| 1a | PE-Core | full | 1e-5 | +5e-4 | 保守 |
| 1b | PE-Core | full | 5e-5 | +5e-4 | 中等 |
| 1c | PE-Core | full | 2e-4 | +5e-4 | 激进 |
| 1d | PE-Core | partial-3 | 5e-5 | +5e-4 | partial 中等 |
| 1e | PE-Core | partial-3 | 2e-4 | +5e-4 | partial 激进 |
| 1f | PE-Core | full | 5e-5 | 无 | SigReg 消融 |
| 0 | SigLIP2 | eval only | — | — | zero-shot 基线 |
| 2a | SigLIP2 | full | 1e-5 | +5e-4 | 保守 |
| 2b | SigLIP2 | full | 5e-5 | +5e-4 | 中等 |
| 2c | SigLIP2 | full | 2e-4 | +5e-4 | 激进 |
| 2d | SigLIP2 | partial-3 | 5e-5 | +5e-4 | partial 中等 |
| 2e | SigLIP2 | partial-3 | 2e-4 | +5e-4 | partial 激进 |
| 2f | SigLIP2 | full | 5e-5 | 无 | SigReg 消融 |

## 第一轮结果（CC3M 代理验证）

### 结论

所有曲线单调下降——CC3M 对预训练模型无正向适配信号（预训练数据 >> CC3M）。但验证了：
- Low-LR + SigReg 可控制衰减在 ~2% 以内
- LiT 对 SigLIP2 崩溃（12层文本塔容量不足）
- Pipeline 可用

### 数据

```
PE-Core lowlr:     51.6 → 50.9  (Δ=-0.7%, 5ep)   | 预训练 T2I=50.2
PE-Core sigreg:    48.9 → 48.0  (Δ=-0.9%, 5ep)   |
SigLIP2 lowlr:    53.7 → 52.4  (Δ=-1.3%, 5ep)   | 预训练 T2I=53.2
SigLIP2 sigreg:   53.0 → 52.2  (Δ=-0.8%, 5ep)   |
SigLIP2 lit:      35.9 → 29.5  (崩溃, 10ep)      |
```

### 反思

1. LiT 不是微调策略：目标是更好的视觉塔，锁住视觉塔与目标矛盾
2. SigReg 是叠加项而非独立选项：应与 LR 组合而非平行对比
3. Partial 并非无效，而是第一轮用错了：保守 LR + partial = 既不学习又破坏对齐

## 第二轮结果

### A. Book 下游域（22K 运动/健身书籍，2637 val）

#### 数据总表

| # | 配置 | I2T R@1 | T2I R@1 | Best T2I@epoch | 备注 |
|---|------|---------|---------|----------------|------|
| 0 | **PE-Core zero-shot** | 3.3% | 1.4% | — | 近随机（域外） |
| 1a | PE full lr=1e-5 +SigReg | 14.4% | 14.2% | 14.2%@95 | 保守，稳定上升未饱和 |
| 1b | PE full lr=5e-5 +SigReg | 17.1% | 17.7% | 17.7%@95 | 中等 |
| 1c | PE full lr=2e-4 +SigReg | 20.2% | 19.5% | **20.1%@70** | 激进，70ep后过拟合下降 |
| 1d | PE partial lr=5e-5 +SigReg | 17.9% | 16.9% | 17.1%@80 | ≈full同LR |
| 1e | PE partial lr=2e-4 +SigReg | 20.1% | 20.3% | **20.4%@85** | partial+激进LR |
| **1f** | **PE full lr=5e-5 无SigReg** | **23.1%** | **22.6%** | **22.6%@95** | **全场最佳** |
| 0 | **SigLIP2 zero-shot** | 1.5% | 1.7% | — | 近随机 |
| 2a | SigLIP2 full lr=1e-5 +SigReg | 11.2% | 12.2% | 12.3%@80 | |
| 2b | SigLIP2 full lr=5e-5 +SigReg | 14.4% | 14.9% | **15.3%@55** | 55ep后轻微下降 |
| 2c | SigLIP2 full lr=2e-4 +SigReg | 0.0% | 0.08% | 0.8%@0 | **崩溃** |
| 2d | SigLIP2 partial lr=5e-5 +SigReg | 4.9% | 5.8% | 6.1%@50 | 效果差 |
| 2e | SigLIP2 partial lr=2e-4 +SigReg | 7.3% | 6.8% | 7.2%@55 | 效果差 |
| 2f | SigLIP2 full lr=5e-5 无SigReg | 14.5% | 14.8% | **15.9%@40** | 比+SigReg略好 |

#### Book 关键发现

1. **微调显著有效**：所有未崩溃的实验均从 ~1.5% 提升到 10-23%，提升 7~16x
2. **PE-Core 全面优于 SigLIP2**：最佳 PE 22.6% vs 最佳 SigLIP2 15.9%（+6.7%绝对值）
3. **SigReg 在小数据微调中是负面的**：PE-Core 无SigReg (22.6%) > 有SigReg (17.7%)，差距 5%
4. **PE-Core 对 LR 鲁棒**：1e-5~2e-4 均正常收敛，只是学习速度不同
5. **SigLIP2 对 LR 极敏感**：lr=2e-4 直接崩溃（T2I→0.08%），只有 lr≤5e-5 可用
6. **Partial 对 SigLIP2 无效**：partial 比 full 差 2-3x，解锁层数太少无法适配
7. **Partial 对 PE-Core 有效但不如 Full**：partial@2e-4≈full@2e-4，但 full@5e-5无SigReg更优

---

### B. CC3M 代理验证（286万对英文，COCO 5cap eval，25K val）

#### 数据总表

| # | 配置 | I2T R@1 | T2I R@1 | vs zero-shot | 备注 |
|---|------|---------|---------|--------------|------|
| 0 | **PE-Core zero-shot** | **69.8%** | **48.7%** | — | 基线 |
| 1a | PE full lr=1e-5 +SigReg | 66.2% | 48.6% | T2I Δ=-0.2% | 几乎无损 |
| 1b | PE full lr=5e-5 +SigReg | 63.0% | 45.4% | T2I Δ=-3.4% | 中等遗忘 |
| 1c | PE full lr=2e-4 +SigReg | 49.2% | 36.8% | T2I Δ=-12.0% | 严重遗忘 |
| 1d | PE partial lr=5e-5 +SigReg | 62.1% | 45.9% | T2I Δ=-2.8% | ≈full同LR |
| 1e | PE partial lr=2e-4 +SigReg | 49.6% | 38.7% | T2I Δ=-10.1% | 遗忘但比full@2e-4好 |
| 1f | PE full lr=5e-5 无SigReg | 62.2% | 46.2% | T2I Δ=-2.6% | SigReg反而略伤 |
| 0 | **SigLIP2 zero-shot** | 39.8% | 35.0% | — | 基线（低于预期） |
| 2a | **SigLIP2 full lr=1e-5 +SigReg** | **67.3%** | **48.8%** | **T2I +13.8%** | **大幅提升！** |
| 2b | SigLIP2 full lr=5e-5 +SigReg | 41.0% | 26.5% | T2I Δ=-8.5% | 遗忘 |
| 2c | SigLIP2 full lr=2e-4 +SigReg | 37.7% | 22.0% | T2I Δ=-13.0% | 严重遗忘 |
| 2d | SigLIP2 partial lr=5e-5 +SigReg | 40.0% | 23.3% | T2I Δ=-11.7% | 崩溃趋势 |
| 2e | SigLIP2 partial lr=2e-4 +SigReg | 36.1% | 21.1% | T2I Δ=-13.9% | 崩溃 |
| 2f | SigLIP2 full lr=5e-5 无SigReg | 54.8% | 38.7% | T2I +3.7% | 有提升但不如2a |

#### CC3M 关键发现

1. **PE-Core 单调下降**（符合预期）：预训练数据 >> CC3M，微调=遗忘。lr=1e-5 几乎无损（Δ=-0.2%）
2. **SigLIP2 lr=1e-5 大幅提升**：T2I 35.0%→48.8%（+13.8%），I2T 39.8%→67.3%（+27.5%）！
   - 说明 SigLIP2 原始文本塔对 COCO 评估的对齐不如 PE-Core，但图-文空间仍有潜力
   - 极低 LR 微调相当于做了**文本塔 alignment tuning**而未破坏视觉特征
3. **SigLIP2 对 LR 极其敏感**：lr=1e-5 是唯一安全区间，5e-5 即崩溃
4. **Partial 对 SigLIP2 在 CC3M 上也无效**：所有 partial 均比同LR full 差
5. **无SigReg 时 SigLIP2@5e-5 也有提升**（+3.7%），但远不如 lr=1e-5+SigReg

---

## 综合分析

### 1. 两个数据集讲了不同的故事

| 维度 | Book（域适配） | CC3M（通用微调） |
|------|---------------|-----------------|
| 目标 | 学习新域知识 | 对齐文本塔（隐含） |
| PE-Core | 大幅提升 (+21%) | 纯遗忘 |
| SigLIP2 | 提升但低于PE (+14%) | lr=1e-5 大幅提升 (+14%) |
| 最佳LR | 5e-5~2e-4 | 1e-5 |
| SigReg 价值 | **负面**（抑制学习） | **中性偏负** |

### 2. SigReg 为何在小数据微调中有害？

SigReg 强制 embedding 保持均匀分布。但在小数据域适配中，模型需要**重组语义聚类结构**来适配新域——SigReg 的均匀性约束直接对抗这一过程。

对比证据：PE-Core full lr=5e-5 无SigReg (22.6%) vs 有SigReg (17.7%)，**SigReg 使性能降低 5 个百分点**。

推论：SigReg 适合从零训练（防止坍缩）或大规模微调，但在**小数据域适配**中是有害的正则化。

### 3. 为什么 PE-Core > SigLIP2 在 book 上？

- PE-Core 使用 softmax loss，梯度信号更稳定，对 LR 容忍范围宽
- SigLIP2 使用 sigmoid loss，每对独立判断，高 LR 时梯度方差大 → 容易崩溃
- PE-Core 文本塔 24层×1024d，容量大于 SigLIP2 的 12层/ctx64
- 中文 token 容量：PE ~14字 vs SigLIP2 ~28字，差异不足以解释 7% 的性能差

### 4. SigLIP2 在 CC3M 上的"意外"提升

SigLIP2 zero-shot 在 COCO eval 上只有 35%（远低于其论文报告），而 lr=1e-5 微调后跳至 48.8%。解释：
- SigLIP2 视觉塔本身很强（来自 WebLI 训练），但其文本塔与 COCO caption 风格不对齐
- CC3M 的 LLaVA-ReCap 英文 caption 风格接近 COCO → 低 LR 微调修正了文本对齐
- 本质上是**文本塔 domain alignment**，不是视觉能力提升

### 5. Partial 策略评估

| 模型 | Partial 有效性 | 原因 |
|------|---------------|------|
| PE-Core | 有效但非最优 | partial@2e-4≈full@2e-4，但 full@5e-5无SigReg更好 |
| SigLIP2 | 无效/有害 | 解锁3层不够，且 sigmoid loss + partial = 梯度不连续 |

Partial 的理论优势（底层稳定+高层激进）只在 PE-Core + 高LR 时兑现。对 SigLIP2 而言，12层视觉塔解锁3层=解锁25%的参数，信号传播受阻。

### 6. 最佳实践建议

**小数据域适配（<50K对）：**
- 首选 PE-Core + full + lr=5e-5 + 无SigReg
- 备选 PE-Core + full/partial + lr=2e-4 + 无SigReg（更快收敛但需要 early stopping）
- 不要用 SigLIP2（除非已确认 lr=1e-5 足够）

**大规模通用微调（百万级）：**
- SigLIP2 + lr=1e-5 + SigReg：可修正文本对齐
- PE-Core + lr=1e-5：几乎不遗忘，但也学不到新东西

---

## 预期验证

| 预期 | 结果 | 判定 |
|------|------|------|
| 域差异大 → 微调有正向增益 | Book: +21% 绝对值 | ✅ 完全验证 |
| LR=5e-5 为最佳平衡候选 | PE 最佳是 lr=5e-5 无SigReg | ✅ 验证 |
| Partial + 高LR 优于 Full + 同LR | PE: partial@2e-4≈full@2e-4，未超越 | ❌ 未超越 |
| SigReg 在 LR=2e-4 时价值最大 | SigReg 在所有 book 实验中均为负面 | ❌ 相反结论 |

---

## 脚本

| 文件 | 用途 |
|------|------|
| `scripts/finetune_pretrained.sh` | CC3M 微调实验（286万对） |
| `scripts/finetune_pretrained_book.sh` | Book 微调实验（2.2万对） |
| `scripts/build_book_tsv.py` | book 数据 JSON→TSV 转换 |
| `scripts/smoke.sh ft` | 冒烟测试 |

## 实验日志

| 目录 | 内容 |
|------|------|
| `logs/20260508_0_ft_book/` | Book 第二轮全部14组实验 |
| `logs/20260508_0_ft_cc3m/` | CC3M 第二轮全部14组实验 |
| `logs/20260507_ft_cc3m/` | 第一轮 CC3M 代理验证 |
