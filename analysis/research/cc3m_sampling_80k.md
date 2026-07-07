# CC3M 采样策略对比实验

> 实验脚本：`experiments/sample_cc3m_80k.sh` / `experiments/sample_cc3m_500k.sh`  
> 采样工具：`scripts/tools/sample_cc3m.py`  
> 评估工具：`scripts/tools/eval_retrieval.py`  
> 开始日期：2026-05-27  
> 最终更新：2026-05-28

---

## 一、背景

COCO curriculum 实验表明，训练样本的呈现顺序对 retrieval 性能有显著影响（最高 +54%）。但 COCO 训练集仅 82K，数据质量和领域都较单一。CC3M 提供了 2.9M 多样化的网络图文对，但全量训练成本高且噪声多。

**核心问题**：如果从 CC3M 采样与 COCO 同量级（80K）的子集，不同采样策略能否超越 COCO 训练效果？

---

## 二、目的

1. 比较 **FPS（多样性最大化）** vs **K-Means（均匀覆盖）** 两种采样策略的训练效果
2. 评估不同 teacher 特征空间对采样质量的影响
3. 确定 CC3M 80K 子集能否匹配或超越 COCO 82K 的 retrieval 性能

---

## 三、方法

### 3.1 采样策略

| 策略 | 原理 | 特点 |
|------|------|------|
| **FPS** | 分层 FPS 排序，取前 80K 索引 | 最大化子集多样性，优先选择特征空间中最远的点 |
| **K-Means** | GPU mini-batch K-Means (K=80000)，每簇取最近质心样本 | 均匀覆盖特征空间，每个代表样本对应一个局部区域 |
| Random | 均匀随机采样 | 基线对照 |

### 3.2 Teacher 特征空间

| Teacher | 模型 | 预训练数据规模 |
|---------|------|---------------|
| pe_core | PE-Core-B-16 | 自研 |
| dinov3 | DINOv3-ViT-B/16 | LVD-1689M |
| siglip2 | ViT-B-16-SigLIP2 | WebLI |
| datacomp | ViT-B-16 | DataComp-XL |
| dfn2b | ViT-B-16 | DFN-2B |
| eva02 | EVA02-B-16 | Merged-30M |
| laion2b | ViT-B-16 | LAION-2B |
| metaclip | ViT-B-16-QuickGELU | MetaCLIP-FullCC |

### 3.3 训练配置

与 COCO curriculum 实验完全一致，**无 curriculum ordering**（纯随机 shuffle 训练）：

| 项目 | 值 |
|------|----|
| Model | PE-Core-B-16-dinov3 |
| Train | CC3M 采样子集 (80K) |
| Val | COCO Karpathy 5cap (5K×5) |
| Loss | SigLIP + SIGReg(cls, w=1e-4) |
| Optimizer | Muon, lr=3.4e-4, muon_lr=0.01 |
| Epochs | 20, warmup 42 steps |
| Batch | 512×8=4096 |
| val_frequency | 1 |

---

## 四、实验矩阵与结果

**Baseline**: COCO train (82K) = **i2t R@1 = 0.0136** (COCO Karpathy 5cap eval, ep19)

### 4.1 COCO Karpathy 5cap Eval (ep19)

#### FPS 采样

| Teacher | i2t R@1 | vs COCO baseline |
|---------|---------|------------------|
| **eva02** | **0.0044** | -68% |
| siglip2 | 0.0038 | -72% |
| pe_core | 0.0036 | -74% |
| dfn2b | 0.0036 | -74% |
| laion2b | 0.0036 | -74% |
| metaclip | 0.0028 | -79% |
| datacomp | 0.0026 | -81% |
| dinov3 | 0.0024 | -82% |

#### K-Means 采样

| Teacher | i2t R@1 | vs COCO baseline |
|---------|---------|------------------|
| **datacomp** | **0.0044** | -68% |
| dinov3 | 0.0036 | -74% |
| laion2b | 0.0036 | -74% |
| siglip2 | 0.0032 | -76% |
| dfn2b | 0.0026 | -81% |
| eva02 | 0.0026 | -81% |
| metaclip | 0.0026 | -81% |
| pe_core | 0.0022 | -84% |

#### Random 基线

| 配置 | i2t R@1 | vs COCO baseline |
|------|---------|------------------|
| random 80K | 0.0034 | -75% |

### 4.2 CC3M Val 内域评估

#### 全量 13K pool (ep20 checkpoint)

所有配置 i2t R@1 ≈ 0.0001（≈ 随机水平 1/13443），无区分度。

#### Bootstrap 评估 (1000-pool × 5 trials, 按 R@50 排序)

| Tag | R@50 mean | ±std | MedR mean | ±std |
|-----|-----------|------|-----------|------|
| fps_dinov3 | **0.0552** | 0.0044 | 503.8 | 16.7 |
| kmeans_datacomp | 0.0546 | 0.0048 | 498.0 | 16.1 |
| kmeans_dinov3 | 0.0544 | 0.0060 | 501.0 | 12.1 |
| kmeans_pe_core | 0.0538 | 0.0093 | 493.6 | 19.4 |
| fps_datacomp | 0.0536 | 0.0061 | 499.6 | 17.0 |
| fps_pe_core | 0.0530 | 0.0050 | 494.6 | 6.4 |
| kmeans_siglip2 | 0.0528 | 0.0032 | 487.4 | 7.8 |
| fps_metaclip | 0.0522 | 0.0026 | 478.0 | 8.6 |
| kmeans_eva02 | 0.0520 | 0.0030 | 511.4 | 9.1 |
| fps_eva02 | 0.0512 | 0.0038 | 499.8 | 9.0 |
| kmeans_laion2b | 0.0510 | 0.0046 | 504.6 | 15.3 |
| kmeans_dfn2b | 0.0494 | 0.0015 | 504.6 | 10.6 |
| fps_siglip2 | 0.0482 | 0.0031 | 504.4 | 10.3 |
| kmeans_metaclip | 0.0478 | 0.0076 | 478.4 | 11.8 |
| fps_laion2b | 0.0476 | 0.0042 | 508.0 | 15.7 |
| **random** | **0.0472** | **0.0049** | 489.8 | 13.3 |
| fps_dfn2b | 0.0454 | 0.0045 | 496.8 | 15.8 |

> **统计检验**：最佳 (0.0552) vs random (0.0472) 差值仅 0.008，约 1.5 个标准差，**不具统计显著性**。MedR 差异同样在噪声范围内（±15）。

### 4.3 全局排行 (COCO eval, 80K 实验)

| Rank | 方法 | Teacher | i2t R@1 | vs COCO baseline |
|------|------|---------|---------|------------------|
| 1 | FPS | eva02 | **0.0044** | -68% |
| 1 | KMeans | datacomp | **0.0044** | -68% |
| 3 | FPS | siglip2 | 0.0038 | -72% |
| 4 | FPS | pe_core/dfn2b/laion2b | 0.0036 | -74% |
| 4 | KMeans | dinov3/laion2b | 0.0036 | -74% |
| — | Random | — | 0.0034 | -75% |
| — | **COCO baseline** | — | **0.0136** | — |

> 80K 实验结论：训练量不足（仅 400 gradient steps），所有方法间差异不显著。

---

## 五、500K 正式实验（修正配置）

### 5.1 配置变更

80K 实验暴露了训练不充分的问题，500K 实验修正如下：

| 项目 | 80K 实验 | 500K 实验 |
|------|---------|-----------|
| 数据量 | 80K | **500K** |
| Gradient steps | 400 | **1220** |
| Epochs | 20 | 10 |
| Loss | SigLIP + SIGReg | SigLIP + SIGReg + **projective neg** |
| K-Means init | random (39% 空簇) | **K-Means++ (0% 空簇)** |
| K-Means 采样 | 最近质心 | **分层随机 (文献最优)** |

K-Means 采样修正依据 Meta FAIR 2024 论文 (arxiv:2405.15613)：簇内随机采样 > 最近质心 > 最远质心。

### 5.2 COCO Karpathy eval (ep9, i2t R@1)

| Rank | Tag | i2t R@1 |
|------|-----|---------|
| **1** | **random** | **0.0432** |
| 2 | kmeans_dinov3 | 0.0418 |
| 3 | kmeans_metaclip | 0.0412 |
| 4 | kmeans_laion2b | 0.0410 |
| 5 | kmeans_pe_core | 0.0408 |
| 6 | fps_metaclip | 0.0400 |
| 7 | fps_pe_core | 0.0394 |
| 8 | kmeans_siglip2 | 0.0392 |
| 9 | fps_datacomp | 0.0388 |
| 10 | fps_siglip2 | 0.0384 |
| 11 | fps_dinov3 | 0.0380 |
| 12 | fps_laion2b | 0.0378 |
| 12 | kmeans_dfn2b | 0.0378 |
| 14 | fps_eva02 | 0.0376 |
| 14 | kmeans_eva02 | 0.0374 |
| 16 | fps_dfn2b | 0.0372 |
| 17 | kmeans_datacomp | 0.0360 |

### 5.3 CC3M Val 内域评估 (1K-pool R@50, 3-trial bootstrap)

| Rank | Tag | R@50 |
|------|-----|------|
| **1** | **random** | **0.0577** |
| 2 | kmeans_dinov3 | 0.0553 |
| 3 | kmeans_datacomp | 0.0530 |
| 4 | kmeans_laion2b | 0.0513 |
| 5 | kmeans_metaclip | 0.0510 |
| 6 | fps_siglip2 | 0.0507 |
| 7 | fps_metaclip | 0.0493 |
| 8 | kmeans_siglip2 | 0.0490 |
| 9 | fps_dinov3 | 0.0483 |
| 10 | fps_pe_core | 0.0480 |
| ... | ... | ... |
| 17 | kmeans_eva02 | 0.0410 |

### 5.4 分析

**两个评估维度一致：Random 采样排第一。**

| 指标 | Random | K-Means (avg) | FPS (avg) |
|------|--------|---------------|-----------|
| COCO i2t R@1 | **0.0432** | 0.0394 | 0.0384 |
| CC3M-val R@50 | **0.0577** | 0.0500 | 0.0464 |

FPS 和 K-Means 策略性采样均不如随机采样。原因：

1. **FPS 的多样性偏差**：FPS 优先选特征空间中最远的点。在 CC3M 这类 web 数据中，"最远"的往往是噪声样本（OCR 图、logo、损坏图片）。过度代表这些低质量长尾数据损害训练。

2. **K-Means 的均匀性偏差**：分层随机采样强制每个簇贡献等比例样本。这意味着低密度区域（通常是噪声/非典型图片）获得了超过其自然比例的权重。

3. **Random 保持原始分布**：CC3M 已经过基本筛选（3M 规模），其自然频率分布本身就是合理的训练信号——高频概念理应被多看到。强制"均匀"或"多样"反而破坏了这个信号。

### 5.5 与文献的对照

FAIR 2024 论文中 hierarchical K-Means 有效的前提是：
- **数据源是数十亿级未筛选网络图片**（严重长尾 + 大量重复）
- 目标是从中选出 100M 均衡子集

我们的场景不同：
- **CC3M 已经是筛选后的 3M 数据**（相对干净、分布相对均匀）
- 从中选 500K（17%）—— 保留率高，random 已近似全集分布

**结论**：对已筛选的中等规模数据做子集选择时，策略性采样（FPS/K-Means）不如随机。这些方法的价值在于**从极大规模脏数据中做高倍率筛选**（如 10B→100M = 1% 保留率）。

---

## 六、最终结论

1. **在 CC3M (3M) → 500K (17%) 场景下，Random 采样是最优策略**
2. FPS 和 K-Means++ 分层随机均不如 Random，差距约 10-15%
3. 原因：CC3M 已经过筛选，其自然分布就是好的训练分布；强制多样性/均匀性引入噪声偏差
4. **FPS/K-Means 采样的适用条件**：数据源极大且未筛选 + 高倍率筛选（<5% 保留率）
5. 对于已筛选数据集的子集选择，应优先考虑**质量过滤**（如 CLIP score 阈值、SemDeDup 去重）而非几何多样性

---

## 七、聚类均衡性分析

### 7.1 指标说明

| 指标 | 含义 | 直觉解释 |
|------|------|---------|
| **Gini 系数** (0~1) | 簇大小分布的不等程度 | 经济学中的"贫富差距"指标。0=完全均等，1=极端不等。如同"一个国家所有财富集中在一个人手里" |
| **CoV (std/mean)** | 变异系数 | 波动相对于均值的比例。CoV=0.5 意味着簇大小的标准差是均值的一半 |
| **P90/P10 比** | 去极端后的头尾差距 | 第 90 百分位簇大小 ÷ 第 10 百分位。排除极端异常后衡量主体分布的不均程度 |
| **Max/Min 比** | 极端值差距 | 最大簇 vs 最小簇。受极端值影响大，反映是否存在巨型或微型异常簇 |
| **Lorenz 曲线** | 累积分布的图形化表达 | X轴=从小到大累计簇的比例，Y轴=这些簇累计包含的样本比例。越贴对角线越均匀 |

### 7.2 CC3M 各 Teacher 特征空间均衡性 (K=5000)

| Teacher | Gini | CoV | P90/P10 | Max/Min | 评价 |
|---------|------|-----|---------|---------|------|
| datacomp | 0.255 | 0.48 | 3.3x | 727x | 均衡 |
| dfn2b | 0.258 | 0.49 | 3.4x | 51x | 均衡 |
| eva02 | 0.257 | 0.48 | 3.3x | 1515x | 均衡 |
| metaclip | 0.258 | 0.48 | 3.4x | 95x | 均衡 |
| laion2b | 0.265 | 0.49 | 3.5x | 171x | 均衡 |
| dinov3 | 0.281 | 0.61 | 3.3x | 90x | 较均衡 |
| pe_core | 0.368 | 0.66 | 9.0x | 2318x | 不均衡 |
| siglip2 | 0.405 | 2.67 | 5.0x | 73129x | 严重不均 |

### 7.3 均衡性 vs 训练效果：无相关性

| Teacher | Gini ↓ | 采样 Avg i2t R@1 | 排名 |
|---------|--------|------------------|------|
| datacomp | 0.255 (最均衡) | 0.0374 | 5/8 |
| metaclip | 0.258 | **0.0406** | **1/8** |
| dfn2b | 0.258 | 0.0375 | 6/8 |
| eva02 | 0.257 | 0.0375 | 6/8 |
| laion2b | 0.265 | 0.0394 | 4/8 |
| dinov3 | 0.281 | 0.0399 | 3/8 |
| pe_core | 0.368 | 0.0401 | **2/8** |
| siglip2 | 0.405 (最不均) | 0.0388 | 4/8 |

**结论**：特征空间的聚类均衡性不能预测采样子集的训练效果。不均衡的 teacher (pe_core, siglip2) 采样效果与均衡 teacher 无显著差异。

产出路径：`/root/.../datas/cc3m-tsv/feature_probe/cluster_balance/`

---

## 八、与文献对比及改进方向

### 8.1 我们的实验 vs FAIR 2024 论文

| 维度 | FAIR 2024 | 我们的实验 | 差距 |
|------|-----------|-----------|------|
| 数据源 | 数十亿未筛选网络图片 | 2.9M CC3M (已筛选) | 1000x 规模 |
| 目标子集 | 100M | 500K | 200x |
| 保留率 | ~1-3% | **17%** | FAIR 的是高倍率筛选 |
| 训练量 | 625K iterations × ViT-L | 1220 steps × ViT-B | 500x |
| 评估 | ImageNet kNN/linear probe | COCO/CC3M retrieval | 不同任务 |
| 结论 | 4-level hier > 1-level > raw | Random > FPS ≈ KMeans | **相反** |

### 8.2 为什么结论相反

FAIR 论文有效的前提条件我们都不满足：

1. **数据极度冗余 + 严重长尾**：他们的原始数据有大量重复和噪声主导概念（如"网页截图"），均衡化去除冗余有巨大价值。CC3M 已经筛选过，冗余度低。
2. **极低保留率 (1-3%)**：从 10B 选 100M，策略性选择避免遗漏稀有概念很关键。我们 17% 的保留率下，random 已经能覆盖绝大多数概念。
3. **大规模训练 (625K iterations)**：模型有足够训练量来"利用"数据均衡性的差异。我们 1220 steps 远不够。

### 8.3 如何改进实验使结论更可靠

| 方案 | 做法 | 预期效果 | 代价 |
|------|------|---------|------|
| **A. 降低保留率** | 从 2.9M 选 30K~50K (1-2%) | 逼近 FAIR 的场景，策略差异应更明显 | 训练效果更差，信号可能仍弱 |
| **B. 增大训练量** | 500K × 100 epochs (12200 steps) 或全量 2.9M 对照 | 模型更强，能感知数据差异 | 10x 训练时间 (~50h) |
| **C. 先去噪再对比** | SemDeDup 去重后再做 FPS/KMeans/Random | 验证"去噪+采样"是否优于纯采样 | 需实现 SemDeDup |
| **D. 用 linear probe 评估** | 在 ImageNet 子集上做 kNN/linear | 更匹配文献评估方式，指标更稳定 | 需准备 ImageNet 数据 |
| **E. 从预训练模型 finetune** | 用 siglip2/datacomp 权重初始化，finetune 5ep | 模型有基础能力，数据差异立刻可见 | 测的是"finetune 数据选择"而非"预训练数据选择" |

**推荐组合**：方案 A + B — 降低保留率到 50K (1.7%) 同时加大训练量到 50 epochs (~6100 steps)。这样同时逼近 FAIR 的两个关键条件。

---

## 九、实验脚本索引

| 脚本/工具 | 功能 |
|-----------|------|
| `scripts/tools/sample_cc3m.py` | CC3M 子集采样: FPS / K-Means++ 分层随机 / Random |
| `scripts/tools/eval_retrieval.py` | Checkpoint → retrieval 评估 (任意 val TSV) |
| `analysis/cluster_balance.py` | CC3M 聚类均衡性分析 + Lorenz 曲线可视化 |
| `experiments/sample_cc3m_80k.sh` | 80K 实验 (训练不足, 无区分度) |
| `experiments/sample_cc3m_500k.sh` | 500K 正式实验 (projective + wds 配置) |
