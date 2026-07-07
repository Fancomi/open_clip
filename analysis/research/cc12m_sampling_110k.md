# CC12M 1/100 采样策略对比实验

> 实验脚本：`experiments/sample_cc12m_50k.sh`  
> 采样工具：`scripts/tools/sample_cc12m.py`  
> 开始日期：2026-05-30  
> 最终更新：2026-05-30

---

## 一、背景

CC3M 500K 实验（保留率 17%）结论：Random 采样最优，FPS/K-Means 策略性采样不如随机。原因分析指向两个关键前提不满足：

1. **数据源不够脏**：CC3M 已筛选，自然分布本身合理
2. **保留率过高**：17% 下 random 已近似全集分布

Meta FAIR 2024 论文 (arxiv:2405.15613) 中策略性采样有效的条件：**数据极脏 + ~1% 保留率**。

CC12M 提供了接近这一条件的验证场景：
- **11M 未精细筛选的 web 图文对**（质量显著低于 CC3M）
- **采样 1/100 → 110K，保留率 ~1%**（对齐 FAIR 论文量级）

---

## 二、目的

1. 在 **1% 保留率 + 高噪声数据** 场景下，验证 FPS/K-Means 是否优于 Random
2. 对齐 FAIR 论文的采样比例 (1/100)，确认策略性采样的有效条件
3. 比较 8 种 teacher 特征空间在低保留率场景下的采样效果差异

---

## 三、方法

### 3.1 采样策略

| 策略 | 原理 | 对齐 FAIR |
|------|------|-----------|
| **FPS** | 分层 FPS (GPU)，最大化子集多样性 | — (对照方法) |
| **K-Means Uniform** | K=11000 簇 K-Means++，**每簇等量** (10) 随机取样 | ✓ 核心方法 |
| Random | 均匀随机 | 基线对照 |

**K-Means Uniform 对齐 FAIR 的关键设计**：
- K-Means++ 初始化 (避免空簇)
- **簇间均匀分配** (每簇取相同样本数) 而非按比例
- **簇内随机** (非最近质心)
- K = n_samples / 10 = 11000 (每簇贡献 ~10 样本)

FAIR 原文使用 4 级分层 K-Means (K=10 per level → 10^4 叶子)，适用于数十亿数据。
对 11M 量级，单级 K=11000 的信息容量等价（叶子簇数同量级）。

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

### 3.3 训练配置（对齐 wm_coco.sh projective）

| 项目 | COCO projective | CC12M 110K 本实验 |
|------|----------------|-------------------|
| Data | COCO 82K | CC12M 子集 110K |
| 保留率 | 100% (COCO 全量) | **~1% (11M → 110K)** |
| Epochs | 20 | **10** |
| Steps/ep | ~20 | ~27 |
| Total steps | ~400 | **~270** |
| Model | PE-Core-B-16-dinov3 | PE-Core-B-16-dinov3 |
| Loss | SigLIP + projective + SIGReg(cls, 1e-4) | 同左 |
| Optimizer | Muon, lr=3.4e-4, muon_lr=0.01 | 同左 |
| Batch | 512×8=4096 | 同左 |
| Warmup | 42 steps | 42 steps |
| Val | COCO Karpathy 5cap | COCO Karpathy 5cap |
| val_frequency | 2 | 2 |

### 3.4 实验设计依据

- **1/100 采样比例**：对齐 FAIR 论文，在该论文中策略性采样在 ~1% 保留率下显著优于 random
- **10 epochs / ~270 steps**：与 COCO 20ep/400steps 同量级，已验证可产生有效区分度
- **Projective loss**：当前最优 loss 配置，确保模型有足够学习能力感知数据差异

---

## 四、实验矩阵与结果

### 4.1 实验矩阵 (17 配置)

| # | Tag | 方法 | Teacher |
|---|-----|------|---------|
| 1 | random | Random | — |
| 2-9 | fps_{teacher} | FPS | 8 teachers |
| 10-17 | kmeans_{teacher} | K-Means++ | 8 teachers |

### 4.2 COCO Karpathy 5cap Eval (best epoch, 按 R@10 排序)

| Rank | Tag | 方法 | Teacher | i2t R@1 | i2t R@10 | Best Epoch |
|------|-----|------|---------|---------|----------|------------|
| 1 | kmeans_uniform_laion2b | K-Means Uniform | laion2b | 0.0062 | **0.0340** | 4 |
| 2 | fps_dinov3 | FPS | dinov3 | 0.0034 | 0.0328 | 4 |
| 3 | **random** | Random | — | 0.0040 | **0.0326** | 4 |
| 4 | kmeans_uniform_dfn2b | K-Means Uniform | dfn2b | 0.0030 | 0.0318 | 4 |
| 5 | fps_pe_core | FPS | pe_core | 0.0036 | 0.0318 | 8 |
| 6 | fps_datacomp | FPS | datacomp | 0.0038 | 0.0316 | 6 |
| 7 | kmeans_uniform_eva02 | K-Means Uniform | eva02 | 0.0036 | 0.0310 | 4 |
| 8 | kmeans_uniform_pe_core | K-Means Uniform | pe_core | 0.0032 | 0.0306 | 4 |
| 9 | kmeans_uniform_metaclip | K-Means Uniform | metaclip | 0.0040 | 0.0306 | 4 |
| 10 | kmeans_uniform_siglip2 | K-Means Uniform | siglip2 | 0.0048 | 0.0304 | 4 |
| 11 | fps_metaclip | FPS | metaclip | 0.0038 | 0.0304 | 4 |
| 12 | fps_laion2b | FPS | laion2b | 0.0042 | 0.0302 | 4 |
| 13 | kmeans_uniform_datacomp | K-Means Uniform | datacomp | 0.0028 | 0.0300 | 2 |
| 14 | fps_dfn2b | FPS | dfn2b | 0.0038 | 0.0300 | 4 |
| 15 | kmeans_uniform_dinov3 | K-Means Uniform | dinov3 | 0.0036 | 0.0296 | 4 |
| 16 | fps_siglip2 | FPS | siglip2 | 0.0030 | 0.0280 | 4 |
| 17 | fps_eva02 | FPS | eva02 | 0.0040 | 0.0266 | 6 |

### 4.3 按方法汇总

| 方法 | Avg R@10 | Best R@10 | Best Teacher |
|------|----------|-----------|--------------|
| **K-Means Uniform** | 0.0305 | **0.0340** | laion2b |
| FPS | 0.0302 | 0.0328 | dinov3 |
| Random | 0.0326 | 0.0326 | — |

### 4.4 分析

1. **K-Means Uniform (FAIR 方法) 的最佳配置 (laion2b) 首次超过 Random**：R@10 0.0340 vs 0.0326 (+4.3%)。但差异仅 0.0014，约 7 张图/5000，**不具统计显著性**。

2. **多数策略性采样仍不优于 Random**：17 个配置中仅 2 个 (kmeans_uniform_laion2b, fps_dinov3) 超过 random，其余 14 个均不及。

3. **方法间差异极小**：最佳 0.0340 vs 最差 0.0266，差距仅 0.0074 (R@10)。所有方法的 median rank 都在 730-860 范围内，差异在噪声水平。

4. **Teacher 特征空间的影响有限**：没有一个 teacher 在两种方法上同时表现最佳。laion2b 在 K-Means Uniform 最优但 FPS 排第 12；dinov3 在 FPS 第 2 但 K-Means Uniform 第 15。

---

## 五、与前序实验的对照

| 维度 | CC3M 500K | CC12M 110K (本实验) | FAIR 2024 |
|------|-----------|---------------------|-----------|
| 数据源质量 | 中等（筛选后） | 低（未精细筛选） | 极低（未筛选） |
| 数据源规模 | 2.9M | 11M | 数十亿 |
| 保留率 | 17% | **~1%** | 1-3% |
| Training steps | 1220 | **270** | 625K |
| 结论 | Random 最优 | **无显著差异** | 策略性采样优 |

---

## 六、结论

1. **即使在 1% 保留率 + 脏数据 (CC12M) 场景下，策略性采样仍未显著优于 Random**
2. K-Means Uniform (对齐 FAIR) 的最佳配置微弱领先 Random (+4.3% R@10)，但不具统计显著性
3. **训练量不足是核心限制**：我们的 270 steps vs FAIR 的 625K steps，差 2300x。模型可能没有足够的训练量来"利用"数据选择带来的质量差异
4. FPS 和 K-Means Uniform 的平均表现与 Random 持平，说明在小训练预算下数据选择策略的边际收益趋近于零
5. **FAIR 方法有效的前提可能不仅是"脏数据 + 低保留率"，还需要"充足的训练量" (≥100K steps)**

### 与 FAIR 论文条件的剩余差距

| 条件 | FAIR | 本实验 | 是否满足 |
|------|------|--------|---------|
| 脏数据源 | 数十亿未筛选 | 11M 未精细筛选 | 部分 ✓ |
| 低保留率 (~1%) | ✓ | ✓ | ✓ |
| 充足训练量 | 625K steps | 270 steps | ✗ (差 2300x) |
| 大模型 | ViT-L | ViT-B | 部分 ✗ |

---

## 七、实验脚本索引

| 脚本/工具 | 功能 |
|-----------|------|
| `scripts/tools/sample_cc12m.py` | CC12M 特征提取 + 子集采样 + 批量导出 |
| `experiments/sample_cc12m_50k.sh` | 完整实验流程 (Phase 1-3) |
| `scripts/tools/eval_retrieval.py` | Checkpoint → retrieval 评估 |
