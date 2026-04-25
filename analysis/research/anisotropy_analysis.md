# 各向异性指标分析与调研方向

> 日期：2026-04-25
> 数据来源：`bash scripts/probe.sh coco` 跑出的 anisotropy 指标（COCO val2014，6 模型）

---

## 现象记录

| 模型     | EffRank | PR     | AvgCos | top4% | top10% | 备注               |
|----------|---------|--------|--------|-------|--------|--------------------|
| DINOv3   | **325** | **最高** | **最低** | **7%** | —      | 全部指标一致：最 isotropic |
| RADIO    | **最低** | **最低** | 高      | **40%** | —      | 全部指标一致：最 anisotropic |
| SigLIP2  | 中      | 中      | **最高** | 中    | —      | AvgCos 最高但 EffRank 非最低 |
| PE-Core  | 中      | 中      | 中      | >15%  | —      | —                  |
| EUPE     | 中      | 中      | 中      | >15%  | —      | —                  |
| TIPSv2   | 中      | 中      | 中      | >15%  | —      | —                  |

视觉观察：DINOv3 在 `image_allmodels.png` 的 PCA 散点图里呈"凹陷四脚锥形"（星形轮廓），
其余模型更接近椭球形云团。

---

## 核心发现：数值"矛盾"的解释

### DINOv3：看起来各向异性，数值说各向同性

这不是矛盾，而是两种不同的分布结构：

```
A. 均匀球面散布（真正 isotropic）   → EffRank 高，top4% 低
B. 高维单纯形（simplex）结构        → EffRank 同样高，top4% 同样低
                                       BUT 分布是多峰的（multi-modal）
```

DINOv3 属于 **B**。DINO 的 centering + sharpening 把特征推向超球面上若干等距"原型方向"
（类似正多面体顶点）。投影到 PC1-PC2 平面时，N 个原型点形成 N 角星/多边形轮廓，
视觉上像"锥形"或"星形"。

**四个现有指标（EffRank / PR / AvgCos / pct_var_top-k）全部量的是"有多少维度被使用"，
无法区分 A 和 B — 这是它们的盲区。**

### RADIO：所有指标一致指向高各向异性

RADIO 是**低秩平滑流形**：少数几个主方向主导大部分方差（top4=40%）。
这与其架构特性吻合：

- 多教师蒸馏（SigLIP2-g + DINOv3-7B + SAM3）的平均效应磨平了各教师的"尖峰"结构
- 平滑低维流形 → COCO 和 CC3M 在 PC1-PC2 完全重合（域不变性好）
- 适合 dense prediction（空间特征连续）

**结论：RADIO 的高 anisotropy 不是退化，而是平滑化的代价。**

### SigLIP2：AvgCos 最高的独立解释

SigLIP2 的随机图像对余弦相似度最高（特征夹角最小），原因是：
对比学习把图像特征往文本特征方向拉，文本语义的主成分方向相对集中，
图像特征整体偏向一个"语义锥"。但其 EffRank 并非最低，说明在这个锥内部仍有高维内部结构。

---

## 现有指标的盲区

```
多峰性（multi-modality）≠ 低秩（low-rank）
```

| 分布类型         | EffRank | pct_var_top4 | 视觉形状          |
|------------------|---------|--------------|-------------------|
| 平滑球面         | 高      | 低           | 圆形云            |
| Simplex / 星形   | 高      | 低           | 星形 / 多角轮廓   |
| 低秩平滑（RADIO）| 低      | 高           | 椭球 / 香蕉形     |
| 完全退化         | 极低    | 接近 100%    | 单点              |

---

## 后续调研方向

### 方向 1：补充多峰性指标

现有指标无法区分"均匀球面"和"simplex"。候选补充指标：

```python
# 方案 A：pairwise cosine 的标准差（最轻量）
std_cos = pairwise_cos_matrix.std()
# DINOv3 的 std_cos 应远高于 RADIO（尽管二者 EffRank 相差很大）

# 方案 B：GMM 拟合最优 K（BIC 准则）
# 反映特征空间有多少"聚类原型"

# 方案 C：pairwise distance histogram 的峰数
# 直方图双峰 → multi-modal；单峰 → unimodal
```

实现位置：`feature_probe.py` → `compute_anisotropy()` 中追加这三个字段。

### 方向 2：RADIO top-4 主成分的语义解释

RADIO top4 解释 40% 方差，这 4 个方向很可能对应跨域语义轴（物体类别、场景类型…）。

实验：
1. 对 RADIO 的 COCO 特征做 PCA，取 top-4 PC
2. 对每个 PC，抽取高/低值端各 20 张图像可视化
3. 是否每个 PC 对应一个可解释的语义轴？

### 方向 3：PCA 维度压缩的 linear probe 性能曲线

DINOv3 top4=7%（压到 4 维丢 93% 信息），RADIO top4=40%（丢 60%）。
问题：哪个模型在低维压缩后下游性能保留更好？

实验设计：
```
d = [4, 8, 16, 32, 64, 128, 256, full]
每个 d：PCA 降维 → kNN or linear probe → COCO 检索 mAP
```

预期假设：
- RADIO 在极低维（d=4~16）保留更多性能（信息集中）
- DINOv3 在高维（d≥64）才开始追上（信息分散但总量更大）

### 方向 4：训练过程中 Anisotropy 的演化

利用已有的 epoch probe 链路观察：

- 从 RADIO 出发继续训练的模型，pct_var_top4 是否逐渐下降（向 DINOv3 靠近）？
- Sharpening 损失对应 simplex 结构形成于哪个 epoch？
- 对应的 `image_allmodels.png` 里星形轮廓是何时出现的？

---

## 一句话结论

> DINOv3 的"锥形"是**高维 simplex 结构在 2D 的投影**，不是低秩。
> 四个现有指标量的是维度数量，**不量分布形状（单峰 vs 多峰）**。
> 加 `std(pairwise cosine)` 或 GMM-K 估计可以把这两类分开；
> RADIO 低秩 + 域不变的组合则值得做 PC 语义解释实验。
