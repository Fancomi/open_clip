# PE-Core 特征空间的 FPS 收敛特性：一个独特的几何结构发现

> 日期：2025-05-20
> 实验代码：`analysis/clip_fps_probe.py`
> 输出目录：`feature_probe/pretrained/clip_fps_compare/`

---

## 一、背景

在对预训练视觉编码器进行特征空间探索时，我们使用 Farthest Point Sampling (FPS) 对特征点云进行顺序分区采样：每次从剩余点中选出 256 个最远点作为一个 batch，移除后继续。通过 GIF 动画观察每个 batch 在 PCA 空间中的分布（标注 batch centroid 和 1σ 方差椭圆），可以直观揭示特征空间的几何结构。

## 二、目的

1. 观察不同 CLIP 模型的 FPS batch 方差演化规律
2. 验证 PE-Core 的 FPS 收敛现象是否为 CLIP 范式通用特性
3. 通过 kNN 密度分布和 centroid 轨迹量化解释观察到的差异

## 三、方法

### 3.1 模型选择

| 模型 | 训练范式 | 数据规模 | 特征维度 (backbone) |
|------|---------|---------|-------------------|
| **PE-Core** | EVA-CLIP + 蒸馏 | 多源混合 | 768 |
| **SigLIP2** | Sigmoid Loss CLIP | WebLI | 768 |
| **DataComp-XL** | DataComp 数据过滤 | CommonPool | 768 |
| **DFN2B** | Data Filtering Networks | 2B 样本 | 768 |
| **EVA02** | EVA 蒸馏 + CLIP | Merged-30M | 768 |
| **LAION2B** | 标准 CLIP | LAION-2B | 768 |
| **MetaCLIP** | Metadata-based CLIP | CommonCrawl | 768 |

所有模型统一提取 **backbone CLS token**（projection head 前，768-dim），确保在同一语义层级对比。

### 3.2 实验流程

1. **特征提取**：COCO val2014 5000 张图 → 各模型 backbone CLS (5000, 768)
2. **FPS 顺序分区**：batch_size=256, n_batches=20 → 每模型 20 组索引
3. **GIF 可视化**：6 个 PC pair 面板 (PC1v2...PC11v12)，每帧标注 batch centroid (X) + 1σ 方差椭圆
4. **kNN 密度分析**：K=50, density = 1/mean_knn_distance
5. **Centroid 轨迹**：每个 batch 的 centroid 在 PC1-PC2 空间的位移轨迹
6. **收敛曲线**：batch centroid 到全局中心的欧氏距离 vs batch index

## 四、实验结果

### 4.1 FPS Batch GIF 观察

**PE-Core（独特行为）**：
- 方差椭圆随 batch 推进**单调缩小**
- Centroid 平滑向全局中心收敛
- 最终 batch 集中在特征空间的核心区域

**其他 6 个模型（共同行为）**：
- 前期/中期方差椭圆相对稳定
- 后期突然"跳入"某个边缘密集簇，方差反而增大或剧烈波动
- Centroid 轨迹不规则，后期出现跳跃

### 4.2 kNN 密度分布

| 模型 | 密度范围 | 动态范围 (max/min) |
|------|---------|-------------------|
| **PE-Core** | [0.165, 0.916] | **5.6x** |
| SigLIP2 | [0.018, 0.081] | 4.5x |
| DataComp | [0.029, 0.058] | 2.0x |
| DFN2B | [0.029, 0.057] | 2.0x |
| EVA02 | [0.041, 0.093] | 2.3x |
| LAION2B | [0.025, 0.054] | 2.1x |
| MetaCLIP | [0.032, 0.065] | 2.0x |

PE-Core 的密度动态范围远超其他模型，说明其特征空间存在显著的**中心密-边缘疏**梯度。

### 4.3 Centroid 收敛曲线

`centroid_convergence.png` 显示：
- **PE-Core**：batch centroid 到全局中心的距离**单调递减**，呈平滑收敛曲线
- **其他模型**：距离曲线不单调，后期出现跳跃或反弹

### 4.4 密度直方图

`density_histogram.png` 显示：
- **PE-Core**：密度分布呈**宽幅单峰**，右尾延伸（高密度核心）
- **标准 CLIP 模型**（DataComp, DFN2B, LAION2B, MetaCLIP）：密度分布**窄幅集中**，接近均匀
- **EVA02, SigLIP2**：介于两者之间

## 五、分析

### 5.1 几何解释

PE-Core 的特征空间呈现**洋葱结构**（onion-like geometry）：
- 外层稀疏：少量离群点分布在远离中心的位置
- 内层密集：大量样本聚集在中心附近
- FPS 自然从外向内"剥洋葱"，每批选完外层后进入更密集的内层，方差单调缩小

其他 CLIP 模型呈现**均匀分布 + 边缘簇**结构：
- 主体分布相对均匀（密度动态范围仅 2x）
- 存在一个或多个边缘密集簇（可能对应特定语义类别的聚集）
- FPS 前期均匀覆盖主体，后期剩余点恰好落入边缘簇，导致方差突变

### 5.2 可能的成因

1. **训练目标差异**：PE-Core 使用 EVA-CLIP 框架 + 多教师蒸馏，可能引入了更强的特征正则化，使分布趋向各向同性的单峰结构
2. **数据分布**：PE-Core 的训练数据混合了多个来源，可能比单一数据源（如 LAION-2B）产生更均匀的覆盖
3. **架构因素**：PE-Core 基于 EVA ViT，其 RoPE + 特殊初始化可能影响特征空间的几何性质

### 5.3 实践意义

- **课程学习 (Curriculum Learning)**：PE-Core 的洋葱结构天然适合 FPS-based 课程学习——从边缘（困难/稀有样本）到中心（典型样本）的渐进式训练
- **主动学习 (Active Learning)**：对于 PE-Core，FPS 采样天然提供从多样性到代表性的平滑过渡
- **数据去重/压缩**：PE-Core 的中心密集区域可能存在大量冗余，适合基于密度的去重

## 六、输出文件

```
clip_fps_compare/
├── batch_fps_pe_core.gif        # PE-Core FPS GIF (收敛行为)
├── batch_fps_datacomp.gif       # DataComp FPS GIF (边缘簇跳跃)
├── batch_fps_dfn2b.gif          # DFN2B
├── batch_fps_eva02.gif          # EVA02
├── batch_fps_laion2b.gif        # LAION2B
├── batch_fps_metaclip.gif       # MetaCLIP
├── batch_fps_siglip2.gif        # SigLIP2
├── density_histogram.png        # kNN 密度分布直方图
├── centroid_trajectory.png      # FPS batch centroid 轨迹 (PC1 vs PC2)
├── centroid_convergence.png     # Centroid-to-center 距离曲线
├── image_allmodels.png          # 所有模型 PC pairs 对比
└── *_img.npz                    # 特征缓存
```

## 七、复现

```bash
# 从 repo root 运行
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
PYTHONPATH=./src:$PYTHONPATH python -m analysis.clip_fps_probe --max-samples 5000 --force
```
