# Curriculum Learning: 基于特征空间度量的样本采样顺序实验

## 背景

CLIP-style 对比学习中，训练数据的呈现顺序通常是完全随机的（每 epoch 独立 shuffle）。然而，课程学习 (Curriculum Learning) 的经典结论表明，训练顺序对模型收敛速度和最终性能有显著影响。

本实验系列探索：**在 PE-Core-B-16-dinov3 + SigLIP + SIGReg + Muon 最优配置基础上，是否可以通过有目的地控制样本呈现顺序来提升最终 retrieval 性能。**

核心创新点：每个 epoch 开始前，对全量训练数据进行特征提取，然后基于特征空间的几何度量计算训练顺序。

## 目的

1. 验证「非随机训练顺序」是否优于「随机 shuffle」
2. 对比 5 种排序策略的效果差异
3. 探索 epoch 0 初始特征源 (DINOv3 / PE-Core / self) 对最终结果的影响
4. 为后续 CC3M 大规模验证提供策略筛选依据

## 方法

### Pipeline

```
for each epoch:
    1. [特征提取] 用当前模型(或epoch0用外部模型)对全量训练集提取 CLS 特征
    2. [排序计算] 基于特征空间度量计算样本顺序 (GPU 加速)
    3. [有序训练] 按计算出的顺序遍历训练样本
```

### 排序策略

| 策略 | 度量 | 直觉 |
|------|------|------|
| **fps** | Farthest Point Sampling | 每 batch 最大化样本多样性, 避免冗余 |
| **density_high** | kNN 密度高→低 | 先学典型样本 (cluster 中心), 后学边缘/噪声 |
| **density_low** | kNN 密度低→高 | 先学边缘/困难样本, 后学典型样本 |
| **curvature_high** | kNN 曲率高→低 | 先学决策边界区域 (高各向异性), 后学平坦区域 |
| **curvature_low** | kNN 曲率低→高 | 先学平坦区域 (低各向异性), 后学边界区域 |

### Epoch 0 初始特征源

| 选项 | 说明 |
|------|------|
| **dinov3** | 用冻结 DINOv3-ViT-B/16 提取, 自监督视觉特征 |
| **pe_core** | 用冻结 PE-Core-B-16 提取, 多模态预训练视觉特征 |
| **self** | 用当前模型 (随机初始化) 提取, 初始特征空间 |

Epoch 1+ 始终用当前模型 checkpoint 提取 (特征空间随训练演化, 排序自适应更新)。

### 技术细节

- **kNN 计算**: GPU 加速 (`torch.cdist` + `topk`), 82K samples × K=50, ~0.6s/epoch
- **FPS 计算**: GPU 逐步选最远点, ~2min/epoch (82K 全排序)
- **曲率计算**: 批量 SVD (`torch.linalg.svdvals`), ~5s/epoch
- **分布式**: Rank 0 计算排序, `dist.broadcast` 同步所有 rank
- **Sampler**: `OrderedDistributedSampler` 按连续块划分给各 GPU

## 实验设计

### 数据
- 训练: COCO clip_train_dedup.tsv (~82K samples)
- 验证: COCO Karpathy 5-caption split (5K images × 5 caps)
- Probe: COCO Karpathy 1-caption

### 超参数 (与 wm_coco.sh baseline 一致)
- Model: PE-Core-B-16-dinov3
- Loss: SigLIP + SIGReg(cls, w=1e-4) + Muon
- Epochs: 20, warmup: 42 steps (~2 epochs)
- BS: 4096 (512×8 GPU), LR: 3.4e-4, Muon LR: 0.01
- Steps/epoch: ~20

### 实验矩阵 (16 runs)

| # | Strategy | Init | Tag |
|---|----------|------|-----|
| 0 | - (random) | - | baseline |
| 1 | fps | dinov3 | fps_dinov3 |
| 2 | fps | pe_core | fps_pecore |
| 3 | fps | self | fps_self |
| 4 | density_high | dinov3 | dhi_dinov3 |
| 5 | density_high | pe_core | dhi_pecore |
| 6 | density_high | self | dhi_self |
| 7 | density_low | dinov3 | dlo_dinov3 |
| 8 | density_low | pe_core | dlo_pecore |
| 9 | density_low | self | dlo_self |
| 10 | curvature_high | dinov3 | chi_dinov3 |
| 11 | curvature_high | pe_core | chi_pecore |
| 12 | curvature_high | self | chi_self |
| 13 | curvature_low | dinov3 | clo_dinov3 |
| 14 | curvature_low | pe_core | clo_pecore |
| 15 | curvature_low | self | clo_self |

## 效果

### Best i2t R@1 排名 (across all epochs)

| Rank | Strategy | Init | best i2t R@1 | best epoch | best t2i R@1 | vs baseline |
|------|----------|------|-------------|------------|-------------|-------------|
| 1 | **fps_reverse** | **dinov3** | **0.0202** | 12 | 0.0147 | **+16.1%** |
| 2 | curvature_high | dinov3 | 0.0194 | 12 | 0.0159 | +11.5% |
| 3 | fps | pe_core | 0.0190 | 6 | 0.0149 | +9.2% |
| 4 | curvature_low | dinov3 | 0.0188 | 14 | 0.0143 | +8.0% |
| 5 | density_low | self | 0.0186 | 14 | 0.0152 | +6.9% |
| 6 | curvature_low | self | 0.0186 | 8 | 0.0149 | +6.9% |
| 7 | fps_reverse | self | 0.0182 | 8 | 0.0157 | +4.6% |
| 8 | curvature_high | pe_core | 0.0182 | 14 | 0.0154 | +4.6% |
| 9 | fps | self | 0.0180 | 18 | 0.0145 | +3.4% |
| 10 | density_low | dinov3 | 0.0178 | 10 | 0.0146 | +2.3% |
| 11 | curvature_low | pe_core | 0.0178 | 8 | 0.0153 | +2.3% |
| - | **baseline** | - | **0.0174** | 10 | 0.0145 | - |
| 12 | density_high | self | 0.0174 | 12 | 0.0139 | +0.0% |
| 13 | density_high | dinov3 | 0.0170 | 14 | 0.0131 | -2.3% |
| 14 | curvature_high | self | 0.0168 | 10 | 0.0152 | -3.4% |
| 15 | fps | dinov3 | 0.0162 | 8 | 0.0151 | -6.9% |
| 16 | fps_reverse | pe_core | 0.0158 | 8 | 0.0142 | -9.2% |
| 17 | density_low | pe_core | 0.0158 | 6 | 0.0156 | -9.2% |
| 18 | density_high | pe_core | 0.0156 | 10 | 0.0139 | -10.3% |
| 14 | density_low | pe_core | 0.0158 | 6 | 0.0156 | -9.2% |
| 15 | density_high | pe_core | 0.0156 | 10 | 0.0139 | -10.3% |

### Final epoch (ep18) 结果

| Strategy | Init | i2t R@1 | t2i R@1 | vs baseline |
|----------|------|---------|---------|-------------|
| baseline | - | 0.0162 | 0.0136 | - |
| fps | dinov3 | 0.0156 | 0.0144 | -3.7% |
| fps | pe_core | 0.0152 | 0.0142 | -6.2% |
| fps | self | **0.0180** | 0.0137 | **+11.1%** |
| density_high | dinov3 | 0.0148 | 0.0120 | -8.6% |
| density_high | pe_core | 0.0134 | 0.0128 | -17.3% |
| density_high | self | 0.0142 | 0.0128 | -12.3% |
| density_low | dinov3 | 0.0172 | 0.0142 | +6.2% |
| density_low | pe_core | 0.0156 | 0.0134 | -3.7% |
| density_low | self | 0.0168 | 0.0152 | +3.7% |
| curvature_high | dinov3 | 0.0144 | 0.0142 | -11.1% |
| curvature_high | pe_core | 0.0168 | 0.0152 | +3.7% |
| curvature_high | self | 0.0150 | 0.0140 | -7.4% |
| curvature_low | dinov3 | 0.0170 | 0.0141 | +4.9% |
| curvature_low | pe_core | 0.0150 | 0.0128 | -7.4% |
| curvature_low | self | 0.0176 | 0.0138 | +8.6% |

## 分析

### 核心发现

1. **新冠军: fps_reverse + dinov3 (+16.1%)**:
   - FPS 反序 = 先学冗余样本（cluster 内部），后学多样性样本（边界/离群点）
   - 与 fps 正序 + dinov3 (-6.9%) 形成鲜明对比：同一排序，方向相反，效果天壤之别
   - 直觉：先用重复样本「打地基」建立稳定表示，再用多样性样本「拓展边界」

2. **FPS 正序 vs 反序对比**:

   | Init | fps (正序) | fps_reverse (反序) | 差异 |
   |------|-----------|-------------------|------|
   | dinov3 | 0.0162 (-6.9%) | **0.0202 (+16.1%)** | 反序大幅领先 |
   | pe_core | 0.0190 (+9.2%) | 0.0158 (-9.2%) | 正序领先 |
   | self | 0.0180 (+3.4%) | 0.0182 (+4.6%) | 持平 |

   - dinov3 init: 反序碾压正序（DINOv3 特征空间中的「冗余」恰好是语义 cluster 中心）
   - pe_core init: 正序碾压反序（PE-Core 特征空间中的「多样性」更有价值）
   - self init: 无显著差异（随机特征的 FPS 排序本身无语义意义）

3. **方向性规律总结**:
   - `density_high`（简单/冗余优先）: 一致负面
   - `fps_reverse`（FPS 反序 = 冗余→多样）: dinov3 下最优，但 pe_core 下最差
   - 关键区别：fps_reverse 是**渐进式**从冗余到多样（有序过渡），density_high 是**一刀切**按密度排序
   - 渐进过渡 > 一刀切排序

4. **Init mode 的决定性作用**:
   - dinov3 init 在 curvature_high 和 fps_reverse 上均为最优 → DINOv3 自监督特征提供了最有意义的几何结构
   - pe_core init 表现不稳定（fps 正序最优，其余多为负面）→ 多模态预训练特征的几何结构与训练目标不完全对齐
   - self init 表现稳定但不极端 → 随机特征的排序提供了一种「温和的正则化」

5. **收敛动态**:
   - fps_reverse_dinov3: ep12 达峰后回落至 ep18 的 0.0156（过拟合）
   - fps_reverse_self: ep8 达峰，ep18 仍保持 0.0174（稳定）
   - 建议：对 dinov3 init 策略配合 early stopping 或 lr cooldown

### 结论

- **最佳配置**: `fps_reverse + dinov3` (best i2t R@1 = 0.0202, +16.1% vs baseline)
- **最稳定配置**: `fps_reverse + self` 或 `curvature_low + self` (+4.6~6.9%, 无需外部模型)
- **核心洞察**: 训练顺序的「方向」比「度量」更重要；渐进式从冗余到多样优于一刀切
- **下一步**: CC3M 验证 (见下方)

---

## CC3M 验证实验 (Shard-level 近似排序)

### 设计

对标基线: `wds_cc3m_pe_dinov3_sigreg_siglip_muon` (i2t R@1=0.2190, t2i R@1=0.1557, 10 epochs)

**重要区别**: CC3M 为 WebDataset 格式，无法做样本级精确排序。采用 **shard 级近似排序**:
- 576 shards × ~5046 samples/shard
- 每 shard 采样 32 张图提取特征 → 计算 shard 质心 (576 × 768)
- 对 576 个质心做精确 FPS/排序 → 按排序后的 shard 顺序喂数据
- 禁用 shard-level shuffle (`detshuffle2` bypass)，保留 sample-level buffer shuffle (5000)
- 排序耗时: ~120s (epoch 0 质心提取) + <1s (FPS on 576 points)

**与 COCO 实验的本质差异**:
- COCO: 样本级精确排序 (82K 样本逐个排序)
- CC3M: shard 级粗粒度排序 (576 个 ~5K 样本块排序，块内仍有 buffer shuffle)
- 粒度差 ~5000x，信号可能被稀释

### 实验矩阵 (5 runs)

| # | Strategy | Init | 选择依据 |
|---|----------|------|---------|
| 0 | baseline | - | 复现对标 |
| 1 | fps_reverse | dinov3 | COCO Top-1 (+16.1%) |
| 2 | curvature_high | dinov3 | COCO Top-2 (+11.5%) |
| 3 | fps_reverse | self | COCO 最稳定 (+4.6%) |
| 4 | fps | self | 正反序对照 |

### 效果

*待实验完成后填写*

| Strategy | Init | i2t R@1 | t2i R@1 | vs baseline (0.2190) |
|----------|------|---------|---------|---------------------|
| baseline | - | | | - |
| fps_reverse | dinov3 | | | |
| curvature_high | dinov3 | | | |
| fps_reverse | self | | | |
| fps | self | | | |
