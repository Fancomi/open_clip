# Modality Gap in CLIP-Style Models: Analysis & Experiments

## 1. 现象描述

对任意 CLIP-范式模型（CLIP、OpenCLIP、SigLIP、PE-Core、TIPSv2）在 COCO 或 CC3M 数据上提取图文特征，对所有特征做 PCA，**PC1 总是图文模态差异**：

- 图像侧投影值集中在 `[+0.4, +0.6]`
- 文本侧投影值集中在 `[-0.6, -0.4]`
- 两侧之间没有连接，PC1 几乎完全可分

该现象在我们的从头训练实验中也完全可以复现。

---

## 2. 理论分析

### 2.1 为什么对比学习会产生模态 gap

设模态方向 **u** = normalize(μ_img - μ_txt)：

```
img_i = a_i * u + r_i   (模态偏置 + 内容残差)
txt_j = b_j * u + s_j
```

相似度分解为：

```
sim(img_i, txt_j) = a_i * b_j  +  r_i · s_j
                    ──────────────  ──────────
                    模态公共偏置      内容残差
```

若所有文本的 `b_j` 近似相同，`a_i * b_j` 成为同行 softmax 的公共常数，被归一化抵消。模态 gap 不直接帮助分类负样本，但 **CLIP loss 也不惩罚它**：

- **alignment** 只拉近正样本图文对
- **uniformity** 在跨模态间推开负样本
- 两者共同允许"各自占球面一侧"的稳定局部极值

### 2.2 五个推论

| # | 推论 | 含义 |
|---|------|------|
| 1 | PC1 在超球面上极度聚集 | L2 norm 不消除 PC1；两族分布各自贴近球面一侧 |
| 2 | 对比 loss 导致的捷径 | 模态身份轴是最容易学到的"免费"分类维度 |
| 3 | "统一图文空间"不充分 | CLIP space 统一的是相似度坐标系，不是分布本身 |
| 4 | 内容 matching 走了捷径 | 跨模态语义信息利用不充分，主要靠内容残差 `r·s` |
| 5 | 最近邻结构偏倚 | 正样本分布在 `)(` 内侧，跨模态最相似样本位于间隙中央 |

### 2.3 目标澄清

**不是**强行让图文分布完全重叠（可能损伤语义）。  
**而是**让跨模态相似度尽量不依赖模态轴，让内容语义承担主要权重。

---

## 3. 实验设计

### 3.1 评估指标（每个实验必须报告）

| 指标 | 含义 |
|------|------|
| `pc1_gap` | PCA-PC1 上图文均值之差（越接近 0 越好） |
| `pc1_var_ratio` | PC1 解释方差比例（越小越好） |
| `modality_clf_acc` | 线性模态分类器准确率（越接近 50% 越好） |
| `i2t_R@1,5,10` | 图到文检索 Recall |
| `t2i_R@1,5,10` | 文到图检索 Recall |
| COCO val R@1 (训练时 | val 曲线) | 主性能指标 |

### 3.2 Baseline

`pe_dinov3_sigreg_cls_probe`（quick.sh）：

```bash
--siglip --sigreg-target cls --sigreg-weight 1e-4
--epochs 10 --warmup 512 --lr ${LR} --probe-data ${PROBE_TSV}
```

### 3.3 实验矩阵

#### Step 0: Post-processing analysis（不训练，纯分析）

对 baseline 已保存的 probe `.npz` 文件运行 `analysis/modality_gap.py`，
测试以下四种后处理变体对检索指标的影响：

| 变体 | 操作 |
|------|------|
| `raw` | 原始特征（对照） |
| `centered` | 各自去均值，re-normalize |
| `gap_remove` | 投影掉模态方向向量，re-normalize |
| `whitened` | PCA whitening on joint pool，re-normalize |

**决策条件**：若 `centered` 或 `gap_remove` 提升 i2t R@1 ≥ 0.5%，则训练时加 gap loss 有据可循。

```bash
python3 analysis/modality_gap.py \
    --probe logs/<run>/probe/step_001740.npz \
    --split proj_features \
    --out   analysis/research/modality_gap_baseline.json
```

#### Step 1: `--modality-gap-weight` λ 消融（无 DINOv3）

`sigreg_cls` 基础上叠加 gap loss，λ ∈ {0.001, 0.005, 0.01, 0.05}：

| 实验名 | λ_gap | 预期 |
|--------|-------|------|
| `mgap_gap001` | 0.001 | 最温和，对主 loss 影响小 |
| `mgap_gap005` | 0.005 | ★ 推荐起点 |
| `mgap_gap01`  | 0.010 | 中强 |
| `mgap_gap05`  | 0.050 | 可能损伤 |

#### Step 2: DINOv3 + `--modality-gap-weight` 消融

在 `pe_dinov3_dinov3_muon_sigreg_probe` 配置上叠加 gap loss，λ ∈ {0.001, 0.005, 0.01}：

| 实验名 | λ_gap |
|--------|-------|
| `mgap_dino_gap001` | 0.001 |
| `mgap_dino_gap005` | 0.005 ★ |
| `mgap_dino_gap01`  | 0.010 |

---

## 4. 实现

### 4.1 `ModalityGapLoss`（`src/open_clip/loss.py`）

```python
class ModalityGapLoss(nn.Module):
    """
    L_gap = || EMA(mean_img) - EMA(mean_txt) ||²
    """
    def forward(self, image_features, text_features) -> torch.Tensor:
        # allreduce batch means across GPUs
        # optional EMA smoothing (ema_momentum=0.999)
        return (mu_img - mu_txt).pow(2).sum()
```

- 支持 EMA 平滑（`--modality-gap-ema`，默认 0.999）
- allreduce batch mean 保证多卡一致
- 纯 batch-level：设 `--modality-gap-ema 1.0`

### 4.2 参数接口

```bash
--modality-gap-weight FLOAT   # λ_gap, 默认 0.0（禁用）
--modality-gap-ema    FLOAT   # EMA momentum，默认 0.999
```

两个 loss 路径均支持：
- `SIGRegContrastiveLoss`（sigreg-only 实验）
- `CLIPWithDINOLoss`（DINOv3 实验）

### 4.3 分析工具

`analysis/modality_gap.py`：

```bash
python3 analysis/modality_gap.py \
    --probe  <path_to_step_XXXXXX.npz> \
    --split  proj_features              # or 'features' for backbone CLS
    --out    analysis/research/results.json
```

输出：

- `pc1_gap`、`pc1_var_ratio`
- `modality_clf_acc`
- `i2t/t2i R@1,5,10`（4 个变体对比）

---

## 5. 假设与预期结果

### 5.1 Post-processing

| 假设 | 验证方式 |
|------|----------|
| `gap_remove` 检索提升 | 说明模态轴不携带内容信息，可安全移除 |
| `gap_remove` 检索下降 | 说明模态差异中混入了有用的频率/内容信号 |
| `whitened` 大幅提升 | 特征各向异性严重，需要完整白化 |

### 5.2 训练时 gap loss

| 假设 | 预期 |
|------|------|
| λ_gap=0.001 | pc1_gap 略降，主 loss 无损 |
| λ_gap=0.005 | ★ 最优：pc1_gap 显著降，R@1 持平或略升 |
| λ_gap=0.050 | 主 loss 受干扰，R@1 可能下降 |
| DINOv3+gap  | KoLeo 已有类似效果，gap loss 增量可能较小 |

### 5.3 关键判断条件

```
λ_gap=0.005 结果  → 若 R@1 不下降 且 pc1_gap 降低 ≥ 30%
                  → 继续做 Step 2 (DINOv3+gap)
                  → 否则停止，gap loss 不值得用于该配置
```

---

## 6. 后续可选实验（根据 Step 1 结果决定）

| 优先级 | 实验 | 触发条件 |
|--------|------|----------|
| A | Mean + Covariance alignment（CORAL-style）| gap loss 有效 |
| B | Modality adversarial（gradient reversal）| gap loss 有效，效果不够强 |
| C | Centered logits（batch-level centering before similarity）| gap 无效，换思路 |
| D | EMA centered logits（running mean centering at inference）| C 有效 |

---

## 7. 运行指令

```bash
# Step 0: Post-processing（需先有 baseline probe）
source /path/to/envs/dino/bin/activate
PYTHONPATH=./src python3 analysis/modality_gap.py \
    --probe logs/<run>/probe/step_001740.npz \
    --out   analysis/research/modality_gap_baseline.json

# Step 1-3: Training experiments
bash experiments/modality_gap.sh
```

---

## 8. 结果记录

| 实验 | pc1_gap | clf_acc | i2t R@1 | t2i R@1 | 备注 |
|------|---------|---------|---------|---------|------|
| baseline | TBD | TBD | TBD | TBD | pe_dinov3_sigreg_cls |
| gap_remove (post) | TBD | TBD | TBD | TBD | 无训练成本 |
| λ=0.001 | TBD | TBD | TBD | TBD | |
| λ=0.005 | TBD | TBD | TBD | TBD | ★ 关键决策点 |
| λ=0.010 | TBD | TBD | TBD | TBD | |
| λ=0.050 | TBD | TBD | TBD | TBD | |
| dino+λ=0.005 | TBD | TBD | TBD | TBD | |

---

*文档版本：2026-05-03 | 代码实现：`src/open_clip/loss.py::ModalityGapLoss`*
