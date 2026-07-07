# Slot Embedding 分析：语义词在特征空间的几何分布

> 数据集：COCO val（Karpathy split），5000 条 caption  
> 特征来源：`logs/20260507_baseline_cc3m/cc3m_pe_dinov3_leproj_muon_lr001_0429_1821/checkpoints/probe/step_007040.npz`  
> 输出目录：`analysis/outputs/slots/coco_val_5000/`  
> 更新日期：2026-05-24

---

## 一、背景与动机

Curriculum Learning 实验中，我们使用 density（局部密度）和 curvature（局部曲率）来描述 embedding 空间的局部几何结构，并以此排序训练样本。这两个度量对不同语义类别的图像是否有差异化响应，目前还不清楚。

本实验的核心问题是：

> **高频语义词（man, sitting, white, on）和低频语义词（apples, staring, sandy, inside of）所对应的图像，在 embedding 空间中的密度和曲率分布是否存在系统性差异？**

这一问题的答案有助于：
1. 解释 curriculum 策略为何对某些语义类别效果更显著
2. 验证 embedding 空间对语义频率的响应是否有规律
3. 为后续按语义类别设计 curriculum 提供依据

---

## 二、Pipeline

### 2.1 整体流程

```
COCO caption (5000)
    ↓
[VLM Slot 抽取] → slots.jsonl
    ↓
[词频统计] → slot_frequencies.json / stats/*.png
    ↓
[Embedding Overlay] → overlay_min10/*.png + slot_geometry_summary.csv
```

### 2.2 Slot 抽取

用 VLM 对每条 caption 抽取 4 类语义 slot：

| Slot 类型 | 说明 | 样本数 | unique 词数 |
|-----------|------|--------|------------|
| `nouns` | 名词（主体、场景） | 15583 | 2397 |
| `verbs` | 动词（动作） | 4626 | 709 |
| `adjectives` | 形容词（属性） | 4049 | 709 |
| `spatial_relations` | 空间关系词 | 4367 | 138 |

VLM 抽取质量：5000/5000 成功，bad 率 = 0。

### 2.3 Embedding Overlay

对每个 slot 类型，分别选 **high 频词**（Top-5）和 **low 频词**（min_count 以上的低频词），在 UMAP/t-SNE 投影空间上 overlay density 和 curvature，观察分布规律。

低频词统一使用 `min_count=10`。旧的 `overlay/`（min_count=1）和
`overlay_min5/` 已清理，避免 singleton 或低计数词带来的噪声。

---

## 三、选中词（min_count=10 版本）

| Slot | high 频词 | low 频词 |
|------|-----------|----------|
| nouns | man, people, woman, table, group | apples, baseball field, benches, bread, drink |
| verbs | sitting, standing, holding, riding, walking | eat, hitting, moving, reading, staring |
| adjectives | white, large, black, young, small | clear, fresh, sandy, wood, bright |
| spatial_relations | on, in, next to, on top of, near | back, beside, past, underneath, inside of |

所有 slot 类型的 feature-caption 对齐率均为 **1.0**（5000/5000 匹配）。

---

## 四、几何度量摘要（overlay_min10）

`slot_geometry_summary.csv` 包含每个选中词的 density 和 curvature 统计。以下为各 slot 类型的代表性数据：

### Nouns

| group | word | n | density_mean | curvature_mean |
|-------|------|---|-------------|----------------|
| high | man | 618 | 0.975 | 0.884 |
| high | people | 317 | 0.960 | 0.885 |
| high | table | 238 | 0.976 | 0.884 |
| low | apples | 10 | 0.931 | 0.891 |
| low | baseball field | 10 | **1.055** | **0.838** |
| low | benches | 10 | 0.936 | 0.876 |

`baseball field` 密度明显偏高（1.055），曲率偏低（0.838），说明对应图像在 embedding 空间中聚集在高密度平坦区域。

### Verbs

| group | word | n | density_mean | curvature_mean |
|-------|------|---|-------------|----------------|
| high | sitting | 547 | 0.975 | 0.883 |
| high | riding | 213 | 1.005 | 0.871 |
| low | hitting | 10 | **1.046** | **0.856** |
| low | staring | 10 | **1.047** | **0.873** |
| low | reading | 10 | 0.946 | 0.885 |

低频动词 hitting/staring 密度偏高，提示这些动作对应的图像落在 embedding 的稠密区（视觉上可能和高频动作场景相似）。

### Adjectives

| group | word | n | density_mean | curvature_mean |
|-------|------|---|-------------|----------------|
| high | white | 339 | 0.989 | 0.875 |
| high | young | 170 | 0.972 | 0.885 |
| low | clear | 10 | 0.936 | **0.891** |
| low | sandy | 10 | 0.966 | 0.886 |

形容词的 density/curvature 差异相对较小，high/low 频词分布较为接近。

### Spatial Relations

| group | word | n | density_mean | curvature_mean |
|-------|------|---|-------------|----------------|
| high | on | 1287 | — | — |
| high | in | 737 | — | — |
| low | past | 12 | — | — |
| low | underneath | 12 | — | — |

空间关系词 unique 数最少（138），但出现率高（on 出现 1287 次），说明 COCO caption 对空间关系描述高度集中。

> 完整数值见 `analysis/outputs/slots/coco_val_5000/overlay_min10/slot_geometry_summary.csv`

---

## 五、可视化产物

每个 slot 类型 × 每个度量，各生成一张 overlay 图，只保留 `overlay_min10/`：

```
overlay_min10/
  slot_overlay_nouns_density.png
  slot_overlay_nouns_curvature.png
  slot_overlay_verbs_density.png
  slot_overlay_verbs_curvature.png
  slot_overlay_adjectives_density.png
  slot_overlay_adjectives_curvature.png
  slot_overlay_spatial_relations_density.png
  slot_overlay_spatial_relations_curvature.png
  slot_selected_words.json
  slot_geometry_summary.csv/json
```

---

## 六、运行方式

```bash
# 完整流程（VLM 抽取 + 统计 + overlay）
bash analysis/run_slot_vlm.sh

# 仅生成 overlay（已有 slots.jsonl 的情况下）
python -m analysis.run \
  --mode overlay_slots \
  --slots-path analysis/outputs/slots/coco_val_5000/slots.jsonl \
  --probe-path <probe.npz> \
  --output-dir analysis/outputs/slots/coco_val_5000/overlay_min10 \
  --min-count 10 \
  --save-geometry-summary
```

核心文件：
- `analysis/slots.py` — slot 抽取逻辑
- `analysis/slot_viz.py` — overlay 可视化
- `analysis/slot_pipeline.py` — 端到端 pipeline
- `analysis/run_slot_vlm.sh` — 一键脚本（默认 MIN_COUNT=10，输出 `overlay_min10/`）

---

## 七、初步结论

1. **低频名词**（如 `baseball field`）密度偏高、曲率偏低，说明它们对应的图像在 embedding 空间中聚集在与高频词相似的平坦稠密区，而非孤立分布。
2. **低频动词**（如 `hitting`, `staring`）密度也偏高，提示动作类低频词在视觉上与高频动作（sitting, riding）共享相似的图像空间。
3. **形容词和空间关系词**的 high/low 频词几何差异相对较小，可能因为这类语义主要由 caption 描述，视觉特征本身区分度不强。
4. 整体来看，词频和 density/curvature 之间**不存在简单的单调关系**——低频词并不系统性地落在 embedding 的稀疏高曲率边缘区，这对 curriculum 设计有一定启示。

---

## 八、后续方向

- [ ] 在 CC3M（更大规模、更多长尾词）上复现本分析，验证结论的泛化性
- [ ] 量化 high 频词和 low 频词的 density/curvature 分布差异（KS 检验或 Wasserstein 距离）
- [ ] 将 slot 类别作为 curriculum 排序的辅助信号，测试是否能改善 retrieval 性能
