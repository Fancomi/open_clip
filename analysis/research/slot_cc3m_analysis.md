# Slot Embedding 分析：CC3M 数据的语义词几何分布

> 数据集：CC3M train（随机抽取 5000 条）  
> 特征来源：PE-Core-B-16 pretrained（`models/timm/PE-Core-B-16/open_clip_model.safetensors`），1024-dim CLIP projection  
> 样本目录：`/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m_slot_sample/`  
> 输出目录：`analysis_outputs/slots/cc3m_5000/`  
> 更新日期：2026-05-24

---

## 一、背景与动机

在 [COCO slot 分析](slot_embedding_analysis.md) 中，我们已对 COCO val 5000 条 caption 做了完整的槽位提取和 embedding 几何分析。本实验将同样的方法应用于 **CC3M 训练集**，目的是：

1. 对比 CC3M 和 COCO 的 caption 词汇分布差异（CC3M 来源于互联网 alt-text，更嘈杂）
2. 验证相同分析方法在更大、更多样的数据集上的稳定性
3. 为 curriculum 研究提供 CC3M 特征空间的几何基准

与 COCO 分析的关键差异：
- **模型**：COCO 分析用的是训练后的 CC3M baseline 模型；CC3M 分析使用 PE-Core **pretrained**（因 baseline 无保存的 .pt 权重）
- **数据**：CC3M 为 alt-text，描述风格更多样（含 `background`, `actor`, `artist` 等媒体词汇）
- **特征维度**：使用 proj_features（1024-dim CLIP projection）与 COCO 分析保持一致

---

## 二、Pipeline

### 2.1 样本提取

从 CC3M WDS tarballs（576 个 `cc3m-train-*.tar`）随机抽取 5000 对 (JPEG, caption) 到磁盘：

```bash
bash analysis/run_cc3m_slot.sh --limit 5000
```

图片保存至：`datas/cc3m_slot_sample/images/`  
TSV 路径：`datas/cc3m_slot_sample/cc3m_sample.tsv`

### 2.2 Probe 生成

使用 PE-Core-B-16 pretrained 对 5000 张图像提取特征，保存为 probe.npz：

```
datas/cc3m_slot_sample/probe_pe_core.npz
  features      (5000, 768)   — backbone CLS
  proj_features (5000, 1024)  — CLIP projection（overlay 使用此 key）
  txt_features  (5000, 1024)  — 文本特征
  paths         (5000,)       — 图片绝对路径
```

### 2.3 VLM Slot 抽取 + Stats + Overlay

与 COCO 流程相同，详见 [slot_embedding_analysis.md](slot_embedding_analysis.md)。

---

## 三、词频分布

| Slot 类型 | 总词次 | unique 词数 | Top-5 |
|-----------|--------|------------|-------|
| `nouns` | 15845 | 4088 | person, background, actor, artist, view |
| `verbs` | 4243 | 1556 | attends, performs, looking, sitting, attend |
| `adjectives` | 5273 | 1516 | white, young, black, red, old |
| `spatial_relations` | 2483 | 146 | on, in, at, from, over |
| `proper_nouns` | 151 | 124 | us, wednesday, imax, london |
| `adverbs` | 563 | 248 | just, very, after, today, also |

**与 COCO 的差异**：
- CC3M nouns Top 词出现 `background`, `actor`, `artist`，具有明显媒体/娱乐风格，COCO 以 `man, people, woman` 为主
- CC3M verbs Top 词为 `attends, performs`（活动/表演场景），COCO 以 `sitting, standing` 等姿势动词为主
- CC3M proper_nouns 比例更高（互联网来源），COCO 几乎没有
- CC3M unique 词数（nouns 4088 vs COCO 2397）远高于 COCO，体现了更大的长尾多样性

---

## 四、选中词（overlay_min10）

| Slot | high 频词 | low 频词 |
|------|-----------|----------|
| nouns | person, background, actor, artist, view | bowl, class, designer, display, door |
| verbs | attends, performs, looking, sitting, attend | driving, eating, filming, find, performing |
| adjectives | white, young, black, red, old | autumn, cold, elderly, flat, huge |
| spatial_relations | on, in, at, from, over | off, between, in the background, next to, before |

所有 slot 对齐率均为 **1.0**（5000/5000 feature 完全匹配）。

---

## 五、可视化产物

```
analysis_outputs/slots/cc3m_5000/
  slot_requests.jsonl               # 5000 lines
  slots.jsonl                       # 5000 lines
  stats/
    slot_frequencies.json
    slot_summary.json
    *.png (16 张频率图)
  overlay_min5/                     # min_count=5，推荐展示
    slot_overlay_{slot}_{density,curvature}.png  (8 张)
    slot_selected_words.json
    slot_geometry_summary.csv/json
  overlay_min10/                    # min_count=10，推荐定量
    slot_overlay_{slot}_{density,curvature}.png  (8 张)
    slot_selected_words.json
    slot_geometry_summary.csv/json
```

---

## 六、运行方式

```bash
# 全量（5000 条）
bash analysis/run_cc3m_slot.sh

# 自定义
bash analysis/run_cc3m_slot.sh --limit 2000 --metric curvature --min-count 10

# 跳过已有 extract/probe
bash analysis/run_cc3m_slot.sh --no-extract --no-probe
```

可通过环境变量覆盖路径：

```bash
CC3M_WDS_DIR=... SAMPLE_DIR=... PROBE_PATH=... OUT_ROOT=... \
  bash analysis/run_cc3m_slot.sh
```

---

## 七、与 COCO 对比的初步发现

1. **词汇多样性更高**：CC3M nouns unique 词数 4088 vs COCO 2397，长尾更明显，体现互联网 alt-text 的自然语言多样性。

2. **语义场景差异**：CC3M 大量媒体/娱乐词（actor, performs, attends），COCO 以日常生活场景为主（sitting, standing, table）。

3. **spatial_relations 差异**：CC3M 出现 `at, from, over` 等更宽泛的介词（COCO 以 `on, in, next to` 为主），说明 CC3M 描述的视角和场景关系更多样。

4. **几何特征**：详见 `overlay_min10/slot_geometry_summary.csv`。CC3M 特征空间基于 pretrained PE-Core，COCO 基于 fine-tuned 模型，二者不直接可比，但内部 high/low 频词的相对几何关系可独立分析。

---

## 八、后续方向

- [ ] 使用相同的 fine-tuned 模型（需保存 CC3M baseline .pt 权重）重新生成 probe，使 CC3M 和 COCO 几何分析在同一模型空间下可比
- [ ] 对 CC3M 的 `proper_nouns` 槽位做单独分析（互联网数据特有）
- [ ] 扩展到更大规模（50K+）验证长尾词的几何分布规律
