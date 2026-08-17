# 评估协议：口径绑定、零重叠、全量报数

*创建: 2026-08-12，更新: 2026-08-17 | 起因: 四次评估事故（口径错配 / 小样本虚高 / 训练集内测试 / epoch 未对齐）*

---

## 0. 四条铁律

1. **口径绑定**：相似度排序口径由训练 `--neg-mode` 决定，评估必须用同一口径。
   错配会让指标系统性归零（精确 0，不是随机值）—— 是评估 bug，不是模型坏。（第 1-2 节）
2. **零重叠**：评测样本与训练集必须零重叠。CC3M 系数据互为训练集，不能互当评测集。
   干净评测只有 COCO karpathy 5cap / IN-1k val / Urban-1k。（第 3 节）
3. **只报全量**：检索指标只报全量（COCO 5000 图 / Urban-1k 1000 图）。小样本虚高 2.7 倍。（第 4 节）
4. **同表同 epoch**：一张对比表里所有模型取同一 epoch 的 checkpoint。
   ep9→ep10 漂移实测 1~1.7 点，与多数消融效应同量级。（第 6 节）

---

## 1. 事故记录（务必读，避免重复踩坑）

### 1.1 现象

`logs/visreg_gemma_gt_gt_base_0811_1318/epoch_9.pt`（E 配方，训练日志记录 i2t R@1 = 23.22%）
用自写脚本按标准 CLIP 方式（`cos` 排序）评估，得到：

```
i2t R@1 = 0.0000   t2i R@1 = 0.0000
img vs 自己的 caption  cos = -0.36   ← 负值
img vs 随机 caption    cos = +0.02   ← 正值
```

### 1.2 排查过程中被误判为"根因"的无关项

以下全部检查过，**全部正常**，浪费了 6 轮排查：

- checkpoint 键匹配（`load_state_dict` missing=0）
- CLIPLeJEPA 包装结构、`clip_model.*` 前缀剥离
- tokenizer 输出（list 输入返回 `[N, 256]`，标量输入 `[1, 256]`）
- preprocess（`train_tr` 随机裁剪 vs `val_tr` CenterCrop）
- 三条图像编码路径一致性（`encode_image` / `forward` / `_get_image_raw`，cos 全 = 1.0）
- 归一化统计量（0.5 vs CLIP 官方 mean/std）

### 1.3 真正的根因

训练配方 `CHAMPION` 含 `--neg-mode projective`（见 `scripts/train/visreg.sh`）。

Projective SigLIP 的目标是 **|cos| → 1**（详见 `mgap_06_projective_siglip.md`），
该目标有两个合法解：`cos → +1` 与 `cos → −1`。**实测模型 99% 收敛到 −1 分支**：

| 统计项（gt_base epoch_9, n=200） | 数值 |
|---|---|
| 正样本 cos 均值 | **−0.3432** |
| 正样本 cos 为负的比例 | **99.0%** |
| 负样本 cos 均值 | −0.0711（\|cos\| = 0.1022） |
| **\|pos cos\| − \|neg cos\|**（projective 间隔） | **+0.2412** ✅ |
| **pos cos − neg cos**（standard 间隔） | **−0.2721** ❌ |

用 `cos` 排序时，正样本因为最负被排到**最后**，所以 R@1 是精确的 0 而非随机值。
用 `|cos|` 排序时，正样本 |cos| = 0.34 显著高于负样本 0.10，检索恢复正常。

---

## 2. 口径绑定规则

### 2.1 排序分数公式（与代码严格一致）

实现见 `scripts/eval/eval_standard.py:apply_neg_mode`，
与 `src/open_clip_train/train.py:get_clip_metrics` 及 `zero_shot.py:run` 完全对齐：

| `--neg-mode` | 正样本目标 | 排序分数 | 等价关系 |
|---|---|---|---|
| `standard` | cos → +1 | `cos` | x ~ x |
| `projective` | \|cos\| → 1 | `\|cos\|` | **x ~ ±x** |
| `antipodal` | cos → −1 | `−cos` | x ~ −x |
| `orthogonal` | cos → +1（负样本推向 0） | `cos` | x ~ x |

`--neg-alpha < 1.0` 时覆盖 `neg-mode`，分数为 `alpha * cos + (1 − alpha) * |cos|`。

### 2.2 铁律

1. **评估的 `--neg-mode` 必须等于训练的 `--neg-mode`**。查 `logs/<run>/params.txt` 的 `neg_mode` 字段。
2. **projective 与 standard 的指标之间不可直接比较绝对值**，但**同口径内的组间对比完全有效**
   （模型推向反侧是 projective 的设计行为，不是缺陷）。
3. 报告指标时**必须标注口径**，例如 `i2t R@1 = 23.50% (projective)`。

### 2.3 对外兼容性：不是阻塞项

projective 训出的模型正样本 cos 为负。这**不构成下游使用的障碍**：

- CLIP 预训练产物本身就不是 plug-and-play 的——下游任务基本都要重新训练，
  重训时符号约定随新目标一起学到。
- 少数免训练直接用的场景（如直接喂 diffusion 文本编码器），只需一次符号调整
  （取 `|cos|` 或翻转投影），属于调整量级，不是研究阻塞项。

因此 **projective 与 standard 是两条平行可比的路线**，可以并行推进、直接对照，
不需要"先统一到 standard 再做研究"。移植业界方案（Long-CLIP / Fix-CLIP / TULIP 等
standard 系）时，建议同时跑 projective 与 standard 两组作为对照。

---

## 3. 训练/评测污染（第三个坑）

**事故**：曾写过 `eval_longtext_retrieval.py`，直接读 `clip_train_dense_256.tsv`
的前 1000 行做 dense-query 检索。那个文件就是训练数据本身 —— 核实后
**评测 1000 图与训练集重叠 1000/1000（100%）**。

产出的数字（pcm_proj dense-query 100% / gt_base 58% / gt_std 51.4%，
以及 gt-query 79.3% vs 98.3%）全部作废，脚本已删除。

**破绽信号**：1000 图检索拿到 98.3%，而同模型在干净 COCO 上只有 23.5%。
**贴近天花板的检索指标几乎一定是训练集内测试**。

**铁律**：
1. 任何检索/分类评测，评测样本必须与训练集**零重叠**。CC3M 系数据（gt / dense /
   dual / mix / concat 各版本）共享同一批图，互为训练集，**不能互相当评测集**。
2. 新写评测脚本时，必须先做重叠检查：把评测 filepath 与训练 TSV 的 filepath
   取交集，重叠 > 0 直接拒跑。
3. 目前**干净的评测有三个**：COCO（karpathy 5cap，短文本）、IN-1k val（短模板分类）、
   **Urban-1k（长文本检索，1000 图 × 平均 132 token 长描述）** —— 三者与 CC3M 训练集均无交集。
4. Urban-1k 是 Long-CLIP (ECCV 2024) 配套 benchmark，入口 `scripts/eval/eval_urban1k.py`。
   数据在 `datas/urban1k/Urban1k/{image,caption}/`。它填补了「长文本能力无干净评测」的空缺。

**防护措施**：`eval_standard.py:assert_no_train_overlap()` 会在跑检索前把评测 filepath
与 `datas/cc3m-tsv/annotations/clip_train*.tsv` 取交集，命中即 `SystemExit` 拒跑。

---

## 4. 样本量陷阱（第二个坑）

同一 checkpoint、同一 projective 口径，仅改变候选池大小：

| COCO 图数 | 候选 caption 数 | i2t R@1 |
|---|---|---|
| 200 | 1,000 | 73.5% |
| 300 | 1,500 | 64.3% |
| **5,000（karpathy 全量）** | **25,000** | **23.5%** |

**小样本虚高 2.7 倍**。规则：

- 检索指标**只报全量**（karpathy 5cap = 5000 图 × 5 caption）。
- 若因资源限制必须采样，**必须在指标旁标注图数**，且只用于相对趋势，不得当作绝对值。
- IN-1k 同理：子集（如 100 类 × 20 图）与全量（1000 类 × 50 图）不可混比。

---

## 4.5 epoch 未对齐（第四个坑）

**事故**：PCM 超参 sweep 的四组在 `epoch_10` 评测，而更早的六组基线在 `epoch_9`，
两批数字被放进同一张对比表。补跑 ep10 后发现漂移不小：

| 模型 | COCO i2t ep9 → ep10 | Urban i2t ep9 → ep10 |
|---|---|---|
| pcm_proj | 23.22% → 22.08%（−1.14） | 49.00% → 47.30%（−1.70） |
| gt_base | 23.50% → 23.14%（−0.36） | 19.70% → 18.30%（−1.40） |

**1~1.7 点的漂移与多数消融声称的效应同量级**，混 epoch 比较会把噪声读成信号。
（本例里结论方向没被推翻，但差值全部要改。）

**铁律**：一张对比表里所有模型必须取同一 epoch。加新组时若旧组缺该 epoch 的评测，
补跑旧组，不要用"差不多"的邻近 epoch 凑。修正后的表见 `longclip_01_pcm.md` §3。

---

## 5. 标准评估入口

唯一评估脚本：`scripts/eval/eval_standard.py`（其余临时脚本已删除，勿再重写）。

```bash
# E 配方 / CHAMPION（projective，默认）
python scripts/eval/eval_standard.py \
    --ckpt logs/visreg_gemma_gt_gt_base_0811_1318/checkpoints/epoch_9.pt \
    --tag gt_base --retrieval --in1k-classes 100

# standard 配方模型
python scripts/eval/eval_standard.py --ckpt ... --tag foo --neg-mode standard --retrieval

# 长描述模板对照（测长文本塔对长模板的响应）
python scripts/eval/eval_standard.py --ckpt ... --tag gemma_dense --long-template

# Urban-1k 长文本检索（1000 图 × 132-token 长描述，训练集外）
python scripts/eval/eval_urban1k.py --ckpt ... --tag pcm_proj --neg-mode projective
```

---

## 6. 干净评测集清单

| 评测集 | 规模 | 测什么 | 入口 |
|---|---|---|---|
| **COCO** karpathy 5cap | 5000 图 × 5 caption（全量） | 短文本检索 | `eval_standard.py --retrieval` |
| **IN-1k** val | 100 类 × 20 图，80 官方模板 | 短模板 zero-shot 分类 | `eval_standard.py --in1k-classes` |
| **Urban-1k** | 1000 图 × 1000 长描述（平均 132 token，1:1） | 长文本检索 | `eval_urban1k.py` |

三者与 CC3M 训练集均零重叠。每个模型必须用与其训练一致的 `--neg-mode` 评估。

**注意**：IN-1k 目前用的是 100 类 × 20 图子集，与全量（1000 类 × 50 图）不可混比；
COCO 与 Urban-1k 都是全量。跨模型对比时三项的采样配置必须一致。

**epoch 对齐**：同一张对比表里所有模型必须取同一 epoch 的 checkpoint（第 4.5 节）。

### 各路线的结果表在哪

本页只管方法论，**不存结果**。具体基线数字见：

- **Long-CLIP / PCM 路线**（11 组：pcm 系 / gt_base / gt_std / mix50 / gemma_dense / concat）
  → `longclip_01_pcm.md` §3
- **VISReg 基座配方消融**（21 组）→ `visreg_all_attempts.md`
- **projective / orthogonal 几何消融** → `mgap_06_projective_siglip.md`


---

## 7. 相关文档

- `longclip_01_pcm.md` — **Long-CLIP / PCM 路线的结果主页**（11 组总表 + 超参消融）
- `mgap_06_projective_siglip.md` — projective SigLIP 的数学推导与消融（|cos| 目标的由来）
- `mgap_05_orthogonal_siglip.md` — 前序 orthogonal 模式
- `scripts/eval/eval_standard.py` — COCO + IN-1k 入口，`apply_neg_mode` 为口径实现
- `scripts/eval/eval_urban1k.py` — Urban-1k 长文本入口

