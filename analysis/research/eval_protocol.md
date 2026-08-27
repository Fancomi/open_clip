# 评估协议：口径绑定、零重叠、全量报数

*创建: 2026-08-12，更新: 2026-08-27（新增 OVSS 局部口径）| 起因: 五次评估事故（口径错配 / 小样本虚高 / 训练集内测试 / epoch 未对齐 / 子集口径翻转排序）*

---

## 0. 四条铁律

1. **口径绑定**：相似度排序口径由训练 `--neg-mode` 决定，评估必须用同一口径。
   错配会让指标系统性归零（精确 0，不是随机值）—— 是评估 bug，不是模型坏。（第 1-2 节）
2. **零重叠**：评测样本与训练集必须零重叠。CC3M 系数据互为训练集，不能互当评测集。
   干净评测只有 COCO karpathy 5cap / IN-1k val / Urban-1k。（第 3 节）
3. **只报全量**：检索指标只报全量（COCO 5000 图 / Urban-1k 1000 图）。小样本虚高 2.7 倍。（第 4 节）
4. **同表同 epoch**：一张对比表里所有模型取同一 epoch 的 checkpoint。
   ep9→ep10 漂移实测 1~1.7 点，与多数消融效应同量级。（第 6 节）
5. **过 2σ 才算结论**：每个口径有实测噪声地板，|Δ| < 2σ 一律判"不可分辨"，
   不得写成"略优/略差"。噪声地板由 **query 数量**决定，不由指标类型决定。（第 4.6 节）
6. **全局口径的结论不能外推到局部表征**（新增 2026-08-27）。
   前四个口径（k-NN / IN-1k / COCO / Urban）**全是"一张图 → 一个向量"的全局口径**，
   它们一致同向也不能说明局部（逐 patch）表征变好了。
   实证：`region_weight` 0.2→2.0 在五个全局口径上单调上升、每跳过 2σ，
   而在 OVSS mIoU 上**单调下降 −14.19**（`region_01_supervision.md` §5.8）。
   → 任何改动**局部**表征的方法（区域监督、dense 蒸馏、patch 级正则），
   必须报一个局部口径，否则结论只在全局那一半成立。（第 6 节表格新增一行）

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

**2026-08-25 整改完成**：IN-1k 已全部重测为全量口径（1000 类 × 50 图 = 50000 图），
`eval_standard.py` 的默认值即全量，输出行自带 `★全量★` / `⚠️子集` 标记。
**2026-08-25 之前所有 IN-1k 数字（2000 图子集）一律作废**，全量总表见
`in1k_fullscope_retest.md`。子集不只是虚高，方向也会翻：E_firstbox 子集 25.50% →
全量 27.15%，因为子集只取前 100 个 wnid，类别难度分布有偏。

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

## 4.6 噪声地板：每个口径的 2σ 门槛

同配方、只改随机种子的重复运行给出的组间散布，就是"配置无差异时也会看到的差"。
`gt_base` 系四次同配方运行（`0806` / `0811` / `s1` / `s2`）标定出下表。
**任何小于 2σ 的差值不构成结论**，写文档时必须标 ✅（显著）/ ❌（噪声内）。

| 口径 | query 数 | 2σ | 标定来源 |
|---|---|---|---|
| COCO t2i R@1 | 25000 caption | **0.13** | 4× gt_base |
| k-NN proj | 50000 图 | **0.29** | 4× gt_base（全量重测 2026-08-25） |
| k-NN backbone | 50000 图 | **0.30** | 同上 |
| IN-1k zero-shot | 50000 图 | **0.32** | 4× gt_base（全量） |
| COCO i2t R@1 | 5000 图 | **0.75** | 4× gt_base |
| Urban-1k i2t R@1 | 1000 图 | **1.92** | 4× gt_base |
| Urban-1k t2i R@1 | 1000 描述 | **1.92**（推定） | ⚠️ 未单独定标，借 i2t —— 见下方推定依据 |
| **VOC-20 OVSS mIoU**（penult） | 1449 图 / ~1.2 亿像素 | **~0.12** | ⚠️ 只有 2× gt_base（n=2） |
| **VOC-20 OVSS mIoU**（last） | 同上 | **~0.16** | ⚠️ 同上 |

**规律：2σ 随 query 数量单调下降，与"测的是分类还是检索"无关。**
同一个 COCO 评测里 t2i（25000 条 caption 当 query）的 2σ 只有 0.13，
而 i2t（5000 张图当 query）是 0.75，差 5.8 倍 —— 纯粹是样本量差 5 倍的结果。
推论：**Urban-1k 只有 1000 图，它的单点结论最不可信；COCO t2i 最灵敏。**

⚠️ **Urban t2i 的 1.92 是借来的，但这个借法有依据。** Urban-1k 是 1000 图 × 1000 长描述的
**1:1 配对**，所以 i2t（1000 图当 query）与 t2i（1000 描述当 query）的 query 数**完全相同**。
按上面那条规律（2σ 只随 query 数走，与测的是 i2t 还是 t2i 无关），借 i2t 的值是**推定**，
不是"偏保守的猜"。**反面例子就在同一张表里**：COCO 的 i2t/t2i query 数差 5 倍
（5000 图 vs 25000 描述），地板就差 5.8 倍（0.75 vs 0.13）—— 那两个之间**绝不能互借**。
判据：**query 数相同 → 可以互借；query 数不同 → 必须各自标定。**
（仍然不等于标过。真要定标只需把 4 个 `gt_base` ckpt 各跑一次 `eval_urban1k.py`，
i2t/t2i 是同一次前向出来的，成本为零 —— 只是历史上没记 t2i 那一列。）

⚠️ **表里每一个地板都只在 `gt_base` 配方下标过，从未在 region / PCM 配方下重标。**
后果分两档：W=2.0 那种 5~27 倍 2σ 的效应不受影响；
但**区域组内部 0.3~0.5 的差**（例如 8 变体消融、相邻 W 的长文本项）建立在一个
可能不适用的地板上 → 引用时要写"按 `gt_base` 地板判"。

⚠️ **OVSS 两行是 n=2 的散布，不是 4 次运行的 2σ**（`gt_base` seed 0 vs seed 1，
配置与数据完全相同，只差 `--seed`）。它给的是量级而不是分布，引用时按"粗地板"用。
它之所以这么小，符合上面那条规律：query 是**像素**而不是图，样本量比其他口径高 4~5 个数量级。
所以 OVSS 上几点的差都远超地板 —— 反过来说，**OVSS 上很小的差也可能是真的**，
不要用其他口径的直觉（"差 1 点不算"）去判它。

⚠️ **2026-08-25 修订**：k-NN 的旧 2σ 是 **0.78**，那是 100 类 × 20 图 = 2000 张
子集下的标定值。全量口径把它压到 0.30（紧 2.6 倍）。**旧口径不只噪声大，还系统性
压缩组间差异**（子集 Δ / 全量 Δ 的比值实测 0.13~0.50），所以此前一切"判为持平"的
k-NN 结论都必须按新 2σ 重判。详见 `longclip_01_pcm.md` §5.8。

---

## 5. 标准评估入口

唯一评估脚本：`scripts/eval/eval_standard.py`（其余临时脚本已删除，勿再重写）。
另有三个专用入口：`eval_urban1k.py`（长文本检索）、`eval_knn_probe.py`（纯图像 k-NN）、
`eval_ovss.py`（开放词表分割 mIoU，**唯一的局部口径**）。

```bash
# E 配方 / CHAMPION（projective，默认；IN-1k 默认即全量 1000 类 × 50 图）
python scripts/eval/eval_standard.py \
    --ckpt logs/visreg_gemma_gt_gt_base_0811_1318/checkpoints/epoch_10.pt \
    --tag gt_base --retrieval --num-workers 14

# standard 配方模型
python scripts/eval/eval_standard.py --ckpt ... --tag foo --neg-mode standard --retrieval

# 长描述模板对照（测长文本塔对长模板的响应）
python scripts/eval/eval_standard.py --ckpt ... --tag gemma_dense --long-template

# Urban-1k 长文本检索（1000 图 × 132-token 长描述，训练集外）
python scripts/eval/eval_urban1k.py --ckpt ... --tag pcm_proj --neg-mode projective

# IN-1k k-NN probe（纯图像口径，无文本参与；全量 1000×50 已是默认，勿再传子集参数）
python scripts/eval/eval_knn_probe.py --ckpt ... --tag pcm_w0.2 --num-workers 12

# OVSS：VOC-2012 val 开放词表分割 mIoU（唯一的**局部**表征口径，全量 1449 图已是默认）
# --dense-mode penult = 与训练时区域分支同一条读出路径；last = 对照。两个都要报。
python scripts/eval/eval_ovss.py --ckpt ... --tag regw2.0
python scripts/eval/eval_ovss.py --ckpt ... --tag regw2.0_LAST --dense-mode last
```

---

## 6. 干净评测集清单

按**文本参与度**从低到高排成一把梯子 —— 这是区分"图像塔变好"与"文本塔更会对齐"的关键：

| 评测集 | 规模 | 测什么 | 文本长度 | 入口 |
|---|---|---|---|---|
| **IN-1k k-NN probe** | 1000 类 × 50 图 = 50000 图（全量） | **纯图像**特征质量（冻结骨干 + 近邻投票） | **无文本** | `eval_knn_probe.py` |
| **IN-1k** val zero-shot | 同上，80 官方模板 | 短模板 zero-shot 分类 | ~8 tok | `eval_standard.py`（全量为默认） |
| **COCO** karpathy 5cap | 5000 图 × 5 caption（全量） | 短文本检索 | ~13 tok | `eval_standard.py --retrieval` |
| **Urban-1k** | 1000 图 × 1000 长描述（1:1） | 长文本检索 | ~132 tok | `eval_urban1k.py` |
| **VOC-2012 val OVSS** | 1449 图，逐**像素**判定 | **局部**表征：patch 级语义可分割性 | 20 个类名 × 80 模板 | `eval_ovss.py` |

⚠️ **OVSS 不在这把梯子上，它在另一根轴上。** 上面四个都是"一张图 → 一个向量 → 比一次"的
**全局**口径，梯子排的是文本参与度；OVSS 是"每个 patch 自己去和类名比"的**局部**口径。
两根轴可以互相矛盾，而且实测就是矛盾的（`region_weight` 在两边方向相反，铁律第 6 条）。
所以**它不是第五级台阶，是第二把梯子的第一级**。
其协议与 SCLIP / ClearCLIP / NACLIP 系列严格一致（短边 336、滑窗 224/stride 112、
20 类 × 80 官方模板、**无任何推理期改造**、无后处理），因此可以与那一族论文的
VOC20 一列对位看 —— 但骨干与类名同义词列表都未对齐，只能定位不能当同条件比较。

五者与 CC3M 训练集均零重叠。每个模型必须用与其训练一致的 `--neg-mode` 评估
（**k-NN probe 例外**：链路里没有文本，不涉及口径绑定，可跨 neg-mode 直接比）。
**k-NN 同时报 backbone（trunk CLS，投影前 768d）与 proj（投影后 1024d）两个数**，
两者之差 Δ(proj − bb) 本身是个诊断量（见 `longclip_01_pcm.md` §5.5）。
⚠️ k-NN probe 是项目自定义口径（余弦 + softmax 温度加权、投影前特征），
**不能与 DINO / DINOv3 论文的 k-NN 数字对标**，只做同源自比。
**口径不要靠 run 名字猜，直接读 `logs/<run>/params.txt` 的 `neg_mode` 字段** ——
2026-08-25 重测时正因为按名字猜，漏判了 `gt_std`（真为 standard），把它评成 8.54%
（真值 23.34%）。

**三项现已全部为全量口径**，跨模型对比无需再对齐采样配置。若为冒烟临时缩小
`--in1k-classes/--in1k-per-class`，输出会打 `⚠️子集(不可与全量混比)`，此类数字禁止入表。

**epoch 对齐**：同一张对比表里所有模型必须取同一 epoch 的 checkpoint（第 4.5 节）。

### 各路线的结果表在哪

本页只管方法论，**不存结果**。具体基线数字见：

- **Long-CLIP / PCM 路线**（11 组：pcm 系 / gt_base / gt_std / mix50 / gemma_dense / concat）
  → `longclip_01_pcm.md` §3，k-NN 全量总表见 §5.2
- **区域-短语监督路线**（12 组：C1~C4 / H / E1~E3 / A′ resize 对照 / G 叠加 / regw0.5 / regw1.0 / regw2.0）
  → `region_01_supervision.md` §4，`region_weight` 扫描见 §5.7
- **IN-1k 全量口径总表**（24 组：基线 / PCM 全系 / region 全系）→ `in1k_fullscope_retest.md`
- **VISReg 基座配方消融**（21 组）→ `visreg_all_attempts.md`
- **projective / orthogonal 几何消融** → `mgap_06_projective_siglip.md`


---

## 7. 相关文档

- `longclip_01_pcm.md` — **Long-CLIP / PCM 路线的结果主页**（11 组总表 + 超参消融 + 2σ 标定）
- `region_01_supervision.md` — **区域-短语监督路线的结果主页**（12 组总表 + `region_weight` 扫描 + resize 对照
  + `region_weight` 扫描）。⚠️ 它 §5.3 记了一条方法论教训：
  **"这条线的旋钮已调平"这个判断，在其中一个旋钮从未被扫过的情况下写过一次，第二天就被推翻。
  下"调平"结论前先确认该旋钮真的被扫过。**
- `in1k_fullscope_retest.md` — IN-1k 全量口径重测（24 组），子集口径作废的直接证据
- `mgap_06_projective_siglip.md` — projective SigLIP 的数学推导与消融（|cos| 目标的由来）
- `mgap_05_orthogonal_siglip.md` — 前序 orthogonal 模式
- `scripts/eval/eval_standard.py` — COCO + IN-1k 入口，`apply_neg_mode` 为口径实现
- `scripts/eval/eval_urban1k.py` — Urban-1k 长文本入口
- `scripts/eval/eval_knn_probe.py` — IN-1k k-NN probe 入口（纯图像口径，文件头有口径声明）

