# 评估协议：口径绑定、零重叠、全量报数

*创建: 2026-08-12，更新: 2026-08-13 | 起因: 三次评估事故（口径错配 / 小样本虚高 / 训练集内测试）*

---

## 0. 三条铁律

1. **口径绑定**：相似度排序口径由训练 `--neg-mode` 决定，评估必须用同一口径。
   错配会让指标系统性归零（精确 0，不是随机值）—— 是评估 bug，不是模型坏。（第 1-2 节）
2. **零重叠**：评测样本与训练集必须零重叠。CC3M 系数据互为训练集，不能互当评测集。
   干净评测目前只有 COCO karpathy 5cap 与 IN-1k val。（第 3 节）
3. **只报全量**：检索指标只报全量 5000 图。小样本虚高 2.7 倍。（第 4 节）

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
3. 目前**干净的评测只有 COCO（karpathy 5cap）与 IN-1k val** —— CC3M 训练从未见过。
4. 长文本能力目前**无干净评测集**。CC3M 之外没有第二份 dense 标注。要评测需先
   建立：CC3M hold-out（需重训）／给 COCO val 生成 dense caption／外部 benchmark
   （Urban-1k / ShareGPT4V / DCI）。

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
```

---

## 6. 可信基线（干净评测：COCO + IN-1k）

全量 COCO karpathy 5cap（5000 图）+ IN-1k 100 类 × 20 图。
**COCO/IN-1k 与 CC3M 训练集零重叠，是当前唯一干净的评测。**

| 模型 | 训练数据 | 口径 | i2t R@1 | i2t R@5 | t2i R@1 | IN-1k top1 |
|---|---|---|---|---|---|---|
| **gt_base** | cc3m gt（256 ctx） | proj | **23.50%** | 47.34% | **16.84%** | **23.20%** |
| **pcm_proj** | dual：dense 主 + gt 短分支 | proj | 23.22% | 47.34% | 15.12% | 22.20% |
| **mix50** | 50% gt + 50% dense（整行替换） | proj | 21.18% | 43.92% | 14.35% | 20.90% |
| **gemma_dense** | 100% gemma dense | proj | 7.74% | 15.96% | 2.09% | 1.85% |
| **concat** | 每行 `gt + dense` 拼接 | proj | 训练即崩 | — | — | 0.50%（ep6） |
| **pcm_std** | dual：dense 主 + gt 短分支 | **std** | 17.18% | 38.36% | 12.09% | 20.75% |
| **gt_std** | cc3m gt（256 ctx） | **std** | 待全量重测¹ | — | — | 待重测¹ |

¹ gt_std 于 2026-08-13 18:21 训练完成（epoch_10），尚未用 `eval_standard.py` 全量重测。
训练日志内值：i2t 21.76% / IN-1k 23.34%（epoch 8）。

训练日志内的旧 baseline（77 ctx gt）：IN-1k 23.48% / COCO i2t 22.84%。

### 从可信数据能得出的结论

1. **PCM 防崩有效**：dense 主导训练下，concat 崩到 0.50%、纯 dense 仅 7.74%，
   PCM 拉到 23.22%。
2. **但 PCM 在短文本任务上未超越 gt_base**：i2t −0.28 / t2i −1.72 / IN-1k −1.00。
   **dense 分支的边际贡献 ≈ 0 或轻微为负。**
3. **projective 优于 standard**：gt 数据上 23.50% vs 21.76%（+1.7）；
   PCM 上 23.22% vs 17.18%（+6.0）。
4. **dense 长文本是否有可泛化价值 —— 尚无结论**（缺干净的长文本评测集，见第 3 节）。

### 被历次修正推翻的旧结论

| 旧结论 | 失效原因 | 修正后 |
|---|---|---|
| gemma_dense「dense-query R@1 = 0，完全未对齐」 | 口径错配 | 弱但非零：i2t 7.74% |
| gemma_dense IN-1k = 0.93% | 口径错配 | 1.85%（proj, 100 类子集） |
| gt_base i2t = 64.3% | 小样本虚高 | 23.50%（全量） |
| mix50 i2t = 60.7% | 小样本虚高 | 21.18%（全量） |
| 「PCM 让 dense 长文本能力 +42 点」 | 训练集内测试 | **无结论**，需干净评测集 |
| 「PCM 让短文本检索掉 19 点」 | 训练集内测试 | 真实代价约 1 点（见上表） |

---

## 7. 相关文档

- `mgap_06_projective_siglip.md` — projective SigLIP 的数学推导与消融（|cos| 目标的由来）
- `mgap_05_orthogonal_siglip.md` — 前序 orthogonal 模式
- `scripts/eval/eval_standard.py` — 唯一评估入口，`apply_neg_mode` 为口径实现

