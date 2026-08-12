# 评估协议：neg-mode 与指标口径的强绑定

*创建: 2026-08-12 | 起因: gt_base 检索指标离奇归零的排查*

---

## 0. 一句话结论

**本仓库的相似度排序口径由训练配方的 `--neg-mode` 决定，评估必须用同一口径。
口径错配会让指标系统性归零（精确 0，不是随机值）——这是评估 bug，不是模型坏。**

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

### 2.3 对外兼容性提醒

projective 训出的模型正样本 cos 为负，**不能 plug-and-play 到假设 `cos → +1` 的下游框架**
（diffusion 文本编码器、标准 CLIP 检索服务、Long-CLIP / Fix-CLIP 等业界方案）。
若需对外交付或与业界数字对齐，需要额外的符号对齐步骤（约束到 +1 分支，或最后做一次 standard 微调）。

---

## 3. 样本量陷阱（同批事故的第二个坑）

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

## 4. 标准评估入口

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

## 5. 经此修正后的可信基线

全量 COCO karpathy 5cap（5000 图）+ IN-1k 100 类 × 20 图，**均为 projective 口径**：

| 模型 | 训练数据 | i2t R@1 | i2t R@5 | t2i R@1 | IN-1k top1 |
|---|---|---|---|---|---|
| **gt_base** | cc3m gt（256 ctx） | **23.50%** | 47.34% | **16.84%** | **23.20%** |
| **mix50** | 50% gt + 50% dense（整行替换） | 21.18% | 43.92% | 14.35% | 20.90% |
| **gemma_dense** | 100% gemma dense | 7.74% | 15.96% | 2.09% | 1.85% |
| **concat** | 每行 `gt + dense` 拼接 | 训练即崩 | — | — | 0.50%（epoch 6） |

训练日志内的 baseline（77 ctx gt）：IN-1k 23.48% / COCO i2t 22.84%。

### 被此次修正推翻的旧结论

| 旧结论（错误口径 / 小样本） | 修正后 |
|---|---|
| gemma_dense「dense-query R@1 = 0，完全未对齐」 | 弱但非零：i2t 7.74% |
| gemma_dense IN-1k = 0.93% | 1.85%（projective, 100 类子集） |
| gt_base i2t = 64.3% | 23.50%（全量） |
| mix50 i2t = 60.7% | 21.18%（全量） |

---

## 6. 相关文档

- `mgap_06_projective_siglip.md` — projective SigLIP 的数学推导与消融（|cos| 目标的由来）
- `mgap_05_orthogonal_siglip.md` — 前序 orthogonal 模式
- `scripts/eval/eval_standard.py` — 唯一评估入口，`apply_neg_mode` 为口径实现
