# Long-CLIP PCM：CC3M 上的复现与超参消融

*创建: 2026-08-17 | 基座: PE-Core-B-16-dinov3 + VISReg-E + 256 ctx | 11 组 10-epoch 训练*

本页是 Long-CLIP 路线（长文本 CLIP）的实现与结果主页。
评测方法论（口径绑定 / 零重叠 / 全量报数）见 [[eval_protocol]]，**不在本页重复**。
projective 几何的数学根基见 `mgap_06_projective_siglip.md`。

---

## 1. 问题

gemma4 重写的 dense 长 caption（平均 132 BPE token）拿来直接训 CLIP，
长文本检索很强但**短文本能力报废**：`gemma_dense` COCO i2t 7.70% / IN-1k 1.70%。
反过来只用 gt 短文（`gt_base`）短文本正常但长文本只有 18.30%。

Long-CLIP (ECCV 2024) 的 **PCM（Primary Component Matching）** 声称能兼顾两者：
主分支用长文本对齐完整图像特征，短分支用**PCA 降维后**的图像特征对齐短 caption。
降维迫使短文本只匹配图像的"主要语义"，从而在长文本主导训练时保住短模板 zero-shot。

本页回答三个问题：
1. PCM 在我们的基座（projective SigLIP + VISReg，非原论文的 standard CLIP）上是否有效？
2. `pcm_weight` / `pcm_dim` 怎么取？
3. 长短能力的 trade-off 是真实存在的，还是超参没调好？

---

## 2. 实现

| 组件 | 位置 |
|---|---|
| PCA 降维（含三级兜底） | `src/open_clip/model.py:CLIPLeJEPA._pca_reduce` |
| 双分支前向（`text` 长 / `text2` 短） | `src/open_clip/model.py:CLIPLeJEPA.forward` |
| `pcm_loss` 累加（复用主损失，含 neg_mode） | `src/open_clip/loss.py:1501` |
| CLI: `--pcm-weight` / `--pcm-dim` / `--csv-caption2-key` | `src/open_clip_train/params.py:593` |
| 启动入口 | `scripts/train/visreg.sh pcm`（`PCM_WEIGHT` / `PCM_DIM` 可覆盖） |
| 数据 | `clip_train_dual.tsv`（`filepath` / `caption_short` / `caption_dense`） |

短分支损失：`pcm_weight × SigLIP(PCA_k(img_feat), txt_short_feat)`，
`PCA_k` = 保留前 k 个主成分后重建回原空间（不改变维度，只降秩）。

**与原论文的三处差异**（都是被基座逼出来的，不是随意改动）：

1. **口径**：原论文是 standard CLIP（`cos → +1`）；我们主线是 projective（`|cos| → 1`）。
   两个口径各跑一组作对照（`pcm_proj` / `pcm_std`），见第 4 节。
2. **PCA 数值稳健性**：`torch.linalg.svd` 在训练后期 embedding 条件数变差时会抛
   `_LinAlgError: failed to converge (512)` —— `pcm_dim=64` 组实测 epoch 5 崩。
   现在是三级兜底：jitter 重试 → `pca_lowrank` → 退化为全维对齐（不中断训练）。
   见 commit `c31b5bc`。
3. **文本塔 batch-max 截断**：短分支 caption 实测最多 34 token，固定跑 256 窗口
   纯属浪费。按 batch 内 EOT 最大位置截断，数值等价（差异 1e-7），
   PCM 训练整体加速 ~1.7×。见 `model.py:_trim_to_batch_max` / commit `a4338fc`。

---

## 3. 总表（全部 epoch_10，干净评测）

固定基座：`PE-Core-B-16-dinov3` + SigLIP + VISReg-E + init-scale ln15 + Muon(0.01)
+ lr 3.4e-4 + bs 8×512 + 10 epoch + `--force-context-length 256`。
只有下表「变量」列所标注的量在变。

评测集与口径（详见 [[eval_protocol]]）：
COCO karpathy 5cap 全量 5000 图 / 25000 caption；IN-1k 100 类 × 20 图 / 80 官方模板；
Urban-1k 1000 图 × 1000 长描述（1:1）。三者与 CC3M 零重叠，已过 `assert_no_train_overlap`。
每个模型用**与其训练一致**的 neg-mode 评估。

| 模型 | 变量 | 训练文本 | 口径 | COCO i2t R@1 | i2t R@5 | COCO t2i R@1 | IN-1k top1 | Urban i2t R@1 | Urban t2i R@1 |
|---|---|---|---|---|---|---|---|---|---|
| **pcm_w0.3** | w=0.3 d=32 | dense 主 + gt 短 | proj | **26.22%** | **50.50%** | 16.27% | 22.15% | **57.70%** | **57.80%** |
| pcm_w0.5 | w=0.5 d=32 | dense 主 + gt 短 | proj | 25.70% | 49.06% | 15.90% | 22.35% | 56.10% | 56.50% |
| pcm_d64 | w=1.0 d=64 | dense 主 + gt 短 | proj | 24.54% | 48.68% | 15.91% | **22.55%** | 47.90% | 49.80% |
| pcm_d16 | w=1.0 d=16 | dense 主 + gt 短 | proj | 22.64% | 44.40% | 13.09% | 20.00% | 50.20% | 46.90% |
| pcm_proj | w=1.0 d=32 | dense 主 + gt 短 | proj | 22.08% | 45.80% | 14.73% | 22.25% | 47.30% | 49.60% |
| pcm_std | w=1.0 d=32 | dense 主 + gt 短 | **std** | 16.78% | 37.08% | 11.73% | 20.75% | 45.60% | 46.20% |
| gemma_dense | 无短分支 | 100% dense | proj | 7.70% | 16.24% | 2.05% | 1.70% | 51.60% | 50.10% |
| mix50 | 无短分支 | 50% gt + 50% dense | proj | 21.06% | 43.72% | 14.04% | 20.65% | 46.10% | 46.40% |
| **gt_base**（对照） | 无长文本 | 100% gt 短 | proj | 23.14% | 46.48% | **16.35%** | 22.80% | 18.30% | 18.60% |
| gt_std（对照） | 无长文本 | 100% gt 短 | **std** | 21.82% | 45.68% | 15.81% | **23.45%** | 16.70% | 17.90% |
| concat | 无短分支 | 每行 `gt+dense` 拼接 | proj | 训练即崩 | — | — | 0.50%（ep6） | — | — |

### 训练编号

| 模型 | `logs/<run>` |
|---|---|
| pcm_w0.3 | `visreg_gemma_pcmw0.3d32_projective_E_0814_1723` |
| pcm_w0.5 | `visreg_gemma_pcmw0.5d32_projective_E_0814_2039` |
| pcm_d16 | `visreg_gemma_pcmw1.0d16_projective_E_0815_0153` |
| pcm_d64 | `visreg_gemma_pcmw1.0d64_projective_E_0815_1606` |
| pcm_proj | `visreg_gemma_pcm32_projective_E_0812_1934` |
| pcm_std | `visreg_gemma_pcm32_standard_E_0813_0332` |
| gemma_dense | `visreg_gemma_dense_256_E_0806_2011` |
| mix50 | `visreg_gemma_mix50_E_0811_1318` |
| gt_base | `visreg_gemma_gt_gt_base_0811_1318` |
| gt_std | `visreg_gemma_gt_gt_std_0813_1133` |
| concat | `visreg_gemma_concat_E_0812_1216`（ep7 被杀，无正式评测） |

`concat` 唯一数字来自训练日志 `out.log`：`imagenet-zeroshot-val-top1` 0.24%→0.55%→0.50%（ep6），
`image_to_text_R@1` 峰值 1.12%（ep4）。不做正式评测，因为已经确定是坏配方。

**epoch 对齐说明**：本表全部取 `epoch_10.pt`。此前版本把 sweep 四组（ep10）与
原始六组（ep9）混在一张表里比 —— ep9→ep10 漂移实测 1~1.7 点（如 pcm_proj
COCO 23.22→22.08、Urban 49.00→47.30），与部分结论声称的效应同量级，**属于真实的方法学缺陷**，
已通过补跑 ep10 修正。今后加新组必须同 epoch 对齐后再进表。

---

## 4. 结论

### 4.1 PCM 有效，但原论文的 `pcm_weight=1.0` 是错的取值

以 `gt_base`（同 epoch、同口径、无长文本）为对照：

| | COCO i2t | COCO t2i | IN-1k | Urban i2t |
|---|---|---|---|---|
| gt_base | 23.14% | 16.35% | 22.80% | 18.30% |
| pcm_w0.3 | **+3.08** | −0.08 | −0.65 | **+39.40** |
| pcm_proj (w1.0) | −1.06 | −1.62 | −0.55 | +29.00 |

`pcm_w0.3` **在短文本和长文本上同时超过 gt_base**：COCO i2t +3.08、Urban i2t +39.40，
代价只有 IN-1k −0.65 和 COCO t2i −0.08（后者在噪声量级）。
也就是说加入长文本不但不用付短文本的代价，短文本还**更好了**。

⚠️ 但这个提升**发生在文本侧，不是图像侧** —— IN-1k k-NN（纯图像口径）显示
`pcm_w0.3` 的视觉特征质量与 `gt_base` 持平（45.25% vs 45.80%），并未变好。
详见 §5.3。

### 4.2 `pcm_weight` 单调：越小越好（两个轴同时）

| w（d=32 固定） | COCO i2t | IN-1k | Urban i2t |
|---|---|---|---|
| **0.3** | **26.22%** | 22.15% | **57.70%** |
| 0.5 | 25.70% | 22.35% | 56.10% |
| 1.0 | 22.08% | 22.25% | 47.30% |

**在测过的范围内两个轴都单调**，这不是 trade-off 曲线上的移动，是整体变好。
机制推测：短分支的 PCA 秩约束（32/1024）本身是对完整 embedding 的强扭曲，
w=1.0 时它与主分支正面冲突；w 小的时候它退化成一个**正则项** ——
足够引导短模板能力，又不足以破坏表示。

**这条曲线还没探到底**：0.3 是本轮最小值，w=0.1 / 0.2 未测。下一步应该往下扫。

### 4.3 `pcm_dim` 影响远小于 weight，且非单调

| d（w=1.0 固定） | COCO i2t | IN-1k | Urban i2t |
|---|---|---|---|
| 16 | 22.64% | 20.00% | 50.20% |
| 32（原论文） | 22.08% | 22.25% | 47.30% |
| 64 | 24.54% | 22.55% | 47.90% |

跨度 COCO 2.5 点 / Urban 2.9 点，**非单调**，且 d=32（原论文值）在 COCO 上反而最差。
我们**没有跑种子重复**，无法给出 run-to-run 噪声的界，所以这个量级的差异不足以排序 ——
只能说 dim 不是主要旋钮，w 才是。若要给 dim 定论，必须先做 2-3 个种子的重复。

### 4.4 PCM 的短分支还**提升**了长文本能力

`pcm_w0.3` Urban i2t **57.70%** > `gemma_dense`（100% dense，无短分支）**51.60%**。
纯长文本训练并不是长文本能力的上限 —— 短分支同时起到正则作用。
这一点原论文未提及（原论文的短分支定位是"保住 zero-shot"）。

### 4.5 PCM 优于朴素混合与拼接

同为"让模型见到 dense"的三种做法：

| 做法 | COCO i2t | Urban i2t |
|---|---|---|
| PCM（w0.3） | **26.22%** | **57.70%** |
| mix50（50% 行整体替换） | 21.06% | 46.10% |
| concat（每行 gt+dense 拼接） | 训练即崩 | — |

PCM 在两个轴上都赢 mix50 ≥5 点。concat 直接崩（IN-1k 0.50%），
推测是拼接后文本分布与两个来源都不一致，等于给了个第三种畸形分布。

### 4.6 projective 在 PCM 上的优势比在 gt 上更大

| 数据 | proj − std（COCO i2t） | proj − std（IN-1k） | proj − std（Urban i2t） |
|---|---|---|---|
| PCM (w1.0 d32) | **+5.30** | +1.50 | +1.70 |
| gt only | +1.32 | −0.65 | +1.60 |

口径本身的收益（gt 上 +1.3）到了 PCM 上放大到 +5.3。
未验证的推测：`|cos|` 的射影等价让长短两个分支更容易共存于同一空间。
**注意**：跨口径不可比绝对值，这里比的是「同数据下两口径的差」，是合法的二阶对比。

### 4.7 dense 长文本确有可泛化价值

Urban-1k 与 CC3M / COCO 均零重叠，排除记忆效应：见过 dense 的模型
（pcm 系 / mix50 / gemma_dense）Urban i2t 全部 ≥45%，只见过 gt 短文的
（gt_base 18.30% / gt_std 16.70%）不到 19%。**分界线干净**。

---

## 5. 纯图像口径：IN-1k k-NN probe

### 5.1 为什么需要它

第 3 节的三个评测**全部把文本当分类器或 query**：COCO 是短句检索、
IN-1k zero-shot 把类名套模板变成文本分类器、Urban-1k 是长文本 query。
所以它们测的都是**图文对齐质量**，无法区分：

- 图像塔学到了更好的视觉特征？
- 还是只是文本塔更会对齐？

k-NN probe 把文本彻底移出链路：冻结骨干 → 提特征 → 近邻投票。
零训练、无可调项（除 k）、DINO 系的标准图像侧指标。

**方法**：IN-1k val 100 类 × 20 图（2000 样本，随机基线 1%）。
留一法 k-NN（k=20，余弦相似度 + `softmax(sim/0.07)` 加权投票，对角排除自身）。
两个特征层都测：

| 层 | 取法 | 维度 | 含义 |
|---|---|---|---|
| `backbone` | `visual.trunk.forward_features(x)[:, 0]` | 768 | 投影头**之前**的原始视觉特征 |
| `proj` | `encode_image(x, normalize=True)` | 1024 | 对齐文本空间**之后**的特征 |

入口 `scripts/eval/eval_knn_probe.py`。全部取 `epoch_10.pt`，与第 3 节同 epoch。

### 5.2 结果

| 模型 | **k-NN backbone** | k-NN proj | Δ(proj−bb) | COCO i2t | IN-1k zs | Urban i2t |
|---|---|---|---|---|---|---|
| **gt_base** | **45.80%** | 43.55% | −2.25 | 23.14% | 22.80% | 18.30% |
| pcm_w0.3 | 45.25% | **44.95%** | −0.30 | **26.22%** | 22.15% | **57.70%** |
| mix50 | 44.35% | 43.00% | −1.35 | 21.06% | 20.65% | 46.10% |
| pcm_w0.5 | 43.95% | 43.40% | −0.55 | 25.70% | 22.35% | 56.10% |
| pcm_d64 | 43.30% | 42.60% | −0.70 | 24.54% | 22.55% | 47.90% |
| pcm_proj (w1.0) | 42.30% | 42.10% | −0.20 | 22.08% | 22.25% | 47.30% |
| pcm_d16 | 41.70% | 40.70% | −1.00 | 22.64% | 20.00% | 50.20% |
| gemma_dense | 39.85% | 40.10% | +0.25 | 7.70% | 1.70% | 51.60% |

### 5.3 dense 长文本没有让图像塔变好 —— 增益在文本侧

`gt_base` 的 k-NN backbone **45.80% 是全场最高**，`pcm_w0.3` 45.25% 略低（−0.55）。

也就是说 `pcm_w0.3` 的 COCO i2t +3.08 / Urban +39.40，**并非因为它的视觉特征更好** ——
图像特征质量与 `gt_base` 持平甚至略低，增益来自图文对齐。

这否证了一个此前基于 COCO 提升做出的推断（「连短文本检索都变强了，说明图像表征本身变好了」）。
纯图像口径直接把它推翻 —— 见第 7 节。

**注意**：0.55 点在已知的 run-to-run 波动量级内（同配置同 seed 的两个 `gt_base`
末期 i2t 相差 0.70），所以准确说法是「**持平，未见提升**」，不是「变差」。

### 5.4 gemma_dense 的谜团：图像塔和文本塔一起弱

这是本节最有价值的对照。`gemma_dense`（100% dense，无短分支）zero-shot 只有 **1.70%**，
接近随机 —— 但此前**无法判断是图像塔坏了还是文本塔坏了**，因为 zero-shot 两者耦合。

k-NN 给出答案：**backbone 39.85%，全场最低，比 `gt_base` 低 5.95 点。**

39.85% 远高于随机（1%），说明图像塔确实学到了实质视觉语义 —— 它没崩。
但它是所有变体里最弱的。所以「纯 dense 训练失败」是**双重失败**：

- 文本塔崩到不可用（zero-shot 1.70%）
- 图像塔同时退化约 6 点

**长文本监督对视觉特征学习本身是有害的**，不只是让文本塔偏离短模板分布。

### 5.5 投影头在丢视觉信息，而 PCM 减少了这个损失

看 Δ 列（proj − backbone，负值 = 投影后判别力下降）：

| 模型 | Δ |
|---|---|
| gt_base | **−2.25** |
| mix50 | −1.35 |
| pcm_d16 | −1.00 |
| pcm_d64 | −0.70 |
| pcm_w0.5 | −0.55 |
| **pcm_w0.3** | **−0.30** |
| pcm_proj (w1.0) | −0.20 |
| gemma_dense | +0.25 |

规律清晰：**带 PCM 短分支的模型，投影头更"保真"**（Δ 普遍在 −0.7 以内），
而无短分支的 `gt_base` 丢掉 2.25 点。

机制上讲得通：PCA 短分支要求**投影后特征的前 k 个主成分仍能对齐短 caption**，
这等于在阻止投影头把视觉判别信息压缩掉。

这个效应此前完全不可见 —— 所有旧评测只看投影后的特征，看不到"投影前后差了多少"。
**这是 PCM 的一个原论文未提及的副作用，且方向是正面的。**

### 5.6 指标间相关性：优化都发生在文本侧

以 k-NN backbone 为自变量，对 8 个模型算 Pearson 相关：

| 与 k-NN backbone 的相关 | r |
|---|---|
| COCO i2t | **+0.748** |
| IN-1k zero-shot | **+0.753** |
| Urban i2t | **−0.398** |

前两个正相关：图像特征质量是短文本任务的基础。

**Urban 负相关是关键**：在我们这批模型里，长文本能力与图像特征质量**反向** ——
越擅长长文本的模型图像特征越弱（`gemma_dense` 是极端点）。
长文本能力主要由文本塔承担，并且**部分是用图像塔的质量换来的**。

另一个观察 —— 各口径的全距：

| 口径 | 最好 − 最差 |
|---|---|
| k-NN backbone | **5.95 点** |
| COCO i2t | 18.52 点 |
| IN-1k zero-shot | 21.10 点 |

图像塔在 8 个变体间只差 6 点，文本侧差 20 点上下。
**本轮所有优化（PCM / mix / concat / 口径）本质上都在动文本塔**，
图像塔的改进空间尚未被触及。

### 5.7 自比评测的四级阶梯

至此形成一套按「文本长度」递进、并带一个零文本基准的自比体系：

| 口径 | 文本参与 | 文本长度 | 测什么 |
|---|---|---|---|
| **IN-1k k-NN** | ❌ 无 | — | 图像特征质量 |
| IN-1k zero-shot | 类名 + 模板 | ~8 token | 短模板对齐 |
| COCO 5cap | 人工短 caption | ~13 token | 短句检索 |
| Urban-1k | 长描述 | ~132 token | 长文本检索 |

四个一起报，才能说清「提升发生在哪一侧」。单看任何一个都会误判 ——
本节的 5.3 就是一个实例。

---

## 6. 未做与待做

1. **噪声地板正在跑**（`gt_base` seed=1 / seed=2，`bash scripts/train/visreg.sh seed-var`）。
   动机：同配置**同 seed** 的两个 `gt_base`（0806 / 0811）末期 i2t 相差 0.70 点。
   在拿到 std 之前，本页所有 ≤1 点的差异都**不能定性**，具体涉及：
   - 5.3 的 `pcm_w0.3` vs `gt_base` k-NN 差 0.55
   - 4.2 的 `w=0.3` vs `w=0.5`（COCO 差 0.52 / IN-1k 差 0.20）
   - 4.3 的整个 dim 扫描（跨度 2.5 点，但非单调）

   ⚠️ 一个已知的不完美：seed-var 两组跑在比原 `gt_base` 更新的 commit 上
   （含 `_trim_to_batch_max` 与 PCA 兜底）。截断已验证数值等价（差异 1e-7），
   PCM 兜底在 `pcm_weight=0` 时不触发，所以理论上不影响 —— 但严格说这两组
   量出的是「新代码的 seed 方差」，与「旧代码 gt_base」之间还差一次同 commit 复现。

2. **w < 0.3 未扫**（4.2 的曲线还在下降）。
3. **d 与 w 未交叉**：4.3 的 dim 扫描全在 `w=1.0` 下做，而 `w=1.0` 已知是坏取值。
4. **epoch 选择规则未统一**：现在固定报 `epoch_10`，但单 run 末期抖动可达 3 点
   （`pcm_proj` ep6→ep9：25.30 → 25.32 → 23.10 → 22.14）。
   「末 3 epoch 均值」可能比「取最后一个」更稳，未验证。
5. **只有 CC3M 一个数据规模**，未验证结论随数据量的稳定性。
6. **Urban-1k 是唯一的长文本评测集**（1000 图，1:1 配对），规模小；
   DCI / ShareGPT4V-1k 尚未接入。
7. **图像侧只有 k-NN**：linear probe / attentive probe / dense probe（分割、深度）
   均未做。5.6 显示图像塔是尚未被触及的方向，这些口径届时才用得上。
8. **concat 未正式评测**（ep7 被杀）。判定依据是训练日志，够用但不严格。

---

## 7. 被本页推翻的旧结论

| 旧结论 | 出处 | 失效原因 | 修正后 |
|---|---|---|---|
| 「长短能力存在真实 trade-off，PCM 只是把曲线右移」 | `eval_protocol.md` §6 旧版 | 只测了 `w=1.0` 这一个（坏）取值 | w=0.3 时**两轴同时超过 gt_base**，trade-off 消失（4.1/4.2） |
| 「PCM 的价值 = 让长文本能力几乎无代价」 | 同上 | 同上 | 不是"无代价"，是**短文本也变好**（+3.08） |
| 「纯 dense 长文本能力最强（51.30%）」 | 同上 | 未与 PCM sweep 对比 | pcm_w0.3 57.70% 更强（4.4） |
| sweep 四组（ep10）与原始六组（ep9）同表对比 | 上一版合并表 | epoch 未对齐，漂移 1~1.7 点与效应同量级 | 全表统一 ep10（§3 说明） |
| 「COCO 短文本也涨了 → 图像表征本身变好了」 | 会话推断，未入文档 | 只看图文对齐指标，无纯图像口径 | k-NN backbone 与 gt_base **持平**（45.25 vs 45.80），增益在文本侧（5.3） |
| 「gemma_dense 是文本塔崩了，图像塔可能是好的」 | 会话推断，未入文档 | zero-shot 无法解耦图/文 | **两塔一起弱**：k-NN 39.85%（全场最低，−5.95）（5.4） |

---

## 8. 相关文档

- [[eval_protocol]] — 评测方法论：口径绑定 / 零重叠 / 全量报数（三次事故记录）
- `mgap_06_projective_siglip.md` — projective SigLIP 的 `|cos|` 目标推导
- [[visreg_all_attempts]] — 基座 VISReg-E 配方的来源
- `anisotropy_analysis.md` — 特征几何指标（EffRank / PR / StableR）。
  注意那是**几何描述而非能力度量**：高 EffRank 不等于特征更有用，判断能力必须跑任务（本页 §5）
- `scripts/eval/eval_standard.py` — COCO + IN-1k 入口
- `scripts/eval/eval_urban1k.py` — Urban-1k 长文本入口
- `scripts/eval/eval_knn_probe.py` — IN-1k k-NN 纯图像口径入口
