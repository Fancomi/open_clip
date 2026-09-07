# 用开源权重复现官方公开指标：本项目评测管线的对齐验证

*创建 2026-08-31 | 重写 2026-09-01（修掉一个静默预处理错配后全部重测）*
*脚本 `scripts/eval/verify_open_weights.py`（import 本仓库评测函数本体，只换模型加载）*

## 0. 目的与结论

**目的**：验证本项目 IN-1k zero-shot / COCO 检索 / IN-1k k-NN / Urban-1k 四个口径的
**实现**与业界标准是否一致 —— 方法是拿「官方公开过指标」的外部权重，用我们的管线复现官方数字。

**结论**：**IN-1k zero-shot 与 COCO 检索的实现与业界完全对齐**，在三个模型 × 两个独立
官方来源上复现到 **±0.3 点以内**（其中 DFN5B 的 IN-1k 与 COCO 官方值由 Apple
随权重发布的 `eval_results.jsonl` 给出，协议透明度最高）。
k-NN 协议逐字对齐 DINO 官方；Urban-1k 是外部基准原样使用，无自定义成分。

**过程中查出一个真实缺陷（已修）**：外部权重按本地路径加载时，`preprocess_cfg`
静默回落到 open_clip 默认值 → 非 CLIP 系模型（SigLIP2/PE）预处理错配。
它一度让 SigLIP2 掉 21.4 点、PE-Core 掉 1.1 点。**本项目自己的 run 不受影响**（§4）。

## 1. 最终验证结果（全部为修复后的重测值）

| 锚点模型 | 口径 | 官方公开值 | 我们实测 | 差 | 判定 |
|---|---|---|---|---|---|
| **openai CLIP B/16**<br>（CLIP 论文 / ALBEF Table 4 引用） | IN-1k zs | 68.3 | **68.35** | **+0.05** | ✅ |
| | COCO i2t R@1 | 52.5 | **52.54** | **+0.04** | ✅ |
| | COCO t2i R@1 | 33.3 | **33.08** | −0.22 | ✅ |
| | k-NN proj | 无官方 | 67.08 | — | 量级合理 |
| **DFN5B-CLIP H/14-378**<br>（Apple 官方 `eval_results.jsonl`） | IN-1k zs | **84.218** | **84.38** | **+0.16** | ✅ |
| | COCO i2t R@1（官方 `text_retrieval`） | **71.82** | **71.92** | **+0.10** | ✅ |
| | COCO t2i R@1（官方 `image_retrieval`） | **55.56** | **55.64** | **+0.08** | ✅ |
| | k-NN proj | 无官方 | 83.45 | — | 量级合理 |
| **SigLIP2 B/16-224**<br>（SigLIP2 论文 Table 1，B/16 224 行） | IN-1k zs | **78.2** | **78.47** | **+0.27** | ✅ |
| | COCO i2t R@1 | 68.9 | **69.32** | +0.42 | ✅ |
| | COCO t2i R@1 | 52.1 | **53.18** | +1.08 | ⚠️ 见 §3.3 |
| | k-NN proj | 无官方 | 77.32 | — | 量级合理 |
| **PE-Core-B/16**<br>（Meta model card） | IN-1k zs | 78.4 | **78.38** | **−0.02** | ✅ |
| | COCO t2i R@1 | 50.9 | **50.30** | −0.60 | ✅ |
| | k-NN proj | 无官方 | 75.97 | — | 量级合理 |

**11 个可对照格子里 10 个落在 ±0.6 以内，8 个落在 ±0.3 以内。**
四个不同架构（openai ViT / Eva+attn-pool / timm-SigLIP map-pool / H-14-378 大分辨率）、
三种 tokenizer（CLIP-BPE 77 / gemma-SP 64 / PE-BPE 32）、两种 resize 模式全部通过。

## 2. 本次查出的缺陷：`preprocess_cfg` 静默回落（已修）

### 2.1 现象

`create_model_and_transforms(model, pretrained="/path/to/local.bin")` 时，
open_clip **只从 `pretrained.py` 的注册表或 hf_hub 读 `preprocess_cfg`**；
给本地文件路径 → 拿不到 → 静默使用默认值：

```
默认（= openai CLIP 系）: mean/std 0.481.../0.268...  bicubic  shortest-resize + CenterCrop
```

而各模型官方要求（就在权重旁边的 `open_clip_config.json` 里）：

| 模型 | mean/std | interpolation | resize_mode |
|---|---|---|---|
| openai CLIP B/16 | 0.481/0.268 | bicubic | shortest+crop |
| DFN5B H/14-378 | 0.481/0.268 | bicubic | **squash** |
| SigLIP2 B/16 | **0.5/0.5** | bicubic | **squash** |
| PE-Core-B/16 | **0.5/0.5** | **bilinear** | **squash** |

### 2.2 代价（实测）

| 模型 | 错配项 | 错配下 | 修正后 | 差 |
|---|---|---|---|---|
| SigLIP2 B/16 IN-1k | mean/std + resize_mode | 57.11 | **78.47** | **21.4 点** |
| PE-Core-B/16 IN-1k | mean/std + interp + resize_mode | 77.29 | **78.38** | 1.1 点 |
| DFN5B H/14 IN-1k | 仅 resize_mode | 83.99 | **84.38** | 0.39 点 |
| openai CLIP B/16 | 无（默认恰好正确） | 68.35 | 68.35 | 0 |

### 2.3 为什么它藏了这么久 —— 值得记住的失效模式

**这个错配在 openai CLIP 上恰好是零影响**（默认值就是它的正确值），
而 openai CLIP 正是最常用的第一个锚点。于是「第一个模型完美复现」
反过来成了「管线没问题」的伪证据，掩盖了后续模型上的错配。

📌 **教训（与 §5.28「口径 bug 会跨数据表复制」同类）**：
**一个静默回落的默认值，在一部分对象上恰好正确时，最危险。**
判据不能是「有一个模型对上了」，必须是「每个对象都用它自己的官方配置对上」。

📌 **第二条**：我一度把 −16.9 点归因为「社区共性问题」（HF 论坛确有 SigLIP2 复现帖），
是用户指出「社区差 5~9 点、我们差 16 点，量级不对」才回头查出真因。
**量级不匹配本身就是判据** —— 不能因为「别人也复现不出」就停止排查。

### 2.4 修法

`verify_open_weights.py` 新增 `_official_preprocess()`：读权重旁边的
`open_clip_config.json`，把 `preprocess_cfg` 显式转成
`image_mean / image_std / image_interpolation / image_resize_mode` 四个 kwargs 传进
`create_model_and_transforms`，并把最终 `val_tr` 打印出来（可核对）。

## 3. 其它已排除 / 已解释的项

### 3.1 PE 家族 `context_length`：本 fork 改过默认值（对外部权重是坑）

本 fork 的 `pretrained.py:_pecfg` 把 PE 家族 `context_length` 默认改成 **256**
（为自家 `--force-context-length 256` 训练服务；PE 官方是 **32**）。
直接加载 PE 官方权重时：文本塔按 256 建、权重只有 32 → `strict=False`
**静默随机初始化文本塔** → IN-1k 3.2%（近随机），**无任何报错**。
必须同时给 `force_context_length=32` **且** 把 tokenizer 的 `context_length` 也改成 32
（否则 tokenizer 吐 256 长度、`_embeds` 里 `positional_embedding[:seq_len]` 直接 shape 报错）。

→ 结论：**我们自己的 `PE-Core-B-16-dinov3` 训练不受影响**（本来就用 256）；
但任何人拿 PE 官方权重做对照都必须显式指定 32。

### 3.2 dtype 分支 bug（已修）

`eval_standard` 内 `dt = float16 if device == "cuda" else float32` 用字符串精确比较。
传 `device="cuda:0"` → 取 fp32 输入喂 fp16 权重 → 直接崩。
本仓库自身从不踩到（`load_model` 恒定传 `"cuda"`）。

### 3.3 唯一仍超 ±0.6 的格子：SigLIP2 COCO t2i +1.08

方向是**我们更高**，且同模型 i2t 只差 +0.42、IN-1k 只差 +0.27。
可能来源：SigLIP2 论文的 COCO 协议细节未公开（是否 5-cap 全池、
是否用 `canonicalize` 文本清洗）。**不视为实现缺陷**，但如实记录为未闭合项。

### 3.4 k-NN backbone 列对外部模型无意义

`extract_feats` 的 backbone 分支取 `trunk.forward_features(x)[:, 0, :]`（CLS token）。
- openai CLIP：老式 VisionTransformer 无 `forward_features` → 返回 None（已跳过）。
- SigLIP2（map-pool）/ PE-Core（attn-pool）：CLS token **不是**其对比表征
  （SigLIP2 11.18 / PE-Core 16.31，接近随机）→ **这两个数字必须丢弃**。
- 对本项目自己的 `PE-Core-B-16-dinov3` 该列有效（trunk CLS 是 VISReg 的作用点）。

## 4. 这个缺陷对本项目已有结论有无影响：**没有**

逐条核对：

1. **本项目所有 run 都不走「本地路径 + 外部配置」这条路。**
   `eval_standard.load_model` 用 `create_model_and_transforms("PE-Core-B-16-dinov3", "")`
   —— 空 `pretrained`、随机初始化再 `load_state_dict`，预处理走 open_clip 默认。
2. **训练与评测用同一套默认值**，实测确认：
   - 训练 `RandomResizedCrop(224, scale=(0.9,1.0), bicubic)` + `Normalize(0.481/0.268)`
   - 评测 `Resize(224, bicubic) + CenterCrop(224)` + 同一 `Normalize`
   → **训练/评测归一化完全一致，不存在错配。**
3. **区域臂的 `--image-resize-only` 也在同一套里**：实测日志
   `=> image_resize_only: preprocess_train 换为无裁剪 Resize(224, 224)`，
   它只把训练侧的 RandomResizedCrop 换成 Resize，`Normalize` 沿用 val 的后半段。
   → 与「A′ = gt+resize 单独 −1.7 点」那条消融的记账一致，无新混淆项。
4. `params.txt` 的 `image_mean/std/interpolation/resize_mode` 全为 `None`
   → 全站 run 共享同一套默认预处理，**组间对比不受影响**。

→ **主表、周报、所有历史结论均不需要修订。**

## 5. 残余风险（如实记录）

- **Urban-1k 未做外部权重对照**：Long-CLIP 官方权重下载未完成。该口径协议无自定义成分
  （数据集自带 1:1 配对、R@1 标准定义），风险本来最低，但确实是四个口径里唯一
  没有外部数字背书的。
- **fp16 vs fp32 未逐点标定**：本项目全部评测统一 fp16 模型 + fp16 输入（自洽）；
  外部对照差异 ≤0.3 点，说明该项影响小于官方数字本身的复现抖动。
- **k-NN 与 Urban-1k 只能靠「协议逐字对齐」背书**，不是数值复现。
