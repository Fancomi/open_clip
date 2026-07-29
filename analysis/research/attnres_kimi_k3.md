# AttnRes（Kimi-K3 注意力残差）在本项目的复现与显存问题定位

结论先行：**旧分支的显存爆炸不是 AttnRes 固有的，是两个可修的实现选择。** Kimi 没有靠堆卡。
K3 的 93 层里只有 8 个锚点（`attn_res_block_size=12`），而参考实现每个调用点物化三份
`(T, S, H)` 中间张量、且全部 fp32。前者是架构参数，后者是纯实现问题。

来源：HF `moonshotai/Kimi-K3` revision `9f62e4e`，`modeling_kimi_linear.py`。
只下载了 `.py` / `config.json` / `model.safetensors.index.json`；权重通过 HTTP Range
读取 safetensors header 加那 374 个 AttnRes 小张量，总计几十 MB。

## 1. K3 里 AttnRes 到底是什么

核心是 `_apply_attn_res`（`modeling_kimi_linear.py:1075`）：对每个 token，把若干个
「深度锚点 + 当前残差流」做 RMSNorm，与一个**学到的固定方向**打分，softmax 出凸组合
权重后加权平均。查询向量是静态的（rank-1，跨 token 共享），所以这不是真 attention，
而是**沿深度轴的单头、静态 query 的加权读取**。

层内调度（`_forward_attn_residual`，`:973`）：

```
prefix_sum = hidden_states                 # 当前残差流
if block_residual:                         # 有锚点才混合
    x = mix([*anchors, prefix_sum])
if layer_idx % block_size == 0:            # 锚点层
    anchors.append(prefix_sum)
    prefix_sum = None                      # 残差流清零重启
prefix_sum = (prefix_sum or 0) + attn(ln_1(x))
x = mix([*anchors, prefix_sum])
prefix_sum = prefix_sum + moe(ln_2(x))
```

架构上的实质：**93 层的一条长 identity 高速路，被换成 8 条短高速路 + softmax 门控的
跨块读取**。注意「清零」是真的——跨块边界没有纯恒等通路。

配置事实（`config.json` → `text_config`）：

| 项 | 值 |
|---|---|
| `num_hidden_layers` | 93 |
| `attn_res_block_size` | 12 |
| 锚点层 | `[0, 12, 24, 36, 48, 60, 72, 84]`，共 **8** 个 |
| slot 数 | 从 1 增长到最多 **9**（不是 93） |
| 每层参数 | `self_attention_res_{norm,proj}` + `mlp_res_{norm,proj}` |
| 全局参数 | `output_attn_res_{norm,proj}`（`norm` 前再混一次，`:1226`） |
| 参数总量 | `93×4×7168 + 2×7168 ≈ 2.68 M`（对 2.8T 免费） |

锚点内容是**块级部分和**：anchor0 = embedding，anchor1 = 层 0–11 输出之和，
anchor2 = 层 12–23 之和，依此类推。

## 2. 从权重里交叉验证

抽读了 `layers.{0,1,2,6,11,12,13,18,24,25,36,37,48,49,60,72,84,85,90,92}` 的
`score_weight = norm.weight * proj.weight`（两者只以逐元素乘积出现）：

- `layers.0.self_attention_res_proj` **精确全零**（absmax 0.0000）。层 0 的 sa 混合分支
  永远不执行（此时锚点列表还空），这条参数是死的——正好说明发布代码的执行路径与训练时一致。
- L2 范数从层 0 的 0.08–0.17 单调升到层 92 的 0.36，`output_attn_res` 最大（0.303）。
- **每个 `%12==0` 的锚点层都是局部峰值**：层 12 是 0.26/0.24 vs 层 11 的 0.20/0.14；
  层 60 是 0.29；层 72 是 0.41。门控确实被训成「在块边界更强地读锚点」，没退化成恒等。
- 视觉塔**没有** AttnRes（`vision_tower` 165 个张量，无 `res_norm`/`res_proj`）。
  AttnRes 纯粹是 LLM 侧的深度方向机制。K3 视觉塔是 27 层 / 1024 dim / patch14。

另外：`block_residual` 在 `KimiLinearModel.forward` 内创建（`:1188`），随层循环传递，
forward 结束即丢弃，**完全不进 KV cache**。对每个 token 独立、无跨 token 依赖，
decode 时每步重建。这是它能在 1M 上下文下不出问题的原因。

发布代码是**纯推理**路径（`KimiSparseMoeBlock` 里 `assert not self.training`），
`fla` 只提供 KDA 的 triton kernel，**AttnRes 的训练 kernel 没有开源**。

## 3. 显存问题的定位

参考实现每个调用点物化 `v`（`cat` 结果）、`v_float`（fp32 拷贝）、`k`（归一化拷贝），
三份 `(T, S, H)` 全部存活到 backward。但**反向传播一份都不需要**：

- `v` 的构成部分本来就在计算图里活着（锚点被所有后续层使用，残差流就是主干）
- `v_float` / `k` / `variance` 全部可由 `v` 重算
- 唯一需要留的是 `probs`，`(T, S)`

本实现 `src/open_clip/attn_res.py` 的做法：把 RMSNorm-then-project 的打分拆成两个
标量归约，归一化后的 slot `k_s` 根本不成形：

```
r_s     = rsqrt(mean(v_s^2) + eps)     # (T,)
dot_s   = v_s @ w                      # (T,)
score_s = r_s * dot_s
```

backward 只需要 `probs`、`r`、`dot`（都是 `(T, S)`）。此外 slot 保持为 list 而非
`cat`，fp32 只用于 `(T, S)` 统计量、softmax 和输出累加器，`(T, H)` 的逐元素运算留在
输入 dtype（`addcmul_` 支持混合 dtype），所以从不分配全尺寸 fp32 拷贝。

### 单调用点实测（H800，K3 尺寸 H=7168 / T=4096 / bf16）

`resid` 是 slot + 其梯度 + 入射 cotangent 的固定开销，两种 kernel 都要付；
`ovh = peak - resid` 才是 kernel 自身的footprint。

| slots | resid MB | naive MB | naive ovh | fast MB | fast ovh | ovh 比 | naive ms | fast ms |
|---|---|---|---|---|---|---|---|---|
| 2 | 280 | 1632 | 1352 | 534 | **254** | 5.3x | 12.99 | 4.14 |
| 3 | 392 | 2360 | 1968 | 645 | **253** | 7.8x | 17.32 | 7.59 |
| 5 | 616 | 3816 | 3200 | 869 | **253** | 12.6x | 25.81 | 11.64 |
| **9**（K3 峰值）| 1064 | 6728 | 5664 | 1318 | **254** | **22.3x** | 40.22 | 17.00 |
| 17 | 1960 | 12552 | 10592 | 2214 | **254** | 41.7x | 71.95 | 31.72 |
| 47 | 5320 | 34392 | 29072 | 5576 | **256** | 113.6x | 191.17 | 90.11 |

fast kernel 的 overhead 是**常数 ~254 MB**（与 slot 数无关），naive 是线性增长。
速度也快 2–3 倍，因为省掉了三次全尺寸拷贝的内存带宽。

### 整塔实测（27 层 / 1024 宽 / T=1024 / B=8 / bf16，fwd+bwd 峰值）

`block=1` 就是旧分支的「每层都保留」变体。

| block | 锚点数 | kernel | peak MB | vs base | ms |
|---|---|---|---|---|---|
| None（基线）| 0 | — | 7794 | 1.00x | 151 |
| **1** | 27 | naive | **OOM** | — | — |
| **1** | 27 | fast | 20327 | 2.61x | 747 |
| 2 | 14 | naive | 38133 | 4.89x | 765 |
| 2 | 14 | fast | 14260 | 1.83x | 476 |
| 3 | 9 | naive | 29678 | 3.81x | 624 |
| 3 | 9 | fast | 12217 | 1.57x | 385 |
| 4 | 7 | naive | 25578 | 3.28x | 551 |
| 4 | 7 | fast | 11219 | **1.44x** | 339 |
| 7 | 4 | naive | 20197 | 2.59x | 460 |
| 7 | 4 | fast | 9916 | 1.27x | 280 |
| 14 | 2 | naive | 16646 | 2.14x | 399 |
| 14 | 2 | fast | 9070 | 1.16x | 241 |

`block=1` + naive 在 80 GB H800 上直接 OOM——这就是旧分支。两个改动各自贡献一个
数量级：`block_size` 从 1 提到 4 把锚点从 27 降到 7，kernel 重写再砍掉 2.3–2.6 倍。

开梯度检查点时对比更极端（基线降到 1694 MB，AttnRes 的锚点无法被 checkpoint 吸收）：

| block | naive | fast | fast vs base |
|---|---|---|---|
| 1 | 57050 | 14262 | 8.42x |
| 4 | 19513 | 5189 | 3.06x |
| 14 | 10582 | 3097 | 1.83x |

额外参数：110.6 K（对 151 M 的 ViT-B-32 是 +0.02%）。

## 4. 一个不在 K3 里、但绕不开的初始化问题

K3 的门控没有 per-slot bias——slot 的身份只来自「内容 vs 共享 query」。所以
`proj_weight = 0` 的新初始化门控输出的是**所有 slot 的均匀平均**，不是残差流。
把 K3 原样搬到已有网络上，第 0 步就会把信号缩小 `1/S` 并混入过期锚点。

`AttnResGate(identity_init=True)` 在 stream slot 上加一个标量 logit（初始 +8），
让 softmax 一开始近似 one-hot 在 stream 上。**这不是恒等变换**——凸组合无法表达
`anchor + stream`，跨块的恒等通路是真没了。它只修正初始的激活**尺度**。

init 时的信号/梯度传播（27 层 / fp32，`out/in` 是输出 RMS 比输入 RMS）：

| block | init | out/in | grad RMS | vs base |
|---|---|---|---|---|
| None | — | 1.623 | 1.4843 | 1.00x |
| 1 | k3-zero | 0.058 | 0.0530 | **0.04x** |
| 1 | identity | 0.449 | 0.6374 | 0.43x |
| 3 | k3-zero | 0.162 | 0.1484 | 0.10x |
| 3 | identity | 0.805 | 0.7519 | 0.51x |
| 7 | k3-zero | 0.325 | 0.2969 | 0.20x |
| 7 | identity | 1.104 | 1.1305 | 0.76x |
| 14 | k3-zero | 0.541 | 0.4948 | 0.33x |
| 14 | identity | 1.618 | 1.6363 | 1.10x |

普通 pre-norm transformer 的 `out/in > 1`（残差累加）；softmax 门控**做不到**——
凸组合不可能超过最大的 slot，所以塔整体是范数收缩的，除非门控学会别的行为。
`block=1` + k3-zero 时梯度只剩基线的 4%，这大概也是旧分支「效果很差」的另一半原因，
和显存问题相互独立。

## 5. 本次落地的代码

| 文件 | 内容 |
|---|---|
| `src/open_clip/attn_res.py` | `attn_res_mix`（省显存 kernel，自定义 autograd）、`attn_res_mix_naive`（K3 逐行转写，做对照）、`AttnResGate` |
| `src/open_clip/transformer.py` | `AttnResTransformer`；`ResidualAttentionBlock.branch_attn/branch_mlp`（拆出无残差加的分支，让 transformer 接管残差记账）；`VisionTransformer` 新增 `attn_res_*` 参数 |
| `src/open_clip/model.py` | `CLIPVisionCfg.attn_res_block_size` / `_identity_init` / `_naive` |
| `src/open_clip/model_configs/ViT-B-32-attnres{4,2}.json`、`-attnres4-k3init.json` | 三个消融配置 |
| `tests/test_attn_res.py` | 21 个测试：fp64 gradcheck、fast vs naive 前反向一致、state_dict 兼容、梯度检查点一致、CUDA 精度 |
| `analysis/attn_res_bench.py` | 上面三张表的生成脚本（`--part gate/tower/signal`） |
| `scripts/train/attnres_cc3m.sh`、`attnres_parallel.sh` | CC3M 四臂对照训练 |

`AttnResTransformer` 保留 `resblocks` 的结构和参数名，普通 `Transformer` 的 checkpoint
可以 `strict=False` 加载（只有门控是新增的，ViT-B-32 上 72 个 key）。

两个踩坑记录：

- K3 硬编码 `.float()`，在 fp64 下会**降精度**，导致 gradcheck 假失败。`_acc_dtype`
  用 `promote_types` 代替。
- `--logs` 不能穿过 torchrun：它的 parser 会把前缀解析到自己的 `--logs-specs`，
  报 `ambiguous option`。脚本改用 `--name` 区分。
- `--train-num-samples` 只对 webdataset 生效，csv 路径会走完整个 TSV。脚本改为
  预先切一份确定性的 head-N 子集，保证各臂看到完全相同的样本。

## 6. 移植到本项目的建议

1. `block_size` 取 `depth / 8` 量级。27 层视觉塔对应 3–4；ViT-B-32（12 层）用 4。
   **别用 1。**
2. 从头训，别嫁接。跨块恒等通路的丢失是架构级的，不是初始化能补的。
3. 若必须嫁接，`identity_init=True` 是底线，且要给门控单独的（更大的）学习率。
4. `prune_intermediate_layers` 在 AttnRes 下已显式 `NotImplementedError`——锚点让
   每一层都耦合到 output gate，截断会静默改变输出。`forward_intermediates` 的
   `stop_early` 同理被忽略。

## 7. CC3M 对照实验

四臂（`base` / `attnres4` / `attnres2` / `k3init`），80 万样本、单卡各一张 GPU 并行，
其余全部对齐（seed 0、LR 1e-3、BS 256、amp_bf16、同一份子集 TSV）。
评测 COCO Karpathy 检索（5000 图 × 5 caption）+ ImageNet zero-shot。

| arm | block | init | train loss | IN top1 | IN top5 | i2t R@1 | i2t R@5 | i2t med rank | t2i R@1 | t2i med rank | batch (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **base** | — | — | **4.776** | **0.60%** | 2.21% | 0.36% | 0.96% | **1299** | 0.18% | 970 | **0.206** |
| k3init | 4 | k3-zero | 4.795 | 0.55% | **2.23%** | **0.42%** | **1.28%** | 1331 | 0.18% | **944** | 0.267 |
| attnres4 | 4 | identity | 4.964 | 0.50% | 1.91% | 0.12% | 0.70% | 1730 | 0.16% | 1179 | 0.267 |
| attnres2 | 2 | identity | 5.028 | 0.42% | 1.73% | 0.12% | 0.62% | 1971 | 0.12% | 1287 | 0.303 |

80 万样本（3125 步）只够跑到 top1 < 1% 的量级，绝对数字都在噪声附近，所以这张表**不能
用来判断 AttnRes 好不好**。能读出来的是三件有方向性的事：

1. **AttnRes 在这个尺度上没有优势，且训练 loss 明确更高**。K3 用 AttnRes 是在 93 层
   深度上解决问题的；12 层的 ViT-B-32 本来就没有深度病，加了只是多一层约束。
   这和第 4 节的信号分析一致——凸组合让塔范数收缩，短塔上纯粹是损失。
2. **`block=2`（6 锚点）比 `block=4`（3 锚点）更差**，两个指标都单调。更多锚点 =
   更频繁地切断恒等通路。这条趋势和「block_size 不该取小」的结论方向一致。
3. **`k3init` 反而比 `identity_init` 好**，且几乎追平基线。这与第 4 节 init 时的梯度
   测量（k3-zero 只有基线 10%）**相反**，说明短塔上「初始梯度大」不是决定因素：
   uniform-mean 初始化把所有 slot 等权混合，本身是一个更强的层间平均/正则，
   而 stream-dominant 初始化把网络推向「块局部 transformer」，反而更难跳出。
   这个反转是本次实验唯一超出预期的结果，值得在深塔上单独复查。

速度开销 +29%（block=4）/ +47%（block=2），与第 3 节整塔基准的趋势吻合。

**没有验证的**：深塔（≥40 层）上的行为，也就是 AttnRes 真正的目标场景。本项目现有的
27 层视觉塔和 12 层 ViT-B-32 都不在那个区间。要判断 AttnRes 值不值得，得在明显更深的
塔上、跑到 loss 有意义的步数才行。
