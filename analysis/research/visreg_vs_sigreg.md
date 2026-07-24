# VISReg vs SIGReg：计算差异与理论依据

## 背景

历史 no-dino 冠军配方 `proj_s15_sigreg`（PE-Core-B-16-dinov3 + SigLIP + projective +
init-scale 15 + Muon + SIGReg cls 1e-4）在 cc3m 上取得 i2t R@1=0.2344。本研究把
**SIGReg**（Sketched Isotropic Gaussian Regularization，LeJEPA，
[2511.08544](https://arxiv.org/abs/2511.08544)）替换为 **VISReg**（Variance-Invariance-
Sketching Regularization，[2606.02572](https://arxiv.org/abs/2606.02572)，官方实现
[HaiyuWu/visreg](https://github.com/HaiyuWu/visreg)），在 cc3m-tsv 上做严格 A/B：
冠军配方其余部分一字不改，唯一变量是正则项。

两者都作用在 `sigreg-target=cls`（backbone CLS raw token `[B, D]`，L2-norm 前）上，
目标一致：**约束视觉 backbone 输出趋向各向同性高斯，防止表征坍缩/各向异性**。
差别在"怎么度量与目标分布的差距"，以及"坍缩时还有没有梯度"。

集成实现见 `src/open_clip/loss.py` 的 `SIGReg` / `VISReg` 两个 `nn.Module`，
经 `--reg-method {sigreg,visreg}` 切换。

---

## 一、两者都想做的事：各向同性高斯约束

LeJEPA 证明：JEPA/自监督嵌入的**最优先验分布是各向同性高斯** N(0, I)——它在给定
二阶矩下熵最大、无偏好方向，能最小化下游预测风险。问题化为：如何用可微、可扩展的
损失，把一个 batch 的高维经验分布 `Z ∈ [N, D]` 推向 N(0, I)？

直接算高维 KL 到 N(0,I) 是 O(D³)（要对协方差矩阵求逆/行列式），不可行。两种方法都用
**Cramér-Wold 定理**降维：高维分布相等 ⟺ 所有一维随机投影上的分布都相等。于是都改为
沿 K 个随机方向 `w_k` 投影到一维，在一维上比较投影分布与目标高斯。**分歧从这里开始。**

---

## 二、SIGReg：频域特征函数检验（Epps-Pulley）

SIGReg 在**频域**用特征函数（characteristic function）检验一维投影是否高斯。

记投影 `p = Z·w ∈ [N]`，其经验特征函数 `φ̂(t) = E[e^{it·p}]`。标准高斯的特征函数是
`φ(t) = e^{-t²/2}`。Epps-Pulley 统计量在若干节点 `t_j`（本实现 17 个，t∈[0,3]）上，
用高斯权重加权积分二者差的模方：

```
L_SIGReg = N · world_size · ∫ |φ̂(t) - φ(t)|² dμ(t)
         = N · world_size · Σ_j w_j [ (cos_mean_j - φ_j)² + sin_mean_j² ]
```

其中 `cos_mean_j = mean_n cos(t_j · p_n)`，`sin_mean_j = mean_n sin(t_j · p_n)`，
`w_j` 为梯形积分权重 × 高斯窗 φ(t_j)。跨卡对 cos_mean/sin_mean 做 all-reduce-mean
得到全局 batch 统计。

**关键性质**：乘了 `N · world_size`——因为特征函数是 batch 均值量（batch-size 不变），
不乘的话大 batch 下正则项在梯度里占比会被稀释，故显式放大到全局样本数量级。

代码：`loss.py:SIGReg.forward`（`x_t = (x@A)·t → cos/sin → err @ weights · N·world_size`）。

---

## 三、VISReg：三项解耦（scale / shape / center）

VISReg 不进频域，而是把正则拆成三个**可独立加权**的项，其中 shape 项在**空域**用
Sliced-Wasserstein 距离对齐分布。给定中心化嵌入 `Ẑ = Z - μ`：

### 3.1 scale（方差项，源自 VICReg variance）

```
L_scale = (1/D) Σ_j (1 - σ_j(Ẑ))²          # σ_j = 第 j 维标准差（biased）
```

只约束每一维的**尺度**≈1，不管分布形状。这一项**坍缩时梯度不消失**：当 σ_j→0，
`d/dσ (1-σ)² = -2(1-σ) → -2`（常数），始终把方差往 1 拉。

### 3.2 shape（分布形状项，Sliced-Wasserstein，替代 VICReg 的 covariance）

先对每维标准化并 **stop-gradient σ**，把形状优化与尺度优化解耦：

```
Z̃ = Ẑ / sg(σ + ε)
```

再沿 K 个随机单位方向 `w_k` 投影，用**一维 2-Wasserstein 距离的闭式解**——
即排序后与标准高斯分位数逐点比较：

```
L_shape = (1/K) Σ_k || sort(Z̃·w_k) - q_N ||²
```

其中 `q_N ∈ [N]` 是标准高斯的固定分位数 `Φ⁻¹(i/(N+1))`（用 erfinv 实现）。
1-D Wasserstein = 分位数函数间的 L_p 距离（排序即最优传输耦合），故排序后逐点作差
就是 W₂²。这比 covariance（只去相关、只管二阶矩）严格更强：它对齐了**每个投影方向上
的完整边缘分布**，而不仅是相关性。

### 3.3 center（居中项）

```
L_center = || μ ||²                          # batch 均值拉向 0
```

论文消融显示 center 对终值影响小（~0.4%），但能加快收敛。

### 3.4 合成

```
L_VISReg = λ_scale · L_scale + λ_shape · L_shape + λ_center · L_center
```

默认 λ 全 = 1。代码：`loss.py:VISReg.forward`。**注意 VISReg 天然 batch-invariant**
（scale/shape/center 都是均值量或已按 N 归一），**不像 SIGReg 乘 N·world_size**——
这直接导致两者裸损失量级差几个数量级，是权重必须重新标定的根因（见第五节）。

<!-- PLACEHOLDER_CONTINUE -->

---

## 三·五、实现细节逐行对照：Epps-Pulley vs Sliced-Wasserstein + 方差

前两节给了公式，这节讲**代码层面到底在算什么、差在哪**，尤其澄清 VISReg 里的
"排序（sort）"是什么、为什么必须排序。

### 共同起点：投影到一维（两边完全相同）

都要判断"一批特征 `Z∈[N,D]` 离标准高斯多远"。高维直接算不可行，靠 Cramér-Wold
定理降维——沿 K 个随机方向 `w_k` 投影到一维，在一维上比较。投影这步两边一致：

```python
p = Z @ W        # [N, K]：Z[N,D] @ W[D,K]，每一列 = 一个方向上 N 个投影值
```

**分歧从"一维上怎么量这 N 个数像不像高斯采样"开始。**

### SIGReg = 频域：比"特征函数"，无排序

任何分布都有特征函数 `φ(t)=E[e^{it·X}]`（分布的傅里叶变换）；标准高斯有闭式
`φ(t)=e^{-t²/2}`。两分布相等 ⟺ 特征函数处处相等。SIGReg 在一组频率节点 `t`（17 个，
t∈[0,3]）上算经验特征函数，与高斯的比：

```python
x_t = (x @ A).unsqueeze(-1) * t   # [N, slices, knots]：每个投影值 × 每个频率 t
cos_mean = x_t.cos().mean(0)      # Re(φ̂) = mean_n cos(t·p_n)
sin_mean = x_t.sin().mean(0)      # Im(φ̂) = mean_n sin(t·p_n)
err = (cos_mean - phi)**2 + sin_mean**2   # |φ̂(t) - φ(t)|²，phi=e^{-t²/2}
loss = (err @ weights).mean() * N * world_size   # 梯形权重×高斯窗，加权积分
```

关键：`e^{it·p}=cos(t·p)+i·sin(t·p)`，`cos_mean/sin_mean` 是经验特征函数实/虚部。
**全程只有 cos/sin/mean，没有排序**。相当于"比两段声音的频谱"，整体比对，不逐点对齐。
因为是均值量，跨卡可 all-reduce 精确合并。

### VISReg shape = 空域：Sliced-Wasserstein，核心是"排序"

思想完全不同：直接在**数值空间**用最优传输距离（Wasserstein）比两个分布。
Wasserstein = "把一堆点搬成另一堆点最省力要搬多远"。高维难算，**但一维有闭式解，
闭式解的关键就是排序**：

- 把 N 个投影值从小到大排序：`p_(1) ≤ p_(2) ≤ … ≤ p_(N)`（`sort`）。
- 取标准高斯的 N 个理论分位点，也升序：`q_i = Φ⁻¹(i/(N+1))`（用 erfinv 实现）。
- **一维最优传输的最优方案 = "最小配最小、最大配最大"**，即排序后逐位对应。故

```
W₂² = (1/N) Σ_i (p_(i) - q_i)²      # 排序后逐点作差平方
```

```python
p_sorted = (z_norm @ W).sort(dim=0).values     # 沿样本维排序 [N, K]
target   = erfinv(2*(i/(N+1)) - 1) * sqrt(2)   # 标准高斯分位数 q_i [N]
shape_loss = (p_sorted - target)**2 .mean()
```

**"排序"的含义**：把这批投影值排队，看第 1 名是否落在高斯"应出现的最小值位置"、
第 2 名是否在第 2 分位……逐名对齐。真高斯采样排序后应贴合高斯分位数曲线。
**为什么必须排序**：Wasserstein 是"按序配对"的距离，不排序就不知道哪个点对应高斯的
哪个分位——排序即建立最优传输的配对。代价 `O(N logN)`，且**跨卡不能对"排好序的数"
求均值**（第 3、5 名的均值无意义），故 VISReg 只能 all-gather 全局 batch 再各卡独立算
（见第五节）。

### VISReg scale/center = 空域矩：连排序都不用

shape 管"形状像不像高斯"，但形状对了尺度仍可能错（各维方差=4 而非 1）。故补两个廉价项：

```python
scale_loss  = (1 - std)**2 .mean()   # 每维标准差拉向 1（std=sqrt(var+ε)）
center_loss = mu**2 .mean()          # 每维均值拉向 0
```

普通一/二阶矩，`O(N·D)`，无投影无排序。scale 项坍缩时梯度恒为 -2（见第四节），
是 VISReg 抗坍缩的来源。

### 三者对照

| | SIGReg | VISReg shape | VISReg scale/center |
|---|--------|--------------|---------------------|
| 空间 | 频域（特征函数）| 空域（数值分布）| 空域（矩）|
| 核心运算 | cos/sin + 加权积分 | **sort** + 分位数比 | mean / std |
| 度量 | 特征函数 L² 差 | 一维 2-Wasserstein | 一/二阶矩 |
| 比喻 | 比频谱 | 排队逐名对齐分位 | 量平均身高/胖瘦 |
| 坍缩梯度 | →0（消失）| 弱 | **恒 -2（强）** |
| 复杂度 | O(N·K·knots) | O(N·K·logN)（排序）| O(N·D) |
| 跨卡 | all-reduce 均值 | all-gather（排序不可 reduce）| all-gather |

**一句话**：SIGReg 在频域用特征函数整体比对；VISReg 的 shape 在空域把投影值**排序后
逐分位对齐高斯**（这就是"排序"，本质是一维最优传输），scale/center 用便宜的矩兜底尺度。



这是 VISReg 论文最强的论点，也是两者最本质的区别。

**SIGReg 坍缩时梯度消失。** 当嵌入坍缩（所有样本趋同，投影 p 的方差→0），经验特征
函数 `φ̂(t) → 1`（所有 e^{it·p_n}≈1），恰好这也是"点分布"的特征函数在小 t 处的行为；
Epps-Pulley 统计量的梯度随之衰减，**在坍缩最严重时趋于 0**——正则失去纠偏能力，
恰恰在最需要它的时候。论文 Figure 2 用 `‖∇L‖` 对特征范数 r 的曲线证明了这点。

**VISReg 坍缩时梯度为常数。** L_scale 的 `(1-σ)²` 在 σ→0 时梯度 → -2（见 3.1），
是一个**恒定的强恢复力**。我们在集成时实测复现了这一性质：

```
输入全零 [64,128]（完全坍缩）:
  SIGReg 梯度范数 ≈ 0
  VISReg 梯度范数 ≈ 13.9（有限、非零）
```

（实现细节：VISReg 的 std 把 ε 放进 sqrt 内 `sqrt(var + ε)` 而非 `sqrt(var)+ε`，
保证 var→0 时梯度有限而非 NaN。见 `loss.py:VISReg.forward` 注释。）

**直观理解**：SIGReg 是"检验分布是否高斯"的统计量，坍缩成一个点时它"检验不出方向"
——梯度模糊；VISReg 的方差项是"直接把每维拉到单位尺度"的机械力，永远知道往哪拉。

## 五、计算复杂度与分布式

| | SIGReg | VISReg |
|---|--------|--------|
| 度量 | 频域特征函数（Epps-Pulley）| 空域 Sliced-Wasserstein + 方差 |
| 主计算 | 投影→cos/sin→积分 | 投影→**排序**→分位数比较 |
| 复杂度 | O(N·D·K + N·K·knots) | O(N·D·K + K·N·logN)（排序）|
| vs VICReg covariance | — | 都是 O(N·D·K)，优于 covariance 的 O(N·D²) |
| 跨卡 | all-reduce cos/sin 均值（可精确合并）| all-gather 全局 batch，各卡独立切片 |
| batch 量级 | 乘 N·world_size | batch-invariant（不乘）|

**分布式的关键差异**：SIGReg 的统计量是均值，能对 cos_mean/sin_mean 做 all-reduce-mean
精确合并成全局 batch 统计。VISReg 的 shape 项含 **sort**，无法对"排好序的分位数"跨卡
求均值，故采用官方做法——**grad-aware all-gather 把 z 汇成全局 batch，每卡用各自独立的
随机切片计算，DDP 平均梯度**，等价 K×world_size 个切片（论文 §3.2）。这既解决了 sort
不可 reduce 的问题，又把"切片数随卡数线性增长"变成免费的扩展性红利。

集成实现：`VISReg.forward` 内 `torch.distributed.nn.all_gather(x)` 后再算三项；
标定脚本 `scripts/tools/calib_visreg_weight.py` 用 `gather=False`（z 已是全局 batch）。

## 六、权重标定：为什么必须重标

因 batch 量级差异（第 3.4 节），VISReg 裸损失量级远小于 SIGReg，直接套用
`--sigreg-weight 1e-4` 会让正则形同虚设。标定思路：**匹配正则项对 backbone 输出
特征 z 的梯度范数 `‖∂L/∂z‖`**——这是 backbone 反传时真正收到的信号，与 backbone
权重的随机初始化无关，可离线在 global_batch=4096 下标定：

```
w_visreg = w_sigreg · ‖∂L_sigreg/∂z‖ / ‖∂L_visreg/∂z‖
```

标定结果 **w_visreg ≈ 1.83e-4**（feat_std 0.5~2 区间稳定在 1.8e-4~4e-4）。脚本：
`scripts/tools/calib_visreg_weight.py`。

## 七、实测结果（cc3m-tsv，冠军配方，只换正则项）

COCO Karpathy 5k 图文互检 + ImageNet-1k zero-shot（IMAGENET_CLASSNAMES + OpenAI 模板），
各组峰值：

### 7.1 组件消融（A–E，10 epoch）

| 组 | 正则配置 | COCO i2t R@1 | IN-1k top1 | top5 |
|----|---------|-------------|-----------|------|
| A | SIGReg 1e-4（锚点）| 22.84 | 21.23 | 40.71 |
| B | VISReg 全项 scale+shape+center | 23.74 | 21.48 | 41.18 |
| C | VISReg scale-only | 23.88 | 22.60 | 42.10 |
| D | VISReg shape-only | **24.64** | 21.59 | 40.94 |
| **E** | **VISReg scale+shape, no-center** | 24.06 | **23.26** | **42.27** |

### 7.2 权重面 sweep（E 基础上扫 scale:shape，均 no-center）

| run | scale:shape | COCO i2t | IN-1k top1 |
|-----|-------------|----------|-----------|
| E    | 1:1 | 24.06 | 23.26 |
| s2sh1 | 2:1 | 23.32 | 23.29 |
| s1sh2 | 1:2 | 23.66 | 23.38 |

### 7.3 正则强度 sweep（固定 E 配方，扫 --sigreg-weight）

| run | weight | COCO i2t | IN-1k top1 |
|-----|--------|----------|-----------|
| w0.5× | 9.15e-5 | 23.88 | 23.23 |
| E (1×) | 1.83e-4 | 24.06 | 23.26 |
| w2×   | 3.66e-4 | 23.84 | 23.31 |

**最优配方 = E：VISReg，scale+shape 等权，去掉 center，weight=1.83e-4。**
相比 SIGReg 锚点：**COCO i2t +1.22，IN-1k top1 +2.03，top5 +1.56**。

## 八、三个旋钮的作用排序（本场景）

三个可调维度全部实测，重要性排序：

1. **center 去留（最大收益）**：B(含center) → E(去center) 同为 1:1，仅去掉 center 项，
   COCO 23.74→24.06、**IN-1k 21.48→23.26（+1.78）**。center 项在 CLIP 场景有害。
   原因：backbone 已被对比损失约束，batch 均值本不飘，center 反而干扰。
2. **scale:shape 配比（1:1 最优，偏置伤检索）**：2:1 / 1:2 的 IN-1k 与 1:1 几乎无差
   （23.26/23.29/23.38，噪声级），但 COCO 明显更低（23.3/23.7 vs 24.06）。等权即最优。
3. **正则总强度（不敏感）**：0.5×–2× 区间 COCO 24.06→23.84/23.88、IN-1k 全在 23.2–23.3。
   落在平坦最优区——**梯度匹配标定法（第六节）选对了量级**。

## 九、场景洞见：与论文侧重的差异

论文强调 shape(SWD) 是核心创新（给纯 JEPA 自监督用，嵌入自由、易坍缩，需 shape
强约束整个分布形状）。我们的 CLIP+SigLIP 场景不同：backbone 已被对比损失约束，坍缩
风险低。实测揭示两个指标各有偏好：

- **shape(SWD) 项 → 检索**：D(shape-only) COCO 最高 24.64（拉开分布、对齐几何利于跨模态检索）
- **scale(方差) 项 → 分类**：C(scale-only) IN-1k 最高 22.60（各向同性/判别性利于线性可分）
- 二者优化方向不同，**等权相加(B/E) 取平衡**；去掉 center 后 E 同时逼近两者峰值，成为
  最佳综合配方。

结论修正：早期"scale 是唯一主力、shape 拖累"的猜测被 D 推翻——**shape 项独立有效**
（检索最强），只是与 scale 侧重不同任务。VISReg 三项中 **center 才是该去掉的那个**。

## 十、结论

- **理论优势**：VISReg 用"解耦的方差 + Sliced-Wasserstein 形状"替代 SIGReg 的单一
  频域检验，核心红利是**坍缩时梯度不消失**（scale 项常数恢复力）+ scale/shape 可分调；
  复杂度线性、跨卡切片免费扩展。
- **实测优势（最优配方 E）**：cc3m-tsv 冠军配方下全面胜 SIGReg——
  **COCO i2t 22.84→24.06（+1.22），IN-1k top1 21.23→23.26（+2.03），top5 +1.56**。
  且 24.06 反超历史 cc3m-wds 上 SIGReg 的 23.44（尽管本次用更劣的 tsv 顺序遍历管线）。
- **工程建议**：CLIP 训练用 VISReg 替代 SIGReg 时，采用 **scale+shape 等权 + 去 center**；
  weight 用梯度匹配标定（~1.83e-4），无需再扫强度。
- **待验证方向**：把 E 配方搬回 cc3m-wds+resampled 强数据管线，看能否把 24.06 再顶高。

## 相关

- [[cc3m_text_dedup]] — 冠军配方 proj_s15_sigreg 与 cc3m 数据管线来源
- [[muon_sigreg_finetune]] — SIGReg 在小数据微调上的负面效应分析
- [[mgap_06_projective_siglip]] — projective neg-mode 配置来源
- 设计 spec：`docs/superpowers/specs/2026-07-23-visreg-integration-design.md`

