# VISReg 集成到 open_clip：以冠军配方在 cc3m-tsv 上对比 SIGReg

日期：2026-07-23
状态：设计已批准，进入实现

## 1. 目标

把 **VISReg**（Variance-Invariance-Sketching Regularization, arXiv 2606.02572,
参考实现 https://github.com/HaiyuWu/visreg）作为一个**可切换的正则器**接入本工程，
使其能在**历史冠军配方**下与 **SIGReg**（LeJEPA, 2511.08544）做严格 A/B 对比。
二者是**替代关系**：冠军配方其余部分一字不改，只替换正则项。

主指标：COCO Karpathy 5k 图文互检 I→T / T→I R@1。

## 2. 冠军配方（对照锚点）

来源：`scripts/train/wds_cc3m.sh` + `analysis/research/cc3m_text_dedup.md`。
历史 no-dino 最优 run `proj_s15_sigreg`：**I→T R@1 = 0.2344 @ep8**。

```
模型   : PE-Core-B-16-dinov3
loss   : --siglip --neg-mode projective --init-logit-scale ln(15)
         --sigreg-target cls --sigreg-weight 1e-4
优化器 : --opt muon --muon-lr 0.01,  --lr 3.4e-4  (GlobalBS=4096=8×512 参考点)
硬件   : 8 GPU × bs512 = GlobalBS 4096
调度   : epochs 10, warmup 512, amp_bf16, grad-checkpointing
val    : COCO karpathy_5cap.tsv, R@1/5/10
```

这条路径不带 `--dinov3`，走 `factory.py` 的 `SIGRegContrastiveLoss` 分支 + `main.py`
的 `CLIPLeJEPA` 包装。正则项 = 对 `cls` raw backbone token `[B, backbone_dim]`（unnorm）
调用 `SIGReg(x)`。**替代点唯一且干净**。

> 数据口径提醒：历史 23.44 在 cc3m-wds + resampled 上取得；本次在 **cc3m-tsv**
> （csv loader，顺序遍历，2,894,192 样本）跑。A 组在 tsv 上未必精确 =23.44，
> 23.44 作参考锚点，不作硬复现目标。A/B 在同一 tsv、同配方下对比才是"替代关系"的公平判据。

## 3. VISReg 算法（照官方实现移植）

输入：unnorm 特征 `z ∈ [N, D]`（与 SIGReg 同一入口）。

- **center**：`L_center = ‖mean(z)‖²`
- **scale**：`z_c = z - μ; std = ‖z_c‖_col / √N + ε; L_scale = mean((std - 1)²)`
- **shape (SWD)**：`z_norm = z_c / sg(std)`；K 个随机单位切片 `W∈[D,K]`；
  `p = sort(z_norm @ W, dim=0)`；目标 `q = Φ⁻¹((i)/(N+1))`（erfinv 实现）；
  `L_shape = mean((p - q)²)`
- 合成：`L_reg = λ_scale·L_scale + λ_shape·L_shape + λ_center·L_center`（默认全 1.0）

`sg(·)` = stop-gradient，解耦 shape 与 scale（论文核心）。

### 分布式（关键，忠实官方）
官方在训练循环中先 **all-gather 全局 batch**，每卡用**各自独立的随机切片**计算 loss，
DDP 平均梯度 ⇒ 等价 `K×M` 切片（论文 §3.2）。因此：
- VISReg.forward 内部 grad-aware all-gather `z` 到全局 batch（`torch.distributed.nn.all_gather`）
- **不**做 seed 同步、**不** all-reduce 统计量（与 SIGReg 的 all-reduce-means 机制不同，
  因为 shape 项含 sort，无法对均值做 reduce）
- 目标分位数按全局 N（=4096）生成

## 4. 实现改动

### 4.1 `src/open_clip/loss.py` — 新增 `VISReg(nn.Module)`
- `__init__(num_slices, lambda_scale, lambda_shape, lambda_center, gather)`
- `forward(x: [N,D]) -> scalar`，逻辑如 §3；内部按需 all-gather
- 复用文件内已有 `_dist_world_size` / distributed helpers

### 4.2 `src/open_clip_train/params.py` — 新增参数
- `--reg-method {sigreg,visreg}`，默认 `sigreg`（现有 baseline 零改动）
- `--visreg-lambda-scale`（默认 1.0）
- `--visreg-lambda-shape`（默认 1.0）
- `--visreg-lambda-center`（默认 1.0）
- 复用 `--sigreg-slices`（VISReg 的 K）与 `--sigreg-weight`（外层 λ）

### 4.3 `src/open_clip/loss.py` + `factory.py` — 接线
`SIGRegContrastiveLoss.__init__` 与 `CLIPWithDINOLoss.__init__` 增加 `reg_method` 及
`visreg_lambda_*`，内部据此实例化 `SIGReg` 或 `VISReg`，赋给同一 `self.sigreg` 属性，
**forward 调用点不变**（`self.sigreg(f)`）。`factory.py` 从 `args` 透传新参数。

> 权重语义差异：论文指出 VISReg batch-invariant（不像 SIGReg 乘 N×world_size），
> 故裸损失量级差很多，`--sigreg-weight` 复用但取值需重新标定（见 §5）。

## 5. 权重标定（先匹配，再扫最优）

1. `scripts/tools/calib_visreg_weight.py`：加载冠军配方模型，跑 ~30 step，
   分别在 SIGReg / VISReg 下测**正则项对 backbone 的梯度范数**。
2. 反推 VISReg 的 `--sigreg-weight`，使其梯度量级 ≈ 冠军 SIGReg(1e-4)。
3. 拿匹配点 `w*` 后，B 组用 `w*`；并围绕它扫 `{0.5w*, w*, 2w*}` 挑最优（若算力允许）。

## 6. 实验矩阵（cc3m-tsv，冠军配方，只改正则项）

`scripts/train/visreg_cc3m.sh`：`--dataset-type csv --csv-separator $'\t'`，
`--train-data .../cc3m-tsv/annotations/clip_train.tsv`，其余 = 冠军配方。

| run | 正则 | 参数 |
|-----|------|------|
| A | SIGReg | `--reg-method sigreg --sigreg-weight 1e-4`（复现锚点）|
| B | VISReg 全项 | `--reg-method visreg --sigreg-weight w*`（λ 全 1）|
| C | VISReg scale-only | `--visreg-lambda-shape 0 --visreg-lambda-center 0` |
| D | VISReg shape-only | `--visreg-lambda-scale 0 --visreg-lambda-center 0` |
| E | VISReg 无 center | `--visreg-lambda-center 0` |

## 7. 验证

1. **smoke**（`scripts/train/smoke_visreg.sh`）：几十 step 跑通 SIGReg / VISReg 两条 loss
   路径 + 强制一次 COCO eval，确认无 NaN、loss 下降、分布式 gather 正常。
2. smoke 通过后正式投递 A–E。

## 8. 不做（YAGNI）

- 不移植官方的 multicrop / DINO probe / ImageNet 数据管线——只取正则器。
- 不改 SIGReg 现有行为、不改冠军配方其余超参。
- 不引入 VICReg / BarlowTwins / SWD 独立项（如需可后续加，接口已可扩展）。
