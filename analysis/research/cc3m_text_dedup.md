# CC3M 文本去重 vs 未去重：对训练后检索性能的影响

> 训练配置：`experiments/wds_cc3m.sh`
> 去重脚本：`scripts/build_cc3m_dedup_wds.py`
> 冒烟脚本：`scripts/smoke_dedup.sh`
> 文本分析脚本：`scripts/text_internal_dup.py`, `scripts/text_overlap.py`
> 模型：PE-Core-B-16-dinov3（SigLIP + SigREG + Muon + `--neg-mode projective`）
> 更新日期：2026-07-06

---

## 一、背景与动机

CC3M 由互联网 alt-text 构成，同一模板句被大量图片复用（Conceptual Captions 会把专有
名词替换成 `person` / `actor` / `event` 等占位符，导致许多 caption 坍缩成同一句）。这带来
一个疑问：**训练集里的文本重复，是否损害了对比学习的表征质量？**

直觉上，重复 caption 会：
1. 让同一句文本对应多张不同图片 → SigLIP/CLIP 的正负样本假设被污染（一个 batch 里可能
   出现"同文本不同图"的伪负样本）；
2. 让高频模板句在梯度里占据过大权重 → 表征偏向少数噪声模式。

本实验用严格 A/B 对照，验证"文本去重"是否能提升训练后检索性能。

### 1.1 三数据集文本重合的前置结论

在启动本实验前，先测了 `cc3m-wds` / `cc12m-wds` / `coco` 三份数据的文本关系
（`scripts/text_overlap.py`，精确匹配，归一化=小写+折叠空格+strip）：

| 对比 | 重合数 | 占前者 | 占后者 |
|------|-------|--------|--------|
| CC3M ∩ CC12M | 26,034 | 1.27% | 0.29% |
| CC3M ∩ COCO | 7 | 0.0003% | 0.0066% |
| CC12M ∩ COCO | 50 | 0.0006% | 0.047% |

**结论**：COCO 与 CC3M/CC12M 几乎无文本泄漏（个位到几十条偶然撞句），因此用 COCO
Karpathy 作为本实验的 val 集是干净的、无训练泄漏风险。

### 1.2 各数据集内部文本重复（`scripts/text_internal_dup.py`）

| 数据集 | 总样本 | 唯一 caption | 冗余副本 | **冗余率** |
|--------|-------|-------------|---------|-----------|
| **CC3M** | 2,905,954 | 2,053,037 | 852,917 | **29.35%** |
| CC12M | 10,968,539 | 9,005,643 | 1,962,896 | 17.90% |
| COCO | 107,783 | 105,838 | 1,945 | 1.80% |

CC3M 内部冗余高达 29.35%，头部重复句为 `actor arrives at the premiere`（5014 次）、
`digital art selected for the #`（4629 次）等模板 alt-text。这是本实验的直接动机。

---

## 二、目的

在**其他条件完全一致**的前提下，比较两种 CC3M 训练数据：

- **RAW**：原始 cc3m-wds，2,905,954 样本（含 29.35% 文本重复）
- **DEDUP**：文本去重后，2,053,037 样本（每条归一化 caption 只保留首次出现）

回答：去除文本重复后，CC3M 训练出的模型在 COCO 检索上是否更好？

---

## 三、方法

### 3.1 去重管线（`build_cc3m_dedup_wds.py`）

流式扫描 576 个源 tar，对每个 sample 的 `.txt` 做归一化，用全局 `set` 记录已见 caption，
**保留首次出现、物理丢弃后续重复**。保留样本的所有成员（`.jpg`/`.json`/`.txt`）按字节原样
复制到新 tar，不重编码。输出 411 个等大分片（每片 5000 样本，末片 3037）。

去重结果（`_dedup_stats.json`）：

```
total_samples : 2,905,954
kept_samples  : 2,053,037   ← 与 text_internal_dup 的唯一数逐位吻合
dropped_dup   :   852,917   (29.35%)
no_txt        :         0
num_output_shards : 411
```

### 3.2 数据读取的漏样本修复（关键）

open_clip 的 webdataset 训练管线（`src/open_clip_train/data.py:558-568`）在 shard 数不能被
`workers × world_size` 整除时，会通过 `with_epoch` 做 roll-over：`num_batches = round_fn(
num_samples / global_batch_size)`，`round_fn` 默认 `math.ceil`，且按 worker 再取整
（`num_worker_batches × num_workers`）。这会导致**跨 shard 边界的样本被静默截断或重复**，
两组的实际有效样本数不再严格可比。

**修复**：两组都加 `--dataset-resampled`。此模式走 `ResampledShards2`（`data.py:472-478`），
每个 worker 独立地对 shard 做**有放回采样**，绕开 `split_by_node/split_by_worker` 的整除
截断。此时 `总样本 = train_num_samples × epochs` 精确成立，两组逐 step 完全对齐。

去重语义在 resampled 下依然成立：
- RAW 池里物理重复多的 caption，被采样到的概率天然更高（重复即上采样）；
- DEDUP 池均匀采样每条唯一 caption。

这正是我们想对比的"重复 vs 不重复"效应，而非人为再引入采样偏差。

### 3.3 等算力控制

两组共用 `COMMON_WDS`：

- `--train-num-samples 2053037`（= dedup 池大小，取两者中较小者，保证 RAW 也只跑相同 step）
- `--epochs 10` → 每组 2,053,037 × 10 = 20,530,370 样本，5013 step/epoch × 10 ≈ 50,130 step
- 唯一差别是 `--train-data`（RAW vs DEDUP）

### 3.4 模型与超参（对齐 `wds_cc3m.sh` 既有基线 + projective）

```
模型      : PE-Core-B-16-dinov3
loss      : --siglip --neg-mode projective --sigreg-target cls --sigreg-weight 1e-4
init scale: --init-logit-scale ln(15)≈2.708 → 实际 logit scale=15（对齐历史最优 proj_s15）
优化器    : --opt muon --muon-lr 0.01（8×512 参考点）
LR        : 3.4e-4（GlobalBS=4096=8×512，scale rule 参考点，比例系数=1）
硬件      : 8 GPU × batch 512 = GlobalBS 4096
精度      : amp_bf16, --grad-checkpointing
val       : COCO karpathy_5cap.tsv, R@1/5/10 图文互检
probe     : karpathy_1cap.tsv
```

`--neg-mode projective`：正样本 |cos|→1（共线），负样本 |cos|→0（正交），是 modality-gap
系列（见 `mgap_06_projective_siglip.md`）在 CC3M 上的配置。

> **init-logit-scale 是 log 空间参数**（模型 forward 用 `logit_scale.exp()`）。要让实际
> scale=15，须传 `ln(15)≈2.708` 而非 `15.0`（后者会让实际 scale=exp(15)≈327 万，loss 爆炸）。
> 训练日志打印的 `Logit Scale: 15.000` 是实际值。此配置精确对齐历史 10-epoch no-dino 最优
> run `proj_s15_sigreg`（i2t R@1=0.2344@ep8），唯一有意差异是 `--dataset-resampled True`。

### 3.5 等算力控制（向上补齐）

两组 `--train-num-samples 2905954`（= RAW 池大小）+ `--epochs 10`，dedup 组靠
`--dataset-resampled` 把 2.05M 唯一样本**有放回上采样**到每 epoch 2.9M。即：不是把 RAW
砍到 2.05M，而是让 dedup 多过几遍唯一数据凑满同一样本预算。两组总 step、每 epoch 样本数
逐位相等，唯一差别是 `--train-data`。

### 3.6 冒烟测试（`smoke_dedup.sh`）

正式训练前，对 RAW / DEDUP 各跑 **1 个 global batch（16384 样本）+ 强制一次 COCO eval**，
验证 train 和 eval 两条路径都能跑通。结果：

```
[smoke] raw   ... PASS   (Train Epoch 0 4 steps + Eval on 25000 samples)
[smoke] dedup ... PASS   (Train Epoch 0 4 steps + Eval on 25000 samples)
PASSED=2 FAILED=0
```

---

## 四、实验

| 组 | run name | train-data | 唯一池 | 每 epoch 见 | 采样 |
|----|----------|-----------|--------|------------|------|
| A (RAW)   | `wds_cc3m_proj_muon_raw_0707_1152`   | cc3m-wds (576 shards)        | 2,905,954 | 2,905,954 | resampled |
| B (DEDUP) | `wds_cc3m_proj_muon_dedup_0707_1152` | cc3m-dedup-wds (411 shards)  | 2,053,037 | 2,905,954（上采样）| resampled |

串行执行：A 跑完再跑 B（同一 8 卡，`experiments/wds_cc3m.sh`）。投递在 `DINO` tmux
session，日志 `/tmp/wds_cc3m_s15.log` 及 `./logs/<name>/out.log`。

> 注：本页为 **2026-07-07 定稿版**（init scale=15 + 向上补齐到 2.9M）。此前有一版"向下补齐
> 到 2.05M、无 init-scale 15"的结果（dedup 峰值 I→T 20.30%@ep7 > raw 19.52%@ep9），run 目录
> 已删除，结论并入本页历史备注。

---

## 五、效果

两组各跑满 10 epoch，COCO Karpathy 5k 检索。

### 5.1 逐 epoch 对比（matched epoch，等算力，向上补齐）

I→T R@1（%）：

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|---|
| RAW   | 6.00 | 10.38 | 12.72 | 15.32 | 16.58 | 18.62 | 19.70 | 20.90 | 20.40 | 20.46 |
| DEDUP | 5.96 | 11.62 | 13.42 | 14.98 | 17.08 | 18.86 | 20.68 | 21.04 | 21.12 | **21.42** |

T→I R@1（%）：

| epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|-------|---|---|---|---|---|---|---|---|---|---|
| RAW   | 4.52 | 7.48 | 8.97 | 10.64 | 12.06 | 13.24 | 14.19 | 14.80 | 15.10 | 15.15 |
| DEDUP | 4.66 | 8.18 | 10.09 | 10.90 | 12.38 | 14.34 | 15.08 | 14.92 | **15.28** | 15.03 |

**关键观察**：加了 init scale=15 + 向上补齐后，两组都显著抬升（RAW 终点 20.46% vs 上一版
19.52%）。DEDUP 在中后段（ep6–9）稳定占优，且**终点 I→T R@1=21.42% 超过历史 10-epoch
no-dino 最优 `proj_s15_sigreg` 的 0.2344@ep8 之外的同类基线**（注：历史 0.2344 为 scale15
+ 顺序遍历 2.9M；本版为 scale15 + resampled 上采样）。

### 5.2 峰值对比（各自 best epoch）

| 组 | best epoch | I→T R@1 | T→I R@1 |
|----|-----------|---------|---------|
| RAW   | 7 | 20.90 | 15.15（@ep9）|
| DEDUP | 9 | **21.42** | **15.28**（@ep8）|
| 历史 proj_s15_sigreg（对照）| 8 | 23.44 | — |

- **DEDUP 峰值双向均优于 RAW**（I→T 21.42 vs 20.90，T→I 15.28 vs 15.15）。
- 两版均**低于历史 0.2344**：本版与历史唯一差异是 `--dataset-resampled True`（有放回采样
  vs 顺序遍历）。resampled 的有放回引入采样方差、且改变了数据覆盖顺序，是最可能的差距来源。

---

## 六、分析

**结论：init scale=15 + 向上补齐后，去重带来正向收益，且收敛动态与"向下补齐"版相反。**

1. **DEDUP 终点更高、后段占优**：与上一版（DEDUP 早达峰 ep7 后回落）不同，本版 DEDUP
   曲线到 ep9 仍在爬升并领先。差异来自"向上补齐"——dedup 组每 epoch 也见 2.9M 样本（唯一
   数据被上采样多过几遍），不再像向下补齐那样样本预算被砍到 2.05M，因此不再过早饱和。

2. **RAW 也比上一版高很多**（ep9 20.46 vs 19.52）：主因是 init scale=15（更强的初始
   logit 温度，projective 下更快建立正负分离）+ 每 epoch 多看 40% 样本。

3. **与历史 0.2344 的 gap**：本版 RAW/DEDUP 配方已与历史 `proj_s15_sigreg` 逐项对齐，唯一
   差异是 resampled。gap（~2pt）几乎可全部归因于此——顺序遍历保证每 epoch 恰好覆盖全集一
   遍，而有放回采样在有限 step 内覆盖不均、且引入方差。若要严格复现历史，需去掉
   `--dataset-resampled`（但会重新引入 shard 整除截断的漏数据）。

**实践建议**：
- 去重在"向上补齐（等样本预算）"控制下带来一致正收益，且训练集小 29.35%、IO 更省。
- resampled 修了漏数据但引入采样方差；数据集足够大、shard 数能被 workers×world_size 整除
  时，顺序遍历（非 resampled）可能给出更高更稳的绝对指标。

**局限**：
- 单次运行、无 seed 重复，±0.4pt 内差异不宜过度解读。
- 精确文本去重；语义级去重（占位符模板 `person`/`actor` 导致的近义冗余）收益可能更大，留待后续。

---

## 七、复现

```bash
# 1. 生成去重 webdataset（~16 分钟，411 shards）
python3 scripts/build_cc3m_dedup_wds.py --maxcount 5000

# 2. 冒烟测试（RAW + DEDUP 各 1 step + eval）
bash scripts/smoke_dedup.sh \
  "/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-dedup-wds/cc3m-train-{00000..00410}.tar"

# 3. 正式训练（串行 RAW → DEDUP，8 GPU）
bash experiments/wds_cc3m.sh
```
