# Scaling Law 分析

*最后更新: 2026-05-18 | 配方: SigLIP + Projective + Muon + SIGReg 1e-4*

---

## 1. 数据点

| Dataset | Unique Samples | Epochs | Total Seen | i2t R@1 | t2i R@1 |
|---------|---------------|--------|-----------|---------|---------|
| COCO | 82K | 20 | 1.66M | 0.0200 | 0.0151 |
| CC3M | 2.9M | 10 | 29M | 0.2278 | 0.1602 |
| CC12M | 11M | 10 | 110M | 0.3958 | 0.2910 |

评估集: COCO Karpathy 5K val (5cap)，模型: PE-Core-B-16-dinov3 (random init)

---

## 2. 拟合模型

### Log-linear (3 点拟合)

```
i2t R@1 = -1.234 + 0.200 * log10(total_seen)
t2i R@1 = -0.901 + 0.146 * log10(total_seen)
```

含义：**每增加一个数量级的数据，i2t +0.20, t2i +0.15**

残差: ±0.02-0.03（相对误差 ~10%），R² ≈ 0.9

---

## 3. 外推预测

| 规模 | Total Seen | i2t 预测 (log-linear) | t2i 预测 (log-linear) |
|------|-----------|----------------------|----------------------|
| CC12M 10ep (实测) | 110M | 0.37 (实际 0.40) | 0.27 (实际 0.29) |
| 400M × 1ep | 400M | 0.49 | 0.35 |
| 400M × 10ep | 4B | 0.69 | 0.50 |
| 2B × 3ep | 6B | 0.72 | 0.53 |
| 12B × 1ep | 12B | 0.78 | 0.57 |
| 12B × 3ep | 36B | 0.88 | 0.64 |

### 校准修正（考虑饱和效应）

Log-linear 假设无饱和，实际 R@1 > 0.6 后增速放缓。参考公开模型校准：

| 规模 | i2t 校准预测 | t2i 校准预测 | 公开模型对标 |
|------|------------|------------|------------|
| 400M × 10ep (~4B seen) | 0.55 - 0.62 | 0.40 - 0.48 | ≈ OpenAI CLIP ViT-B/16 (0.58) |
| 2B × 3ep (~6B seen) | 0.58 - 0.65 | 0.42 - 0.50 | ≈ OpenCLIP ViT-B/16 (0.62) |
| 12B × 3ep (~36B seen) | 0.65 - 0.72 | 0.48 - 0.55 | ≈ SigLIP ViT-B/16 (0.65) |

---

## 4. 参考：公开模型 COCO 5K i2t R@1

| 模型 | 数据 | i2t R@1 |
|------|------|---------|
| OpenAI CLIP ViT-B/16 | WIT-400M, 32ep | ~0.58 |
| OpenCLIP ViT-B/16 | LAION-2B, 32ep | ~0.62 |
| SigLIP ViT-B/16 | WebLI-10B | ~0.65 |
| TIPS/PE-Core ViT-B/16 | proprietary | ~0.68 |

---

## 5. 局限性与注意事项

1. **仅 3 个数据点**：log-linear 拟合对外推 2 个数量级以上的预测不可靠
2. **数据质量差异未建模**：COCO(精标) > CC3M(网络对) > CC12M(噪声大)，质量下降会拖低实际表现
3. **COCO 用 20ep 其余 10ep**：非完全统一条件，COCO 点可能偏高
4. **模型规模固定**：所有实验用 ViT-B/16 (~86M params)，更大模型可能有不同 scaling 斜率
5. **饱和效应**：R@1 接近 0.7+ 后通常亚线性增长（log-linear 会高估）
6. **Epoch 效率递减**：数据重复看的边际收益递减（12B×1ep vs 400M×30ep 效果不同）

---

## 6. 结论

- 当前配方在 CC12M (110M seen) 达到 i2t=0.40，处于 scaling 曲线的陡峭区间
- 如果 scaling 趋势保持，400M 数据 10ep 应能达到 **i2t ≈ 0.55-0.62**，接近 OpenAI CLIP 水平
- **数据是当前最大瓶颈**：同一配方从 CC3M→CC12M 提升了 +74%（0.23→0.40），远超任何超参调整

---

## 附录 A. 实验脚本配置存档

> 归档已删除的三个实验脚本（`experiments/modality_gap.sh`、`mgap_scale_bias.sh`、`wm_coco.sh`）的必要复现配置，仅保留活跃 run 与关键结论；完整历史 run 见 git log。

### A.0 公共环境与路径

```bash
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"; export TZ='Asia/Shanghai'

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"; PROBE_TSV="${COCO}/karpathy_1cap.tsv"
CC3M_TRAIN="/dev/shm/cc3m_wds/{00000..00280}.tar"; CC3M_N_TRAIN=2857622  # 先 cp 到 /dev/shm

GPUS=8; GlobalBS=$((512 * GPUS))                       # = 4096
LR=$(python3 -c "import math; print(3.4e-4 * math.sqrt(${GlobalBS}/4096))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(${GlobalBS}/4096))")

BASE="--precision amp_bf16 --workers 32 --batch-size 512 --lr ${LR} \
    --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 --delete-previous-checkpoint"
SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"
```

模型统一 `PE-Core-B-16-dinov3`，入口 `torchrun --nproc_per_node=8 -m open_clip_train.main`。

### A.1 `modality_gap.sh`（CC3M，10 epoch，正式实验）

矩阵：Step 0 post-processing 分析（`analysis/modality_gap.py`，不训练）；Step 1/2 img/txt-only within-modal repulsion，λ∈{0.25,0.5,0.75,1.0,1.5,2.0,2.5,5.0,7.5}；Step 3 both-sides，λ∈{0.25,0.75,1.0,1.5,2.0}；参考区 baseline + gap loss λ∈{0.001,0.005,0.01,0.05}。当前仅 baseline 活跃。

```bash
torchrun --nproc_per_node=8 --master_port=29555 -m open_clip_train.main \
    --model PE-Core-B-16-dinov3 --train-data "${CC3M_TRAIN}" --val-data "${VAL}" \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
    --warmup 512 --epochs 10 ${BASE} ${SIGREG_BASE} --name "mgap_baseline_${TS}" < /dev/null

# within-modal 模板: ... ${SIGREG_BASE} --within-modal-weight <λ> [--within-modal-sides img|txt]
# gap loss 模板:     ... ${SIGREG_BASE} --modality-gap-weight <λ>
```

**结论**：CC3M gap loss 最优 λ=0.005（+0.92% i2t R@1 vs baseline）；λ=0.05 过强损害对齐。

### A.2 `mgap_scale_bias.sh`（CC3M，1 epoch，logit scale/bias 消融）

新增 CLI：`--init-logit-scale`（log-space）、`--init-logit-bias`、`--freeze-logit-params`。固定 1 epoch，warmup 512，其余同 A.0。run 模板在 `${BASE} ${SIGREG_BASE}` 后追加 `${EXTRA}`，`--epochs 1`。

```bash
# A. Bias sweep (scale=ln10 默认): bias ∈ {-5,-8,-10(baseline),-12,-15,-20}
#    EXTRA="--init-logit-bias <b>"
# B. Scale sweep (bias=-10 默认): --init-logit-scale {1.6094=ln5, 2.9957=ln20, 3.9120=ln50}
# C. Freeze: --freeze-logit-params [× default | (ln20,-15) | (ln5,-5)]
# D. Cross combo: (ln20,-15), (ln5,-5)
# smoke 模式 (bash mgap_scale_bias.sh smoke): 单卡 synthetic 64 样本 BS=8 1ep，不用 muon
```

### A.3 `wm_coco.sh`（COCO quick，20 epoch，设计消融迭代）

COCO 平台：`--train-data ${COCO}/clip_train_dedup.tsv`（~82K），`--dataset-type csv`，`--epochs 20 --warmup 42`，save/val frequency=2。当前活跃 21 runs：

```bash
MUON_BASE="--siglip --sigreg-target cls --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

# Round 4 — SIGReg 关联消融 (12 runs, MUON_BASE 不带默认 sigreg-weight)
#  A. weight sweep:  ${MUON_BASE} --sigreg-weight {0,1e-5,1e-4,1e-3,1e-2}
#  B. 无 SIGReg+单一正则: --sigreg-weight 0 + {--koleo-weight 0.05 | --uniformity-weight 0.5 |
#       --modality-gap-weight 0.001 | --koleo-weight 0.03 --uniformity-weight 0.3}
#  C. KoLeo×SIGReg: --koleo-weight 0.05 --sigreg-weight {1e-5,1e-3,1e-2}
# Antipodal/Orthogonal/Projective (各 3 runs): ${SIGREG_BASE} --neg-mode {antipodal|orthogonal|projective}
#   × {纯 | --koleo-weight 0.05 | --uniformity-weight 0.5}
```

**结论**：COCO baseline i2t=0.0168；Round 1 Top-3：koleo0.05（+15.1% i2t/-6.4% t2i）、gap0.001（+15.1%/-2.9%）、uni0.5（+10.5%/-2.9%）。有效区间：KoLeo [0.005,1.0]、Uniformity [0.5,1.0]、Gap Loss 仅 0.001。σ 饱和：bias=-10 时 λ=5 的 within-modal 贡献仅 ~0.03% of main loss。

---

*文档版本: 2026-05-18 v1*
