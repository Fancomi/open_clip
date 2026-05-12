#!/bin/bash
# experiments/wm_coco.sh
#
# Within-modal repulsion 快速验证实验 —— COCO 数据集
#
# 用 COCO 替代 CC3M 做快速迭代：
#   COCO train: ~82K samples, steps/epoch ≈ 20 @ BS4096
#   20 epoch ≈ 400 steps total，一次实验约 10~15 分钟，适合设计消融
#
# 确认设计可行后再用 modality_gap.sh (CC3M) 做正式实验
#
# 用法:
#   bash experiments/wm_coco.sh
#
# 运行顺序由上到下，任一失败 set -e 中止。
# 按需取消注释消融行，baseline 始终保留在最前。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_dedup.tsv"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

# COCO: steps/epoch ≈ 20 (82783/4096), warmup=42 ≈ 2 epoch
BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 2 --grad-checkpointing \
    --log-every-n-steps 2 --val-frequency 2 \
    --delete-previous-checkpoint"

COMMON="--warmup 42 ${BASE} --epochs 20 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="wmc_${TAG}_${TS}"
    echo "======== [wmc] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# SIGREG_BASE: SigLIP + SIGReg(cls) + Muon，与 modality_gap.sh 保持一致
SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} \
    --probe-data ${PROBE_TSV}"

# ════════════════════════════════════════════════════════════════════════════
# Baseline: SigLIP + SIGReg + Muon，无 within-modal
# ════════════════════════════════════════════════════════════════════════════
# run "baseline" "PE-Core-B-16-dinov3" 29520 "${SIGREG_BASE}"

# ════════════════════════════════════════════════════════════════════════════
# img-only within-modal repulsion  (within_modal_sides=img)
# txt-only within-modal repulsion  (within_modal_sides=txt) # Best: wmc_txt2000_0506_1633, 但是有效秩很低
# ════════════════════════════════════════════════════════════════════════════
# run "img550" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 5.0  --within-modal-sides img"
# run "txt550" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 5.0 --within-modal-sides txt"

# run "img750" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 7.5  --within-modal-sides img"
# run "txt750" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 7.5 --within-modal-sides txt"

# run "img250" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 2.5  --within-modal-sides img"
# run "txt250" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 2.5 --within-modal-sides txt"

# run "img2000" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 20.0  --within-modal-sides img"
# run "txt2000" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 20.0 --within-modal-sides txt"

# run "img1500" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 15.0  --within-modal-sides img"
# run "txt1500" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 15.0 --within-modal-sides txt"

# (已完成) replace 模式实验
# run "img1000" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 10.0 --within-modal-sides img"
# run "txt2500" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 25.0 --within-modal-sides txt"

# run "img250" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 2.5 --within-modal-sides img"
# run "txt3000" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 30.0 --within-modal-sides txt"

# ════════════════════════════════════════════════════════════════════════════
# Phase 1: auxiliary 模式 — 保留 full SigLIP + 叠加 within-modal
#
#   L = standard_siglip(img, txt, scale, bias)   [完整 N×N, 含 cross-neg]
#     + λ × within_modal(txt/img, scale, bias)   [额外正则]
#
# 预期：img eff_rank 不崩塌（cross-neg 维持判别信号），txt 更均匀 → gap 缩小
# 注意：由于 sigmoid 饱和（bias=-10），λ=5 的 within-modal 贡献仅 ~0.03% of main loss
#       需要极大 λ 或 Phase 2（解耦 bias）才能产生有效梯度
# ════════════════════════════════════════════════════════════════════════════

# ── txt-only replace 模式，指数搜索峰值 ─────────────────────────────────
#
# 已知：λ=30 (txt3000) = 0.0192，仍在爬坡，尚未饱和
# 策略：指数步距先定数量级（30→60→150→400→1000→3000）
#       找到第一个下降点后，下一批次在该区间二分
#
# 预计运行时间：6 runs × 30min ≈ 3h
# ─────────────────────────────────────────────────────────────────────────

# ── 今晚批次（~12h, ~24 runs） ──────────────────────────────────────────
# # 核心验证：auxiliary txt-only，扫 λ
# run "aux_txt5"    "PE-Core-B-16-dinov3" 29551 "${SIGREG_BASE} --within-modal-weight 5.0   --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux_txt20"   "PE-Core-B-16-dinov3" 29552 "${SIGREG_BASE} --within-modal-weight 20.0  --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux_txt50"   "PE-Core-B-16-dinov3" 29553 "${SIGREG_BASE} --within-modal-weight 50.0  --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux_txt200"  "PE-Core-B-16-dinov3" 29554 "${SIGREG_BASE} --within-modal-weight 200.0 --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux_txt500"  "PE-Core-B-16-dinov3" 29555 "${SIGREG_BASE} --within-modal-weight 500.0 --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux_txt1000" "PE-Core-B-16-dinov3" 29556 "${SIGREG_BASE} --within-modal-weight 1000.0 --within-modal-sides txt --within-modal-mode auxiliary"

# # 对照：auxiliary both-sides
# run "aux_both5"   "PE-Core-B-16-dinov3" 29557 "${SIGREG_BASE} --within-modal-weight 5.0   --within-modal-mode auxiliary"
# run "aux_both50"  "PE-Core-B-16-dinov3" 29558 "${SIGREG_BASE} --within-modal-weight 50.0  --within-modal-mode auxiliary"
# run "aux_both200" "PE-Core-B-16-dinov3" 29559 "${SIGREG_BASE} --within-modal-weight 200.0 --within-modal-mode auxiliary"

# # 对照：auxiliary img-only（验证 img repulsion 在有 cross-neg 时是否还崩）
# run "aux_img50"   "PE-Core-B-16-dinov3" 29560 "${SIGREG_BASE} --within-modal-weight 50.0  --within-modal-sides img --within-modal-mode auxiliary"
# run "aux_img200"  "PE-Core-B-16-dinov3" 29561 "${SIGREG_BASE} --within-modal-weight 200.0 --within-modal-sides img --within-modal-mode auxiliary"

# # baseline（重跑一次确保对齐，可选）
# run "baseline2"   "PE-Core-B-16-dinov3" 29562 "${SIGREG_BASE}"

# ════════════════════════════════════════════════════════════════════════════
# (已完成) Binary search round 1: [30, 60]
#   结论: txt4000(λ=40) i2t=0.0154 异常低谷, txt5000(λ=50) i2t=0.0172 回升
#         λ=40 疑似噪声/不稳定区，需补 λ=45 确认
# ════════════════════════════════════════════════════════════════════════════
# run "txt4000"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 40.0 --within-modal-sides txt"
# run "txt5000"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 50.0 --within-modal-sides txt"

# ════════════════════════════════════════════════════════════════════════════
# (已完成) Round 2: 填充 [30,60] 空洞
#   结论: λ=30 全局峰(0.0192), λ=35 急跌至 0.0158, λ=45-60 宽平台 ~0.016
#         峰型尖窄，跌幅远超噪声，需确认峰是否在 λ=28-30 附近
# ════════════════════════════════════════════════════════════════════════════
# run "txt3500"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 35.0 --within-modal-sides txt"
# run "txt4500"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 45.0 --within-modal-sides txt"
# run "txt5500"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 55.0 --within-modal-sides txt"

# ════════════════════════════════════════════════════════════════════════════
# (已完成) Round 3: 精细扫描 λ ∈ [25, 35]
#   结论: 30>28>32, λ=27 暴跌低于 25 和 28
#         噪声分析: R@1 差异仅 3-5 张图(5K val set), COCO 400 steps 噪声底不可忽视
#         λ=30 是否真实峰值存疑，需复现性验证
# ════════════════════════════════════════════════════════════════════════════
# run "txt2700"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 27.0 --within-modal-sides txt"
# run "txt2800"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 28.0 --within-modal-sides txt"
# run "txt3200"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 32.0 --within-modal-sides txt"

# ════════════════════════════════════════════════════════════════════════════
# (已完成) Round 4: 复现性验证 — λ=30 两次复现一致，COCO 调参到此为止
# ════════════════════════════════════════════════════════════════════════════
# run "txt3000b" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 30.0 --within-modal-sides txt"
# run "txt2800b" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 28.0 --within-modal-sides txt"
# run "txt3200b" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 32.0 --within-modal-sides txt"

# ════════════════════════════════════════════════════════════════════════════
# (已完成) Adaptive 小规模验证（3 runs，已被下方 20-run 全量实验取代）
# ════════════════════════════════════════════════════════════════════════════
# run "ada1"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 1.0  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada05" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 0.5  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada2"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 2.0  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]

# ════════════════════════════════════════════════════════════════════════════
# ★ 20-run 全量一步到位：覆盖所有关键参数空间 (~10h)
#
# 三条探索线：
#   A. Adaptive replace txt  (14 runs): λ ∈ [0.02, 1000]，5 个数量级，确认最优区间
#   B. Auxiliary txt 超高 λ  ( 4 runs): λ ∈ [3k, 200k]，验证 sigmoid 饱和边界
#   C. Baseline 方差估计     ( 2 runs): 确认 COCO 随机性基线
#
# 参照点（已有）：
#   baseline:          i2t=0.0168
#   txt3000 (λ=30):    i2t=0.0192  ← 当前 non-adaptive 最优
#   aux_txt1000:       [已有数据]
# ════════════════════════════════════════════════════════════════════════════

# ── A. Adaptive replace txt：独立可学习 (scale_wm, bias_wm)，init from λ=30 equiv ──
# λ 越小 → cross_pos 权重越大，接近纯正样本对齐
# λ 越大 → wm_txt 权重越大，scale_wm 会自动放大来维持均衡
# 预期：存在一个最优区间，两侧性能下降
# run "ada002"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 0.02  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada005"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 0.05  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada01"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 0.1   --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada02"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 0.2   --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada05"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 0.5   --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada1"    "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 1.0   --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada2"    "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 2.0   --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada5"    "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 5.0   --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada10"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 10.0  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada20"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 20.0  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada50"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 50.0  --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada100"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 100.0 --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada300"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 300.0 --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]
# run "ada1k"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 1000.0 --within-modal-sides txt"  # [REMOVED: --within-modal-adaptive]

# ── B. Auxiliary txt 超高 λ：验证 sigmoid 饱和的真实上限 ────────────────
# 已知 aux_txt1000 有效，更高 λ 是否继续受益？
# run "aux3k"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 3000.0   --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux10k"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 10000.0  --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux50k"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 50000.0  --within-modal-sides txt --within-modal-mode auxiliary"
# run "aux200k" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 200000.0 --within-modal-sides txt --within-modal-mode auxiliary"

# ── C. Baseline 方差估计：量化 COCO 随机波动底线 ─────────────────────────
# run "base2"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE}"
# run "base3"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE}"

# ════════════════════════════════════════════════════════════════════════════
# both-sides within-modal repulsion  (within_modal_sides=both)
# ════════════════════════════════════════════════════════════════════════════
# (已完成) both-sides replace 模式
# run "wm15"  "PE-Core-B-16-dinov3" 29545 "${SIGREG_BASE} --within-modal-weight 1.5"
# run "wm2"   "PE-Core-B-16-dinov3" 29546 "${SIGREG_BASE} --within-modal-weight 2.0"
# run "wm025" "PE-Core-B-16-dinov3" 29542 "${SIGREG_BASE} --within-modal-weight 0.25"
# run "wm075" "PE-Core-B-16-dinov3" 29543 "${SIGREG_BASE} --within-modal-weight 0.75"

# ════════════════════════════════════════════════════════════════════════════
# ★ 表示空间优化实验：Gap Loss + Uniformity + KoLeo
#
# 三条线：
#   D. Gap Loss COCO 扫参       (5 runs): λ ∈ {0.001, 0.005, 0.01, 0.05, 0.1}
#   E. Uniformity Loss 扫权重   (5 runs): w ∈ {0.01, 0.05, 0.1, 0.5, 1.0}, t=2.0
#   F. KoLeo Loss 扫权重        (5 runs): w ∈ {0.01, 0.05, 0.1, 0.5, 1.0}
#
# 预计运行时间：15 runs × ~15min ≈ 4h
# ════════════════════════════════════════════════════════════════════════════

# ── D. Gap Loss：batch mean distance loss ──────────────────────────────────
# CC3M 最优 λ=0.005 (+0.92% i2t R@1)，COCO baseline 无 gap，验证是否仍有 regularizer 收益
# run "gap001"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --modality-gap-weight 0.001"
# run "gap005"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --modality-gap-weight 0.005"
# run "gap01"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --modality-gap-weight 0.01"
# run "gap05"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --modality-gap-weight 0.05"
# run "gap1"    "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --modality-gap-weight 0.1"

# ── E. Uniformity Loss (Wang & Isola 2020)：log(mean(exp(-t*||z_i-z_j||^2))) ──
# 直接优化 hypersphere 均匀性，不依赖 gap 存在与否
# t=2.0 是论文默认值，先固定 t 扫 weight
# run "uni001"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.01"
# run "uni005"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.05"
# run "uni01"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.1"
# run "uni05"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.5"
# run "uni1"    "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 1.0"

# ── F. KoLeo Loss：-log(dist_to_nn).mean()，近邻熵正则 ─────────────────────
# 已在 DINOv3 中验证有效，现在接入 CLIP 路径
# run "koleo001" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.01"
# run "koleo005" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.05"
# run "koleo01"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.1"
# run "koleo05"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.5"
# run "koleo1"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 1.0"

echo "======== wm_coco representation space sweep done (15 runs × ~15min ≈ 4h) ========"

# ════════════════════════════════════════════════════════════════════════════
# ★ Round 2: 系数细调 + 混合实验
#
# 基于 Round 1 结果：
#   Top-3:  koleo005 (w=0.05, +15%), gap001 (λ=0.001, +15%), uni05 (w=0.5, +10.5%)
#   KoLeo 有效区间宽: [0.005, 1.0]
#   Uniformity 最优区间: [0.5, 1.0]
#   Gap Loss 仅 λ=0.001 有效
#
# 策略：
#   G. KoLeo 细调      (5 runs): w ∈ {0.02, 0.03, 0.05, 0.07, 0.1} 细化峰
#   H. Uniformity 细调  (3 runs): w ∈ {0.3, 0.5, 0.7} 确认峰
#   I. 混合实验         (6 runs): 最优组合
#
# 预计运行时间：14 runs × ~15min ≈ 3.5h
# ════════════════════════════════════════════════════════════════════════════

# ── G. KoLeo 细调：确认 w=0.05 是否为真实峰 ─────────────────────────────
# run "koleo002" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.02"
# run "koleo003" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.03"
# run "koleo005b" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.05"
# run "koleo007" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.07"
# run "koleo015" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.15"

# ── H. Uniformity 细调：确认 w=0.5 附近 ──────────────────────────────────
# run "uni03"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.3"
# run "uni05b" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.5"
# run "uni07"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.7"

# ── I. 混合实验：top performers 组合 ─────────────────────────────────────
# I1: KoLeo + Gap (两个 +15% 的组合)
# run "mix_koleo005_gap001" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.05 --modality-gap-weight 0.001"
# I2: KoLeo + Uniformity
# run "mix_koleo005_uni05"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.05 --uniformity-weight 0.5"
# I3: Uniformity + Gap
# run "mix_uni05_gap001"    "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --uniformity-weight 0.5 --modality-gap-weight 0.001"
# I4: 三合一
# run "mix_all3"            "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.05 --uniformity-weight 0.5 --modality-gap-weight 0.001"
# I5: KoLeo + Uniformity (降低 uni 防止冲突)
# run "mix_koleo005_uni03"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.05 --uniformity-weight 0.3"
# I6: KoLeo higher + Uniformity
# run "mix_koleo1_uni05"    "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 1.0 --uniformity-weight 0.5"

echo "======== Round 2: fine-tune + mix done (14 runs × ~15min ≈ 3.5h) ========"

# ════════════════════════════════════════════════════════════════════════════
# ★ Round 3: 小比例混合
#
# Round 2 混合用各自最优权重，总正则化压力翻倍导致过强。
# 现在每个组分减半或更小，保持总正则化压力 ≈ 单一最优。
#
# 参照单一最优：
#   koleo w=0.05   (+15.1% i2t, -6.4% t2i)
#   uni   w=0.5    (+10.5% i2t, -2.9% t2i)
#   gap   λ=0.001  (+15.1% i2t, -2.9% t2i)
# ════════════════════════════════════════════════════════════════════════════

# ── J. KoLeo + Uniformity 减半 ──────────────────────────────────────────
# run "hmix_k025_u025"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.025 --uniformity-weight 0.25"
# run "hmix_k003_u015"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.03  --uniformity-weight 0.15"
# run "hmix_k001_u03"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.01  --uniformity-weight 0.3"

# ── K. KoLeo + Gap 减半 ─────────────────────────────────────────────────
# run "hmix_k025_g0005" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.025 --modality-gap-weight 0.0005"
# run "hmix_k003_g0005" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.03  --modality-gap-weight 0.0005"

# ── L. 三合一 减半/三分 ─────────────────────────────────────────────────
# run "hmix_all_half"   "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.025 --uniformity-weight 0.25 --modality-gap-weight 0.0005"
# run "hmix_all_third"  "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --koleo-weight 0.015 --uniformity-weight 0.15 --modality-gap-weight 0.0003"

echo "======== Round 3: half-weight mix done (7 runs) ========"

# ════════════════════════════════════════════════════════════════════════════
# ★ Round 4: SIGReg 关联实验 — 四者关系全面消融
#
# 核心问题：SIGReg (pre-norm) 与 KoLeo/Uni/Gap (post-norm) 是否冗余？
#
# 已有参照：
#   baseline (sig=1e-4):           i2t=0.0172  t2i=0.0140
#   koleo005 (sig=1e-4, k=0.05):  i2t=0.0198  t2i=0.0131
#
# 实验矩阵：
#   A. SIGReg weight sweep (无 post-norm reg)     5 runs
#   B. post-norm reg 替代 SIGReg (sigreg=0)       4 runs
#   C. KoLeo × SIGReg 交互                        3 runs
#   共 12 runs，~3h
# ════════════════════════════════════════════════════════════════════════════

# sigreg-target=cls 保留（控制模型结构），weight=0 仅禁用 loss
MUON_BASE="--siglip --sigreg-target cls --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

# ── A. SIGReg weight sweep（无 post-norm 正则）────────────────────────────
run "no_reg"    "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 0"
run "sig_1e5"   "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 1e-5"
run "sig_1e4"   "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 1e-4"
run "sig_1e3"   "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 1e-3"
run "sig_1e2"   "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 1e-2"

# ── B. 无 SIGReg + 单一 post-norm 正则 ──────────────────────────────────
run "k005_nosig"     "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 0 --koleo-weight 0.05"
run "u05_nosig"      "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 0 --uniformity-weight 0.5"
run "g001_nosig"     "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 0 --modality-gap-weight 0.001"
run "k003_u03_nosig" "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 0 --koleo-weight 0.03 --uniformity-weight 0.3"

# ── C. KoLeo × SIGReg weight 交互 ───────────────────────────────────────
run "k005_sig1e5" "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 1e-5 --koleo-weight 0.05"
run "k005_sig1e3" "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 1e-3 --koleo-weight 0.05"
run "k005_sig1e2" "PE-Core-B-16-dinov3" 29537 "${MUON_BASE} --sigreg-weight 1e-2 --koleo-weight 0.05"

echo "======== Round 4: SIGReg ablation done (12 runs × ~15min ≈ 3h) ========"

# ════════════════════════════════════════════════════════════════════════════
# ★ Antipodal SigLIP: cos=-1 对齐正样本，cos≠-1 推离负样本
#
# 核心思路：不再对抗模态鸿沟，而是「承认」它——
#   正样本 (img_i, txt_i) 推向超球面完全相反方向 (cos=-1)
#   负样本 (img_i, txt_j) 远离相反方向
#
# 实现：仅取反 SigLIP 余弦相似度（logits = -scale*cos + bias），
#       标签不变，分布式逻辑不变
#
# 预期：
#   - sim_pos 应从 ~0 趋向 -1（而非 +1）
#   - 更均匀的特征分布（每个 text 推远 1 个正样本、容忍 N-1 个负样本）
#   - 模态方向反转，但强度/结构可能不同
# ════════════════════════════════════════════════════════════════════════════

run "antipodal"           "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode antipodal"
run "anti_koleo005"       "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode antipodal --koleo-weight 0.05"
run "anti_uni05"          "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode antipodal --uniformity-weight 0.5"

echo "======== Antipodal SigLIP done (3 runs × ~15min ≈ 45min) ========"

# ════════════════════════════════════════════════════════════════════════════
# ★ Orthogonal SigLIP: 正样本 cos→+1，负样本 cos→0 (正交)
#
# 核心思路：负样本与正样本无关，应正交而非对立
#   正样本 (img_i, txt_i) 推向 cos=+1（标准对齐）
#   负样本 (img_i, txt_j) 推向 cos=0（正交，而非 cos=-1）
#
# 实现：负样本 logits 使用 |cos| 代替 cos（偏离0即受罚）
#
# 预期：
#   - 模态鸿沟消失（负样本无法利用反方向逃逸）
#   - 有效秩提升（正交约束迫使利用更多维度）
#   - 双向提升（i2t/t2i 无 trade-off）
# ════════════════════════════════════════════════════════════════════════════

run "orthogonal"          "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode orthogonal"
run "ortho_koleo005"      "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode orthogonal --koleo-weight 0.05"
run "ortho_uni05"         "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode orthogonal --uniformity-weight 0.5"

echo "======== Orthogonal SigLIP done (3 runs × ~15min ≈ 45min) ========"

# ════════════════════════════════════════════════════════════════════════════
# ★ Projective SigLIP: 共线=相关 (|cos|→1), 正交=无关 (|cos|→0)
#
# 核心思路：射影空间 RP^n —— x 和 -x 视为同一语义
#   正样本 (img_i, txt_i) 推向共线 |cos|=1（±方向皆可）
#   负样本 (img_i, txt_j) 推向正交 |cos|=0
#
# 实现：ALL logits 使用 |cos|（正负样本统一）
#
# 预期：
#   - 双倍容量：同一维度可编码两个方向的语义
#   - 模态鸿沟消失（同 orthogonal 原理）
#   - 信息论优势：D 维空间有效容量翻倍
# ════════════════════════════════════════════════════════════════════════════

run "projective"          "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode projective"
run "proj_koleo005"       "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode projective --koleo-weight 0.05"
run "proj_uni05"          "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --neg-mode projective --uniformity-weight 0.5"

echo "======== Projective SigLIP done (3 runs × ~15min ≈ 45min) ========"
