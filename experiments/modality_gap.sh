#!/bin/bash
# experiments/modality_gap.sh
#
# 模态差异消除实验
# Baseline: pe_dinov3_sigreg_cls_probe (quick.sh)
#
# 实验矩阵:
#   Step 0 — 后处理分析 (纯分析，不训练)
#   Step 1 — --within-modal-sides img  λ 消融 (λ ∈ {0.25, 0.5, 0.75, 1.0})
#   Step 2 — --within-modal-sides txt  λ 消融 (λ ∈ {0.25, 0.5, 0.75, 1.0})
#   Step 3 — --within-modal-sides both λ 消融 (λ ∈ {0.25, 0.75, 1.0, 1.5, 2.0})
#
# 本脚本仅包含训练实验 (Step 1-3)。
# Step 0 后处理分析请使用: analysis/modality_gap.py
#
# 用法:
#   bash experiments/modality_gap.sh
#
# 运行顺序由上到下，任一失败 set -e 中止。
# 可注释掉不需要的行逐步推进。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

CC3M="/dev/shm/cc3m_wds"
CC3M_TRAIN="${CC3M}/{00000..00280}.tar"
CC3M_N_TRAIN=2857622

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 --grad-checkpointing \
    --log-every-n-steps 1 --val-frequency 1 \
    --delete-previous-checkpoint"

COMMON_WDS="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="mgap_${TAG}_${TS}"
    echo "======== [mgap] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON_WDS} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ── CC3M copy ────────────────────────────────────────────────────────────────
if [ ! -d "${CC3M}" ]; then
    echo "[mgap] Loading CC3M to memory ..."
    cp -r "/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds" "${CC3M}"
    echo "[mgap] Done, $(du -sh ${CC3M} | cut -f1)"
else
    echo "[mgap] Found ${CC3M}, skip copy"
fi

# ════════════════════════════════════════════════════════════════════════════
# Step 0: Post-processing analysis (no training)
# 先跑 baseline probe，然后用 analysis/modality_gap.py 分析
# ════════════════════════════════════════════════════════════════════════════
# 用法（在 baseline probe 目录有 step_*.npz 后运行）：
#
#   source /root/.../envs/dino/bin/activate
#   PYTHONPATH=./src python3 analysis/modality_gap.py \
#       --probe logs/cc3m_pe_dinov3_sigreg_cls_probe_<TS>/probe/step_001740.npz \
#       --split proj_features \
#       --out   analysis/research/modality_gap_baseline.json
#
SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} \
    --probe-data ${PROBE_TSV}"


# ════════════════════════════════════════════════════════════════════════════
# Step 1: img-only within-modal repulsion  (within_modal_sides=img)
# Step 2: txt-only within-modal repulsion  (within_modal_sides=txt)
#
#   cross-modal: positive pairs only (diagonal)
#   within-modal: img-img all-negative repulsion only (no txt-txt)
#   within-modal: txt-txt all-negative repulsion only (no img-img)
#
#   Hypothesis: txt tower keeps full cross-modal supervision signal;
#   img tower gets within-modal repulsion → forces img cluster to spread,
#   transitively reduces modality gap without collapsing discrimination.
#
#   λ sweep: 0.25, 0.5, 0.75, 1.0
# ════════════════════════════════════════════════════════════════════════════

# # run "img050" "PE-Core-B-16-dinov3" 29531 "${SIGREG_BASE} --within-modal-weight 0.5  --within-modal-sides img"
# # run "txt050" "PE-Core-B-16-dinov3" 29535 "${SIGREG_BASE} --within-modal-weight 0.5  --within-modal-sides txt"

# # run "img100" "PE-Core-B-16-dinov3" 29532 "${SIGREG_BASE} --within-modal-weight 1.0  --within-modal-sides img"
# # run "txt100" "PE-Core-B-16-dinov3" 29536 "${SIGREG_BASE} --within-modal-weight 1.0  --within-modal-sides txt"

# # run "img150" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 1.5  --within-modal-sides img"
# # run "txt150" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 1.5 --within-modal-sides txt"

# run "img550" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 5.0  --within-modal-sides img"
# run "txt550" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 5.0 --within-modal-sides txt"

# run "img750" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 7.5  --within-modal-sides img"
# run "txt750" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 7.5 --within-modal-sides txt"

# run "img250" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 2.5  --within-modal-sides img"
# run "txt250" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 2.5 --within-modal-sides txt"

# # run "img025" "PE-Core-B-16-dinov3" 29530 "${SIGREG_BASE} --within-modal-weight 0.25 --within-modal-sides img"
# # run "txt025" "PE-Core-B-16-dinov3" 29534 "${SIGREG_BASE} --within-modal-weight 0.25 --within-modal-sides txt"


# ════════════════════════════════════════════════════════════════════════════
# Step 3: Within-modal SigLIP repulsion  (DESIGN v3)
#
#   cross-modal : positive pairs only (diagonal B-vector)
#                 -logsigmoid(s·<img_i,txt_i> + b) / B
#   within-modal: SigLIP_all_neg(img, img) + SigLIP_all_neg(txt, txt)
#                 same logit_scale / logit_bias / sum-B normalisation
#
#   Loss = cross_pos + λ * 0.5 * (SigLIP_wm_img + SigLIP_wm_txt)
#
#   Gradient balance (λ=0.5 → within-modal neg pairs = cross-modal neg pairs):
#     λ=0.25 : within-modal ~0.5× cross-modal neg pressure
#     λ=0.75 : within-modal ~1.5× cross-modal neg pressure
#     λ=1.0  : within-modal ~2×  cross-modal neg pressure
#     λ=1.5  : within-modal ~3×  cross-modal neg pressure
#     λ=2.0  : within-modal ~4×  cross-modal neg pressure
#
#   Active experiments (running):
#     wm025 : λ=0.25
#     wm075 : λ=0.75
#     wm1   : λ=1.0
#     wm15  : λ=1.5
#     wm2   : λ=2.0
# ════════════════════════════════════════════════════════════════════════════
# # run "wm1"   "PE-Core-B-16-dinov3" 29544 "${SIGREG_BASE} --within-modal-weight 1.0"
# run "wm15"  "PE-Core-B-16-dinov3" 29545 "${SIGREG_BASE} --within-modal-weight 1.5"
# run "wm2"   "PE-Core-B-16-dinov3" 29546 "${SIGREG_BASE} --within-modal-weight 2.0"
# run "wm025" "PE-Core-B-16-dinov3" 29542 "${SIGREG_BASE} --within-modal-weight 0.25"
# run "wm075" "PE-Core-B-16-dinov3" 29543 "${SIGREG_BASE} --within-modal-weight 0.75"


# echo "======== modality_gap experiments done ========"


# ════════════════════════════════════════════════════════════════════════════
# 参考实验记录（按需取消注释单独运行，勿直接追加到上方流水线）
# ════════════════════════════════════════════════════════════════════════════

# ── Baseline: 纯 SigLIP + SIGReg，无 gap 干预 ────────────────────────────
# logs/mgap_baseline_<TS>
run "baseline" "PE-Core-B-16-dinov3" 29555 "${SIGREG_BASE}"

# ── Gap loss 消融（batch mean distance loss，梯度流过 batch mean）─────────
# λ=0.001 — 轻微惩罚，对收敛几乎无影响
# run "gap001" "PE-Core-B-16-dinov3" 29551 "${SIGREG_BASE} --modality-gap-weight 0.001"
# λ=0.005 — 历史最佳 (+0.92% i2t R@1 vs baseline)
# run "gap005" "PE-Core-B-16-dinov3" 29552 "${SIGREG_BASE} --modality-gap-weight 0.005"
# λ=0.01
# run "gap01"  "PE-Core-B-16-dinov3" 29553 "${SIGREG_BASE} --modality-gap-weight 0.01"
# λ=0.05 — 过强，损害对齐
# run "gap05"  "PE-Core-B-16-dinov3" 29554 "${SIGREG_BASE} --modality-gap-weight 0.05"
