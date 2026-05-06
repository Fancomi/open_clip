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
# txt-only within-modal repulsion  (within_modal_sides=txt)
# ════════════════════════════════════════════════════════════════════════════
# run "img550" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 5.0  --within-modal-sides img"
# run "txt550" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 5.0 --within-modal-sides txt"

# run "img750" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 7.5  --within-modal-sides img"
# run "txt750" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 7.5 --within-modal-sides txt"

# run "img250" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 2.5  --within-modal-sides img"
# run "txt250" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 2.5 --within-modal-sides txt"

run "img2000" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 20.0  --within-modal-sides img"
run "txt2000" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 20.0 --within-modal-sides txt"

run "img1500" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 15.0  --within-modal-sides img"
run "txt1500" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 15.0 --within-modal-sides txt"

run "img1000" "PE-Core-B-16-dinov3" 29533 "${SIGREG_BASE} --within-modal-weight 10.0  --within-modal-sides img"
run "txt1000" "PE-Core-B-16-dinov3" 29537 "${SIGREG_BASE} --within-modal-weight 10.0 --within-modal-sides txt"

# ════════════════════════════════════════════════════════════════════════════
# both-sides within-modal repulsion  (within_modal_sides=both)
# ════════════════════════════════════════════════════════════════════════════
run "wm15"  "PE-Core-B-16-dinov3" 29545 "${SIGREG_BASE} --within-modal-weight 1.5"
run "wm2"   "PE-Core-B-16-dinov3" 29546 "${SIGREG_BASE} --within-modal-weight 2.0"
run "wm025" "PE-Core-B-16-dinov3" 29542 "${SIGREG_BASE} --within-modal-weight 0.25"
run "wm075" "PE-Core-B-16-dinov3" 29543 "${SIGREG_BASE} --within-modal-weight 0.75"

echo "======== wm_coco all done ========"
