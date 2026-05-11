#!/bin/bash
# Weighted Multi-Teacher: validate that per-teacher weighting fixes gradient dilution
#
# 配置: 3T (PE + SigLIP2 + DataComp), same as M1, with varying weight ratios
#   W1 = 1.0, 0.2, 0.2  (PE 5:1 ratio)
#   W2 = 1.0, 0.1, 0.1  (PE 10:1 ratio)
#   W3 = 1.0, 0.5, 0.5  (PE 2:1 ratio)

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ── Data ─────────────────────────────────────────────────────────────────────
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"

CC3M_WDS="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds"
CC3M_TRAIN="${CC3M_WDS}/cc3m-train-{0000..0575}.tar"
CC3M_N_TRAIN=2905954

# ── Compute ──────────────────────────────────────────────────────────────────
GPUS=8
BS=512
GlobalBS=$(python3 -c "print(${BS} * ${GPUS})")

BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(${GlobalBS} / (8 * 512)))")

# ── Models ───────────────────────────────────────────────────────────────────
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"

PE="local-dir:${MODEL_DIR}/PE-Core-B-16"
SIGLIP2="local-dir:${MODEL_DIR}/ViT-B-16-SigLIP2"
DATACOMP="local-dir:${MODEL_DIR}/DataComp-XL-B-16"

TEACHERS="${PE},${SIGLIP2},${DATACOMP}"

# ── Common args ──────────────────────────────────────────────────────────────
BASE="--precision amp_bf16 --workers 32 --batch-size ${BS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

COMMON="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run() {
    local TAG=$1 PORT=$2 WEIGHTS=$3
    local NAME="mtw_${TAG}_${TS}"
    echo "======== [multi-teacher-weighted] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "PE-Core-B-16" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        --siglip \
        --multi-teacher \
        --teachers "${TEACHERS}" \
        --teacher-weights "${WEIGHTS}" \
        --name "${NAME}" < /dev/null
}

# ══════════════════════════════════════════════════════════════════════════════

# W1: PE dominant (5:1)
run "W1_5to1" 29610 "1.0,0.2,0.2"

# W2: PE extreme (10:1)
run "W2_10to1" 29611 "1.0,0.1,0.1"

# W3: PE mild (2:1)
run "W3_2to1" 29612 "1.0,0.5,0.5"

echo "======== multi_teacher_weighted all done ========"
