#!/bin/bash
# Multi-Teacher: one image encoder aligned with N frozen text encoders (from-scratch)
#
# 实验矩阵：
#   M1 = PE + SigLIP2 + DataComp            (3T)
#   M2 = PE + SigLIP2 + DataComp + LAION2B + DFN2B  (5T)
#   M3 = PE + SigLIP2 + DataComp + DFN2B + EVA02 + LAION2B + MetaCLIP  (7T)
#   M4 = PE + DataComp + DFN2B + LAION2B + MetaCLIP   (5T, no SigLIP2)
#   M5 = PE + SigLIP2 + EVA02             (3T)

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ── Data ─────────────────────────────────────────────────────────────────────
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

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
DFN2B="local-dir:${MODEL_DIR}/DFN2B-ViT-B-16"
EVA02="local-dir:${MODEL_DIR}/EVA02-B-16"
LAION2B="local-dir:${MODEL_DIR}/LAION2B-B-16"
METACLIP="local-dir:${MODEL_DIR}/MetaCLIP-FullCC-B-16"

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
    local TAG=$1 PORT=$2 TEACHERS=$3
    local NAME="mt_${TAG}_${TS}"
    echo "======== [multi-teacher] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "PE-Core-B-16" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        --siglip \
        --multi-teacher \
        --teachers "${TEACHERS}" \
        --name "${NAME}" < /dev/null
}

# ══════════════════════════════════════════════════════════════════════════════

# M1: PE + SigLIP2 + DataComp (3T)
run "M1_3T_pe_sig_dc" 29590 \
    "${PE},${SIGLIP2},${DATACOMP}"

# M2: PE + SigLIP2 + DataComp + LAION2B + DFN2B (5T)
run "M2_5T_pe_sig_dc_la_dfn" 29591 \
    "${PE},${SIGLIP2},${DATACOMP},${LAION2B},${DFN2B}"

# M3: PE + SigLIP2 + DataComp + DFN2B + EVA02 + LAION2B + MetaCLIP (7T)
run "M3_7T_all" 29592 \
    "${PE},${SIGLIP2},${DATACOMP},${DFN2B},${EVA02},${LAION2B},${METACLIP}"

# M4: PE + DataComp + DFN2B + LAION2B + MetaCLIP (5T, no SigLIP2)
run "M4_5T_no_sig" 29593 \
    "${PE},${DATACOMP},${DFN2B},${LAION2B},${METACLIP}"

# M5: PE + SigLIP2 + EVA02 (3T alternative)
run "M5_3T_pe_sig_eva" 29594 \
    "${PE},${SIGLIP2},${EVA02}"
