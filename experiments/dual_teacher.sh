#!/bin/bash
# Dual-Teacher: one image encoder aligned with two frozen text encoders (PE-Core + SigLIP2)
#
# 配置矩阵：
#   E1 = single CLS + 2 heads, image scratch
#   E2 = dual CLS (2 MAP pools) + 2 heads, image scratch
#   E3 = single CLS + 2 heads, image pretrained (PE-Core)
#   E4 = dual CLS + 2 heads, image pretrained (PE-Core)
#   E5 = single CLS + 2 heads, image pretrained (SigLIP2)

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
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"
SIGLIP_CKPT="${MODEL_DIR}/ViT-B-16-SigLIP2/open_clip_model.safetensors"
SIGLIP_MODEL="local-dir:${MODEL_DIR}/ViT-B-16-SigLIP2"

# ── Common args ──────────────────────────────────────────────────────────────
BASE="--precision amp_bf16 --workers 32 --batch-size ${BS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

COMMON="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

DUAL="--dual-teacher \
    --teacher-pe-ckpt ${PE_CKPT} \
    --teacher-sig-ckpt ${SIGLIP_CKPT} \
    --teacher-sig-model ${SIGLIP_MODEL}"

run() {
    local TAG=$1 PORT=$2 EXTRA=$3
    local NAME="dt_${TAG}_${TS}"
    echo "======== [dual-teacher] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "PE-Core-B-16" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        --siglip \
        ${DUAL} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ══════════════════════════════════════════════════════════════════════════════

# # E1: Single CLS + dual head, image from scratch
# run "E1_1cls_scratch" 29580 ""

# # E2: Dual CLS + dual head, image from scratch
# run "E2_2cls_scratch" 29581 "--dual-cls"

# # E3: Single CLS + dual head, image pretrained (PE-Core)
# run "E3_1cls_pt_pe" 29582 "--pretrained-image-init ${PE_CKPT}"

# # E4: Dual CLS + dual head, image pretrained (PE-Core)
# run "E4_2cls_pt_pe" 29583 "--dual-cls --pretrained-image-init ${PE_CKPT}"

# E5: Dual CLS + dual head, image pretrained (PE-Core), longer training
run "E5_2cls_pt_pe_20ep" 29534 "--dual-cls --pretrained-image-init ${PE_CKPT} --epochs 20"

echo "======== dual_teacher all done ========"
