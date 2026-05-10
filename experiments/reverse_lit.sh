#!/bin/bash
# Reverse-LiT: pretrained text tower + from-scratch image tower
# 研究问题：预训练文本编码器能否作为锚点，从头训练图像编码器？
#
# 配置矩阵：
#   A  = lock pretrained text + scratch image (核心 reverse-LiT)
#   B  = lock pretrained text + MLP bridge + scratch image
#   C  = tune pretrained text + scratch image (Tu-Ti)
#   C2 = partial lock text (last 4 layers tuned) + scratch image
#   D1 = scratch baseline (both from scratch)
#   D2 = standard LiT (lock pretrained image, tune text)

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
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="rlit_${TAG}_${TS}"
    echo "======== [reverse-lit] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        --siglip \
        --probe-data "${PROBE_TSV}" \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ══════════════════════════════════════════════════════════════════════════════
# PE-Core-B-16 experiments
# ══════════════════════════════════════════════════════════════════════════════

# D1: Baseline - both from scratch
run "D1_scratch_pe" "PE-Core-B-16" 29570 ""

# A: Reverse-LiT (lock pretrained text, scratch image)
run "A_rlit_pe" "PE-Core-B-16" 29571 \
    "--pretrained-text-path ${PE_CKPT} --lock-text"

# B: Reverse-LiT + MLP bridge
run "B_rlit_mlp_pe" "PE-Core-B-16" 29572 \
    "--pretrained-text-path ${PE_CKPT} --lock-text --text-proj-type mlp"

# C: Tu-Ti (tune pretrained text + scratch image)
run "C_tuti_pe" "PE-Core-B-16" 29573 \
    "--pretrained-text-path ${PE_CKPT}"

# C2: Partial lock text (unlock last 4 groups)
run "C2_partial_pe" "PE-Core-B-16" 29574 \
    "--pretrained-text-path ${PE_CKPT} --lock-text --lock-text-unlocked-layers 4"

# D2: Standard LiT (lock pretrained image, tune text from pretrained)
run "D2_lit_pe" "PE-Core-B-16" 29575 \
    "--pretrained ${PE_CKPT} --lock-image"

# ══════════════════════════════════════════════════════════════════════════════
# ViT-B-16-SigLIP2 experiments
# Note: uses local-dir to avoid network dependency for HF tokenizer.
# Image tower starts from pretrained (not scratch) — this is "lock-text + tune-image" variant.
# ══════════════════════════════════════════════════════════════════════════════
SIGLIP_DIR="local-dir:${MODEL_DIR}/ViT-B-16-SigLIP2"

# A: Lock text + tune image (pretrained both, lock text only)
run "A_rlit_sig2" "${SIGLIP_DIR}" 29576 \
    "--lock-text"

# B: Lock text + MLP bridge + tune image
run "B_rlit_mlp_sig2" "${SIGLIP_DIR}" 29577 \
    "--lock-text --text-proj-type mlp"

echo "======== reverse_lit all done ========"
