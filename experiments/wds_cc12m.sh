#!/bin/bash
# Baseline: pe_dinov3_sigreg_siglip_muon，训练数据换用 cc12m-wds（标准 HuggingFace 格式）
#
# 数据：/root/paddlejob/workspace/env_run/penghaotian/datas/cc12m-wds/
#   格式：webdataset (.jpg + .txt per sample)
#   train shards: cc12m-train-{0000..2175}.tar  (10,968,539 samples)
#   无自带 val split → 继续用 COCO karpathy val
#
# 规模说明：CC12M 约是 CC3M 的 3.8x，epoch 相同时 total_steps 也 3.8x。
#   steps/epoch ≈ 10968539/4096 ≈ 2678
#   warmup 512 steps ≈ 0.19 epoch（与 CC3M 基线比例相当，保持不变）
#   epoch 数维持 10（可按需调整 --epochs）

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

CC12M_WDS="/root/paddlejob/workspace/env_run/penghaotian/datas/cc12m-wds"
CC12M_TRAIN="${CC12M_WDS}/cc12m-train-{0000..2175}.tar"
CC12M_N_TRAIN=10968539

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")

BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

COMMON_WDS="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --train-num-samples ${CC12M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run_wds() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="wds_cc12m_${TAG}_${TS}"
    echo "======== [wds_cc12m] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC12M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON_WDS} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ── baseline replicated on cc12m-wds ─────────────────────────────────────────
run_wds "pe_dinov3_sigreg_siglip_muon" "PE-Core-B-16-dinov3" 29561 \
    "--siglip --sigreg-target cls --sigreg-weight 1e-4 \
     --epochs 10 --warmup 512 \
     --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
     --probe-data ${PROBE_TSV}"

echo "======== wds_cc12m all done ========"
