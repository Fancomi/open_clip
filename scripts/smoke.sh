#!/bin/bash
# smoke.sh — 快速冒烟测试，任一失败即 set -e 中止
#
# 非 DINOv3：1 GPU + COCO karpathy_1cap CSV，~10 steps/epoch，2 epoch
# DINOv3   ：8 GPU + CC3M webdataset（仅 5 步），与 quick.sh 配置一致

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/karpathy_1cap.tsv"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

CC3M="/dev/shm/cc3m_wds"
CC3M_TRAIN="${CC3M}/{00000..00280}.tar"

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

# 5 steps × 8GPU × BS512
SMOKE_N=$(python3 -c "print(${PreGpuBS} * ${GPUS} * 5)")

# ── CSV 版（非 DINOv3，1 GPU）────────────────────────────────────────────────
BASE_CSV="--precision amp_bf16 --workers 4 --epochs 2 --batch-size 512 \
    --lr 3.4e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 0 --log-every-n-steps 1 --val-frequency 1"
COMMON_CSV="--warmup 2 ${BASE_CSV} \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

# ── WDS 版（DINOv3，8 GPU，5 步）────────────────────────────────────────────
BASE_WDS="--precision amp_bf16 --workers 32 --epochs 1 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 0 --log-every-n-steps 1 --val-frequency 0 \
    --grad-checkpointing"
COMMON_WDS="--warmup 2 ${BASE_WDS} \
    --dataset-type webdataset --train-num-samples ${SMOKE_N} \
    --csv-img-key filepath --csv-caption-key caption"

run_smoke_csv() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="smoke_${TAG}_${TS}"
    echo ""; echo "======== [smoke/csv 1GPU] ${TAG} ========"
    torchrun --nproc_per_node=1 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" --train-data "${TRAIN}" --val-data "${VAL}" \
        ${COMMON_CSV} ${EXTRA} --name "${NAME}" < /dev/null
    echo "======== [smoke/csv] ${TAG} PASSED ========"
}

run_smoke_wds() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="smoke_${TAG}_${TS}"
    echo ""; echo "======== [smoke/wds 8GPU] ${TAG} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" --train-data "${CC3M_TRAIN}" --val-data "${VAL}" \
        ${COMMON_WDS} ${EXTRA} --name "${NAME}" < /dev/null
    echo "======== [smoke/wds] ${TAG} PASSED ========"
}

# # ── 1. sigreg cls（CLS raw Identity，无 DINOv3）────────────────────────────
# run_smoke_csv "sigreg_cls"      "PE-Core-B-16-dinov3" 29515 \
#     "--siglip --sigreg-target cls --sigreg-weight 1e-4 --probe-data ${PROBE_TSV}"

# # ── 2. sigreg cls_proj（CLS raw → MLP，无 DINOv3）──────────────────────────
# run_smoke_csv "sigreg_cls_proj" "PE-Core-B-16-dinov3" 29516 \
#     "--siglip --sigreg-target cls_proj --sigreg-weight 1e-4 --probe-data ${PROBE_TSV}"

# ── 3. DINOv3 + sigreg cls（与 KoLeo 同位，无额外参数）────────────────────
run_smoke_wds "dinov3_sigreg"   "PE-Core-B-16-dinov3" 29517 \
    "--siglip --sigreg-target cls --sigreg-weight 1e-4 \
     --opt muon --muon-lr ${MUON_LR} \
     --dinov3 --dino-n-global-crops 1 --dino-local-crops-number 8 \
     --dino-head-prototypes 8192 --dino-warmup-teacher-temp-epochs 1"


# ── 3. DINOv3 + sigreg cls（与 KoLeo 同位，无额外参数）────────────────────
run_smoke_wds "dinov3_sigreg_proj"   "PE-Core-B-16-dinov3" 29517 \
    "--siglip --sigreg-target cls_proj --sigreg-weight 1e-4 \
     --opt muon --muon-lr ${MUON_LR} \
     --dinov3 --dino-n-global-crops 1 --dino-local-crops-number 8 \
     --dino-head-prototypes 8192 --dino-warmup-teacher-temp-epochs 1"

# ── 4. DINOv3 only（修复后的基线）──────────────────────────────────────────
run_smoke_wds "dinov3"          "PE-Core-B-16-dinov3" 29518 \
    "--siglip --opt muon --muon-lr ${MUON_LR} \
     --dinov3 --dino-n-global-crops 1 --dino-local-crops-number 8 \
     --dino-head-prototypes 8192 --dino-warmup-teacher-temp-epochs 1"

echo ""; echo "======== smoke 全部通过 ========"
