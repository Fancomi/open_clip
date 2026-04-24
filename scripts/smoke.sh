#!/bin/bash
# smoke.sh — 快速冒烟测试，单 GPU + COCO karpathy_1cap（5000条）
# ~10 steps/epoch，train+val+probe 合计约 2 分钟
#
# 配置：
#   train: karpathy_1cap.tsv（5000条）→ ~10 steps/epoch (BS=512×1GPU)
#   val:   karpathy_5cap.tsv（完整，评估全流程）
#   epochs: 2（跑完 1 个 val+probe 周期）
#   硬件: 1x GPU

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/karpathy_1cap.tsv"   # 5000条 → ~10 steps/epoch
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

BASE_SMOKE="--precision amp_bf16 --workers 4 --epochs 2 --batch-size 512 \
    --lr 3.4e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 0 \
    --log-every-n-steps 1 --val-frequency 1"

COMMON_SMOKE="--warmup 2 ${BASE_SMOKE} \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run_smoke() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="smoke_${TAG}_${TS}"
    echo "======== [smoke] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=1 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON_SMOKE} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

run_smoke "pe_dinov3_leproj" "PE-Core-B-16-dinov3" 29515 "--siglip --lejepa --lejepa-proj --probe-data ${PROBE_TSV}"
# run_smoke "vit_leproj"       "ViT-B-16-exp"        29513 "--siglip --lejepa --lejepa-proj"
# run_smoke "dinov3_leproj"    "DINOv3-B-16-ape"     29517 "--siglip --lejepa --lejepa-proj"
# run_smoke "vit_le"           "ViT-B-16-exp"        29513 "--siglip --lejepa"
# run_smoke "pe_dinov3_le"     "PE-Core-B-16-dinov3" 29515 "--siglip --lejepa"
# run_smoke "dinov3_le"        "DINOv3-B-16-ape"     29517 "--siglip --lejepa"
# run_smoke "vit"              "ViT-B-16-exp"        29513 "--siglip"
# run_smoke "pe_dinov3"        "PE-Core-B-16-dinov3" 29515 "--siglip"
# run_smoke "dinov3"           "DINOv3-B-16-ape"     29517 "--siglip"

echo "======== smoke 全部通过 ========"
