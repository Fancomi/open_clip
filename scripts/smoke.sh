#!/bin/bash
# smoke.sh — 最小完整流程验证（10 步 train + 小 eval）
#
# 实验矩阵（2 组，SigLIP only）：
#   PE-Core-B × SigLIP | ViT-B × SigLIP
#
# 数据：
#   train: clip_train_smoke.tsv (2560 行 = 10 steps × 128 bs × 2 gpu)
#   val:   clip_val_smoke.tsv   (512 行)
# 硬件：2x A800-80GB

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'  # Use Beijing time for timestamps

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_smoke.tsv"
VAL="${COCO}/clip_val_smoke.tsv"

COMMON="--dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --precision amp_bf16 --workers 4 --epochs 1 --batch-size 128 \
    --lr 4e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --warmup 1 \
    --save-frequency 1 --save-most-recent --delete-previous-checkpoint \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
    --siglip"

run() {
    local TAG=$1 MODEL=$2 PORT=$3
    local NAME="smoke_${TAG}_${TS}"
    echo "======== [smoke] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=2 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        --name "${NAME}"
}


# run "vit_siglip"    "ViT-B-16-exp"         29511
run "pe_cls_siglip" "PE-Core-B-16-cls"    29510

echo "======== smoke 全部完成 ========"
