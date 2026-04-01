#!/bin/bash
# quick.sh — 全量 COCO SigLIP 对比训练
#
# 实验矩阵（2 组，SigLIP only）：
#   PE-Core-B × SigLIP | ViT-B × SigLIP
#
# 数据：clip_train_dedup.tsv (113,287 行，去重) / clip_val.tsv (25,010 行)
# global batch = 2048/GPU × 2 = 4096
# steps/epoch ≈ 28, 30 epochs = 840 steps total
# schedule: warmup 50 → stable at peak LR (700 steps, 25 epochs) → cooldown (140 steps, 5 epochs)
# const-cooldown: 两个架构都在 peak LR 充分训练，不因 cosine 提前衰减错过有效窗口
#
# 硬件：2x A800-80GB

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'  # Use Beijing time for timestamps

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_dedup.tsv"
VAL="${COCO}/clip_val.tsv"

COMMON="--dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --precision amp_bf16 --workers 6 --epochs 30 --batch-size 2048 \
    --lr 7e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --warmup 50 \
    --save-frequency 5 --save-most-recent \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 5 \
    --siglip"

run() {
    local TAG=$1 MODEL=$2 PORT=$3
    local NAME="quick_${TAG}_${TS}"
    echo "======== [quick] ${TAG} => ${NAME} ========"
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


echo "======== quick 全部完成 ========"
