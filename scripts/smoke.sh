#!/bin/bash
# smoke.sh — 最小完整流程验证（精确 10 步 train + 小 eval）
#
# 实验矩阵（4 组）：
#   PE-Core-B × CLIP | PE-Core-B × SigLIP
#   ViT-B     × CLIP | ViT-B     × SigLIP
#
# 数据：
#   train: clip_train_smoke.tsv (2560 行 = 10 steps × 128 bs × 2 gpu)
#   val:   clip_val_smoke.tsv   (512 行，快速验证 eval 流程)
# 硬件：2x A800-80GB

set -e
export PYTHONPATH="./src:${PYTHONPATH}"

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_smoke.tsv"
VAL="${COCO}/clip_val_smoke.tsv"

COMMON="--dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --precision bf16 --workers 4 --epochs 1 --batch-size 128 \
    --lr 5e-4 --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.2 --warmup 2 \
    --save-frequency 1 --save-most-recent --delete-previous-checkpoint \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

run() {
    local TAG=$1 MODEL=$2 EXTRA=$3 PORT=$4
    local NAME="smoke_${TAG}_${TS}"
    echo "======== [smoke] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=2 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} ${EXTRA} \
        --name "${NAME}"
}

run "pe_clip"    "PE-Core-B-16-exp" "--local-loss --gather-with-grad"  29510
run "pe_siglip"  "PE-Core-B-16-exp" "--siglip"                        29511
run "vit_clip"   "ViT-B-16-exp"     "--local-loss --gather-with-grad"  29512
run "vit_siglip" "ViT-B-16-exp"     "--siglip"                        29513

echo "======== smoke 全部完成 ========"
