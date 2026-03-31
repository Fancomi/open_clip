#!/bin/bash
# quick.sh — 1/10 数据 scaling law 对比
#
# 实验矩阵（4 组）：
#   PE-Core-B × CLIP | PE-Core-B × SigLIP
#   ViT-B     × CLIP | ViT-B     × SigLIP
#
# 数据：
#   train: clip_train_quick.tsv (56,675 行 ≈ 1/10 全量)
#   val:   clip_val.tsv (25,010 行，quick 用全量 val 评估)
#   每 epoch ≈ 221 steps (56675 / 256)，2 epochs ≈ 442 steps
# 硬件：2x A800-80GB

set -e
export PYTHONPATH="./src:${PYTHONPATH}"

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_quick.tsv"
VAL="${COCO}/clip_val.tsv"

COMMON="--dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --precision bf16 --workers 4 --epochs 2 --batch-size 128 \
    --lr 5e-4 --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.2 --warmup 50 \
    --save-frequency 1 --save-most-recent \
    --grad-checkpointing --log-every-n-steps 10 --val-frequency 1"

run() {
    local TAG=$1 MODEL=$2 EXTRA=$3 PORT=$4
    local NAME="quick_${TAG}_${TS}"
    echo "======== [quick] ${TAG} => ${NAME} ========"
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

echo "======== quick 全部完成 ========"
