#!/bin/bash
# quick.sh — 全量 COCO 对比训练，大 batch
#
# 实验矩阵（4 组）：
#   PE-Core-B × CLIP | PE-Core-B × SigLIP
#   ViT-B     × CLIP | ViT-B     × SigLIP
#
# 数据：clip_train.tsv (566,747 行全量) / clip_val.tsv (25,010 行)
# global batch = 4096/GPU × 2 = 8192
# steps/epoch ≈ 69，10 epochs ≈ 690 steps
#
# 超参参考 h14_84_8_pretrain.sh（demo 为 batch=32768, lr=2.048e-3）
#   lr = 1e-3（按 batch 比例缩放）
#   beta2 = 0.95（demo 一致）
#   warmup = 50 steps（690 总步的 ~7%）
#
# 硬件：2x A800-80GB（4096/GPU ≈ 57GB，grad-ckpt 开启）

set -e
export PYTHONPATH="./src:${PYTHONPATH}"

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train.tsv"
VAL="${COCO}/clip_val.tsv"

COMMON="--dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --precision bf16 --workers 6 --epochs 10 --batch-size 4096 \
    --lr 1e-3 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --warmup 5 \
    --save-frequency 2 --save-most-recent \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 2"

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
