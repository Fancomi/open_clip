#!/bin/bash
# quick.sh — PE-CLS vs PE-DINOv3 vs ViT with SigLIP/CLIP loss comparison
#
# 实验矩阵（6 组）：
#   PE-CLS    + SigLIP (baseline)
#   PE-CLS    + CLIP
#   PE-DINOv3 + SigLIP (DINOv3-aligned)
#   PE-DINOv3 + CLIP
#   ViT       + SigLIP
#   ViT       + CLIP
#
# 数据：clip_train_dedup.tsv (113,287 行，去重) / clip_val.tsv (25,010 行)
# global batch = 2048/GPU × 2 = 4096
# steps/epoch ≈ 28, 30 epochs = 840 steps total
# schedule: warmup 10 → cosine decay to 0
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
    --lr 3e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --warmup 20 \
    --save-frequency 0 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 5"

run() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="quick_${TAG}_${TS}"
    echo "======== [quick] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=2 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        ${EXTRA} \
        --name "${NAME}"
}

# PE-DINOv3 with SigLIP
run "pe_dinov3_siglip[retry+trunc_normal]" "PE-Core-B-16-dinov3" 29512 "--siglip"
# PE-CLS with SigLIP (baseline)
run "pe_cls_siglip[retry+trunc_normal]"    "PE-Core-B-16-cls"    29510 "--siglip" # 确保和之前效果一致
# # ViT with CLIP
# run "vit_clip"         "ViT-B-16-exp"        29515 ""
# # PE-CLS with CLIP
# run "pe_cls_clip"      "PE-Core-B-16-cls"    29511 ""
# # PE-DINOv3 with CLIP
# run "pe_dinov3_clip"   "PE-Core-B-16-dinov3" 29513 ""
# # ViT with SigLIP
# run "vit_siglip"       "ViT-B-16-exp"        29514 "--siglip"


echo "======== quick 全部完成 ========"
