#!/bin/bash
# eval_pretrained.sh — 直接跑 eval，不训练，验证预训练权重在 COCO 上的检索指标
#
# 评估协议：官方 Karpathy test split（5K 图 × 5cap = 25K 对）
#   I2T R@1 = 给定图，检索文本，5条中至少1条在top-1（CLIP-style binary）
#   T2I R@1 = 给定文本，检索图，正确图在top-1
#
# 复现论文数值（本项目实测 vs 论文报告）：
#   PE-Core-B-16:     T2I R@1=50.2  I2T R@1=71.1   论文报告 T2I=50.9 ✅
#   ViT-B-16-SigLIP2: T2I R@1=53.2  I2T R@1=69.4   论文报告 I2T≈68.9 ✅

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
KAR_5CAP="${COCO}/karpathy_5cap.tsv"
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
GPUS=8

eval_only() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="eval_${TAG}_${TS}"
    echo "======== [eval] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --val-data "${KAR_5CAP}" \
        --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
        --workers 8 \
        --batch-size 256 \
        --precision amp_bf16 \
        --val-num-captions-per-image 5 \
        ${EXTRA} \
        --name "${NAME}"
}

# ---------- PE-Core-B-16 ----------
eval_only "pe_core" \
    "local-dir:${MODEL_DIR}/PE-Core-B-16" \
    29600 ""

# ---------- ViT-B-16-SigLIP2 ----------
eval_only "siglip2" \
    "local-dir:${MODEL_DIR}/ViT-B-16-SigLIP2" \
    29601 "--siglip"

echo "======== eval_pretrained 完成 ========"
