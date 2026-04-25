#!/bin/bash
# run_pca_demo.sh — 一键运行 4 模型 PCA 对比可视化
# 用法: bash scripts/run_pca_demo.sh [num_images] [output_dir]

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"

NUM_IMAGES=${1:-6}
OUTPUT_DIR=${2:-scripts/pca_demo}

IMG_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/images/val2014"

echo "======== PCA Demo: ${NUM_IMAGES} images → ${OUTPUT_DIR} ========"

python3 scripts/model_pca_demo.py \
    --img_dir    "${IMG_DIR}" \
    --num_images "${NUM_IMAGES}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42

echo "======== Done: ${OUTPUT_DIR} ========"
