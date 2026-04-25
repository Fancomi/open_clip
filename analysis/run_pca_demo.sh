#!/bin/bash
# PCA patch-feature visualization for DINOv3 / RADIO / TIPSv2 / EUPE
# Usage: bash analysis/run_pca_demo.sh [num_images] [output_dir]

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"

NUM_IMAGES=${1:-6}
OUTPUT_DIR=${2:-analysis/pca_demo}
IMG_DIR="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/images/val2014"

echo "======== PCA Demo: ${NUM_IMAGES} images → ${OUTPUT_DIR} ========"
python3 -m analysis.pca_demo \
    --img_dir    "${IMG_DIR}" \
    --num_images "${NUM_IMAGES}" \
    --output_dir "${OUTPUT_DIR}" \
    --seed 42
echo "======== Done: ${OUTPUT_DIR} ========"
