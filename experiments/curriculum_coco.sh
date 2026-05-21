#!/bin/bash
# experiments/curriculum_coco.sh
#
# Curriculum Learning 实验 —— COCO 快速验证
#
# 基于 pe_dinov3_sigreg_siglip_muon 最优配置，验证 5 种采样顺序策略:
#   fps            — Farthest Point Sampling (多样性优先)
#   density_high   — kNN 密度高优先 (简单→难)
#   density_low    — kNN 密度低优先 (难→简单)
#   curvature_high — kNN 曲率高优先 (决策边界优先)
#   curvature_low  — kNN 曲率低优先 (平坦区域优先)
#
# Epoch 0 初始特征源: dinov3 / pe_core / self
# Epoch 1+: 始终用当前模型
#
# COCO: ~82K samples, steps/epoch ≈ 20 @ BS4096, 20 epochs ≈ 400 steps
# 预计运行时间: 16 runs × ~18min ≈ 5h
#
# 用法:
#   bash experiments/curriculum_coco.sh

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_dedup.tsv"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

BASE="--precision amp_bf16 --workers 8 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 2 --grad-checkpointing \
    --log-every-n-steps 2 --val-frequency 2 \
    --delete-previous-checkpoint"

COMMON="--warmup 42 ${BASE} --epochs 20 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} \
    --probe-data ${PROBE_TSV}"

run() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="cur_${TAG}_${TS}"
    echo "======== [curriculum] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ════════════════════════════════════════════════════════════════════════════
# Baseline (无 curriculum, 纯随机顺序)
# ════════════════════════════════════════════════════════════════════════════
run "baseline" "PE-Core-B-16-dinov3" 29570 "${SIGREG_BASE}"

# ════════════════════════════════════════════════════════════════════════════
# FPS 策略 (3 init modes)
# ════════════════════════════════════════════════════════════════════════════
run "fps_dinov3" "PE-Core-B-16-dinov3" 29571 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init dinov3"
run "fps_pecore" "PE-Core-B-16-dinov3" 29572 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init pe_core"
run "fps_self"   "PE-Core-B-16-dinov3" 29573 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init self"

# ════════════════════════════════════════════════════════════════════════════
# Density High (高密度优先 = 简单样本先学)
# ════════════════════════════════════════════════════════════════════════════
run "dhi_dinov3" "PE-Core-B-16-dinov3" 29574 "${SIGREG_BASE} --curriculum-strategy density_high --curriculum-init dinov3"
run "dhi_pecore" "PE-Core-B-16-dinov3" 29575 "${SIGREG_BASE} --curriculum-strategy density_high --curriculum-init pe_core"
run "dhi_self"   "PE-Core-B-16-dinov3" 29576 "${SIGREG_BASE} --curriculum-strategy density_high --curriculum-init self"

# ════════════════════════════════════════════════════════════════════════════
# Density Low (低密度优先 = 困难样本先学)
# ════════════════════════════════════════════════════════════════════════════
run "dlo_dinov3" "PE-Core-B-16-dinov3" 29577 "${SIGREG_BASE} --curriculum-strategy density_low --curriculum-init dinov3"
run "dlo_pecore" "PE-Core-B-16-dinov3" 29578 "${SIGREG_BASE} --curriculum-strategy density_low --curriculum-init pe_core"
run "dlo_self"   "PE-Core-B-16-dinov3" 29579 "${SIGREG_BASE} --curriculum-strategy density_low --curriculum-init self"

# ════════════════════════════════════════════════════════════════════════════
# Curvature High (高曲率优先 = 决策边界区域先学)
# ════════════════════════════════════════════════════════════════════════════
run "chi_dinov3" "PE-Core-B-16-dinov3" 29580 "${SIGREG_BASE} --curriculum-strategy curvature_high --curriculum-init dinov3"
run "chi_pecore" "PE-Core-B-16-dinov3" 29581 "${SIGREG_BASE} --curriculum-strategy curvature_high --curriculum-init pe_core"
run "chi_self"   "PE-Core-B-16-dinov3" 29582 "${SIGREG_BASE} --curriculum-strategy curvature_high --curriculum-init self"

# ════════════════════════════════════════════════════════════════════════════
# Curvature Low (低曲率优先 = 平坦区域先学)
# ════════════════════════════════════════════════════════════════════════════
run "clo_dinov3" "PE-Core-B-16-dinov3" 29583 "${SIGREG_BASE} --curriculum-strategy curvature_low --curriculum-init dinov3"
run "clo_pecore" "PE-Core-B-16-dinov3" 29584 "${SIGREG_BASE} --curriculum-strategy curvature_low --curriculum-init pe_core"
run "clo_self"   "PE-Core-B-16-dinov3" 29585 "${SIGREG_BASE} --curriculum-strategy curvature_low --curriculum-init self"

echo "======== Curriculum Learning done (16 runs) ========"
