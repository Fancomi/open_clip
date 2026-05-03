#!/bin/bash
# experiments/modality_gap.sh
#
# 模态差异消除实验
# Baseline: pe_dinov3_sigreg_cls_probe (quick.sh)
#
# 实验矩阵:
#   Step 0 — 后处理分析 (纯分析，不训练)
#   Step 1 — --modality-gap-weight λ 消融 (λ ∈ {0.001, 0.005, 0.01, 0.05})
#   Step 2 — --modality-gap-weight + --sigreg-target cls 联合
#   Step 3 — DINOv3 + --modality-gap-weight 消融
#
# 本脚本仅包含训练实验 (Step 1-3)。
# Step 0 后处理分析请使用: analysis/modality_gap.py
#
# 用法:
#   bash experiments/modality_gap.sh
#
# 运行顺序由上到下，任一失败 set -e 中止。
# 可注释掉不需要的行逐步推进。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

CC3M="/dev/shm/cc3m_wds"
CC3M_TRAIN="${CC3M}/{00000..00280}.tar"
CC3M_N_TRAIN=2857622

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 --grad-checkpointing \
    --log-every-n-steps 1 --val-frequency 1"

COMMON_WDS="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="mgap_${TAG}_${TS}"
    echo "======== [mgap] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON_WDS} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ── CC3M copy ────────────────────────────────────────────────────────────────
if [ ! -d "${CC3M}" ]; then
    echo "[mgap] Loading CC3M to memory ..."
    cp -r "/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds" "${CC3M}"
    echo "[mgap] Done, $(du -sh ${CC3M} | cut -f1)"
else
    echo "[mgap] Found ${CC3M}, skip copy"
fi

# ════════════════════════════════════════════════════════════════════════════
# Step 0: Post-processing analysis (no training)
# 先跑 baseline probe，然后用 analysis/modality_gap.py 分析
# ════════════════════════════════════════════════════════════════════════════
# 用法（在 baseline probe 目录有 step_*.npz 后运行）：
#
#   source /root/.../envs/dino/bin/activate
#   PYTHONPATH=./src python3 analysis/modality_gap.py \
#       --probe logs/cc3m_pe_dinov3_sigreg_cls_probe_<TS>/probe/step_001740.npz \
#       --split proj_features \
#       --out   analysis/research/modality_gap_baseline.json
#
# ════════════════════════════════════════════════════════════════════════════
# Step 1: modality-gap-weight λ 消融（sigreg cls only，无 DINOv3）
# 固定 sigreg-target=cls，只改 lambda_gap
# ════════════════════════════════════════════════════════════════════════════
SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 --probe-data ${PROBE_TSV}"

run "gap001"  "PE-Core-B-16-dinov3" 29571 "${SIGREG_BASE} --modality-gap-weight 0.001"
run "gap005"  "PE-Core-B-16-dinov3" 29572 "${SIGREG_BASE} --modality-gap-weight 0.005"
run "gap01"   "PE-Core-B-16-dinov3" 29573 "${SIGREG_BASE} --modality-gap-weight 0.01"
run "gap05"   "PE-Core-B-16-dinov3" 29574 "${SIGREG_BASE} --modality-gap-weight 0.05"

# ════════════════════════════════════════════════════════════════════════════
# Step 2: DINOv3 + modality-gap-weight 消融
# 与 quick.sh pe_dinov3_dinov3_muon_sigreg_probe 对齐，加 gap loss
# ════════════════════════════════════════════════════════════════════════════
DINO_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} \
    --dinov3 --dino-n-global-crops 1 --dino-local-crops-number 8 \
    --dino-head-prototypes 8192 --dino-warmup-teacher-temp-epochs 3 \
    --probe-data ${PROBE_TSV}"

run "dino_gap001" "PE-Core-B-16-dinov3" 29575 "${DINO_BASE} --modality-gap-weight 0.001"
run "dino_gap005" "PE-Core-B-16-dinov3" 29576 "${DINO_BASE} --modality-gap-weight 0.005"
run "dino_gap01"  "PE-Core-B-16-dinov3" 29577 "${DINO_BASE} --modality-gap-weight 0.01"

echo "======== modality_gap experiments done ========"
