#!/bin/bash
# Baseline: pe_dinov3_sigreg_siglip_muon，训练数据换用 cc3m-wds（标准 HuggingFace 格式）
#
# 数据：/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds/
#   格式：webdataset (.jpg + .txt per sample)
#   train shards: cc3m-train-{0000..0575}.tar  (2,905,954 samples)
#   val   shards: cc3m-validation-{0000..0015}.tar (13,443 samples)
#   注：val 用自带 val split，不再依赖 COCO karpathy；若需要 COCO val 请换回 --val-data

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

CC3M_WDS="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds"
CC3M_TRAIN="${CC3M_WDS}/cc3m-train-{0000..0575}.tar"
CC3M_N_TRAIN=2905954

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")

BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

# CC3M-wds: steps/epoch ≈ 2905954/4096 ≈ 709, warmup 10% ≈ 71 → 与旧 CC3M shm 版同 epoch 数(10)
COMMON_WDS="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run_wds() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="wds_cc3m_${TAG}_${TS}"
    echo "======== [wds_cc3m] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON_WDS} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ── baseline replicated on cc3m-wds ──────────────────────────────────────────

# ORRI VIT
# run_wds "vit"  "ViT-B-16-exp" 29562 \
#     "--siglip \
#     --epochs 10 --warmup 512 \
#     --probe-data ${PROBE_TSV}"

# ORI
# run_wds "pe_dinov3_siglip" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip \
#     --epochs 10 --warmup 512 \
#     --probe-data ${PROBE_TSV}"

# + Muon
# run_wds "pe_dinov3_siglip_muon" "PE-Core-B-16-dinov3" 29561 \
#     "--siglip \
#     --epochs 10 --warmup 512 \
#     --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#     --probe-data ${PROBE_TSV}"

# + Muon + SigREG
# run_wds "pe_dinov3_sigreg_siglip_muon" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip --sigreg-target cls --sigreg-weight 1e-4 \
#      --epochs 10 --warmup 512 \
#      --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#      --probe-data ${PROBE_TSV}"

# + Muon + SigREG + dino
# run_wds "pe_dinov3_sigreg_siglip_muon_dino" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip --sigreg-target cls --sigreg-weight 1e-4 \
#      --epochs 10 --warmup 512 \
#      --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#      --dinov3 --dino-n-global-crops 1 --dino-local-crops-number 8 --dino-head-prototypes 8192 --dino-warmup-teacher-temp-epochs 3 \
#      --probe-data ${PROBE_TSV}"

# ════════════════════════════════════════════════════════════════════════════
# Antipodal SigLIP — CC3M 正式验证
#
# 对标 pe_dinov3_sigreg_siglip_muon，仅加 --neg-mode antipodal
# CC3M 有模态 GAP（COCO 无），这才是 antipodal 的真正战场
# ════════════════════════════════════════════════════════════════════════════

run_wds "anti_sigreg_muon" "PE-Core-B-16-dinov3" 29560 \
    "--siglip --neg-mode antipodal --sigreg-target cls --sigreg-weight 1e-4 \
     --epochs 10 --warmup 512 \
     --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
     --probe-data ${PROBE_TSV}"

# ════════════════════════════════════════════════════════════════════════════
# Orthogonal SigLIP — CC3M 正式验证
#
# 对标 pe_dinov3_sigreg_siglip_muon，仅加 --neg-mode orthogonal
# CC3M 有模态 GAP（COCO 无），orthogonal 的核心假设是消除 gap
# ════════════════════════════════════════════════════════════════════════════

run_wds "ortho_sigreg_muon" "PE-Core-B-16-dinov3" 29560 \
    "--siglip --neg-mode orthogonal --sigreg-target cls --sigreg-weight 1e-4 \
     --epochs 10 --warmup 512 \
     --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
     --probe-data ${PROBE_TSV}"

echo "======== wds_cc3m all done ========"
