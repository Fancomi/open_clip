#!/bin/bash
# CC12M 110K 50ep 实验: 验证充足训练量下采样策略差异
# 只跑 3 个代表性配置: random / kmeans_uniform_laion2b / fps_dinov3
# 目标: 13500 steps (50ep × 27 steps/ep), 对比 10ep 的 270 steps
set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
CC12M_SUBSETS="/root/paddlejob/workspace/env_run/penghaotian/datas/cc12m-wds/subsets"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8
PreGpuBS=512
GlobalBS=$((PreGpuBS * GPUS))
LR=$(python3 -c "import math; print(3.4e-4 * math.sqrt($GlobalBS / (8*512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt($GlobalBS / (8*512)))")

EPOCHS=50
WARMUP=42
NAME_PREFIX="cc12m_110k_50ep"

COMMON="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 10 --grad-checkpointing \
    --log-every-n-steps 2 --val-frequency 5 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 --epochs ${EPOCHS} --warmup ${WARMUP} \
    --delete-previous-checkpoint"

LOSS_OPTS="--siglip --neg-mode projective --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

run() {
    local TAG=$1 PORT=$2 TRAIN_TSV=$3
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [cc12m_50ep] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN_TSV}" --val-data "${VAL}" \
        ${COMMON} ${LOSS_OPTS} --name "${NAME}" < /dev/null
}

PORT=29880

run "random"               ${PORT} "${CC12M_SUBSETS}/random_110k.tsv"
PORT=$((PORT + 1))
run "kmeans_uniform_laion2b" ${PORT} "${CC12M_SUBSETS}/kmeans_uniform_laion2b_110k.tsv"
PORT=$((PORT + 1))
run "fps_dinov3"           ${PORT} "${CC12M_SUBSETS}/fps_dinov3_110k.tsv"

echo "======== CC12M 110K 50ep done ========"
