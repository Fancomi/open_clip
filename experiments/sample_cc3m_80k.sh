#!/bin/bash
# CC3M 80K 采样实验: 比较 FPS vs K-Means 采样策略 × 不同 teacher 特征空间
# SMOKE=1 冒烟测试 (1 epoch, karpathy_1cap)
set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

SMOKE=${SMOKE:-0}
TS=$(date +%m%d_%H%M)
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
CC3M_SUBSETS="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/subsets"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8
PreGpuBS=512
LR=0.00034
MUON_LR=0.01
TEACHERS="pe_core dinov3 siglip2 datacomp dfn2b eva02 laion2b metaclip"

if [ "${SMOKE}" = "1" ]; then
    N_SAMPLES=1000
    MAX_IMAGES="--max-images 5000"
    EPOCHS=1; WARMUP=0; SAVE_FREQ=0; WORKERS=8; LOG_STEPS=1
    NAME_PREFIX="smoke_cc3m_sample"
else
    N_SAMPLES=80000
    MAX_IMAGES=""
    EPOCHS=20; WARMUP=42; SAVE_FREQ=2; WORKERS=32; LOG_STEPS=2
    NAME_PREFIX="cc3m_sample"
fi

COMMON="--precision amp_bf16 --workers ${WORKERS} --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency ${SAVE_FREQ} --grad-checkpointing \
    --log-every-n-steps ${LOG_STEPS} --val-frequency 1 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 --epochs ${EPOCHS} --warmup ${WARMUP}"

if [ "${SMOKE}" != "1" ]; then
    COMMON="${COMMON} --delete-previous-checkpoint"
fi

SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: 生成采样子集 TSV
# ══════════════════════════════════════════════════════════════════════════════
echo "======== Phase 1: Sampling CC3M subsets (${N_SAMPLES} per config) ========"

NK=$((N_SAMPLES / 1000))

# Random baseline (instant)
python tools/sample_cc3m.py --teacher random --method random --n-samples ${N_SAMPLES} ${MAX_IMAGES}

# 并行提特征: 分两批 × 4 GPU 并行 (避免 IO 瓶颈)
BATCH1="pe_core dinov3 siglip2 datacomp"
BATCH2="dfn2b eva02 laion2b metaclip"

GPU_ID=0
for teacher in ${BATCH1}; do
    echo "[Phase1] Extracting ${teacher} on cuda:${GPU_ID} ..."
    python tools/sample_cc3m.py --teacher ${teacher} --method fps --n-samples ${N_SAMPLES} \
        --device cuda:${GPU_ID} ${MAX_IMAGES} &
    GPU_ID=$((GPU_ID + 1))
done
wait

GPU_ID=0
for teacher in ${BATCH2}; do
    echo "[Phase1] Extracting ${teacher} on cuda:${GPU_ID} ..."
    python tools/sample_cc3m.py --teacher ${teacher} --method fps --n-samples ${N_SAMPLES} \
        --device cuda:${GPU_ID} ${MAX_IMAGES} &
    GPU_ID=$((GPU_ID + 1))
done
wait
echo "======== Phase 1a: FPS done, features cached ========"

# K-Means 复用已缓存特征 (单 GPU 够用, 顺序执行)
for teacher in ${TEACHERS}; do
    python tools/sample_cc3m.py --teacher ${teacher} --method kmeans --n-samples ${N_SAMPLES} ${MAX_IMAGES}
done

echo "======== Phase 1 done: all subsets generated ========"

echo "======== Phase 1 done: all subsets generated ========"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: 训练
# ══════════════════════════════════════════════════════════════════════════════
echo "======== Phase 2: Training ========"

run() {
    local TAG=$1 PORT=$2 TRAIN_TSV=$3
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [cc3m_sample] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN_TSV}" --val-data "${VAL}" \
        ${COMMON} ${SIGREG_BASE} --name "${NAME}" < /dev/null
}

PORT=29850

# Random baseline
run "random" ${PORT} "${CC3M_SUBSETS}/random_${NK}k.tsv"
PORT=$((PORT + 1))

# FPS and K-Means for each teacher
for teacher in ${TEACHERS}; do
    run "fps_${teacher}" ${PORT} "${CC3M_SUBSETS}/fps_${teacher}_${NK}k.tsv"
    PORT=$((PORT + 1))
    run "kmeans_${teacher}" ${PORT} "${CC3M_SUBSETS}/kmeans_${teacher}_${NK}k.tsv"
    PORT=$((PORT + 1))
done

echo "======== All CC3M sampling experiments done ========"
