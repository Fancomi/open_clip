#!/bin/bash
# CC3M 50万采样实验: FPS vs K-Means vs Random, 使用 wds_cc3m 正式配置
# 配置: pe_dinov3_sigreg_siglip_muon + --neg-mode projective
# 数据量: 500K (对比 80K 的 3120 steps → 此处 ~1220 steps/ep × 10ep = 12200 steps)
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
GlobalBS=$((PreGpuBS * GPUS))
LR=$(python3 -c "import math; print(3.4e-4 * math.sqrt($GlobalBS / (8*512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt($GlobalBS / (8*512)))")

TEACHERS="pe_core dinov3 siglip2 datacomp dfn2b eva02 laion2b metaclip"

if [ "${SMOKE}" = "1" ]; then
    N_SAMPLES=5000
    MAX_IMAGES="--max-images 20000"
    EPOCHS=1; WARMUP=0; SAVE_FREQ=0; WORKERS=8; LOG_STEPS=1
    NAME_PREFIX="smoke_cc3m_500k"
else
    N_SAMPLES=500000
    MAX_IMAGES=""
    EPOCHS=10; WARMUP=512; SAVE_FREQ=1; WORKERS=32; LOG_STEPS=1
    NAME_PREFIX="cc3m_500k"
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

LOSS_OPTS="--siglip --neg-mode projective --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: 采样 (复用已缓存特征, 只需重新采 500K)
# ══════════════════════════════════════════════════════════════════════════════
echo "======== Phase 1: Sampling CC3M subsets (${N_SAMPLES} per config) ========"

NK=$((N_SAMPLES / 1000))

# Random baseline
python tools/sample_cc3m.py --teacher random --method random --n-samples ${N_SAMPLES} ${MAX_IMAGES}

# FPS (复用已有特征缓存, 顺序即可)
for teacher in ${TEACHERS}; do
    python tools/sample_cc3m.py --teacher ${teacher} --method fps --n-samples ${N_SAMPLES} ${MAX_IMAGES}
done

# K-Means
for teacher in ${TEACHERS}; do
    python tools/sample_cc3m.py --teacher ${teacher} --method kmeans --n-samples ${N_SAMPLES} ${MAX_IMAGES}
done

echo "======== Phase 1 done ========"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: 训练
# ══════════════════════════════════════════════════════════════════════════════
echo "======== Phase 2: Training (${EPOCHS} epochs, ~$((N_SAMPLES / GlobalBS * EPOCHS)) steps) ========"

run() {
    local TAG=$1 PORT=$2 TRAIN_TSV=$3
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [cc3m_500k] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN_TSV}" --val-data "${VAL}" \
        ${COMMON} ${LOSS_OPTS} --name "${NAME}" < /dev/null
}

PORT=29860

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

echo "======== All CC3M 500K sampling experiments done ========"
