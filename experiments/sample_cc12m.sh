#!/bin/bash
# CC12M 1/100 采样实验: 验证低保留率(~1%)下 FPS/K-Means 是否优于 Random
# 参考 Meta FAIR 2024 (arxiv:2405.15613) 的 1/100 采样比例
# 训练配置对齐 wm_coco.sh projective 实验
set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

SMOKE=${SMOKE:-0}
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

TEACHERS="pe_core dinov3 siglip2 datacomp dfn2b eva02 laion2b metaclip"

# CC12M ~11M, 1/100 = 110K
if [ "${SMOKE}" = "1" ]; then
    N_SAMPLES=5000
    EPOCHS=1; WARMUP=0; SAVE_FREQ=0; WORKERS=8; LOG_STEPS=1; VAL_FREQ=1
    NAME_PREFIX="smoke_cc12m"
else
    N_SAMPLES=110000
    # 110K / 4096 ≈ 27 steps/ep × 10ep = 270 steps (对齐 COCO 400 steps 量级)
    EPOCHS=10; WARMUP=42; SAVE_FREQ=2; WORKERS=32; LOG_STEPS=2; VAL_FREQ=2
    NAME_PREFIX="cc12m_110k"
fi

# 对齐 wm_coco.sh projective 配置
COMMON="--precision amp_bf16 --workers ${WORKERS} --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency ${SAVE_FREQ} --grad-checkpointing \
    --log-every-n-steps ${LOG_STEPS} --val-frequency ${VAL_FREQ} \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 --epochs ${EPOCHS} --warmup ${WARMUP} \
    --delete-previous-checkpoint"

LOSS_OPTS="--siglip --neg-mode projective --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: 采样
# ══════════════════════════════════════════════════════════════════════════════
echo "======== Phase 1: Sampling (${N_SAMPLES}, 1/100 of CC12M) ========"
NK=$((N_SAMPLES / 1000))

python tools/sample_cc12m.py sample --teacher random --method random --n-samples ${N_SAMPLES}
for teacher in ${TEACHERS}; do
    python tools/sample_cc12m.py sample --teacher ${teacher} --method fps --n-samples ${N_SAMPLES}
    python tools/sample_cc12m.py sample --teacher ${teacher} --method kmeans_uniform --n-samples ${N_SAMPLES}
done
echo "======== Phase 1 done ========"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 2: 导出 TSV (一次遍历)
# ══════════════════════════════════════════════════════════════════════════════
echo "======== Phase 2: Export TSVs ========"
python tools/sample_cc12m.py export_all
echo "======== Phase 2 done ========"

# ══════════════════════════════════════════════════════════════════════════════
# Phase 3: 训练 (17 configs × 10ep × ~27 steps/ep = ~270 steps each)
# ══════════════════════════════════════════════════════════════════════════════
echo "======== Phase 3: Training (${EPOCHS}ep, ~$((N_SAMPLES / GlobalBS * EPOCHS)) steps/config) ========"

run() {
    local TAG=$1 PORT=$2 TRAIN_TSV=$3
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [cc12m] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN_TSV}" --val-data "${VAL}" \
        ${COMMON} ${LOSS_OPTS} --name "${NAME}" < /dev/null
}

PORT=29870

run "random" ${PORT} "${CC12M_SUBSETS}/random_${NK}k.tsv"
PORT=$((PORT + 1))

for teacher in ${TEACHERS}; do
    run "fps_${teacher}" ${PORT} "${CC12M_SUBSETS}/fps_${teacher}_${NK}k.tsv"
    PORT=$((PORT + 1))
    run "kmeans_uniform_${teacher}" ${PORT} "${CC12M_SUBSETS}/kmeans_uniform_${teacher}_${NK}k.tsv"
    PORT=$((PORT + 1))
done

echo "======== All CC12M 1/100 sampling experiments done ========"
