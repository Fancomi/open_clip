#!/bin/bash
# 数据采样实验: 比较 FPS / K-Means / Random 采样策略 × teacher 特征空间
# 用法:
#   bash scripts/train/sample.sh cc3m_80k     # CC3M 80K, 20ep, csv 全 teacher
#   bash scripts/train/sample.sh cc3m_500k    # CC3M 500K, 10ep, projective
#   bash scripts/train/sample.sh cc12m_110k   # CC12M 1/100(110K), 10ep, projective
#   bash scripts/train/sample.sh cc12m_50ep   # CC12M 110K, 50ep, 仅 3 代表配置
#   SMOKE=1 bash scripts/train/sample.sh cc3m_80k   # 冒烟 (1ep, 小样本)
set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

PRESET="${1:?用法: sample.sh <cc3m_80k|cc3m_500k|cc12m_110k|cc12m_50ep>}"
SMOKE=${SMOKE:-0}
TS=$(date +%m%d_%H%M)
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8; PreGpuBS=512; GlobalBS=$((PreGpuBS * GPUS))
LR=$(python3 -c "import math; print(3.4e-4 * math.sqrt($GlobalBS / (8*512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt($GlobalBS / (8*512)))")
TEACHERS="pe_core dinov3 siglip2 datacomp dfn2b eva02 laion2b metaclip"

# ── 预设参数 ──────────────────────────────────────────────────────────────────
# DATASET: cc3m|cc12m  KMETHOD: kmeans|kmeans_uniform  NEG: 空|--neg-mode projective
# EXPORT: 1 表示需 export_all(cc12m) PORT: 起始端口
case "${PRESET}" in
    cc3m_80k)
        DATASET=cc3m; KMETHOD=kmeans; NEG=""; EXPORT=0; PORT=29850
        N_FULL=80000; EP_FULL=20; WARM_FULL=42; SAVE_FULL=2; LOG_FULL=2; VF_FULL=1
        N_SMOKE=1000; MAXIMG="--max-images 5000" ;;
    cc3m_500k)
        DATASET=cc3m; KMETHOD=kmeans; NEG="--neg-mode projective"; EXPORT=0; PORT=29860
        N_FULL=500000; EP_FULL=10; WARM_FULL=512; SAVE_FULL=1; LOG_FULL=1; VF_FULL=1
        N_SMOKE=5000; MAXIMG="--max-images 20000" ;;
    cc12m_110k)
        DATASET=cc12m; KMETHOD=kmeans_uniform; NEG="--neg-mode projective"; EXPORT=1; PORT=29870
        N_FULL=110000; EP_FULL=10; WARM_FULL=42; SAVE_FULL=2; LOG_FULL=2; VF_FULL=2
        N_SMOKE=5000; MAXIMG="" ;;
    cc12m_50ep)
        DATASET=cc12m; KMETHOD=kmeans_uniform; NEG="--neg-mode projective"; EXPORT=0; PORT=29880
        N_FULL=110000; EP_FULL=50; WARM_FULL=42; SAVE_FULL=10; LOG_FULL=2; VF_FULL=5
        REPRESENTATIVE=1 ;;   # 仅跑 random / kmeans_uniform_laion2b / fps_dinov3
    *) echo "未知预设: ${PRESET}"; exit 1 ;;
esac

SUBSETS="/root/paddlejob/workspace/env_run/penghaotian/datas/${DATASET}-$([ $DATASET = cc3m ] && echo tsv || echo wds)/subsets"
SAMPLER="tools/sample_${DATASET}.py"
[ "${DATASET}" = "cc12m" ] && SAMPLE_CMD="sample" || SAMPLE_CMD=""

# ── SMOKE / FULL 运行参数 ─────────────────────────────────────────────────────
if [ "${SMOKE}" = "1" ]; then
    N_SAMPLES=${N_SMOKE}; EPOCHS=1; WARMUP=0; SAVE_FREQ=0; WORKERS=8; LOG_STEPS=1; VAL_FREQ=1
    NAME_PREFIX="smoke_${PRESET}"; DEL_CKPT=""
else
    N_SAMPLES=${N_FULL}; EPOCHS=${EP_FULL}; WARMUP=${WARM_FULL}; SAVE_FREQ=${SAVE_FULL}
    WORKERS=32; LOG_STEPS=${LOG_FULL}; VAL_FREQ=${VF_FULL}
    NAME_PREFIX="${PRESET}"; DEL_CKPT="--delete-previous-checkpoint"
    MAXIMG=""
fi
NK=$((N_SAMPLES / 1000))

COMMON="--precision amp_bf16 --workers ${WORKERS} --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency ${SAVE_FREQ} --grad-checkpointing \
    --log-every-n-steps ${LOG_STEPS} --val-frequency ${VAL_FREQ} \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 --epochs ${EPOCHS} --warmup ${WARMUP} ${DEL_CKPT}"
LOSS_OPTS="--siglip ${NEG} --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

smp() { python "${SAMPLER}" ${SAMPLE_CMD} "$@"; }

run() {
    local TAG=$1 P=$2 TRAIN_TSV=$3
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [${PRESET}] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${P} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN_TSV}" --val-data "${VAL}" \
        ${COMMON} ${LOSS_OPTS} --name "${NAME}" < /dev/null
}

# ── 代表性配置模式 (cc12m_50ep): 跳过采样, 直接训练 3 个已有子集 ─────────────────
if [ "${REPRESENTATIVE:-0}" = "1" ]; then
    run "random"                 ${PORT}       "${SUBSETS}/random_${NK}k.tsv"
    run "kmeans_uniform_laion2b" $((PORT + 1)) "${SUBSETS}/kmeans_uniform_laion2b_${NK}k.tsv"
    run "fps_dinov3"             $((PORT + 2)) "${SUBSETS}/fps_dinov3_${NK}k.tsv"
    echo "======== ${PRESET} done ========"; exit 0
fi

# ── Phase 1: 采样 ─────────────────────────────────────────────────────────────
echo "======== Phase 1: Sampling ${DATASET} (${N_SAMPLES}/config) ========"
smp --teacher random --method random --n-samples ${N_SAMPLES} ${MAXIMG}
for t in ${TEACHERS}; do
    smp --teacher ${t} --method fps     --n-samples ${N_SAMPLES} ${MAXIMG}
    smp --teacher ${t} --method ${KMETHOD} --n-samples ${N_SAMPLES} ${MAXIMG}
done
[ "${EXPORT}" = "1" ] && { echo "======== Phase 1b: Export TSVs ========"; python "${SAMPLER}" export_all; }
echo "======== Phase 1 done ========"

# ── Phase 2: 训练 (17 configs) ────────────────────────────────────────────────
echo "======== Phase 2: Training (${EPOCHS}ep) ========"
run "random" ${PORT} "${SUBSETS}/random_${NK}k.tsv"; PORT=$((PORT + 1))
for t in ${TEACHERS}; do
    run "fps_${t}"            ${PORT} "${SUBSETS}/fps_${t}_${NK}k.tsv"; PORT=$((PORT + 1))
    run "${KMETHOD}_${t}"     ${PORT} "${SUBSETS}/${KMETHOD}_${t}_${NK}k.tsv"; PORT=$((PORT + 1))
done
echo "======== All ${PRESET} sampling experiments done ========"
