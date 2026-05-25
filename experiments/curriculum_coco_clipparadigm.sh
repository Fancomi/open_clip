#!/bin/bash
# COCO sample-level FPS curriculum: PE epoch0-only vs frozen PE, plus CLIP-paradigm feature extractors.
# SMOKE=1 runs all configs on karpathy_1cap for 1 epoch train+eval.

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

SMOKE=${SMOKE:-0}
TS=$(date +%m%d_%H%M)
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN_FULL="${COCO}/clip_train_dedup.tsv"
TRAIN_SMOKE="${COCO}/karpathy_1cap.tsv"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8
PreGpuBS=512
LR=0.00034
MUON_LR=0.01

if [ "${SMOKE}" = "1" ]; then
    TRAIN="${TRAIN_SMOKE}"
    EPOCHS=1
    WARMUP=0
    SAVE_FREQ=0
    WORKERS=8
    LOG_STEPS=1
    NAME_PREFIX="smoke_coco_clipfps"
else
    TRAIN="${TRAIN_FULL}"
    EPOCHS=20
    WARMUP=42
    SAVE_FREQ=2
    WORKERS=32
    LOG_STEPS=2
    NAME_PREFIX="coco_clipfps"
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

run() {
    local TAG=$1 PORT=$2 EXTRA=$3
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [coco_clipfps] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN}" --val-data "${VAL}" \
        ${COMMON} ${EXTRA} --name "${NAME}" < /dev/null
}

# Controls: epoch0-only vs frozen PE every epoch, and direction.
run "baseline"           29600 "${SIGREG_BASE}"
run "fps_pe_e0_random"   29601 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init pe_core --curriculum-epochs 1"
run "fps_pe_frozen_all"  29602 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init pe_core_always"
run "fpsrev_pe_e0_random"  29603 "${SIGREG_BASE} --curriculum-strategy fps_reverse --curriculum-init pe_core --curriculum-epochs 1"
run "fpsrev_pe_frozen_all" 29604 "${SIGREG_BASE} --curriculum-strategy fps_reverse --curriculum-init pe_core_always"

# CLIP paradigm sweep from analysis/clip_fps_probe.py: compare FPS direction across pretrained geometries.
for INIT in siglip2 datacomp dfn2b eva02 laion2b metaclip random_init; do
    run "fps_${INIT}"    29605 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init ${INIT}"
    run "fpsrev_${INIT}" 29606 "${SIGREG_BASE} --curriculum-strategy fps_reverse --curriculum-init ${INIT}"
done

echo "======== COCO CLIP-paradigm FPS curriculum done ========"
