#!/bin/bash
# CC3M sample-level curriculum experiments.
# Uses cc3m-tsv (extracted from cc3m-wds) so curriculum is exact sample-level,
# not shard-level approximation.

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

CC3M_TSV="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations/clip_train.tsv"
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8
PreGpuBS=512
LR=0.00034
MUON_LR=0.01

COMMON="--warmup 512 --epochs 10 \
    --precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 \
    --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

run() {
    local TAG=$1 PORT=$2 EXTRA=$3
    local NAME="cc3m_csv_cur_${TAG}_${TS}"
    echo "======== [cc3m_csv_cur] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${CC3M_TSV}" --val-data "${VAL}" \
        ${COMMON} ${EXTRA} --name "${NAME}" < /dev/null
}

# CSV-loader baseline: same data, no curriculum. Needed to separate loader-format effects from curriculum effects.
run "baseline" 29570 "${SIGREG_BASE}"

# COCO top: FPS reverse with DINOv3 epoch-0 geometry.
run "fpsrev_dinov3" 29571 "${SIGREG_BASE} --curriculum-strategy fps_reverse --curriculum-init dinov3"

# COCO strong FPS variant: PE-Core geometry, forward FPS.
run "fps_pecore" 29572 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init pe_core"

# Stable no-external-model variants.
run "fpsrev_self" 29573 "${SIGREG_BASE} --curriculum-strategy fps_reverse --curriculum-init self"
run "fps_self"    29574 "${SIGREG_BASE} --curriculum-strategy fps --curriculum-init self"

echo "======== CC3M CSV sample-level curriculum done (5 runs) ========"
