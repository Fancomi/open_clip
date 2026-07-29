#!/bin/bash
# attnres_cc3m.sh -- AttnRes vs baseline on CC3M, matched everything else.
#
# Two arms, same seed / data / optimizer / step count:
#   base      ViT-B-32
#   attnres4  ViT-B-32 + AttnRes, block_size=4 (12 layers -> 3 anchors)
#
# Env knobs:
#   GPUS=1 SAMPLES=200000 BS=256 ./scripts/train/attnres_cc3m.sh
#   ARMS="attnres4" ./scripts/train/attnres_cc3m.sh     # single arm
#
# Note: --logs is NOT passed through torchrun -- its parser resolves the prefix
# against its own --logs-specs and dies with "ambiguous option". Runs land in
# ./logs/<TAG>_<arm>; set TAG to separate experiments.
set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

DATA_ROOT="/root/paddlejob/workspace/env_run/penghaotian/datas"
CC3M_TSV="${DATA_ROOT}/cc3m-tsv/annotations/clip_train.tsv"
VAL="${DATA_ROOT}/coco/annotations/karpathy_5cap.tsv"
IN_VAL="${DATA_ROOT}/imagenet-val"

GPUS=${GPUS:-1}
BS=${BS:-256}
SAMPLES=${SAMPLES:-400000}
EPOCHS=${EPOCHS:-1}
LR=${LR:-1e-3}
PORT=${PORT:-29610}
ARMS=${ARMS:-"base attnres4"}
TAG=${TAG:-attnres}

# --train-num-samples only gates webdataset; the csv path walks the whole TSV.
# Cut a deterministic head-N subset so both arms see exactly the same samples.
SUBSET="/tmp/attnres_cc3m_${SAMPLES}.tsv"
if [ ! -s "${SUBSET}" ]; then
    head -n $((SAMPLES + 1)) "${CC3M_TSV}" > "${SUBSET}"
    echo "[data] wrote ${SUBSET} ($(wc -l < "${SUBSET}") lines incl. header)"
fi

BASE_ARGS="--precision amp_bf16 --workers 8 --batch-size ${BS} \
    --lr ${LR} --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.2 \
    --warmup 500 --epochs ${EPOCHS} \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 --val-frequency 1 --zeroshot-frequency 1 \
    --log-every-n-steps 20 --seed 0"

run_arm() {
    local ARM=$1 MODEL=$2 EXTRA=$3
    echo "======== [${ARM}] model=${MODEL} ========"
    torchrun --nproc_per_node="${GPUS}" --master_port="${PORT}" \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${SUBSET}" --val-data "${VAL}" --imagenet-val "${IN_VAL}" \
        ${BASE_ARGS} ${EXTRA} --name "${TAG}_${ARM}" < /dev/null
    PORT=$((PORT + 1))
}

for ARM in ${ARMS}; do
    case "${ARM}" in
        base)     run_arm base     "ViT-B-32" "" ;;
        attnres4) run_arm attnres4 "ViT-B-32-attnres4" "" ;;
        attnres2) run_arm attnres2 "ViT-B-32-attnres2" "" ;;
        k3init)   run_arm k3init   "ViT-B-32-attnres4-k3init" "" ;;
        *) echo "unknown arm: ${ARM}" >&2; exit 1 ;;
    esac
done
