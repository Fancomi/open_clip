#!/bin/bash
# visreg_sweep.sh — VISReg scale/shape 权重 sweep（基于 A-D 结论：B 等权次优，
#   C(scale)强分类、D(shape)强检索，center 疑似有害）。全部 no-center，扫权重面 + 量级。
#
# 目标：找同时 ≥C 分类(22.60) 且 ≥D 检索(24.64) 的甜点。全部 --imagenet-val 出双指标。
#
# 冠军配方（除正则外固定）：PE-Core-B-16-dinov3 + siglip + projective + init-scale ln15
#   + muon(0.01) + lr 3.4e-4 + cc3m-tsv + 10 epoch。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
ROOT=/root/paddlejob/workspace/env_run/penghaotian
COCO="${ROOT}/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"
IMNVAL="${ROOT}/datas/imagenet-val"
CC3M_TSV="${ROOT}/datas/cc3m-tsv/annotations/clip_train.tsv"
CC3M_N_TRAIN=2894191

GPUS=8; PreGpuBS=512
GlobalBS=$((PreGpuBS * GPUS))
LR=$(python3 -c "import math; print(3.4e-4 * math.sqrt(${GlobalBS}/(8*512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(${GlobalBS}/(8*512)))")
INIT_LS=$(python3 -c "import math; print(math.log(15))")
VISREG_W="${VISREG_W:-1.83e-4}"

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"
COMMON="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type csv --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
    --imagenet-val ${IMNVAL}"
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --sigreg-target cls --reg-method visreg \
    --lr ${LR} --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

run() {  # run TAG PORT WEIGHT LSCALE LSHAPE
    local TAG=$1 PORT=$2 W=$3 LS=$4 LSH=$5
    local NAME="visreg_sweep_${TAG}_${TS}"
    echo "======== [sweep] ${TAG} (w=$W scale=$LS shape=$LSH) => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${CC3M_TSV}" --val-data "${VAL}" \
        ${COMMON} ${CHAMPION} \
        --sigreg-weight ${W} \
        --visreg-lambda-scale ${LS} --visreg-lambda-shape ${LSH} --visreg-lambda-center 0.0 \
        --name "${NAME}" < /dev/null
}

# E 中心点（1:1 no-center）——先跑，作为 sweep 参照
run "E_s1sh1"  29580 ${VISREG_W} 1.0 1.0
# 偏分类（放大 scale，C 方向）
run "s2sh1"    29581 ${VISREG_W} 2.0 1.0
# 偏检索（放大 shape，D 方向）
run "s1sh2"    29582 ${VISREG_W} 1.0 2.0

echo "======== visreg_sweep all done ========"
