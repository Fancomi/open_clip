#!/bin/bash
# smoke_visreg.sh — VISReg 集成冒烟测试
#
# 正式投递前，对 SIGReg / VISReg 两条 loss 路径各跑几步 + 强制一次 COCO eval，
# 验证：训练与 eval 两条路径跑通、loss 有限（无 NaN）、分布式 gather 正常。
# 用极小 train-num-samples 让每个 epoch 只有几步。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"
CC3M_TSV="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations/clip_train.tsv"

GPUS=${GPUS:-8}
PreGpuBS=${PreGpuBS:-256}
GlobalBS=$((PreGpuBS * GPUS))
SMOKE_N=$((GlobalBS * 4))   # 4 steps/epoch
INIT_LS=$(python3 -c "import math; print(math.log(15))")
LR=3.4e-4
MUON_LR=0.01
VISREG_W="${VISREG_W:-1.83e-4}"

BASE="--precision amp_bf16 --workers 8 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 --epochs 1 --warmup 2 \
    --dataset-type csv --train-num-samples ${SMOKE_N} \
    --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5"

CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --sigreg-target cls --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

PASS=0; FAIL=0
smoke() {
    local TAG=$1 PORT=$2 REG=$3
    local NAME="smoke_visreg_${TAG}"
    echo "======== [smoke] ${TAG} ========"
    if torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${CC3M_TSV}" --val-data "${VAL}" \
        ${BASE} ${CHAMPION} ${REG} --name "${NAME}" < /dev/null 2>&1 | tail -40; then
        echo "[smoke] ${TAG} ... PASS"; PASS=$((PASS+1))
    else
        echo "[smoke] ${TAG} ... FAIL"; FAIL=$((FAIL+1))
    fi
}

smoke "sigreg" 29580 "--reg-method sigreg --sigreg-weight 1e-4"
smoke "visreg" 29581 "--reg-method visreg --sigreg-weight ${VISREG_W} \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 1.0"

echo "======== PASSED=${PASS} FAILED=${FAIL} ========"
[ "${FAIL}" -eq 0 ]
