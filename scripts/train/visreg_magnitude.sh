#!/bin/bash
# visreg_magnitude.sh — VISReg 正则总强度 sweep（基于 sweep 结论：E=scale+shape 1:1
#   no-center 是最佳综合配方，配比不是瓶颈；唯一未探维度 = 正则总强度 weight）。
#
# 固定 E 配方（1:1 no-center），只扫 --sigreg-weight。当前 1.83e-4（梯度匹配 SIGReg）。
# 全部 --imagenet-val 出双指标。目标：看正则更强/更弱能否再顶高 E 的 24.06/23.26。

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

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"
COMMON="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type csv --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
    --imagenet-val ${IMNVAL}"
# E 配方：scale+shape 1:1, no-center
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --sigreg-target cls --reg-method visreg \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0 \
    --lr ${LR} --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

run() {  # run TAG PORT WEIGHT
    local TAG=$1 PORT=$2 W=$3
    local NAME="visreg_mag_${TAG}_${TS}"
    echo "======== [mag] ${TAG} (w=$W) => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${CC3M_TSV}" --val-data "${VAL}" \
        ${COMMON} ${CHAMPION} --sigreg-weight ${W} \
        --name "${NAME}" < /dev/null
}

# 0.5× / 2× 当前 1.83e-4（1× = E 已跑，不重复）
run "w0p5x" 29585 9.15e-5
run "w2x"   29586 3.66e-4

echo "======== visreg_magnitude all done ========"
