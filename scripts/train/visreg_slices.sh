#!/bin/bash
# visreg_slices.sh — K（切片数）敏感性实验
#
# 目的：判断「投影方向的数量/质量」是否是当前瓶颈。
#   若 K=32 相比 K=256 几乎不掉点 ⟹ 本场景对方向不敏感，top-K 挑方向等
#   方向工程的收益上限很低，不值得投入。
#   若明显掉点 ⟹ 方向确实是瓶颈，值得做 top-K / 正交化 / 学习 slicing 分布。
#
# 单变量：只改 --sigreg-slices，其余 = 已确认的最优配方 E_s1sh1
#   （VISReg, scale:shape=1:1, no-center, weight=1.83e-4）
#
# 注意：实际等效方向数 = K × 8卡（各卡独立采样）
#   K=32  → 等效 256
#   K=256 → 等效 2048（基线）

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
# 最优配方 E_s1sh1：scale:shape=1:1，no-center
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --sigreg-target cls --reg-method visreg \
    --sigreg-weight ${VISREG_W} \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0 \
    --lr ${LR} --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

run() {  # run TAG PORT SLICES
    local TAG=$1 PORT=$2 K=$3
    local NAME="visreg_slices_${TAG}_${TS}"
    echo "======== [slices] ${TAG} (K=${K}, 等效 $((K*GPUS))) => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${CC3M_TSV}" --val-data "${VAL}" \
        ${COMMON} ${CHAMPION} \
        --sigreg-slices ${K} \
        --name "${NAME}" < /dev/null
}

# K=32：等效 256 方向（1/8 于基线）。若不掉点 → 方向不是瓶颈
run "k32"  29590 32
# K=8 已取消：在「总杠杆仅 1.7pt」的前提下掉点空间被压在噪声级，信息量低。
#   算力让给 visreg_wsweep.sh（跨数量级权重扫描）。
# run "k8"   29591 8

echo "======== visreg_slices all done ========"
echo "对照基线：visreg_sweep_E_s1sh1_* (K=256, 等效 2048) COCO 24.06 / IN-1k 23.26"
