#!/bin/bash
# visreg_wsweep.sh — 权重大范围对数扫描（跨数量级），回答「上升空间有多大」
#
# 动机（scripts/tools/probe_grad_ratio.py 的诊断）：
#   在最优 ckpt 上实测，正则项对 backbone 的梯度只有对比损失的 2.1e-07。
#   即：杠杆几乎完全没被拉动。此前的 sweep 只在 1.83e-4 周围 ±2× 打转，
#   全都落在同一个「无效区」里，难怪 0.5×/1×/2× 差异都在噪声级。
#
#   本实验跨 4 个数量级扫权重，目的不是"微调"，而是找出这个机制的
#   有效区间与崩溃点 —— 崩在哪里本身就回答了天花板在哪。
#
# 注：backbone 权重矩阵走 Muon（正交化更新，步长由 muon_lr 定），
#   梯度幅度被归一化，故"梯度占比"只能作量级参考，实际影响须由本实验确定。
#
# 单变量：只改 --sigreg-weight，其余 = 最优配方 E_s1sh1
#   （VISReg, scale:shape=1:1, no-center, K=256, 冠军超参, 10 epoch, 双指标）

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
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --opt muon --muon-lr ${MUON_LR} --lr ${LR} --probe-data ${PROBE_TSV} \
    --reg-method visreg --sigreg-slices 256 \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

run() {  # run TAG PORT TARGET WEIGHT
    local TAG=$1 PORT=$2 TGT=$3 W=$4
    local NAME="visreg_w_${TAG}_${TS}"
    echo "======== [wsweep] ${TAG} (target=${TGT} w=${W}) => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${CC3M_TSV}" --val-data "${VAL}" \
        ${COMMON} ${CHAMPION} \
        --sigreg-target ${TGT} --sigreg-weight ${W} \
        --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 失败/崩溃（本身是有效信息，继续下一组）"
}

# ── 第一阶段：cls 上跨数量级扫权重（基线 1.83e-4，梯度占比 2.1e-07）──────────
run "cls_1e2x"  29600 cls 1.83e-2     # 100×    → 占比 ~2e-05
run "cls_1e4x"  29601 cls 1.83e0      # 1e4×    → 占比 ~2e-03
# cls_1e6x 已取消：1e4× 时 IN-1k 已掉 1.7pt（崩溃点已定位），1e6× 无新信息。
#   算力让给 visreg_stage2.sh（机制改进：top-K / 混合高斯目标）。
# run "cls_1e6x"  29602 cls 1.83e2

# ── 第二阶段：cls_proj（MLP projector 缓冲，正则不直接压 backbone）───────────
# 诊断显示 cls_proj 裸 loss 大 21×，但透到 backbone 的梯度只大 1.8× → 需更大权重
run "proj_1e4x" 29603 cls_proj 1.83e0
run "proj_1e6x" 29604 cls_proj 1.83e2

echo "======== visreg_wsweep all done ========"
echo "对照基线 visreg_sweep_E_s1sh1_*: COCO i2t 24.06 / IN-1k top1 23.26"
