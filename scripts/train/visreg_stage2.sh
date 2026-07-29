#!/bin/bash
# visreg_stage2.sh — 第二阶段：在「正则真正起作用」的高权重区验证机制改进
#
# 为什么现在才跑这些方案（重要的方法论纠正）：
#   probe_grad_ratio.py 实测：1× 权重下正则对 backbone 的梯度只占对比损失 2.1e-07。
#   在那个工作点上，任何关于正则「估计质量」的改进（top-K 挑方向、正交化、
#   闭式解）都会被那个近零系数抹掉 —— 之前的小规模否定测试全都做在了错误的工作点。
#   k32 的结果印证了这点：K 砍到 1/8，COCO 24.06→23.66、IN 23.26→23.23，几乎无变化。
#
#   第一阶段权重扫描找到了有效区：100× 时 IN-1k 23.58（>基线 23.26），
#   1e4× 时裸 loss 从 0.235 压到 0.0065（正则第一次真正约束住分布），
#   但 IN 掉到 21.58 —— 说明强正则先伤分类判别性。
#
# 本阶段的三个假设：
#   1. cls_1e3x  : 填 100×–1e4× 的空隙，定出 IN-1k 的权重峰值
#   2. topk      : 方向质量只在有杠杆时才显现 → 在最优权重上开 top-K
#   3. mixture   : 实测真实特征多岛（同簇率 66-68%），而标准高斯目标是单峰。
#                  1e4× 时 IN 崩 1.7pt 很可能正是强行压成单峰毁了语义聚类。
#                  换成混合高斯目标 → 高权重下不伤分类（唯一可能同时改善两指标的路）
#
# 单变量原则：2、3 都在 1 定出的最优权重上跑，其余 = 最优配方 E_s1sh1。

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

# 第一阶段最优权重区（100× 在 IN 上最好）。W_BEST 可 override。
W_BEST="${W_BEST:-1.83e-2}"

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"
COMMON="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type csv --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
    --imagenet-val ${IMNVAL}"
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --opt muon --muon-lr ${MUON_LR} --lr ${LR} --probe-data ${PROBE_TSV} \
    --sigreg-target cls --reg-method visreg --sigreg-slices 256 \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

run() {  # run TAG PORT WEIGHT EXTRA
    local TAG=$1 PORT=$2 W=$3 EXTRA=$4
    local NAME="visreg_s2_${TAG}_${TS}"
    echo "======== [stage2] ${TAG} (w=${W} ${EXTRA}) => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${CC3M_TSV}" --val-data "${VAL}" \
        ${COMMON} ${CHAMPION} \
        --sigreg-weight ${W} ${EXTRA} \
        --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 失败/崩溃（本身是信息，继续）"
}

# ① 填权重空隙：1e3×，定 IN-1k 峰值位置
run "w1e3x"      29610 1.83e-1 ""

# ② top-K 挑方向 @ 有效权重（池 1024 挑 256）
run "topk"       29611 ${W_BEST} "--visreg-topk-pool 1024"

# ③ 混合高斯目标 @ 有效权重（5 分量，间距 2σ）
run "mix5"       29612 ${W_BEST} "--visreg-mixture 5 --visreg-mixture-sep 2.0"

# ④ 混合高斯 @ 高权重（1e4×）—— 检验"多岛目标能否救回强正则下崩掉的分类"
run "mix5_1e4x"  29613 1.83e0   "--visreg-mixture 5 --visreg-mixture-sep 2.0"

echo "======== visreg_stage2 all done ========"
echo "对照：E_s1sh1(1×) COCO 24.06 / IN 23.26 ; cls_1e2x(100×) COCO 23.58 / IN 23.58"
