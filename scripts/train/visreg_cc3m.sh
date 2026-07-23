#!/bin/bash
# visreg_cc3m.sh — VISReg vs SIGReg，冠军配方在 cc3m-tsv 上的 A/B + 组件消融
#
# 目标：把 VISReg（2606.02572）作为 SIGReg（LeJEPA 2511.08544）的替代正则器，
#       在历史 no-dino 冠军配方 proj_s15_sigreg（I→T R@1=0.2344@ep8）下对比。
#       冠军配方其余部分一字不改，唯一变量 = 正则项。
#
# 冠军配方（来源 wds_cc3m.sh + analysis/research/cc3m_text_dedup.md）：
#   PE-Core-B-16-dinov3 + --siglip --neg-mode projective --init-logit-scale ln(15)
#   --sigreg-target cls  --opt muon --muon-lr 0.01  --lr 3.4e-4  epochs10 warmup512
#
# 数据：cc3m-tsv（csv loader，制表符分隔，绝对路径，2,894,191 样本）
#   注：历史 23.44 在 cc3m-wds+resampled 上取得；本次用 tsv（顺序遍历）。
#   A 组是本 tsv 上的 SIGReg 锚点，A/B 同 tsv 同配方对比才是"替代关系"判据。
#
# VISReg 权重：由 scripts/tools/calib_visreg_weight.py 标定，使其 ‖∂L/∂z‖ ≈ SIGReg(1e-4)。
#   global_batch=4096 下标定结果 ≈ 1.83e-4（feat_std 0.5~2 区间稳定在 1.8e-4~4e-4）。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

# ── cc3m-tsv 数据源 ──────────────────────────────────────────────────────────
CC3M_TSV="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations/clip_train.tsv"
CC3M_N_TRAIN=2894191

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")

BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")
INIT_LS=$(python3 -c "import math; print(math.log(15))")

# 标定所得 VISReg 权重（可 override：VISREG_W=... ./visreg_cc3m.sh）
VISREG_W="${VISREG_W:-1.83e-4}"

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

# csv loader：制表符分隔（params 默认即 \t，不显式传 --csv-separator——
# 在双引号变量里 $'\t' 不会被解释为制表符，会当成字面量导致列名解析失败）
COMMON="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type csv --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

# 冠军配方（除正则项外全部固定）
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --sigreg-target cls \
    --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
    --probe-data ${PROBE_TSV}"

# run TAG PORT REG_EXTRA
run() {
    local TAG=$1 PORT=$2 REG_EXTRA=$3
    local NAME="visreg_cc3m_${TAG}_${TS}"
    echo "======== [visreg_cc3m] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${CC3M_TSV}" \
        --val-data "${VAL}" \
        ${COMMON} ${CHAMPION} ${REG_EXTRA} \
        --name "${NAME}" < /dev/null
}

# ── A：SIGReg 锚点（复现冠军，cc3m-tsv 上）────────────────────────────────────
run "A_sigreg"        29570 "--reg-method sigreg --sigreg-weight 1e-4"

# ── B：VISReg 全项（标定权重，λ 全 1）──────────────────────────────────────────
run "B_visreg_full"   29571 "--reg-method visreg --sigreg-weight ${VISREG_W} \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 1.0"

# ── C：VISReg scale-only（只留方差项）─────────────────────────────────────────
run "C_visreg_scale"  29572 "--reg-method visreg --sigreg-weight ${VISREG_W} \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 0.0 --visreg-lambda-center 0.0"

# ── D：VISReg shape-only（只留 SWD 分布形状项）────────────────────────────────
run "D_visreg_shape"  29573 "--reg-method visreg --sigreg-weight ${VISREG_W} \
    --visreg-lambda-scale 0.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

# ── E：VISReg 无 center（scale+shape）─────────────────────────────────────────
run "E_visreg_nocenter" 29574 "--reg-method visreg --sigreg-weight ${VISREG_W} \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

echo "======== visreg_cc3m all done ========"
