#!/bin/bash
# smoke.sh — 快速冒烟测试，每个 epoch 仅 10 步，秒级发现问题
# 与 quick.sh 完全相同的实验矩阵和参数，仅缩减数据量
#
# 配置：
#   train: CC3M 第 1 个 tar，train-num-samples=5120 → 10 steps/epoch (BS=64×8GPU)
#   val:   clip_val.tsv（完整，评估全流程）
#   epochs: 2（跑完 1 个 val 周期）
#   硬件: 8x H800

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"

CC3M_SRC="/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds"

SMOKE_DIR="/dev/shm/cc3m_smoke"

_cleanup() {
    echo "[smoke] 清理 /dev/shm/cc3m_smoke ..."
    rm -rf /dev/shm/cc3m_smoke
}
trap _cleanup EXIT INT TERM

if [ ! -d "${SMOKE_DIR}" ]; then
    echo "[smoke] 加载前 8 个 tar 到内存 (~4GB) ..."
    mkdir -p "${SMOKE_DIR}"
    cp "${CC3M_SRC}"/{00000..00007}.tar "${SMOKE_DIR}/"
    echo "[smoke] 加载完成"
else
    echo "[smoke] 检测到 ${SMOKE_DIR} 已存在，跳过复制"
fi

GPUS=8
BS=64
# workers=1, world_size=8 → 需要 >= 1*8=8 个 shard，恰好满足
# 10 steps/epoch: 10 * 64 * 8 = 5120
N_TRAIN=$(python3 -c "print(10 * ${BS} * ${GPUS})")

SMOKE_TRAIN="${SMOKE_DIR}/{00000..00007}.tar"

BASE_SMOKE="--precision amp_bf16 --workers 1 --epochs 2 --batch-size ${BS} \
    --lr 1e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

COMMON_SMOKE="--warmup 2 ${BASE_SMOKE} \
    --dataset-type webdataset --train-num-samples ${N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

run_smoke() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="smoke_${TAG}_${TS}"
    echo "======== [smoke] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${SMOKE_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON_SMOKE} \
        ${EXTRA} \
        --name "${NAME}"
}

# run_smoke "vit_leproj"       "ViT-B-16-exp"        29513 "--siglip --lejepa --lejepa-proj"
# run_smoke "pe_cls_leproj"    "PE-Core-B-16-cls"    29514 "--siglip --lejepa --lejepa-proj"
run_smoke "pe_dinov3_leproj" "PE-Core-B-16-dinov3" 29515 "--siglip --lejepa --lejepa-proj"
# run_smoke "dinov3_leproj"    "DINOv3-B-16-ape"     29517 "--siglip --lejepa --lejepa-proj"

# run_smoke "vit_le"           "ViT-B-16-exp"        29513 "--siglip --lejepa"
# run_smoke "pe_cls_le"        "PE-Core-B-16-cls"    29514 "--siglip --lejepa"
# run_smoke "pe_dinov3_le"     "PE-Core-B-16-dinov3" 29515 "--siglip --lejepa"
run_smoke "dinov3_le"        "DINOv3-B-16-ape"     29517 "--siglip --lejepa"

run_smoke "vit"              "ViT-B-16-exp"        29513 "--siglip"
# run_smoke "pe_cls"           "PE-Core-B-16-cls"    29514 "--siglip"
# run_smoke "pe_dinov3"        "PE-Core-B-16-dinov3" 29515 "--siglip"
# run_smoke "dinov3"           "DINOv3-B-16-ape"     29517 "--siglip"

# DINOv3 联合训练 smoke test（SigLIP + DINO + iBOT + KoLeo）
run_smoke "pe_dinov3_clip" "PE-Core-B-16" 29513 \
    "--siglip --dinov3 \
     --dino-local-crops-number 2 \
     --dino-head-prototypes 65536 \
     --dino-loss-weight 1.0 --ibot-loss-weight 1.0 --koleo-loss-weight 0.1 \
     --freeze-last-layer-epochs 1"

echo "======== smoke 全部通过 ========"
