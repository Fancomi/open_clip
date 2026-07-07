#!/bin/bash
# train_video_frames.sh — PE-Core 视频帧对比学习训练
#
# 使用 muscle_wiki 去重+FPS 划分后的数据（split.json）
# PE-Core Stage 1 方案：每视频随机采 1 帧，标准 SigLIP 对比学习
#
# 前置: 先运行 scripts/video/split_video_data.py 生成 split.json

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ============ 数据 ============
VIDEO_ROOT="/root/paddlejob/workspace/env_run/penghaotian/datas/muscle_wiki"
SPLIT_FILE="${VIDEO_ROOT}/split.json"
if [ ! -f "${SPLIT_FILE}" ]; then
    echo "[错误] ${SPLIT_FILE} 不存在, 请先运行 scripts/video/split_video_data.py"
    exit 1
fi
# 从 split.json 读取 train 样本数
N_TRAIN=$(python3 -c "import json; d=json.load(open('${SPLIT_FILE}')); print(len(d['train']))")
echo "[数据] train=${N_TRAIN}, split=${SPLIT_FILE}"

# ============ 模型 ============
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"

# ============ 硬件 ============
GPUS=${GPUS:-1}
BS=${BS:-64}
EPOCHS=${EPOCHS:-10}
LR=${LR:-5e-6}
# warmup 10% of first epoch
WARMUP=$(python3 -c "import math; print(max(10, math.ceil(${N_TRAIN} / (${BS} * ${GPUS}) * 0.1)))")

# ============ 公共参数 ============
BASE="--precision amp_bf16 --workers 8 --batch-size ${BS} \
    --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.05 \
    --save-frequency 5 --grad-checkpointing --log-every-n-steps 10 --val-frequency 1 \
    --dataset-type video_frame --train-num-samples ${N_TRAIN} \
    --epochs ${EPOCHS} --warmup ${WARMUP}"

# ============ 运行 ============
NAME="pe_video_ft_${TS}"
echo "======== [video] ${NAME}  warmup=${WARMUP} ========"
torchrun --nproc_per_node=${GPUS} --master_port=29670 \
    -m open_clip_train.main \
    --model "PE-Core-B-16" \
    --pretrained "${PE_CKPT}" \
    --train-data "${VIDEO_ROOT}" --val-data "${VIDEO_ROOT}" \
    ${BASE} \
    --siglip --lr ${LR} \
    --name "${NAME}" "$@"
