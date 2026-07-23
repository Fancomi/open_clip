#!/bin/bash
# eval_imagenet_zeroshot.sh — 对已完成的 A–E run 补测 ImageNet-1k zero-shot top1/top5
#
# open_clip 的 evaluate() 在 'train' not in data 时进入纯评测分支（main.py:765），
# 加 --imagenet-val 即触发 zero_shot_eval（IMAGENET_CLASSNAMES + OpenAI 模板）。
# 用 --resume 加载每组 best/last checkpoint，单卡评测（无 NCCL 依赖）。
#
# cls sigreg-target 用 nn.Identity projector，无额外参数，ckpt 可直接 load 进 CLIPLeJEPA。
#
# 用法：bash eval_imagenet_zeroshot.sh [epoch]   # epoch 默认取各组 COCO 峰值 epoch

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

ROOT=/root/paddlejob/workspace/env_run/penghaotian
IMNVAL="${ROOT}/datas/imagenet-val"
INIT_LS=$(python3 -c "import math; print(math.log(15))")
LOGROOT="./logs"

# run 目录 → COCO 峰值 epoch（来自实测；用 best epoch 的 ckpt 评 ImageNet）
declare -A PEAK=(
  [visreg_cc3m_A_sigreg_0723_0911]=7
  [visreg_cc3m_B_visreg_full_0723_0911]=7
  [visreg_cc3m_C_visreg_scale_0723_0911]=8
  [visreg_cc3m_D_visreg_shape_0723_0911]=7
)

eval_one() {  # eval_one <run_dir> <reg_method> <epoch> <gpu>
  local RUN=$1 RM=$2 EP=$3 GPU=$4
  local CKPT="${LOGROOT}/${RUN}/checkpoints/epoch_${EP}.pt"
  [ -f "$CKPT" ] || { echo "[skip] $RUN ep$EP 无 ckpt"; return 0; }
  echo "======== [imagenet-zs] ${RUN} ep${EP} (${RM}) ========"
  CUDA_VISIBLE_DEVICES=${GPU} python3 -m open_clip_train.main \
    --model "PE-Core-B-16-dinov3" \
    --imagenet-val "${IMNVAL}" \
    --resume "${CKPT}" \
    --precision amp_bf16 --workers 8 --batch-size 256 \
    --siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
    --sigreg-target cls --reg-method ${RM} \
    --zeroshot-frequency 1 \
    --name "imnzs_${RUN}_ep${EP}" < /dev/null 2>&1 \
    | grep -aE "imagenet-zeroshot-val-top1|imagenet-zeroshot-val-top5|Starting zero-shot"
}

# GPU 0/1 空闲（其余在训 E）；串行避免与训练抢卡
eval_one visreg_cc3m_A_sigreg_0723_0911      sigreg 7 0
eval_one visreg_cc3m_B_visreg_full_0723_0911 visreg 7 0
eval_one visreg_cc3m_C_visreg_scale_0723_0911 visreg 8 0
eval_one visreg_cc3m_D_visreg_shape_0723_0911 visreg 7 0

echo "======== imagenet-zs all done ========"
