#!/bin/bash
# verify_imagenet_baseline.sh — 用两个开源预训练权重校准本仓库的 ImageNet-1k zero-shot 评测口径
#
# 目的：确认 src/open_clip_train/zero_shot.py 这条评测链路（IMAGENET_CLASSNAMES +
# OPENAI_IMAGENET_TEMPLATES + ImageFolder 类序 + preprocess）能复现 model card / paper 数字。
# 若能对齐，则我们自训 run 的 imagenet 指标口径可信。
#
# 参考值：
#   PE-Core-B-16      IN-1k 78.4  (model card README.md)
#   ViT-B-16-SigLIP2  IN-1k 78.2  (SigLIP 2 paper 2502.14786, B/16 @224)
#
# 走 local-dir: schema，模型结构 + preprocess_cfg + tokenizer 全部取自权重目录里的
# open_clip_config.json，不经过我们改过的 model_configs/*.json，是干净的上游口径。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'
export HF_HUB_OFFLINE=1

ROOT=/root/paddlejob/workspace/env_run/penghaotian
IMNVAL="${ROOT}/datas/imagenet-val"
TIMM="${ROOT}/models/timm"

eval_one() {  # eval_one <model_dir> <tag> <gpu> [extra args...]
  local DIR=$1 TAG=$2 GPU=$3; shift 3
  echo "======== [verify-imnzs] ${TAG} ========"
  CUDA_VISIBLE_DEVICES=${GPU} python3 -m open_clip_train.main \
    --model "local-dir:${DIR}" \
    --imagenet-val "${IMNVAL}" \
    --precision amp_bf16 --workers 8 --batch-size 256 \
    --zeroshot-frequency 1 \
    --name "verify_imnzs_${TAG}" "$@" < /dev/null 2>&1 \
    | grep -aE "imagenet-zeroshot-val-top1|imagenet-zeroshot-val-top5|Final image preprocessing|Using .*Tokenizer|Error|Traceback"
}

GPU=${1:-0}
eval_one "${TIMM}/PE-Core-B-16"     pe_core_b16 ${GPU}
eval_one "${TIMM}/ViT-B-16-SigLIP2" siglip2_b16 ${GPU}

echo "======== verify done ========"
