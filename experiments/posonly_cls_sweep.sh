#!/bin/bash
set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
GPUS=8; BS=512; LR=3.4e-4; MLR=0.01
PE_CKPT='/root/paddlejob/workspace/env_run/penghaotian/models/timm/PE-Core-B-16/open_clip_model.safetensors'
COCO='/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations'
TRAIN="${COCO}/clip_train_dedup.tsv"
VAL="${COCO}/karpathy_5cap.tsv"
P="${COCO}/karpathy_1cap.tsv"

BASE="--precision amp_bf16 --workers 32 --batch-size ${BS} \
  --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
  --save-frequency 4 --grad-checkpointing --log-every-n-steps 4 --val-frequency 4 \
  --delete-previous-checkpoint --warmup 42 \
  --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
  --val-num-captions-per-image 5 --opt muon --muon-lr ${MLR} \
  --probe-data ${P} --siglip --grad-clip-norm 1.0 \
  --pretrained-text-path ${PE_CKPT} --lock-text \
  --pos-only sigmoid --sigreg-target cls"

run() {
  local TAG=$1; shift
  local NAME="wmc_${TAG}_${TS}"
  echo "======== ${TAG} => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=29537 \
    -m open_clip_train.main --model PE-Core-B-16-dinov3 \
    --train-data "${TRAIN}" --val-data "${VAL}" \
    ${BASE} "$@" --name "${NAME}" < /dev/null || true
}

echo '======== cls target sweep ========'

# SIGReg weight sweep (cls target, 50ep)
run "cls_w1e5"    --sigreg-weight 0.00001 --epochs 50
run "cls_w5e5"    --sigreg-weight 0.00005 --epochs 50
run "cls_w5e4"    --sigreg-weight 0.0005  --epochs 50

# 100ep (看是否继续涨)
run "cls_100ep"   --sigreg-weight 0.0001  --epochs 100

# LR sweep (cls, 50ep, w=0.0001)
run "cls_lr5e4"   --sigreg-weight 0.0001  --epochs 50 --lr 5e-4 --muon-lr 0.015
run "cls_lr2e4"   --sigreg-weight 0.0001  --epochs 50 --lr 2e-4 --muon-lr 0.007

# joint vs separate (cls维度img/txt不同, joint会fallback到sep, 但确认一下)
run "cls_sep"     --sigreg-weight 0.0001  --epochs 50
run "cls_joint"   --sigreg-weight 0.0001  --epochs 50 --sigreg-joint

echo '======== ALL 8 DONE ========'
