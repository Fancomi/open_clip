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
  --delete-previous-checkpoint --warmup 42 --epochs 50 \
  --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
  --val-num-captions-per-image 5 --opt muon --muon-lr ${MLR} \
  --probe-data ${P} --siglip --grad-clip-norm 1.0 \
  --pretrained-text-path ${PE_CKPT} --lock-text \
  --pos-only sigmoid --sigreg-weight 0.0001"

run() {
  local TAG=$1; shift
  local NAME="wmc_${TAG}_${TS}"
  echo "======== ${TAG} => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=29537 \
    -m open_clip_train.main --model PE-Core-B-16-dinov3 \
    --train-data "${TRAIN}" --val-data "${VAL}" \
    ${BASE} "$@" --name "${NAME}" < /dev/null || true
}

echo '======== SIGReg target ablation (50ep) ========'

# cls: backbone CLS token (768d), image only (text是frozen不同维度)
run "po_cls" --sigreg-target cls

# cls_proj: MLP projector on CLS (512d)
run "po_cls_proj" --sigreg-target cls_proj

# 对照: clip (已有结果 ~0.0298 @50ep)

echo '======== DONE ========'
