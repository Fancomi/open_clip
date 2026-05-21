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
  --save-frequency 2 --grad-checkpointing --log-every-n-steps 2 --val-frequency 2 \
  --delete-previous-checkpoint --warmup 42 --epochs 20 \
  --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
  --val-num-captions-per-image 5 --opt muon --muon-lr ${MLR} \
  --probe-data ${P} --siglip --grad-clip-norm 1.0 \
  --pretrained-text-path ${PE_CKPT} --lock-text --sigreg-joint"

run() {
  local TAG=$1; shift
  local NAME="wmc_${TAG}_${TS}"
  echo "======== ${TAG} => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=29537 \
    -m open_clip_train.main --model PE-Core-B-16 \
    --train-data "${TRAIN}" --val-data "${VAL}" \
    ${BASE} "$@" --name "${NAME}" < /dev/null || true
}

echo '======== Pretrained Text Pos-Only (対標 rlit_A = 0.2560) ========'

run "po_sig_pt_w001" --sigreg-target clip --sigreg-weight 0.01 --pos-only sigmoid
run "po_sig_pt_w0001" --sigreg-target clip --sigreg-weight 0.001 --pos-only sigmoid
run "po_mse_pt_w001" --sigreg-target clip --sigreg-weight 0.01 --pos-only mse
run "po_mse_pt_w0001" --sigreg-target clip --sigreg-weight 0.001 --pos-only mse

# 对照: 标准 SigLIP (有负样本) frozen text
run "rlit_siglip_ref" --sigreg-target clip --sigreg-weight 1e-4

echo '======== ALL 5 PT EXPERIMENTS DONE ========'
