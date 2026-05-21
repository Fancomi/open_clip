#!/bin/bash
set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
GPUS=8; BS=512; LR=3.4e-4; MLR=0.01
PE_CKPT='/root/paddlejob/workspace/env_run/penghaotian/models/timm/PE-Core-B-16/open_clip_model.safetensors'
COCO='/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations'
VAL="${COCO}/karpathy_5cap.tsv"
P="${COCO}/karpathy_1cap.tsv"
CC3M_WDS='/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds'
CC3M_TRAIN="${CC3M_WDS}/cc3m-train-{0000..0575}.tar"
CC3M_N=2905954

NAME="wds_cc3m_posonly_sig_w0001_${TS}"
echo "======== ${NAME} ========"
torchrun --nproc_per_node=${GPUS} --master_port=29560 \
  -m open_clip_train.main --model PE-Core-B-16-dinov3 \
  --train-data "${CC3M_TRAIN}" --val-data "${VAL}" \
  --dataset-type webdataset --train-num-samples ${CC3M_N} \
  --csv-img-key filepath --csv-caption-key caption \
  --val-num-captions-per-image 5 \
  --precision amp_bf16 --workers 32 --batch-size ${BS} \
  --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
  --opt muon --muon-lr ${MLR} \
  --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
  --warmup 512 --epochs 10 \
  --siglip --grad-clip-norm 1.0 \
  --pretrained-text-path "${PE_CKPT}" --lock-text --sigreg-joint \
  --sigreg-target clip --sigreg-weight 0.0001 --pos-only sigmoid \
  --probe-data "${P}" \
  --name "${NAME}"
echo "======== DONE ========"
