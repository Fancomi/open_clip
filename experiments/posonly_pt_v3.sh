#!/bin/bash
# Pos-only V3: 基于最优配置 (sigmoid + pt frozen + sigreg joint w=0.0005) 继续探索
# 核心问题: t2i 已追平对照, 但 i2t 差 3x (image间缺区分度)
# 策略: 
#   A. 调参: 更小sigreg / 更多epoch / 不同lr
#   B. 新设计: 加轻量image-image排斥力 (within-modal) 补i2t
#   C. 新设计: 加 KoLeo (推散image最近邻) 补i2t
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
  --delete-previous-checkpoint --warmup 42 \
  --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
  --val-num-captions-per-image 5 --opt muon --muon-lr ${MLR} \
  --probe-data ${P} --siglip --grad-clip-norm 1.0 \
  --pretrained-text-path ${PE_CKPT} --lock-text --sigreg-joint"

run() {
  local TAG=$1; shift
  local NAME="wmc_${TAG}_${TS}"
  echo "======== ${TAG} => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=29537 \
    -m open_clip_train.main --model PE-Core-B-16-dinov3 \
    --train-data "${TRAIN}" --val-data "${VAL}" \
    ${BASE} "$@" --name "${NAME}" < /dev/null || true
}

echo '======== V3: 从最优出发探索 i2t 提升 ========'

# A. 调参空间
# A1. 更小 sigreg (接近0, 让对齐力完全主导)
run "po3_sig_w0001" --sigreg-target clip --sigreg-weight 0.0001 --pos-only sigmoid --epochs 20

# A2. 50 epoch (更长训练)
run "po3_sig_50ep" --sigreg-target clip --sigreg-weight 0.0005 --pos-only sigmoid --epochs 50

# B. pos-only + within-modal image 排斥 (补 i2t 区分度)
# within-modal 只对 image 做排斥, 替代负样本的 image-image 区分力
run "po3_sig_wm_img05" --sigreg-target clip --sigreg-weight 0.0005 --pos-only sigmoid --epochs 20 \
  --within-modal-weight 0.5 --within-modal-sides img --within-modal-mode auxiliary

run "po3_sig_wm_img1" --sigreg-target clip --sigreg-weight 0.0005 --pos-only sigmoid --epochs 20 \
  --within-modal-weight 1.0 --within-modal-sides img --within-modal-mode auxiliary

# C. pos-only + KoLeo (推散 image 最近邻)
run "po3_sig_koleo01" --sigreg-target clip --sigreg-weight 0.0005 --pos-only sigmoid --epochs 20 \
  --koleo-weight 0.1

run "po3_sig_koleo05" --sigreg-target clip --sigreg-weight 0.0005 --pos-only sigmoid --epochs 20 \
  --koleo-weight 0.5

# D. pos-only + projective (|cos| 让 image 更均匀分布在球面)
run "po3_sig_proj" --sigreg-target clip --sigreg-weight 0.0005 --pos-only sigmoid --epochs 20 \
  --neg-mode projective

echo '======== ALL 8 V3 DONE ========'
