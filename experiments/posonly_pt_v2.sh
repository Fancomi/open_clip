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

# 用 dinov3 模型（和 baseline 对齐）
BASE="--precision amp_bf16 --workers 32 --batch-size ${BS} \
  --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
  --save-frequency 2 --grad-checkpointing --log-every-n-steps 2 --val-frequency 2 \
  --delete-previous-checkpoint --warmup 42 --epochs 20 \
  --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
  --val-num-captions-per-image 5 --opt muon --muon-lr ${MLR} \
  --probe-data ${P} --siglip --grad-clip-norm 1.0"

run() {
  local TAG=$1; shift
  local NAME="wmc_${TAG}_${TS}"
  echo "======== ${TAG} => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=29537 \
    -m open_clip_train.main --model PE-Core-B-16-dinov3 \
    --train-data "${TRAIN}" --val-data "${VAL}" \
    ${BASE} "$@" --name "${NAME}" < /dev/null || true
}

echo '======== V2: 用dinov3模型 + 更多SIGReg权重探索 ========'

# 1. pos-only sigmoid, pretrained text, sigreg 0.001 (用dinov3模型对齐baseline)
run "po2_sig_pt_w0001" \
  --pretrained-text-path "${PE_CKPT}" --lock-text \
  --sigreg-target clip --sigreg-weight 0.001 --pos-only sigmoid --sigreg-joint

# 2. 更小的sigreg 0.0005
run "po2_sig_pt_w00005" \
  --pretrained-text-path "${PE_CKPT}" --lock-text \
  --sigreg-target clip --sigreg-weight 0.0005 --pos-only sigmoid --sigreg-joint

# 3. sigreg分开计算(不joint)
run "po2_sig_pt_w0001_sep" \
  --pretrained-text-path "${PE_CKPT}" --lock-text \
  --sigreg-target clip --sigreg-weight 0.001 --pos-only sigmoid

# 4. 不冻结text (双方都训练), 但加pretrained初始化
run "po2_sig_pt_w0001_nolock" \
  --pretrained-text-path "${PE_CKPT}" \
  --sigreg-target clip --sigreg-weight 0.001 --pos-only sigmoid --sigreg-joint

# 5. 无pretrained text (从零, 对标nolock之前最好的0.0070)
run "po2_sig_nolock_w001" \
  --sigreg-target clip --sigreg-weight 0.01 --pos-only sigmoid --sigreg-joint

# 6. 对照: 有负样本 SigLIP + pretrained text (rlit对照)
run "po2_rlit_ref" \
  --pretrained-text-path "${PE_CKPT}" --lock-text \
  --sigreg-target clip --sigreg-weight 1e-4

echo '======== ALL 6 V2 DONE ========'
