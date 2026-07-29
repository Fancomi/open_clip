#!/bin/bash
# run_chain.sh — 串行链：wsweep 剩余的 proj 两组 → stage2 四组
#
# 说明：cls 三组已完成（1e2x ✓ / 1e4x ✓ / 1e6x 已按计划取消）。
# 本链只跑 proj_1e4x、proj_1e6x，然后进 stage2。
# 用 setsid 隔离进程组，避免误杀父 driver。

cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
export PATH="$HOME/.local/bin:$PATH"
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
ROOT=/root/paddlejob/workspace/env_run/penghaotian
COCO="${ROOT}/datas/coco/annotations"
IMNVAL="${ROOT}/datas/imagenet-val"
CC3M_TSV="${ROOT}/datas/cc3m-tsv/annotations/clip_train.tsv"
GPUS=8; BS=512
LR=3.4e-4; MUON_LR=0.01
INIT_LS=$(python3 -c "import math; print(math.log(15))")

COMMON="--precision amp_bf16 --workers 32 --batch-size ${BS} \
  --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
  --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
  --warmup 512 --epochs 10 --dataset-type csv --train-num-samples 2894191 \
  --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
  --imagenet-val ${IMNVAL}"
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
  --opt muon --muon-lr ${MUON_LR} --probe-data ${COCO}/karpathy_1cap.tsv \
  --reg-method visreg --sigreg-slices 256 \
  --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

go() {  # go TAG PORT TARGET WEIGHT EXTRA
  local TAG=$1 PORT=$2 TGT=$3 W=$4 EXTRA=$5
  local NAME="visreg_q_${TAG}_${TS}"
  echo "======== [queue] ${TAG} (target=${TGT} w=${W} ${EXTRA}) => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
    --model "PE-Core-B-16-dinov3" \
    --train-data "${CC3M_TSV}" --val-data "${COCO}/karpathy_5cap.tsv" \
    ${COMMON} ${CHAMPION} \
    --sigreg-target ${TGT} --sigreg-weight ${W} ${EXTRA} \
    --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 崩溃（是有效信息，继续）"
}

W_BEST="${W_BEST:-1.83e-2}"   # 100×，第一阶段 IN-1k 最优点

# ── 阶段一余下：cls_proj（MLP 缓冲，崩溃点应更靠后）──────────────────
go "proj_1e4x"  29603 cls_proj 1.83e0  ""
go "proj_1e6x"  29604 cls_proj 1.83e2  ""

# ── 阶段二：机制改进 ────────────────────────────────────────────────
go "w1e3x"      29610 cls 1.83e-1      ""                                        # 填权重空隙
go "topk"       29611 cls ${W_BEST}    "--visreg-topk-pool 1024"                 # top-K 挑方向
go "mix5"       29612 cls ${W_BEST}    "--visreg-mixture 5 --visreg-mixture-sep 2.0"  # 混合高斯目标
go "mix5_1e4x"  29613 cls 1.83e0       "--visreg-mixture 5 --visreg-mixture-sep 2.0"  # 多岛目标救强正则

echo "[queue] $(date '+%F %T') 全部完成"
