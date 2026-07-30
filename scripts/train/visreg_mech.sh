#!/bin/bash
# visreg_mech.sh — 第三阶段：机制（形式）改进，而非强度
#
# 背景（第二阶段的负面结论，已落文档）：
#   权重跨 4 个数量级 1x→1e4x，裸 loss 从 0.235 压到 0.0065（-97%，正则确实生效），
#   但 COCO 24.06/23.58/23.98、IN 23.26/23.58/23.49 —— 两指标均在 ±0.5pt 噪声内，
#   无趋势。⇒「推向各向同性高斯」的**强度**与 CLIP 下游表现无强因果。
#   故本阶段只改**形式**，不再扫强度。
#
# 顺序按对比干净度排（mix5_1e4x 最前）：
#   ① mix5_1e4x : 与已知的 cls_1e4x(单峰,COCO 23.98/IN 23.49) 同强度，只换目标形状
#                 → 差异完全归因于「单峰 vs 多岛目标」。高权重下正则真在起作用，
#                   目标形状的影响最易显现。
#   ② mix5      : 100x 下同样对比（对照 cls_1e2x COCO 23.58/IN 23.58）
#   ③ topk      : 方向质量属"形式"，未被强度结论排除

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
ROOT=/root/paddlejob/workspace/env_run/penghaotian
COCO="${ROOT}/datas/coco/annotations"
IMNVAL="${ROOT}/datas/imagenet-val"
CC3M_TSV="${ROOT}/datas/cc3m-tsv/annotations/clip_train.tsv"
GPUS=8; BS=512
INIT_LS=$(python3 -c "import math; print(math.log(15))")

COMMON="--precision amp_bf16 --workers 32 --batch-size ${BS} \
  --lr 3.4e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
  --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
  --warmup 512 --epochs 10 --dataset-type csv --train-num-samples 2894191 \
  --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
  --imagenet-val ${IMNVAL}"
CHAMPION="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
  --opt muon --muon-lr 0.01 --probe-data ${COCO}/karpathy_1cap.tsv \
  --sigreg-target cls --reg-method visreg --sigreg-slices 256 \
  --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

go() {  # go TAG PORT WEIGHT EXTRA
  local TAG=$1 PORT=$2 W=$3 EXTRA=$4
  local NAME="visreg_m_${TAG}_${TS}"
  echo "======== [mech] ${TAG} (w=${W} ${EXTRA}) => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
    --model "PE-Core-B-16-dinov3" \
    --train-data "${CC3M_TSV}" --val-data "${COCO}/karpathy_5cap.tsv" \
    ${COMMON} ${CHAMPION} --sigreg-weight ${W} ${EXTRA} \
    --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 崩溃（是信息，继续）"
}

MIX="--visreg-mixture 5 --visreg-mixture-sep 2.0"
go "mix5_1e4x" 29620 1.83e0  "${MIX}"                    # ① 对照 cls_1e4x
go "mix5_1e2x" 29621 1.83e-2 "${MIX}"                    # ② 对照 cls_1e2x
go "topk_1e2x" 29622 1.83e-2 "--visreg-topk-pool 1024"   # ③ 方向质量

echo "[mech] $(date '+%F %T') 全部完成"
echo "对照: cls_1e4x COCO 23.98/IN 23.49 ; cls_1e2x COCO 23.58/IN 23.58 ; 1x基线 24.06/23.26"
