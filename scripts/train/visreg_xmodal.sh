#!/bin/bash
# visreg_xmodal.sh — 第四阶段：跨模态「相对目标」+ 正则侧消融
#
# 背景（第十节定论）：单塔分布正则是**绝对目标**（推向 N(0,I)），实测与 CLIP 下游
#   无强因果 —— 强度调 4 个数量级、换目标形状、换方向选择，全部 ±0.5pt 不动。
#   本阶段换思路：约束图/文两塔的**相对关系**（= modality gap），与检索有明确因果链。
#
# 三组实验：
#   ① img_only   : VISReg 只作用视觉塔。注意现行配方 text tower 一直被正则着
#                  （CLIPLeJEPA 同时产出 image_proj/text_proj，loss 对两者都算），
#                  本组拆开 text 侧正则的贡献。
#   ② xm_pair    : SigLIP + 逐对投影对齐（无 sort，保留配对身份）。
#                  等价于「K 个随机子空间上的逐对 MSE」，比直接 MSE 温和。
#   ③ xm_dist    : SigLIP + sorted shape/scale 对齐（分布对齐）。
#                  sort 置换不变、不含配对信息（实测打乱 loss 变化 <1e-8），
#                  故只能小权重作辅助，绝不能替代对比损失（否则退化解：两塔各自
#                  推成 N(0,I)，检索掉到随机 1/N）。
#
# xmatch 需两塔同维 → 用 --sigreg-target clip（1024 维 CLIP 空间），而非 cls。
#   ⚠ 这使 ②③ 与最优配方（cls）有两处差异（target + xmatch），故补一组
#     clip_base 作为纯 target 对照，保证单变量可归因。
#
# 权重由梯度匹配标定（probe 得：占比 1e-2 时 pair w=2.3、dist w=7.8）——
#   吸取第十节教训，先确保不落进「梯度占比 2e-07」那种无效区。

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
ROOT=/root/paddlejob/workspace/env_run/penghaotian
COCO="${ROOT}/datas/coco/annotations"
IMNVAL="${ROOT}/datas/imagenet-val"
CC3M="${ROOT}/datas/cc3m-tsv/annotations/clip_train.tsv"
GPUS=8; BS=512
INIT_LS=$(python3 -c "import math; print(math.log(15))")

COMMON="--precision amp_bf16 --workers 32 --batch-size ${BS} \
  --lr 3.4e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
  --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
  --warmup 512 --epochs 10 --dataset-type csv --train-num-samples 2894191 \
  --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
  --imagenet-val ${IMNVAL}"
# 最优配方 E：VISReg scale+shape 等权、no-center、w=1.83e-4、K=256
BASE_REG="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
  --opt muon --muon-lr 0.01 --probe-data ${COCO}/karpathy_1cap.tsv \
  --reg-method visreg --sigreg-slices 256 --sigreg-weight 1.83e-4 \
  --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

go() {  # go TAG PORT EXTRA
  local TAG=$1 PORT=$2 EXTRA=$3
  local NAME="visreg_x_${TAG}_${TS}"
  echo "======== [xmodal] ${TAG} (${EXTRA}) => ${NAME} ========"
  torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
    --model "PE-Core-B-16-dinov3" \
    --train-data "${CC3M}" --val-data "${COCO}/karpathy_5cap.tsv" \
    ${COMMON} ${BASE_REG} ${EXTRA} \
    --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 崩溃（是信息，继续）"
}

# ① 正则侧消融：只正则视觉塔（对照 E=both）
go "img_only"  29630 "--sigreg-target cls --reg-sides img"

# 纯 target 对照：clip 空间 + 双塔正则，无 xmatch（②③ 的正确对照组）
go "clip_base" 29631 "--sigreg-target clip"

# ② 逐对投影对齐（保留配对）
go "xm_pair"   29632 "--sigreg-target clip --xmatch-weight 2.3 --xmatch-mode pair"

# ③ 分布对齐（置换不变，小权重辅助）
go "xm_dist"   29633 "--sigreg-target clip --xmatch-weight 7.8 --xmatch-mode dist"

echo "[xmodal] $(date '+%F %T') 全部完成"
echo "对照: E(cls,both) COCO 24.06 / IN 23.26"
