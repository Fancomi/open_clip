#!/bin/bash
# finetune_pretrained.sh — 对预训练 CLIP 模型 (PE-Core / SigLIP2) 进行微调
#
# 核心问题：直接高 LR 微调导致表征崩溃
# 解决方案：4 种策略梯度递进，从安全到激进
#   1. lit       — 锁视觉塔，只训文本对齐（最安全）
#   2. partial   — 解锁视觉塔最后 N 组
#   3. lowlr     — 全参数极低 LR + 长 warmup
#   4. sigreg    — 全参数 + SigReg 防坍缩正则
#
# 对比基准：
#   PE-Core  pretrained:  T2I R@1=50.2  I2T R@1=71.1
#   SigLIP2  pretrained:  T2I R@1=53.2  I2T R@1=69.4
#   from-scratch baseline: pe_dinov3_sigreg_siglip_muon (quick.sh)

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ============ 数据 ============
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

CC3M_SRC="/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds"
CC3M="/dev/shm/cc3m_wds"
CC3M_TRAIN="${CC3M}/{00000..00280}.tar"
CC3M_N_TRAIN=2857622

# ============ 模型权重 ============
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"
SIG2_CKPT="${MODEL_DIR}/ViT-B-16-SigLIP2/open_clip_model.safetensors"

# ============ 硬件与 batch ============
GPUS=8
BS=512
GLOBAL_BS=$((BS * GPUS))

# ============ 学习率 ============
# 微调 LR 远低于从零训练：base=3.4e-6（从零的 1/100）
# 再按 sqrt(GlobalBS/ref) 缩放
FT_BASE_LR=3.4e-6
FT_LR=$(python3 -c "import math; print(${FT_BASE_LR} * math.sqrt(${GLOBAL_BS} / 4096))")
# LiT / partial 可稍高（冻结层不受梯度影响）
LIT_LR=$(python3 -c "print(${FT_LR} * 10)")  # ~3.4e-5

# CC3M: steps/epoch=174 (2857622 / 4096 / accum)
# warmup 50% for fine-tuning stability
WARMUP_HALF=435  # 5ep * 174 * 0.5

# ============ 公共参数 ============
BASE="--precision amp_bf16 --workers 32 --batch-size ${BS} \
    --beta1 0.9 --beta2 0.98 --eps 1e-6 \
    --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

# ============ 数据加载到内存 ============
if [ ! -d "${CC3M}" ]; then
    echo "[ft] Loading CC3M to /dev/shm ..."
    cp -r "${CC3M_SRC}" "${CC3M}"
    echo "[ft] Done: $(du -sh ${CC3M} | cut -f1)"
else
    echo "[ft] CC3M already in /dev/shm, skip"
fi

# ============ 通用 run 函数 ============
run_ft() {
    local TAG=$1 MODEL=$2 PORT=$3; shift 3
    local NAME="ft_${TAG}_${TS}"
    echo "======== [ft] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        --probe-data "${PROBE_TSV}" \
        ${BASE} \
        "$@" \
        --name "${NAME}" < /dev/null
}

# ============================================================
# 实验组 A：PE-Core-B-16（标准 CLIP softmax loss）
# ============================================================
# A1: LiT — 锁视觉，只训文本
run_ft "pe_lit" "PE-Core-B-16" 29580 \
    --pretrained "${PE_CKPT}" \
    --lock-image \
    --lr "${LIT_LR}" --wd 0.1 --epochs 10 --warmup 174

# A2: Partial — 解锁视觉最后 3 组（proj + last_block + 倒数第二 block）
run_ft "pe_partial3" "PE-Core-B-16" 29581 \
    --pretrained "${PE_CKPT}" \
    --lock-image --lock-image-unlocked-groups 3 \
    --lr "${LIT_LR}" --wd 0.1 --epochs 10 --warmup 174

# A3: Full low-LR
run_ft "pe_lowlr" "PE-Core-B-16" 29582 \
    --pretrained "${PE_CKPT}" \
    --lr "${FT_LR}" --wd 0.01 --epochs 5 --warmup "${WARMUP_HALF}"

# A4: Full + SigReg 正则
run_ft "pe_sigreg" "PE-Core-B-16" 29583 \
    --pretrained "${PE_CKPT}" \
    --sigreg-target cls --sigreg-weight 5e-4 \
    --lr "${FT_LR}" --wd 0.05 --epochs 5 --warmup "${WARMUP_HALF}"

# ============================================================
# 实验组 B：ViT-B-16-SigLIP2（SigLIP sigmoid loss）
# ============================================================
# B1: LiT
run_ft "sig2_lit" "ViT-B-16-SigLIP2" 29584 \
    --pretrained "${SIG2_CKPT}" --siglip \
    --lock-image \
    --lr "${LIT_LR}" --wd 0.1 --epochs 10 --warmup 174

# B2: Partial unlock 3
run_ft "sig2_partial3" "ViT-B-16-SigLIP2" 29585 \
    --pretrained "${SIG2_CKPT}" --siglip \
    --lock-image --lock-image-unlocked-groups 3 \
    --lr "${LIT_LR}" --wd 0.1 --epochs 10 --warmup 174

# B3: Full low-LR
run_ft "sig2_lowlr" "ViT-B-16-SigLIP2" 29586 \
    --pretrained "${SIG2_CKPT}" --siglip \
    --lr "${FT_LR}" --wd 0.01 --epochs 5 --warmup "${WARMUP_HALF}"

# B4: Full + SigReg
run_ft "sig2_sigreg" "ViT-B-16-SigLIP2" 29587 \
    --pretrained "${SIG2_CKPT}" --siglip \
    --sigreg-target cls --sigreg-weight 5e-4 \
    --lr "${FT_LR}" --wd 0.05 --epochs 5 --warmup "${WARMUP_HALF}"

echo "======== finetune_pretrained all done ========"
