#!/bin/bash
# finetune_pretrained_v3.sh — 第三轮微调实验（CC3M）
#
# 补充探索：
#   1. 更低 LR（3e-6 / 5e-6）：PE + SigLIP2
#   2. Partial 层数（unlock 1 / 2）
#   3. SigReg weight（1e-5 / 5e-5）
#
# v2 关键发现：SigLIP2 lr=1e-5 一个 epoch 后从 35%→70%（I2T），但持续遗忘
# PE 所有配置单调下降，lr=1e-5 几乎无损

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ============ 数据 ============
CC3M_SRC="/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds"
CC3M="/dev/shm/cc3m_wds"
CC3M_TRAIN="${CC3M}/{00000..00280}.tar"
CC3M_N_TRAIN=2857622

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"

# ============ 模型 ============
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"
SIG2_CKPT="${MODEL_DIR}/ViT-B-16-SigLIP2/open_clip_model.safetensors"

# ============ 硬件 ============
GPUS=8
BS=512
# CC3M: 2857622 / (512*8) ≈ 697 steps/epoch
# 5 epochs = 3485 steps, warmup 10% = 348
EPOCHS=5
WARMUP=348

# ============ 公共参数 ============
BASE="--precision amp_bf16 --workers 32 --batch-size ${BS} \
    --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.05 \
    --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 \
    --epochs ${EPOCHS} --warmup ${WARMUP}"

# ============ CC3M 加载到内存 ============
if [ ! -d "${CC3M}" ]; then
    echo "[ft] Loading CC3M to /dev/shm ..."
    cp -r "${CC3M_SRC}" "${CC3M}"
    echo "[ft] Done: $(du -sh ${CC3M} | cut -f1)"
else
    echo "[ft] CC3M already in /dev/shm"
fi

# ============ 运行函数 ============
run() {
    local TAG=$1 MODEL=$2 PORT=$3; shift 3
    local NAME="ft_${TAG}_${TS}"
    echo "======== [ft] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC3M_TRAIN}" --val-data "${VAL}" \
        ${BASE} "$@" \
        --name "${NAME}" < /dev/null
}

SIG="--sigreg-target cls --sigreg-weight 5e-4"

# ============================================================
# 1. 更低 LR（PE + SigLIP2）
# ============================================================
echo ""; echo "============ 1. Lower LR ============"

run "pe_cc3m_lr3e6" "PE-Core-B-16" 29650 \
    --pretrained "${PE_CKPT}" --lr 3e-6 ${SIG}

run "pe_cc3m_lr5e6" "PE-Core-B-16" 29651 \
    --pretrained "${PE_CKPT}" --lr 5e-6 ${SIG}

run "sig2_cc3m_lr3e6" "ViT-B-16-SigLIP2" 29652 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 3e-6 ${SIG}

run "sig2_cc3m_lr5e6" "ViT-B-16-SigLIP2" 29653 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-6 ${SIG}

# ============================================================
# 2. Partial 层数（unlock 1 / 2，对比已有的 unlock 3）
# ============================================================
echo ""; echo "============ 2. Partial Layers ============"

# PE-Core: unlock 1 / 2，用 lr=5e-5
run "pe_cc3m_partial1_lr5e5" "PE-Core-B-16" 29654 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${SIG} \
    --lock-image --lock-image-unlocked-groups 1

run "pe_cc3m_partial2_lr5e5" "PE-Core-B-16" 29655 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${SIG} \
    --lock-image --lock-image-unlocked-groups 2

# SigLIP2: unlock 1 / 2，用 lr=1e-5（SigLIP2 安全区间）
run "sig2_cc3m_partial1_lr1e5" "ViT-B-16-SigLIP2" 29656 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 1e-5 ${SIG} \
    --lock-image --lock-image-unlocked-groups 1

run "sig2_cc3m_partial2_lr1e5" "ViT-B-16-SigLIP2" 29657 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 1e-5 ${SIG} \
    --lock-image --lock-image-unlocked-groups 2

# ============================================================
# 3. SigReg weight（1e-5 / 5e-5）
# ============================================================
echo ""; echo "============ 3. SigReg Weight ============"

# PE: lr=1e-5（CC3M 上最安全的 LR）
run "pe_cc3m_lr1e5_sig1e5" "PE-Core-B-16" 29658 \
    --pretrained "${PE_CKPT}" --lr 1e-5 \
    --sigreg-target cls --sigreg-weight 1e-5

run "pe_cc3m_lr1e5_sig5e5" "PE-Core-B-16" 29659 \
    --pretrained "${PE_CKPT}" --lr 1e-5 \
    --sigreg-target cls --sigreg-weight 5e-5

# SigLIP2: lr=1e-5
run "sig2_cc3m_lr1e5_sig1e5" "ViT-B-16-SigLIP2" 29660 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 1e-5 \
    --sigreg-target cls --sigreg-weight 1e-5

run "sig2_cc3m_lr1e5_sig5e5" "ViT-B-16-SigLIP2" 29661 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 1e-5 \
    --sigreg-target cls --sigreg-weight 5e-5

echo "======== finetune_pretrained_v3 all done ========"
