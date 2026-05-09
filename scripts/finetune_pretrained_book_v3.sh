#!/bin/bash
# finetune_pretrained_book_v3.sh — 第三轮微调实验（book 域）
#
# 基于 v2 结论的补充探索：
#   1. 更低 LR（3e-6 / 5e-6）：PE + SigLIP2
#   2. Partial 层数（unlock 1 / 2）：诊断"哪层需要适配"
#   3. SigReg weight（1e-5 / 5e-5）：找不抑制学习的最小正则
#   4. 更长训练（200ep / 300ep）：PE lr=5e-5 无SigReg 是否饱和
#
# v2 最佳：PE full lr=5e-5 无SigReg = 22.6% T2I（95ep 仍上升）

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ============ 数据 ============
BOOK="/root/paddlejob/workspace/env_run/penghaotian/datas/book_20260507/annotations"
TRAIN="${BOOK}/train.tsv"
VAL="${BOOK}/val.tsv"

if [ ! -f "${TRAIN}" ] || [ ! -f "${VAL}" ]; then
    echo "[ft] TSV not found, generating..."
    python3 scripts/build_book_tsv.py
fi

# ============ 模型 ============
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"
SIG2_CKPT="${MODEL_DIR}/ViT-B-16-SigLIP2/open_clip_model.safetensors"

# ============ 硬件 ============
GPUS=8
BS=512
# book: 22687 / (512*8) ≈ 6 steps/epoch

# ============ 公共参数（不含 epochs/warmup，由各实验指定） ============
BASE_COMMON="--precision amp_bf16 --workers 8 --batch-size ${BS} \
    --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.05 \
    --save-frequency 10 --grad-checkpointing --log-every-n-steps 1 --val-frequency 5 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 1"

# 100ep 版本
BASE100="${BASE_COMMON} --epochs 100 --warmup 60"
# 200ep 版本
BASE200="${BASE_COMMON} --epochs 200 --warmup 120"
# 300ep 版本
BASE300="${BASE_COMMON} --epochs 300 --warmup 180"

# ============ 运行函数 ============
run() {
    local TAG=$1 MODEL=$2 PORT=$3; shift 3
    local NAME="ft_${TAG}_${TS}"
    echo "======== [ft] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" --val-data "${VAL}" \
        "$@" \
        --name "${NAME}" < /dev/null
}

SIG="--sigreg-target cls --sigreg-weight 5e-4"

# ============================================================
# 1. 更低 LR（PE + SigLIP2）
# ============================================================
echo ""; echo "============ 1. Lower LR ============"

run "pe_book_lr3e6" "PE-Core-B-16" 29630 \
    --pretrained "${PE_CKPT}" --lr 3e-6 ${SIG} ${BASE100}

run "pe_book_lr5e6" "PE-Core-B-16" 29631 \
    --pretrained "${PE_CKPT}" --lr 5e-6 ${SIG} ${BASE100}

run "sig2_book_lr3e6" "ViT-B-16-SigLIP2" 29632 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 3e-6 ${SIG} ${BASE100}

run "sig2_book_lr5e6" "ViT-B-16-SigLIP2" 29633 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-6 ${SIG} ${BASE100}

# ============================================================
# 2. Partial 层数（unlock 1 / 2，对比已有的 unlock 3）
# ============================================================
echo ""; echo "============ 2. Partial Layers ============"

# PE-Core: unlock 1 = 只解锁 proj head
run "pe_book_partial1_lr5e5" "PE-Core-B-16" 29634 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 1

run "pe_book_partial1_lr2e4" "PE-Core-B-16" 29635 \
    --pretrained "${PE_CKPT}" --lr 2e-4 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 1

# PE-Core: unlock 2 = proj + last block
run "pe_book_partial2_lr5e5" "PE-Core-B-16" 29636 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 2

run "pe_book_partial2_lr2e4" "PE-Core-B-16" 29637 \
    --pretrained "${PE_CKPT}" --lr 2e-4 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 2

# SigLIP2: unlock 1
run "sig2_book_partial1_lr5e5" "ViT-B-16-SigLIP2" 29638 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 1

run "sig2_book_partial1_lr1e5" "ViT-B-16-SigLIP2" 29639 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 1e-5 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 1

# SigLIP2: unlock 2
run "sig2_book_partial2_lr5e5" "ViT-B-16-SigLIP2" 29640 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 2

run "sig2_book_partial2_lr1e5" "ViT-B-16-SigLIP2" 29641 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 1e-5 ${SIG} ${BASE100} \
    --lock-image --lock-image-unlocked-groups 2

# ============================================================
# 3. SigReg weight（1e-5 / 5e-5，对比已有 5e-4 和 0）
# ============================================================
echo ""; echo "============ 3. SigReg Weight ============"

# PE: 用最佳 LR=5e-5
run "pe_book_lr5e5_sig1e5" "PE-Core-B-16" 29642 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${BASE100} \
    --sigreg-target cls --sigreg-weight 1e-5

run "pe_book_lr5e5_sig5e5" "PE-Core-B-16" 29643 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${BASE100} \
    --sigreg-target cls --sigreg-weight 5e-5

# SigLIP2: 用最佳 LR=5e-5
run "sig2_book_lr5e5_sig1e5" "ViT-B-16-SigLIP2" 29644 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 ${BASE100} \
    --sigreg-target cls --sigreg-weight 1e-5

run "sig2_book_lr5e5_sig5e5" "ViT-B-16-SigLIP2" 29645 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 ${BASE100} \
    --sigreg-target cls --sigreg-weight 5e-5

# ============================================================
# 4. 更长训练（PE lr=5e-5 无SigReg，v2最佳配置）
# ============================================================
echo ""; echo "============ 4. Longer Training ============"

run "pe_book_lr5e5_nosig_200ep" "PE-Core-B-16" 29646 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${BASE200}

run "pe_book_lr5e5_nosig_300ep" "PE-Core-B-16" 29647 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${BASE300}

echo "======== finetune_pretrained_book_v3 all done ========"
