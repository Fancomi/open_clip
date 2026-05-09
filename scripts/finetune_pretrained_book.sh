#!/bin/bash
# finetune_pretrained.sh — 预训练模型在目标域（book）数据上的微调实验
#
# 实验轴设计（正交）：
#   主轴：学习率 1e-5 / 5e-5 / 2e-4
#   冻结轴：full（全参数）/ partial（锁视觉，解锁最后 N 组）
#   叠加项：SigReg 5e-4（默认开启）
#   消融：同 LR 去掉 SigReg / 同 LR 对比 full vs partial
#
# 目标：在下游域数据上获得比预训练 zero-shot 更好的视觉塔

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ============ 数据（自动生成 TSV） ============
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
# 100 epochs = 600 steps, warmup 10% = 60 steps
EPOCHS=100
WARMUP=60

# ============ 公共参数 ============
BASE="--precision amp_bf16 --workers 8 --batch-size ${BS} \
    --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.05 \
    --save-frequency 10 --grad-checkpointing --log-every-n-steps 1 --val-frequency 5 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 1 \
    --epochs ${EPOCHS} --warmup ${WARMUP}"

# ============ 运行函数 ============
run() {
    local TAG=$1 MODEL=$2 PORT=$3; shift 3
    local NAME="ft_${TAG}_${TS}"
    echo "======== [ft] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" --val-data "${VAL}" \
        ${BASE} "$@" \
        --name "${NAME}" < /dev/null
}

eval_only() {
    local TAG=$1 MODEL=$2 PORT=$3; shift 3
    local NAME="eval_${TAG}_${TS}"
    echo "======== [eval] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=1 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --val-data "${VAL}" \
        --precision amp_bf16 --batch-size 512 --workers 8 \
        --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
        --val-num-captions-per-image 1 \
        "$@" --name "${NAME}" < /dev/null
}

# ============ SigReg 公共选项 ============
SIG="--sigreg-target cls --sigreg-weight 5e-4"

# ============================================================
# 0. 预训练 zero-shot 基线
# ============================================================
eval_only "pe_book_zeroshot"   "PE-Core-B-16"     29600 --pretrained "${PE_CKPT}"
eval_only "sig2_book_zeroshot" "ViT-B-16-SigLIP2" 29601 --pretrained "${SIG2_CKPT}" --siglip

# ============================================================
# 1. PE-Core-B-16
# ============================================================
# --- Full fine-tune: LR 梯度 ---
run "pe_book_lr1e5" "PE-Core-B-16" 29610 \
    --pretrained "${PE_CKPT}" --lr 1e-5 ${SIG}

run "pe_book_lr5e5" "PE-Core-B-16" 29611 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${SIG}

run "pe_book_lr2e4" "PE-Core-B-16" 29612 \
    --pretrained "${PE_CKPT}" --lr 2e-4 ${SIG}

# --- Partial: 锁视觉，解锁最后 3 组（proj + 2 blocks），LR 可更高 ---
run "pe_book_partial_lr5e5" "PE-Core-B-16" 29613 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${SIG} \
    --lock-image --lock-image-unlocked-groups 3

run "pe_book_partial_lr2e4" "PE-Core-B-16" 29614 \
    --pretrained "${PE_CKPT}" --lr 2e-4 ${SIG} \
    --lock-image --lock-image-unlocked-groups 3

# --- 消融：无 SigReg ---
run "pe_book_lr5e5_nosig" "PE-Core-B-16" 29615 \
    --pretrained "${PE_CKPT}" --lr 5e-5

# ============================================================
# 2. ViT-B-16-SigLIP2
# ============================================================
# --- Full fine-tune ---
run "sig2_book_lr1e5" "ViT-B-16-SigLIP2" 29620 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 1e-5 ${SIG}

run "sig2_book_lr5e5" "ViT-B-16-SigLIP2" 29621 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 ${SIG}

run "sig2_book_lr2e4" "ViT-B-16-SigLIP2" 29622 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 2e-4 ${SIG}

# --- Partial ---
run "sig2_book_partial_lr5e5" "ViT-B-16-SigLIP2" 29623 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 ${SIG} \
    --lock-image --lock-image-unlocked-groups 3

run "sig2_book_partial_lr2e4" "ViT-B-16-SigLIP2" 29624 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 2e-4 ${SIG} \
    --lock-image --lock-image-unlocked-groups 3

# --- 消融 ---
run "sig2_book_lr5e5_nosig" "ViT-B-16-SigLIP2" 29625 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5

echo "======== finetune_pretrained all done ========"
