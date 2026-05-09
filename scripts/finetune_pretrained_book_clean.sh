#!/bin/bash
# finetune_pretrained_book_clean.sh — book_20260508_clean 数据上的微调实验
#
# 基于三轮经验，选择：
#   A. 最佳配置复刻：PE full lr=5e-5/2e-4 nosig，200/300ep
#   B. 低 LR 补足训练：PE/SigLIP2 lr=3e-6/5e-6，300ep（v3 100ep 未收敛）
#   C. SigLIP2 最佳：full lr=5e-5，sig=1e-5，200/300ep
#   D. Muon：PE/SigLIP2 + 不同 muon_lr，对比 AdamW 最佳

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

# ============ 数据 ============
BOOK="/root/paddlejob/workspace/env_run/penghaotian/datas/book_20260508_clean/annotations"
TRAIN="${BOOK}/train.tsv"
VAL="${BOOK}/val.tsv"

if [ ! -f "${TRAIN}" ] || [ ! -f "${VAL}" ]; then
    echo "[ft] TSV not found, generating..."
    python3 scripts/build_book_tsv.py \
        --data-root /root/paddlejob/workspace/env_run/penghaotian/datas/book_20260508_clean
fi

# ============ 模型 ============
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"
SIG2_CKPT="${MODEL_DIR}/ViT-B-16-SigLIP2/open_clip_model.safetensors"

# ============ 硬件 ============
GPUS=8
BS=512
# clean: 18499 / (512*8) ≈ 4.5 steps/epoch → warmup 按 10% 取

# ============ 公共参数 ============
BASE_COMMON="--precision amp_bf16 --workers 8 --batch-size ${BS} \
    --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.05 \
    --save-frequency 10 --grad-checkpointing --log-every-n-steps 1 --val-frequency 5 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 1"

BASE100="${BASE_COMMON} --epochs 100 --warmup 50"
BASE200="${BASE_COMMON} --epochs 200 --warmup 100"
BASE300="${BASE_COMMON} --epochs 300 --warmup 150"

# Muon 用：AdamW 部分（embed/bias/norm）的 LR 沿用微调尺度，Muon 部分约 ×30
# GlobalBS=4096 时 quick.sh 比例：muon_lr=0.01, lr=3.4e-4（从零训练）
# 微调取保守值：adam_lr=5e-5 / muon_lr=1.5e-3，及更激进 muon_lr=5e-3
ADAM_LR_FT="5e-5"
MUON_LR_LOW="1.5e-3"   # ≈ adam×30，保守
MUON_LR_HIGH="5e-3"    # ≈ adam×100，激进

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

# ============================================================
# 0. Zero-shot 基线（新数据上重测）
# ============================================================
echo ""; echo "============ 0. Zero-shot baselines ============"

eval_only "pe_clean_zeroshot"   "PE-Core-B-16"     29700 --pretrained "${PE_CKPT}"
eval_only "sig2_clean_zeroshot" "ViT-B-16-SigLIP2" 29701 --pretrained "${SIG2_CKPT}" --siglip

# ============================================================
# A. 最佳配置复刻（PE full nosig，200/300ep）
# ============================================================
echo ""; echo "============ A. Best configs ============"

# v3 最佳：PE full lr=5e-5 nosig 300ep（旧数据 24.7%）
run "pe_clean_lr5e5_nosig_200ep" "PE-Core-B-16" 29702 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${BASE200}

run "pe_clean_lr5e5_nosig_300ep" "PE-Core-B-16" 29703 \
    --pretrained "${PE_CKPT}" --lr 5e-5 ${BASE300}

# 快速收敛对照：lr=2e-4 nosig
run "pe_clean_lr2e4_nosig_100ep" "PE-Core-B-16" 29704 \
    --pretrained "${PE_CKPT}" --lr 2e-4 ${BASE100}

run "pe_clean_lr2e4_nosig_200ep" "PE-Core-B-16" 29705 \
    --pretrained "${PE_CKPT}" --lr 2e-4 ${BASE200}

# ============================================================
# B. 低 LR 补足训练（v3 100ep 未收敛，需要 300ep）
# ============================================================
echo ""; echo "============ B. Low LR long training ============"

run "pe_clean_lr5e6_nosig_300ep" "PE-Core-B-16" 29706 \
    --pretrained "${PE_CKPT}" --lr 5e-6 ${BASE300}

run "sig2_clean_lr5e6_300ep" "ViT-B-16-SigLIP2" 29707 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-6 \
    --sigreg-target cls --sigreg-weight 1e-5 ${BASE300}

# ============================================================
# C. SigLIP2 最佳区间（lr=5e-5，轻量 SigReg）
# ============================================================
echo ""; echo "============ C. SigLIP2 best zone ============"

run "sig2_clean_lr5e5_sig1e5_200ep" "ViT-B-16-SigLIP2" 29708 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 \
    --sigreg-target cls --sigreg-weight 1e-5 ${BASE200}

run "sig2_clean_lr5e5_sig1e5_300ep" "ViT-B-16-SigLIP2" 29709 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 \
    --sigreg-target cls --sigreg-weight 1e-5 ${BASE300}

# nosig 对照（v2/v3 SigLIP2 nosig≈sig_small，验证新数据上是否一致）
run "sig2_clean_lr5e5_nosig_200ep" "ViT-B-16-SigLIP2" 29710 \
    --pretrained "${SIG2_CKPT}" --siglip --lr 5e-5 ${BASE200}

# ============================================================
# D. Muon（对比 AdamW 最佳配置）
# ============================================================
echo ""; echo "============ D. Muon ============"
# Muon 对 hidden weight matrices 用 Nesterov momentum + 正交化更新，
# 其余参数（embed/bias/norm/logit）用 AdamW（lr=ADAM_LR_FT）
# muon_lr 控制矩阵权重更新幅度，需单独调

# PE-Core: 两档 muon_lr，nosig，200ep
run "pe_clean_muon_mlr1e3_200ep" "PE-Core-B-16" 29711 \
    --pretrained "${PE_CKPT}" \
    --opt muon --lr ${ADAM_LR_FT} --muon-lr ${MUON_LR_LOW} \
    ${BASE200}

run "pe_clean_muon_mlr5e3_200ep" "PE-Core-B-16" 29712 \
    --pretrained "${PE_CKPT}" \
    --opt muon --lr ${ADAM_LR_FT} --muon-lr ${MUON_LR_HIGH} \
    ${BASE200}

# SigLIP2: muon_lr 保守档，sig=1e-5，200ep
run "sig2_clean_muon_mlr1e3_200ep" "ViT-B-16-SigLIP2" 29713 \
    --pretrained "${SIG2_CKPT}" --siglip \
    --opt muon --lr ${ADAM_LR_FT} --muon-lr ${MUON_LR_LOW} \
    --sigreg-target cls --sigreg-weight 1e-5 \
    ${BASE200}

run "sig2_clean_muon_mlr5e3_200ep" "ViT-B-16-SigLIP2" 29714 \
    --pretrained "${SIG2_CKPT}" --siglip \
    --opt muon --lr ${ADAM_LR_FT} --muon-lr ${MUON_LR_HIGH} \
    --sigreg-target cls --sigreg-weight 1e-5 \
    ${BASE200}

echo "======== finetune_pretrained_book_clean all done ========"