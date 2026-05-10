#!/bin/bash
# finetune_pretrained_book_clean_v2.sh — PE-only 精细搜索（clean 数据）
#
# 基于 v1 结论（PE lr=2e-4 nosig 100ep = 20.3% best@80ep）：
#   A. AdamW 高 LR 短训：lr=5e-4/1e-3 + 更短 ep，找真正的 peak
#   B. Muon 低 muon_lr：mlr=1e-4/3e-4/5e-4 + adam_lr=2e-4，warmup 翻倍
#   C. SigReg 极小权重：weight=1e-6/5e-6/1e-5，看是否对 clean 数据有益
#   D. Muon + SigReg 最佳档组合（胜者对决）

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
PE_CKPT="/root/paddlejob/workspace/env_run/penghaotian/models/timm/PE-Core-B-16/open_clip_model.safetensors"

# ============ 硬件 ============
GPUS=8
BS=512
# clean: 18499 / (512*8) ≈ 4.5 steps/epoch

# ============ 公共参数 ============
BASE_COMMON="--precision amp_bf16 --workers 8 --batch-size ${BS} \
    --beta1 0.9 --beta2 0.98 --eps 1e-6 --wd 0.05 \
    --save-frequency 10 --grad-checkpointing --log-every-n-steps 1 --val-frequency 5 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 1"

# warmup 约 10% epochs（min 20 steps）
BASE50="${BASE_COMMON}  --epochs  50 --warmup  25"
BASE100="${BASE_COMMON} --epochs 100 --warmup  50"
BASE200="${BASE_COMMON} --epochs 200 --warmup 100"

# ============ 运行函数 ============
run() {
    local TAG=$1 PORT=$2; shift 2
    local NAME="ft_${TAG}_${TS}"
    echo "======== [ft] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "PE-Core-B-16" \
        --pretrained "${PE_CKPT}" \
        --train-data "${TRAIN}" --val-data "${VAL}" \
        "$@" \
        --name "${NAME}" < /dev/null
}

# ============================================================
# A. AdamW 高 LR 短训（搜索 peak 位置）
# v1: lr=2e-4 在 80ep 达峰，200ep 过拟合（18.9%）
# 假设 lr=5e-4/1e-3 会更快达峰
# ============================================================
echo ""; echo "============ A. AdamW high-LR short run ============"

# 基线：v1 最佳（对照）
run "pe_clean2_lr2e4_nosig_100ep" 29800 \
    --lr 2e-4 ${BASE100}

# 更高 LR：峰值前移，减少 ep
run "pe_clean2_lr5e4_nosig_50ep" 29801 \
    --lr 5e-4 ${BASE50}

run "pe_clean2_lr5e4_nosig_100ep" 29802 \
    --lr 5e-4 ${BASE100}

run "pe_clean2_lr1e3_nosig_50ep" 29803 \
    --lr 1e-3 ${BASE50}

# ============================================================
# B. Muon 低 muon_lr（寻找有益区间）
# v1: mlr=1.5e-3→17.3%, mlr=5e-3→15.9%，均差于 AdamW
# 理论：fine-tuning 的 muon_lr 应远小于 from-scratch（×30 法则不适用）
# 实测：用 adam_lr=2e-4（v1最佳 AdamW LR），warmup 翻倍（100步）
# ============================================================
echo ""; echo "============ B. Muon low muon_lr ============"

# adam_lr 对齐 v1 最佳 AdamW（2e-4），muon_lr 从 5e-4 往下探
run "pe_clean2_muon_mlr5e4_200ep" 29804 \
    --opt muon --lr 2e-4 --muon-lr 5e-4 ${BASE200}

run "pe_clean2_muon_mlr3e4_200ep" 29805 \
    --opt muon --lr 2e-4 --muon-lr 3e-4 ${BASE200}

run "pe_clean2_muon_mlr1e4_200ep" 29806 \
    --opt muon --lr 2e-4 --muon-lr 1e-4 ${BASE200}

# 也测 adam_lr=5e-5 + mlr=5e-4（与 v1 adam_lr 对齐，只降 muon_lr）
run "pe_clean2_muon_alr5e5_mlr5e4_200ep" 29807 \
    --opt muon --lr 5e-5 --muon-lr 5e-4 ${BASE200}

# ============================================================
# C. SigReg 极小权重（clean 数据验证中性点）
# v1 旧数据：weight=1e-5 几乎无害（-0.3%），weight=5e-4 有害（-5%）
# clean 数据还没测过 sigreg，这次用极小权重搜索
# ============================================================
echo ""; echo "============ C. SigReg tiny weight ============"

# 用 lr=2e-4 100ep（v1最佳配置），仅改 sigreg_weight
run "pe_clean2_lr2e4_sig1e6_100ep" 29808 \
    --lr 2e-4 --sigreg-target cls --sigreg-weight 1e-6 ${BASE100}

run "pe_clean2_lr2e4_sig5e6_100ep" 29809 \
    --lr 2e-4 --sigreg-target cls --sigreg-weight 5e-6 ${BASE100}

run "pe_clean2_lr2e4_sig1e5_100ep" 29810 \
    --lr 2e-4 --sigreg-target cls --sigreg-weight 1e-5 ${BASE100}

# sigreg_target=clip（作用于 CLIP 空间，而非原始 CLS）
run "pe_clean2_lr2e4_sig_clip_1e5_100ep" 29811 \
    --lr 2e-4 --sigreg-target clip --sigreg-weight 1e-5 ${BASE100}

# ============================================================
# D. Muon + SigReg 最佳档组合（等 B/C 结果后判断）
# 预设：用 B 最佳 muon_lr + C 最佳 sigreg_weight，200ep
# 若 B/C 均无益，此组可跳过
# ============================================================
# （注释掉，等 B/C 结果出来手动决定是否加跑）
# run "pe_clean2_muon_best_sig_best_200ep" 29812 \
#     --opt muon --lr 2e-4 --muon-lr <best_mlr> \
#     --sigreg-target cls --sigreg-weight <best_sig> ${BASE200}

echo "======== finetune_pretrained_book_clean_v2 all done ========"
