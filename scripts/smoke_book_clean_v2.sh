#!/bin/bash
# smoke_book_clean_v2.sh — finetune_pretrained_book_clean_v2.sh 冒烟测试
#
# 覆盖所有实验配置：
#   A. AdamW 高 LR (lr=5e-4/1e-3)
#   B. Muon 低 muon_lr (mlr=1e-4/3e-4/5e-4, 两种 adam_lr)
#   C. SigReg 极小权重 (weight=1e-6/5e-6/1e-5, target=cls/clip)
#
# 每个测试：book csv 数据，8GPU，1 step (batch=8/GPU)，train + val 均跑

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

PE_CKPT="/root/paddlejob/workspace/env_run/penghaotian/models/timm/PE-Core-B-16/open_clip_model.safetensors"
BOOK="/root/paddlejob/workspace/env_run/penghaotian/datas/book_20260508_clean/annotations"
TRAIN="${BOOK}/train.tsv"
VAL="${BOOK}/val.tsv"

GPUS=8
PASSED=0; FAILED=0

run_smoke() {
    local TAG=$1 PORT=$2; shift 2
    local NAME="smoke_bcv2_${TAG}_${TS}"
    echo -n "[smoke] ${TAG} ... "
    local LOG="/tmp/smoke_bcv2_${TAG}.log"
    if torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "PE-Core-B-16" \
        --pretrained "${PE_CKPT}" \
        --train-data "${TRAIN}" --val-data "${VAL}" \
        --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
        --val-num-captions-per-image 1 \
        --batch-size 8 --epochs 1 --warmup 0 --workers 4 \
        --precision amp_bf16 --wd 0.05 --beta1 0.9 --beta2 0.98 --eps 1e-6 \
        --save-frequency 0 --log-every-n-steps 1 --val-frequency 1 \
        "$@" --name "${NAME}" > "${LOG}" 2>&1; then
        echo "PASS"; PASSED=$((PASSED + 1))
    else
        echo "FAIL"; FAILED=$((FAILED + 1))
        tail -5 "${LOG}"
    fi
    rm -rf "./logs/${NAME}" 2>/dev/null || true
}

# ── A. AdamW 高 LR ──────────────────────────────────────────────────────────
echo ""; echo "============ A. AdamW high LR ============"

run_smoke "a_lr2e4_nosig"   29820 --lr 2e-4
run_smoke "a_lr5e4_nosig"   29821 --lr 5e-4
run_smoke "a_lr1e3_nosig"   29822 --lr 1e-3

# ── B. Muon 低 muon_lr ───────────────────────────────────────────────────────
echo ""; echo "============ B. Muon low muon_lr ============"

run_smoke "b_muon_alr2e4_mlr5e4" 29823 --opt muon --lr 2e-4 --muon-lr 5e-4
run_smoke "b_muon_alr2e4_mlr3e4" 29824 --opt muon --lr 2e-4 --muon-lr 3e-4
run_smoke "b_muon_alr2e4_mlr1e4" 29825 --opt muon --lr 2e-4 --muon-lr 1e-4
run_smoke "b_muon_alr5e5_mlr5e4" 29826 --opt muon --lr 5e-5 --muon-lr 5e-4

# ── C. SigReg 极小权重 ───────────────────────────────────────────────────────
echo ""; echo "============ C. SigReg tiny weight ============"

run_smoke "c_sig_cls_1e6"  29827 --lr 2e-4 --sigreg-target cls  --sigreg-weight 1e-6
run_smoke "c_sig_cls_5e6"  29828 --lr 2e-4 --sigreg-target cls  --sigreg-weight 5e-6
run_smoke "c_sig_cls_1e5"  29829 --lr 2e-4 --sigreg-target cls  --sigreg-weight 1e-5
run_smoke "c_sig_clip_1e5" 29830 --lr 2e-4 --sigreg-target clip --sigreg-weight 1e-5

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo ""
echo "============ Smoke Results ============"
echo "  PASSED: ${PASSED}"
echo "  FAILED: ${FAILED}"
echo "======================================="
[ ${FAILED} -eq 0 ] && echo "[smoke] All passed." || { echo "[smoke] FAILED!"; exit 1; }
