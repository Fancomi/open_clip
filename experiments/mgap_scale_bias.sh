#!/bin/bash
# experiments/mgap_scale_bias.sh
#
# 模态 gap 与 logit_scale / logit_bias 初始化消融
#
# 假设: 模态 gap 由负样本推力>正样本拉力引起，可通过调节 scale/bias 消除
# 平台: CC3M (2.86M), 1 epoch, 8×H800, BS=4096
# 预计单次耗时: ~35min
#
# 用法:
#   bash experiments/mgap_scale_bias.sh          # 全部
#   bash experiments/mgap_scale_bias.sh smoke    # 冒烟测试

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
MODE="${1:-run}"  # smoke / run

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

CC3M="/dev/shm/cc3m_wds"
CC3M_TRAIN="${CC3M}/{00000..00280}.tar"
CC3M_N_TRAIN=2857622

MODEL="PE-Core-B-16-dinov3"
GPUS=8
PreGpuBS=512
GlobalBS=$((PreGpuBS * GPUS))
LR=$(python3 -c "import math; print(3.4e-4 * math.sqrt(${GlobalBS} / 4096))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(${GlobalBS} / 4096))")

# ── 公共参数 ──
BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 --grad-checkpointing \
    --log-every-n-steps 1 --val-frequency 1 \
    --delete-previous-checkpoint"

SIGREG="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

# smoke 模式不用 muon (单卡无 dist)
SIGREG_SMOKE="--siglip --sigreg-target cls --sigreg-weight 1e-4"

# ── 运行函数 ──
PASSED=0; FAILED=0

run() {
    local TAG=$1 PORT=$2 EXTRA=$3
    local NAME="sb_${TAG}_${TS}"
    if [ "${MODE}" = "smoke" ]; then
        echo -n "[smoke] ${TAG} ... "
        local LOG="/tmp/smoke_sb_${TAG}.log"
        if torchrun --nproc_per_node=1 --master_port=${PORT} \
            -m open_clip_train.main \
            --model "${MODEL}" \
            --dataset-type synthetic --train-num-samples 64 \
            --batch-size 8 --epochs 1 --warmup 0 --workers 0 \
            --precision amp_bf16 --lr 1e-5 --wd 0.01 \
            --save-frequency 0 --log-every-n-steps 1 \
            ${SIGREG_SMOKE} ${EXTRA} --name "${NAME}" > "${LOG}" 2>&1; then
            echo "PASS"; PASSED=$((PASSED + 1))
        else
            echo "FAIL (see ${LOG})"; FAILED=$((FAILED + 1))
            tail -5 "${LOG}"
        fi
        rm -rf "./logs/${NAME}" 2>/dev/null || true
    else
        echo "======== [sb] ${TAG} => ${NAME} ========"
        torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
            -m open_clip_train.main \
            --model "${MODEL}" \
            --train-data "${CC3M_TRAIN}" --val-data "${VAL}" \
            --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
            --csv-img-key filepath --csv-caption-key caption \
            --val-num-captions-per-image 5 \
            --warmup 512 --epochs 1 \
            ${BASE} ${SIGREG} ${EXTRA} \
            --name "${NAME}" < /dev/null
    fi
}

# ── CC3M copy (仅 run 模式) ──
if [ "${MODE}" = "run" ]; then
    if [ ! -d "${CC3M}" ]; then
        echo "[sb] Loading CC3M to memory ..."
        cp -r "/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds" "${CC3M}"
        echo "[sb] Done, $(du -sh ${CC3M} | cut -f1)"
    else
        echo "[sb] Found ${CC3M}, skip copy"
    fi
fi

# ═══════════════════════════════════════════════════════════════════════
# Group A: Bias sweep  (scale=ln(10), bias varies)
# ═══════════════════════════════════════════════════════════════════════
# A0: baseline (scale=ln10, bias=-10)  — 对照
run "bias_m10"  29601 ""

# A1: bias=-5  (更弱的负样本抑制，初始负梯度更大)
run "bias_m05"  29602 "--init-logit-bias -5"

# A2: bias=-8
run "bias_m08"  29603 "--init-logit-bias -8"

# A3: bias=-12
run "bias_m12"  29604 "--init-logit-bias -12"

# A4: bias=-15 (更强的负样本抑制，几乎消除负梯度)
run "bias_m15"  29605 "--init-logit-bias -15"

# A5: bias=-20
run "bias_m20"  29606 "--init-logit-bias -20"

# ═══════════════════════════════════════════════════════════════════════
# Group B: Scale sweep  (bias=-10, scale varies)
# ═══════════════════════════════════════════════════════════════════════
# B1: scale=ln(5)≈1.61  (温和梯度)
run "scale_05"  29611 "--init-logit-scale 1.6094"

# B2: scale=ln(20)≈3.0  (更锐利)
run "scale_20"  29612 "--init-logit-scale 2.9957"

# B3: scale=ln(50)≈3.91  (极锐利)
run "scale_50"  29613 "--init-logit-scale 3.9120"

# ═══════════════════════════════════════════════════════════════════════
# Group C: Freeze (不让 scale/bias 学习)
# ═══════════════════════════════════════════════════════════════════════
# C1: 默认 init 冻结
run "freeze_default"  29621 "--freeze-logit-params"

# C2: (scale=ln(20), bias=-15) 冻结 — 强抑制负样本
run "freeze_hi"  29622 "--init-logit-scale 2.9957 --init-logit-bias -15 --freeze-logit-params"

# C3: (scale=ln(5), bias=-5) 冻结 — 弱抑制
run "freeze_lo"  29623 "--init-logit-scale 1.6094 --init-logit-bias -5 --freeze-logit-params"

# ═══════════════════════════════════════════════════════════════════════
# Group D: Cross combo (感兴趣的组合)
# ═══════════════════════════════════════════════════════════════════════
# D1: scale↑ + bias↓ (大 scale 配深 bias)
run "cross_s20_bm15" 29631 "--init-logit-scale 2.9957 --init-logit-bias -15"

# D2: scale↓ + bias↑ (小 scale 配浅 bias)
run "cross_s05_bm05" 29632 "--init-logit-scale 1.6094 --init-logit-bias -5"

# ══════════════════════════════════════════════════════════════════════
echo ""
if [ "${MODE}" = "smoke" ]; then
    echo "============ Smoke Results ============"
    echo "  PASSED: ${PASSED}"
    echo "  FAILED: ${FAILED}"
    echo "======================================="
    [ ${FAILED} -eq 0 ] && echo "[sb] All smoke passed." || { echo "[sb] SMOKE FAILED!"; exit 1; }
else
    echo "======== [sb] All experiments done ========"
fi
