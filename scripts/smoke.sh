#!/bin/bash
# smoke.sh — 快速冒烟测试，验证训练 pipeline 关键路径
#
# 三组测试：
#   A. Fine-tune: pretrained 加载 + lock + sigreg（synthetic 数据，1GPU）
#   B. DINOv3:   8GPU + CC3M webdataset（仅 5 步）
#
# 用法：
#   bash scripts/smoke.sh          # 全部
#   bash scripts/smoke.sh ft       # 只跑微调组
#   bash scripts/smoke.sh dinov3   # 只跑 DINOv3 组

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
GROUP="${1:-all}"  # ft / dinov3 / all

# ============ 路径 ============
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"
TRAIN="${COCO}/karpathy_1cap.tsv"

CC3M="/dev/shm/cc3m_wds"
CC3M_TRAIN="${CC3M}/{00000..00280}.tar"

MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"
SIG2_CKPT="${MODEL_DIR}/ViT-B-16-SigLIP2/open_clip_model.safetensors"

# ============ 硬件参数 ============
GPUS=8
PreGpuBS=512
GlobalBS=$((PreGpuBS * GPUS))
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(${GlobalBS} / 4096))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(${GlobalBS} / 4096))")
SMOKE_N=$((PreGpuBS * GPUS * 5))

# ============ 计数 ============
PASSED=0; FAILED=0; SKIPPED=0

# ============ 运行函数 ============
# 单卡 synthetic（无数据依赖，最轻量）
run_smoke_syn() {
    local TAG=$1 MODEL=$2 PORT=$3; shift 3
    local NAME="smoke_${TAG}_${TS}"
    echo -n "[smoke/syn] ${TAG} ... "
    local LOG="/tmp/smoke_${TAG}.log"
    if torchrun --nproc_per_node=1 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --dataset-type synthetic --train-num-samples 64 \
        --batch-size 8 --epochs 1 --warmup 0 --workers 0 \
        --precision amp_bf16 --lr 1e-5 --wd 0.01 \
        --save-frequency 0 --log-every-n-steps 1 \
        "$@" --name "${NAME}" > "${LOG}" 2>&1; then
        echo "PASS"; PASSED=$((PASSED + 1))
    else
        echo "FAIL (see /tmp/smoke_${TAG}.log)"; FAILED=$((FAILED + 1))
        tail -3 "${LOG}"
    fi
    rm -rf "./logs/${NAME}" 2>/dev/null || true
}

# 8卡 WDS
run_smoke_wds() {
    local TAG=$1 MODEL=$2 PORT=$3; shift 3
    local NAME="smoke_${TAG}_${TS}"
    echo -n "[smoke/wds] ${TAG} ... "
    local LOG="/tmp/smoke_${TAG}.log"
    if torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" --train-data "${CC3M_TRAIN}" --val-data "${VAL}" \
        --dataset-type webdataset --train-num-samples ${SMOKE_N} \
        --csv-img-key filepath --csv-caption-key caption \
        --batch-size ${PreGpuBS} --epochs 1 --warmup 2 --workers 32 \
        --precision amp_bf16 --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
        --save-frequency 0 --log-every-n-steps 1 --val-frequency 0 \
        --grad-checkpointing \
        "$@" --name "${NAME}" > "${LOG}" 2>&1; then
        echo "PASS"; PASSED=$((PASSED + 1))
    else
        echo "FAIL (see /tmp/smoke_${TAG}.log)"; FAILED=$((FAILED + 1))
        tail -3 "${LOG}"
    fi
    rm -rf "./logs/${NAME}" 2>/dev/null || true
}

# 条件跳过：检查文件是否存在
need() {
    local WHAT=$1 PATH_=$2
    if [ ! -e "${PATH_}" ]; then
        echo "[smoke] SKIP ($WHAT not found: ${PATH_})"
        SKIPPED=$((SKIPPED + 1)); return 1
    fi
    return 0
}

# ======================================================================
# A. Fine-tune 组
# ======================================================================
if [ "${GROUP}" = "ft" ] || [ "${GROUP}" = "all" ]; then
    echo ""; echo "============ A. Fine-tune Smoke Tests ============"
    # 注意：PE-Core-B-16 使用标准 BPE tokenizer，无网络依赖
    # ViT-B-16-SigLIP2 需要 HF tokenizer，依赖网络或 local-dir

    # A1: 从零训练基线（PE-Core，无网络依赖）
    run_smoke_syn "scratch_pe" "PE-Core-B-16" 29700

    # A2: lock-image 冻结机制
    run_smoke_syn "lock_img" "PE-Core-B-16" 29701 --lock-image

    # A3: lock-image + partial unlock
    run_smoke_syn "lock_partial" "PE-Core-B-16" 29702 \
        --lock-image --lock-image-unlocked-groups 2

    # A4: sigreg（无 pretrained，验证 loss 兼容）
    run_smoke_syn "sigreg_cls" "PE-Core-B-16" 29703 \
        --sigreg-target cls --sigreg-weight 1e-4

    # A5-A7: 带 pretrained 权重微调（需要 checkpoint 文件）
    if need "PE_CKPT" "${PE_CKPT}"; then
        run_smoke_syn "pe_ft_lit" "PE-Core-B-16" 29704 \
            --pretrained "${PE_CKPT}" --lock-image

        run_smoke_syn "pe_ft_partial" "PE-Core-B-16" 29705 \
            --pretrained "${PE_CKPT}" --lock-image --lock-image-unlocked-groups 3

        run_smoke_syn "pe_ft_sigreg" "PE-Core-B-16" 29706 \
            --pretrained "${PE_CKPT}" --sigreg-target cls --sigreg-weight 1e-4
    fi

    # A8-A9: SigLIP2（需要 checkpoint + tokenizer）
    if need "SIG2_CKPT" "${SIG2_CKPT}"; then
        run_smoke_syn "sig2_ft_lit" "ViT-B-16-SigLIP2" 29707 \
            --pretrained "${SIG2_CKPT}" --siglip --lock-image

        run_smoke_syn "sig2_ft_sigreg" "ViT-B-16-SigLIP2" 29708 \
            --pretrained "${SIG2_CKPT}" --siglip \
            --sigreg-target cls --sigreg-weight 1e-4
    fi

    # A10-A11: Antipodal SigLIP（正样本推向 cos=-1）
    run_smoke_syn "antipodal_siglip" "PE-Core-B-16" 29709 --siglip --antipodal

    run_smoke_syn "antipodal_sigreg" "PE-Core-B-16" 29710 \
        --siglip --antipodal --sigreg-target cls --sigreg-weight 1e-4
fi

# ======================================================================
# B. DINOv3 组
# ======================================================================
if [ "${GROUP}" = "dinov3" ] || [ "${GROUP}" = "all" ]; then
    echo ""; echo "============ B. DINOv3 Smoke Tests ============"

    if ! need "CC3M" "${CC3M}"; then
        true  # skipped
    else
        DINO_COMMON="--siglip --opt muon --muon-lr ${MUON_LR} \
            --dinov3 --dino-n-global-crops 1 --dino-local-crops-number 8 \
            --dino-head-prototypes 8192 --dino-warmup-teacher-temp-epochs 1"

        run_smoke_wds "dinov3_sigreg" "PE-Core-B-16-dinov3" 29710 \
            ${DINO_COMMON} --sigreg-target cls --sigreg-weight 1e-4

        run_smoke_wds "dinov3_sigreg_proj" "PE-Core-B-16-dinov3" 29711 \
            ${DINO_COMMON} --sigreg-target cls_proj --sigreg-weight 1e-4

        run_smoke_wds "dinov3_only" "PE-Core-B-16-dinov3" 29712 \
            ${DINO_COMMON}
    fi
fi

# ============ 汇总 ============
echo ""
echo "============ Results ============"
echo "  PASSED:  ${PASSED}"
echo "  FAILED:  ${FAILED}"
echo "  SKIPPED: ${SKIPPED}"
echo "================================="
[ ${FAILED} -eq 0 ] && echo "[smoke] All passed." || { echo "[smoke] FAILED!"; exit 1; }
