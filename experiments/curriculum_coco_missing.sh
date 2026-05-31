#!/bin/bash
# 补全 curriculum 三策略矩阵的缺失实验：
#   - Strategy C for pe_core (新config，可与A/B直接比较)
#   - Strategy A + C for dinov3 (新config)
#   - Strategy C for 6个外部CLIP teacher (使用 _c 后缀)
# SMOKE=1 在 karpathy_1cap (~5k) 跑1epoch验证

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

SMOKE=${SMOKE:-0}
TS=$(date +%m%d_%H%M)
COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN_FULL="${COCO}/clip_train_dedup.tsv"
TRAIN_SMOKE="${COCO}/karpathy_1cap.tsv"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

GPUS=8
PreGpuBS=512
LR=0.00034
MUON_LR=0.01

if [ "${SMOKE}" = "1" ]; then
    TRAIN="${TRAIN_SMOKE}"
    EPOCHS=1; WARMUP=0; SAVE_FREQ=0; WORKERS=8; LOG_STEPS=1
    NAME_PREFIX="smoke_coco_cur_missing"
else
    TRAIN="${TRAIN_FULL}"
    EPOCHS=20; WARMUP=42; SAVE_FREQ=2; WORKERS=32; LOG_STEPS=2
    NAME_PREFIX="coco_cur_missing"
fi

COMMON="--precision amp_bf16 --workers ${WORKERS} --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency ${SAVE_FREQ} --grad-checkpointing \
    --log-every-n-steps ${LOG_STEPS} --val-frequency 1 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5 --epochs ${EPOCHS} --warmup ${WARMUP}"

if [ "${SMOKE}" != "1" ]; then
    COMMON="${COMMON} --delete-previous-checkpoint"
fi

SIGREG_BASE="--siglip --sigreg-target cls --sigreg-weight 1e-4 \
    --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV}"

run() {
    local TAG=$1 PORT=$2 EXTRA=$3
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [cur_missing] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN}" --val-data "${VAL}" \
        ${COMMON} ${SIGREG_BASE} ${EXTRA} --name "${NAME}" < /dev/null
}

# ── Strategy C for pe_core (新config) ─────────────────────────────────────────
# --curriculum-init pe_core (无epoch限制): epoch0用pe_core，epoch1+用自身特征
run "c_fps_pecore"    29830 "--curriculum-strategy fps         --curriculum-init pe_core"
run "c_fpsrev_pecore" 29831 "--curriculum-strategy fps_reverse --curriculum-init pe_core"

# ── Strategy A for dinov3 (新config) ──────────────────────────────────────────
# epoch0用dinov3特征，epoch1+恢复随机
run "a_fps_dinov3"    29832 "--curriculum-strategy fps         --curriculum-init dinov3 --curriculum-epochs 1"
run "a_fpsrev_dinov3" 29833 "--curriculum-strategy fps_reverse --curriculum-init dinov3 --curriculum-epochs 1"

# ── Strategy C for dinov3 (新config) ──────────────────────────────────────────
# epoch0用dinov3，epoch1+用自身特征
run "c_fps_dinov3"    29834 "--curriculum-strategy fps         --curriculum-init dinov3"
run "c_fpsrev_dinov3" 29835 "--curriculum-strategy fps_reverse --curriculum-init dinov3"

# ── Strategy C for 外部CLIP teachers (_c后缀) ─────────────────────────────────
# epoch0用外部teacher特征，epoch1+用自身特征
run "c_fps_siglip2"    29836 "--curriculum-strategy fps         --curriculum-init siglip2_c"
run "c_fpsrev_siglip2" 29837 "--curriculum-strategy fps_reverse --curriculum-init siglip2_c"
run "c_fps_datacomp"   29838 "--curriculum-strategy fps         --curriculum-init datacomp_c"
run "c_fpsrev_datacomp" 29839 "--curriculum-strategy fps_reverse --curriculum-init datacomp_c"
run "c_fps_dfn2b"      29840 "--curriculum-strategy fps         --curriculum-init dfn2b_c"
run "c_fpsrev_dfn2b"   29841 "--curriculum-strategy fps_reverse --curriculum-init dfn2b_c"
run "c_fps_eva02"      29842 "--curriculum-strategy fps         --curriculum-init eva02_c"
run "c_fpsrev_eva02"   29843 "--curriculum-strategy fps_reverse --curriculum-init eva02_c"
run "c_fps_laion2b"    29844 "--curriculum-strategy fps         --curriculum-init laion2b_c"
run "c_fpsrev_laion2b" 29845 "--curriculum-strategy fps_reverse --curriculum-init laion2b_c"
run "c_fps_metaclip"   29846 "--curriculum-strategy fps         --curriculum-init metaclip_c"
run "c_fpsrev_metaclip" 29847 "--curriculum-strategy fps_reverse --curriculum-init metaclip_c"

echo "======== curriculum missing runs done ========"
