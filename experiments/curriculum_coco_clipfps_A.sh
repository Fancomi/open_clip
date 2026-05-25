#!/bin/bash
# COCO FPS 策略A扫描: epoch0-only pretrained teacher, 之后恢复随机采样
# 对比 curriculum_coco_clipparadigm.sh 中策略B (每epoch frozen) 的结果
# 覆盖全部6个CLIP-paradigm teacher: siglip2, datacomp, dfn2b, eva02, laion2b, metaclip
# SMOKE=1 在 karpathy_1cap (~5k) 跑 1 epoch train+eval 做冒烟验证

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
    EPOCHS=1
    WARMUP=0
    SAVE_FREQ=0
    WORKERS=8
    LOG_STEPS=1
    NAME_PREFIX="smoke_coco_clipfps_A"
else
    TRAIN="${TRAIN_FULL}"
    EPOCHS=20
    WARMUP=42
    SAVE_FREQ=2
    WORKERS=32
    LOG_STEPS=2
    NAME_PREFIX="coco_clipfps_A"
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
    local TAG=$1 PORT=$2 INIT=$3 DIR=$4
    local NAME="${NAME_PREFIX}_${TAG}_${TS}"
    echo "======== [coco_clipfps_A] ${TAG} (init=${INIT}, dir=${DIR}) => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main --model PE-Core-B-16-dinov3 \
        --train-data "${TRAIN}" --val-data "${VAL}" \
        ${COMMON} ${SIGREG_BASE} \
        --curriculum-strategy ${DIR} \
        --curriculum-init ${INIT} \
        --curriculum-epochs 1 \
        --name "${NAME}" < /dev/null
}

# 策略A: epoch0用外部CLIP teacher特征排序, epoch1+恢复随机采样 (--curriculum-epochs 1)
# 对比策略B (curriculum_coco_clipparadigm.sh): 每epoch都用同一frozen teacher
run "fps_siglip2_e0"    29810 siglip2  fps
run "fpsrev_siglip2_e0" 29811 siglip2  fps_reverse
run "fps_datacomp_e0"   29812 datacomp fps
run "fpsrev_datacomp_e0" 29813 datacomp fps_reverse
run "fps_dfn2b_e0"      29814 dfn2b    fps
run "fpsrev_dfn2b_e0"   29815 dfn2b    fps_reverse
run "fps_eva02_e0"      29816 eva02    fps
run "fpsrev_eva02_e0"   29817 eva02    fps_reverse
run "fps_laion2b_e0"    29818 laion2b  fps
run "fpsrev_laion2b_e0" 29819 laion2b  fps_reverse
run "fps_metaclip_e0"   29820 metaclip fps
run "fpsrev_metaclip_e0" 29821 metaclip fps_reverse

echo "======== COCO clipfps Strategy-A sweep done ========"
