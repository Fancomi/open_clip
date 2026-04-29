#!/bin/bash

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'  # Use Beijing time for timestamps
source /etc/default/locale

TS=$(date +%m%d_%H%M)


COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_dedup.tsv"
VAL="${COCO}/karpathy_5cap.tsv"       # 官方 Karpathy test split，与论文数值对齐
PROBE_TSV="${COCO}/karpathy_1cap.tsv" # per-epoch feature probe dataset

# ============ CC3M 数据加载到内存 ============
CC3M_SRC="/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds"
CC3M="/dev/shm/cc3m_wds"

# _cleanup() {
#     echo "[quick] 清理 /dev/shm/cc3m_wds ..."
#     rm -rf /dev/shm/cc3m_wds
# }
# trap _cleanup EXIT INT TERM


CC3M_TRAIN="${CC3M}/{00000..00280}.tar"
CC3M_N_TRAIN=2857622

# GPU2 - BS2048 - 30Epoch - Best LR: 2e-4 . ln(4096)+1 = 9.3177661667
# GPU8 - BS2048 - 10Epoch - Best LR: 4e-4 . ln(16384)+1 = 10.7040605278
# GPU8 - BS512  - 10Epoch - Best LR: 3.4e-4 (BASE_LR reference point)
GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")

# BASE_LR: best lr at reference config (8GPU x 512BS = GlobalBS 4096)
# Scale rule: lr ∝ sqrt(GlobalBS / ref_GlobalBS)
# GPU8 - BS512 - 10Epoch - Best LR: 3.4e-4
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (${GPUS} * ${PreGpuBS})))")
MUON_LR=$(python3 -c "import math; print(0.02 * math.sqrt(($GlobalBS) / (${GPUS} * ${PreGpuBS})))")

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

# COCO: steps/epoch=21(82783/4096), total_steps(20ep)=420, warmup=10%=42
# CC3M: steps/epoch=174, total_steps(20ep)=3480, warmup=10%=348
COMMON="--warmup 42 ${BASE} --epochs 20 \
    --dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"
run() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="quick_${TAG}_${TS}"
    echo "======== [quick] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

COMMON_WDS="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"
run_cc3m() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="cc3m_${TAG}_${TS}"
    echo "======== [cc3m] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${CC3M_TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON_WDS} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ============ CC3M 数据加载到内存 ============
MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"
PE_CKPT="${MODEL_DIR}/PE-Core-B-16/open_clip_model.safetensors"
SIG2_CKPT="${MODEL_DIR}/ViT-B-16-SigLIP2/open_clip_model.safetensors"

if [ ! -d "${CC3M}" ]; then
    echo "[quick] Loading CC3M to memory (${CC3M_SRC} -> ${CC3M}) ..."
    cp -r "${CC3M_SRC}" "${CC3M}"
    echo "[quick] Done, total $(du -sh ${CC3M} | cut -f1)"
else
    echo "[quick] Found ${CC3M} already exists, skip copy"
fi

# ---- 超参消融（只动 pe_dinov3_leproj，其余 ablation 目的见注释）----
# 问题：R@1 在 epoch 6~8 达峰后下跌；疑似原因：epoch 太多 / wd 过强 / LR 过高
# 实验设计：2×2 矩阵（epochs × wd）+ LR 轴，每次只动一个变量

# baseline: epochs=20, warmup=348, lr=4e-4, wd=0.2  ← 已跑 cc3m_pe_dinov3_leproj_0416_0031
# run_cc3m "pe_dinov3_leproj"   "PE-Core-B-16-dinov3" 29550       "--siglip --lejepa --lejepa-proj"

# # [1] epoch 轴：减半 epoch
# run_cc3m "pe_dinov3_leproj_e10"    "PE-Core-B-16-dinov3" 29549  "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 174"
# run_cc3m "pe_dinov3_leproj_e12_warm384" "PE-Core-B-16-dinov3" 29549  "--siglip --lejepa --lejepa-proj --epochs 12 --warmup 384"
# run_cc3m "pe_dinov3_leproj_e12_warm512" "PE-Core-B-16-dinov3" 29550  "--siglip --lejepa --lejepa-proj --epochs 12 --warmup 512"


# ---- finetune from pretrained: visual+text 双塔全部从预训练权重初始化 ----
LR_1_100=$(python3 -c "print($LR / 100)")
LR_1_50=$(python3 -c "print($LR / 50)")

# # run_cc3m "pe_dinov3_e10_warm768_LR_dinov3" "PE-Core-B-16-dinov3" 29540  "--siglip --epochs 10 --warmup 768 --lr ${LR} --dinov3 --dino-local-crops-number 2 --dino-head-prototypes 8192"

# # Bestv0:  e20 warm384 LR*1.0 (basic = 2e-4 ... bs )
# # BestNow: e10 warm512 LR*1.7 lejepa-weight: 2e-4(from 1e-4)
# # run_cc3m "pe_dinov3_leproj_e8_warm768_LR" "PE-Core-B-16-dinov3" 29541  "--siglip --lejepa --lejepa-proj --epochs 8 --warmup 768  --lr ${LR}"
# run_cc3m "pe_dinov3_leproj_e10_warm768_LR" "PE-Core-B-16-dinov3" 29542  "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 768  --lr ${LR}"

# run_cc3m "pe_dinov3_leproj_e10_warm512_LR_LE2e-4" "PE-Core-B-16-dinov3" 29544  "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --lejepa-weight 2e-4"
# run_cc3m "pe_dinov3_leproj_e10_warm512_LR_LE4e-4" "PE-Core-B-16-dinov3" 29544  "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --lejepa-weight 4e-4"
# run_cc3m "pe_dinov3_leproj_e10_warm512_LR_LE5e-5" "PE-Core-B-16-dinov3" 29544  "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --lejepa-weight 5e-5"


# # # ViT-B-16-SigLIP2: COCO I2T R@1 68.9；visual+text 双塔均已对齐
# # # BestNow: e10 LR / 100
# run_cc3m "siglip2_LR_1_100"          "ViT-B-16-SigLIP2"   29548 "--siglip --pretrained ${SIG2_CKPT}  --wd 0.0 --lr ${LR_1_100} --epochs 10"
# run_cc3m "siglip2_LR_1_50"           "ViT-B-16-SigLIP2"   29549 "--siglip --pretrained ${SIG2_CKPT}  --wd 0.0 --lr ${LR_1_50} --epochs 10"


# # PE-Core-B-16: COCO I2T R@1 71.0
# run_cc3m "pe_core_LR_1_100"          "PE-Core-B-16"        29551 "--siglip --pretrained ${PE_CKPT}  --wd 0.0 --lr ${LR_1_100} --epochs 10"
# run_cc3m "pe_core_LR_1_50"           "PE-Core-B-16"        29552 "--siglip --pretrained ${PE_CKPT}  --wd 0.0 --lr ${LR_1_50} --epochs 10"

# ===
# ORI
# run_cc3m "pe_dinov3_leproj_probe"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --probe-data ${PROBE_TSV}"
# run_cc3m "pe_dinov3_dinov3_probe" "PE-Core-B-16-dinov3" 29540  "--siglip --epochs 10 --warmup 512 --lr ${LR} --dinov3 --dino-local-crops-number 2 --dino-head-prototypes 8192  --probe-data ${PROBE_TSV}"
# run_cc3m "pe_dinov3_dinov3_probe_clip" "PE-Core-B-16-dinov3" 29541  "--epochs 10 --warmup 512 --lr ${LR} --dinov3 --dino-local-crops-number 2 --dino-head-prototypes 8192  --probe-data ${PROBE_TSV} --probe-freq-steps 176"

run_cc3m "pe_dinov3_leproj_muon_lr001"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --opt muon --muon-lr ${MUON_LR}  --probe-data ${PROBE_TSV} --probe-freq-steps 176"
# run_cc3m "pe_dinov3_leproj_muon_lr0005"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --opt muon --muon-lr 0.005  --probe-data ${PROBE_TSV} --probe-freq-steps 176"
# run_cc3m "pe_dinov3_leproj_muon_lr0002"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --opt muon --muon-lr 0.002  --probe-data ${PROBE_TSV} --probe-freq-steps 176"
# run_cc3m "pe_dinov3_leproj_muon_lr001"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --opt muon --muon-lr ${MUON_LR}  --probe-data ${PROBE_TSV} --probe-freq-steps 176"
# run_cc3m "pe_dinov3_leproj_muon_lr001_ep15"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 15 --warmup 512  --lr ${LR} --opt muon --muon-lr ${MUON_LR}  --probe-data ${PROBE_TSV} --probe-freq-steps 176"
# run_cc3m "pe_dinov3_leproj_muon_lr002"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} --opt muon --muon-lr 0.02  --probe-data ${PROBE_TSV} --probe-freq-steps 176"
# run_cc3m "vit_muon_lr002" "ViT-B-16-exp" 29566 "--siglip --epochs 10 --warmup 512 --lr ${LR} --opt muon --muon-lr 0.02"
# run_cc3m "vit_muon"    "ViT-B-16-exp"        29565 "--siglip --epochs 10 --warmup 512  --lr ${LR} --opt muon --muon-lr ${MUON_LR}"


run_cc3m "vit"         "ViT-B-16-exp"        29562 "--siglip --epochs 10 --warmup 512  --lr ${LR} "
run_cc3m "pe_dinov3"   "PE-Core-B-16-dinov3" 29563 "--siglip --epochs 10 --warmup 512  --lr ${LR} "
run_cc3m "dinov3"    "DINOv3-B-16-ape"       29564 "--siglip --epochs 10 --warmup 512  --lr ${LR} "

run_cc3m "pe_dinov3_leproj"   "PE-Core-B-16-dinov3"     29560 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR}"
run_cc3m "vit_leproj"         "ViT-B-16-exp"        29557 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} "
run_cc3m "dinov3_leproj"    "DINOv3-B-16-ape"       29558 "--siglip --lejepa --lejepa-proj --epochs 10 --warmup 512  --lr ${LR} "

run_cc3m "vit_le"         "ViT-B-16-exp"        29559 "--siglip --lejepa --epochs 10 --warmup 512  --lr ${LR} "
run_cc3m "pe_dinov3_le"   "PE-Core-B-16-dinov3" 29560 "--siglip --lejepa --epochs 10 --warmup 512  --lr ${LR} "
run_cc3m "dinov3_le"    "DINOv3-B-16-ape"       29561 "--siglip --lejepa --epochs 10 --warmup 512  --lr ${LR} "


echo "======== quick all done ========"
