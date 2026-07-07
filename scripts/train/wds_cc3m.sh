#!/bin/bash
# CC3M 文本去重 vs 未去重 对照实验 (raw vs dedup)
#
# 目的：验证文本去重是否提升 CC3M 训练后检索性能。
#   cc3m-wds 内部有 ~29.35% 文本冗余（2,905,954 → 2,053,037 唯一 caption）。
#
# 数据：
#   raw   : /root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds/       (2,905,954 samples)
#   dedup : /root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-dedup-wds/ (~2,053,037 samples)
#           由 scripts/build_cc3m_dedup_wds.py 生成（归一化 caption 首次出现保留）
#
# 严格 A/B：两组唯一差别是 --train-data。等算力控制 = 相同 train-num-samples × epochs。
#
# 漏数据修复（webdataset shard 切分不整除会静默截断/重复）：
#   两组都加 --dataset-resampled → 每 worker 放回采样，绕开 data.py:558-568 的整除截断，
#   总样本数 = train-num-samples × epochs 精确相等，两组逐 step 对齐。
#
# val: COCO karpathy_5cap.tsv（已验证与 cc3m/cc12m 几乎无文本泄漏），R@1/5/10 图文互检。

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"

# ── raw vs dedup 数据源 ──────────────────────────────────────────────────────
CC3M_RAW_WDS="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-wds"
CC3M_RAW_TRAIN="${CC3M_RAW_WDS}/cc3m-train-{0000..0575}.tar"

CC3M_DEDUP_WDS="/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-dedup-wds"
# shard 范围来自 build_cc3m_dedup_wds.py 的 _dedup_stats.json（411 shards, 2,053,037 samples）
CC3M_DEDUP_TRAIN="${CC3M_DEDUP_WDS}/cc3m-train-{00000..00410}.tar"

# 等算力：两组都以 RAW 池大小为准（向上补齐）。dedup 组在 resampled 下把 2.05M
# 唯一样本有放回采样到每 epoch 2.9M，保证两组每 epoch 见到的样本数、总 step 数逐位相等。
CC3M_N_TRAIN=2905954

GPUS=8
PreGpuBS=512
GlobalBS=$(python3 -c "print(${PreGpuBS} * ${GPUS})")

BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(($GlobalBS) / (8 * 512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(($GlobalBS) / (8 * 512)))")

MODEL_DIR="/root/paddlejob/workspace/env_run/penghaotian/models/timm"

BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
    --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
    --save-frequency 1 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"

# CC3M-wds: steps/epoch ≈ 2905954/4096 ≈ 709, warmup 512, epoch 数(10) 与既往一致
# --dataset-resampled 修复 wds shard 整除截断的漏数据问题（两组一致）
COMMON_WDS="--warmup 512 ${BASE} --epochs 10 \
    --dataset-type webdataset --dataset-resampled --train-num-samples ${CC3M_N_TRAIN} \
    --csv-img-key filepath --csv-caption-key caption \
    --val-num-captions-per-image 5"

# run_wds TAG MODEL PORT TRAIN_DATA EXTRA
run_wds() {
    local TAG=$1 MODEL=$2 PORT=$3 TRAIN_DATA=$4 EXTRA=$5
    local NAME="wds_cc3m_${TAG}_${TS}"
    echo "======== [wds_cc3m] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN_DATA}" \
        --val-data "${VAL}" \
        ${COMMON_WDS} \
        ${EXTRA} \
        --name "${NAME}" < /dev/null
}

# ── baseline replicated on cc3m-wds ──────────────────────────────────────────

# ORRI VIT
# run_wds "vit"  "ViT-B-16-exp" 29562 \
#     "--siglip \
#     --epochs 10 --warmup 512 \
#     --probe-data ${PROBE_TSV}"

# ORI
# run_wds "pe_dinov3_siglip" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip \
#     --epochs 10 --warmup 512 \
#     --probe-data ${PROBE_TSV}"

# + Muon
# run_wds "pe_dinov3_siglip_muon" "PE-Core-B-16-dinov3" 29561 \
#     "--siglip \
#     --epochs 10 --warmup 512 \
#     --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#     --probe-data ${PROBE_TSV}"

# + Muon + SigREG
# run_wds "pe_dinov3_sigreg_siglip_muon" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip --sigreg-target cls --sigreg-weight 1e-4 \
#      --epochs 10 --warmup 512 \
#      --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#      --probe-data ${PROBE_TSV}"

# + Muon + SigREG + dino
# run_wds "pe_dinov3_sigreg_siglip_muon_dino" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip --sigreg-target cls --sigreg-weight 1e-4 \
#      --epochs 10 --warmup 512 \
#      --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#      --dinov3 --dino-n-global-crops 1 --dino-local-crops-number 8 --dino-head-prototypes 8192 --dino-warmup-teacher-temp-epochs 3 \
#      --probe-data ${PROBE_TSV}"

# ════════════════════════════════════════════════════════════════════════════
# 旧的 antipodal / orthogonal 验证（本次实验不跑，已注释）
# ════════════════════════════════════════════════════════════════════════════
# run_wds "anti_sigreg_muon" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip --neg-mode antipodal --sigreg-target cls --sigreg-weight 1e-4 \
#      --epochs 10 --warmup 512 \
#      --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#      --probe-data ${PROBE_TSV}"
# run_wds "ortho_sigreg_muon" "PE-Core-B-16-dinov3" 29560 \
#     "--siglip --neg-mode orthogonal --sigreg-target cls --sigreg-weight 1e-4 \
#      --epochs 10 --warmup 512 \
#      --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
#      --probe-data ${PROBE_TSV}"

# ════════════════════════════════════════════════════════════════════════════
# 本次实验：CC3M 文本去重 vs 未去重（projective + Muon + SigREG）
#
# 精确对齐历史 10-epoch no-dino 最优 run `proj_s15_sigreg`（i2t R@1=0.2344 @ep8）：
#   SigLIP + projective + Muon + SIGReg cls 1e-4 + --init-logit-scale 15.0
# 两组共用 COMMON_WDS（--dataset-resampled + train-num-samples=2905954 + epochs=10），
# 唯一差别是 --train-data。串行执行：A(raw) 先跑完再跑 B(dedup)。
#
# 与历史最优仅剩的差异：--dataset-resampled True（历史 False）——这是有意的漏数据修复。
# ════════════════════════════════════════════════════════════════════════════

# 精确对齐历史最优的 init logit scale=15.0（实际 scale）。
# 注意：--init-logit-scale 是 log 空间参数（模型 forward 用 logit_scale.exp()），
# 默认 SigLIP=ln(10)。历史 proj_s15 打印 "Logit Scale: 15.000" 对应 raw=ln(15)。
INIT_LS=$(python3 -c "import math; print(math.log(15))")

EXP_EXTRA="--siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
     --sigreg-target cls --sigreg-weight 1e-4 \
     --epochs 10 --warmup 512 \
     --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
     --probe-data ${PROBE_TSV}"

# A. RAW（未去重，2.9M 池，resampled 下物理重复样本被采样概率更高）
run_wds "proj_muon_raw"   "PE-Core-B-16-dinov3" 29560 "${CC3M_RAW_TRAIN}"   "${EXP_EXTRA}"

# B. DEDUP（去重，2.05M 池，均匀采样）
run_wds "proj_muon_dedup" "PE-Core-B-16-dinov3" 29561 "${CC3M_DEDUP_TRAIN}" "${EXP_EXTRA}"

echo "======== wds_cc3m all done ========"
