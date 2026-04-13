#!/bin/bash
# quick.sh — PE-CLS vs PE-DINOv3 vs ViT with SigLIP/CLIP loss comparison
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

# 实验矩阵（6 组）：
#   PE-CLS    + SigLIP (baseline)
#   PE-CLS    + CLIP
#   PE-DINOv3 + SigLIP (DINOv3-aligned)
#   PE-DINOv3 + CLIP
#   ViT       + SigLIP
#   ViT       + CLIP
#
# 数据：clip_train_dedup.tsv (113,287 行，去重) / clip_val.tsv (25,010 行)
# global batch = 2048/GPU × 2 = 4096
# steps/epoch ≈ 28, 30 epochs = 840 steps total
# schedule: warmup 10 → cosine decay to 0
#
# 硬件：2x A800-80GB
#
# ============ LeJEPA 正则化 ============
# LeJEPA SIGReg (https://arxiv.org/abs/2511.08544): 约束 embedding 服从各向同性高斯
# 两种模式（模型自动包装，始终传 unnormalized features 给 SIGReg）:
#   --lejepa          : SIGReg 直接作用于 backbone raw embedding（轻量）
#   --lejepa --lejepa-proj: SIGReg 作用于 MLP projector 输出（更强正则化）
#
# 关键参数:
#   --lejepa-weight   : SIGReg 权重 λ，推荐 1e-4 ~ 1e-2
#   --lejepa-num-slices: 随机切片数，默认 256
#   --lejepa-proj-dim : projector 输出维度（仅 --lejepa-proj 模式）

set -e
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'  # Use Beijing time for timestamps

TS=$(date +%m%d_%H%M)

COCO="/root/paddlejob/workspace/env_run/penghaotian/datas/coco/annotations"
TRAIN="${COCO}/clip_train_dedup.tsv"
VAL="${COCO}/clip_val.tsv"

COMMON="--dataset-type csv --csv-img-key filepath --csv-caption-key caption \
    --precision amp_bf16 --workers 6 --epochs 30 --batch-size 1024 \
    --lr 2e-4 --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --warmup 20 \
    --save-frequency 0 \
    --grad-checkpointing --log-every-n-steps 1 --val-frequency 5"

# bs:2048 GPU:2,  math.log(2048*2)+1 9.317766166719343
# bs:1024 GPU:2,  math.log(1024*2)+1 8.624

run() {
    local TAG=$1 MODEL=$2 PORT=$3 EXTRA=$4
    local NAME="quick_${TAG}_${TS}"
    echo "======== [quick] ${TAG} => ${NAME} ========"
    torchrun --nproc_per_node=2 --master_port=${PORT} \
        -m open_clip_train.main \
        --model "${MODEL}" \
        --train-data "${TRAIN}" \
        --val-data "${VAL}" \
        ${COMMON} \
        ${EXTRA} \
        --name "${NAME}"
}

# ============ 原有实验 ============
# DINOv3 with SigLIP (ape)
#run "dinov3_siglip"     "DINOv3-B-16-ape"    29510 "--siglip"
# PE-DINOv3 with SigLIP
#run "pe_dinov3_siglip" "PE-Core-B-16-dinov3" 29511 "--siglip"
# PE-CLS with SigLIP (baseline)
#run "pe_cls_siglip"    "PE-Core-B-16-cls"    29512 "--siglip"
# ViT with CLIP
#run "vit_clip"         "ViT-B-16-exp"        29513 ""
# PE-CLS with CLIP
#run "pe_cls_clip"      "PE-Core-B-16-cls"    29514 ""
# PE-DINOv3 with CLIP
#run "pe_dinov3_clip"   "PE-Core-B-16-dinov3" 29515 ""
# ViT with SigLIP
#run "vit_siglip"       "ViT-B-16-exp"        29516 "--siglip"
# DINOv3 with CLIP (ape)
#run "dinov3_clip"    "DINOv3-B-16-ape"       29517 ""

# # ============ LeJEPA 正则化实验 ============
# # 模式1: 无 Projector (轻量级，直接在 CLIP 特征上应用 SIGReg)
# # 推荐用于快速实验验证效果
# run "pe_dinov3_siglip_lejepa"     "PE-Core-B-16-dinov3"     29528 "--siglip --lejepa"

# # 模式2: 有 Projector (更强正则化，通过 Projector 间接约束 backbone) # 原版就是有proj
# # 推荐用于追求更好的正则化效果
# run "pe_dinov3_siglip_lejepa_proj" "PE-Core-B-16-dinov3"    29529 "--siglip --lejepa --lejepa-proj"
# run "dinov3_siglip_lejepa"         "DINOv3-B-16-ape"        29530 "--siglip --lejepa"
# run "dinov3_siglip_lejepa_proj"    "DINOv3-B-16-ape"        29531 "--siglip --lejepa --lejepa-proj"

# run "pe_dinov3_siglip_lejepa_proj_e-3" "PE-Core-B-16-dinov3"    29536 "--siglip --lejepa --lejepa-proj --lejepa-weight 1e-3"
# run "pe_dinov3_siglip_lejepa_e-3"      "PE-Core-B-16-dinov3"    29537 "--siglip --lejepa --lejepa-weight 1e-3"

# run "pe_dinov3_siglip_lejepa_proj_e-2" "PE-Core-B-16-dinov3"    29538 "--siglip --lejepa --lejepa-proj --lejepa-weight 1e-2"
# run "pe_dinov3_siglip_lejepa_e-2"      "PE-Core-B-16-dinov3"    29539 "--siglip --lejepa --lejepa-weight 1e-2"

# 对比全部的无le和有le
# run "vit_leproj"         "ViT-B-16-exp"        29513 "--siglip --lejepa --lejepa-proj"
# run "pe_cls_leproj"      "PE-Core-B-16-cls"    29514 "--siglip --lejepa --lejepa-proj"
# run "pe_dinov3_leproj"   "PE-Core-B-16-dinov3" 29515 "--siglip --lejepa --lejepa-proj"
# run "dinov3_leproj"    "DINOv3-B-16-ape"       29517 "--siglip --lejepa --lejepa-proj"
# run "vit_le"         "ViT-B-16-exp"        29513 "--siglip --lejepa"
# run "pe_cls_le"      "PE-Core-B-16-cls"    29514 "--siglip --lejepa"
# run "pe_dinov3_le"   "PE-Core-B-16-dinov3" 29515 "--siglip --lejepa"
# run "dinov3_le"    "DINOv3-B-16-ape"       29517 "--siglip --lejepa"

# run "pe_dinov3" "PE-Core-B-16-dinov3"    29529 "--siglip"
# run "pe_dinov3_attnres_b3" "PE-Core-B-16-dinov3" 29540 "--siglip --attn-res --attn-res-block-size 2"
# run "pe_dinov3_lejepa" "PE-Core-B-16-dinov3"    29529 "--siglip --lejepa --lejepa-proj"

# 对比全部的无AttnRes和有AttnRes
run "pe_dinov3_attnres2"   "PE-Core-B-16-dinov3" 29515 "--siglip --attn-res --attn-res-block-size 2"
run "vit_attnres2"         "ViT-B-16-exp"        29513 "--siglip --attn-res --attn-res-block-size 2"
run "pe_cls_attnres2"      "PE-Core-B-16-cls"    29514 "--siglip --attn-res --attn-res-block-size 2"
run "dinov3_attnres2"    "DINOv3-B-16-ape"       29517 "--siglip --attn-res --attn-res-block-size 2"

# run "pe_dinov3_leproj_attnres2" "PE-Core-B-16-dinov3" 29540 "--siglip --attn-res --attn-res-block-size 2 --lejepa --lejepa-proj"
# run "dinov3_le_attnres2" "DINOv3-B-16-ape"       29541 "--siglip --attn-res --attn-res-block-size 2 --lejepa"
# run "vit_leproj_attnres2" "ViT-B-16-exp"         29542 "--siglip --attn-res --attn-res-block-size 2 --lejepa --lejepa-proj"
# run "pe_cls_le_attnres2" "PE-Core-B-16-cls"      29543 "--siglip --attn-res --attn-res-block-size 2 --lejepa"

# run "pe_dinov3"   "PE-Core-B-16-dinov3" 29515 "--siglip"
# run "vit"         "ViT-B-16-exp"        29513 "--siglip"
# run "pe_cls"      "PE-Core-B-16-cls"    29514 "--siglip"
# run "dinov3"    "DINOv3-B-16-ape"       29517 "--siglip"

echo "======== quick 全部完成 ========"
