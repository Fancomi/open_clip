#!/bin/bash
# visreg.sh — VISReg 训练统一入口（合并原 visreg_* 8 个脚本）
#
# 背景：VISReg 研究已于 2026-07-30 全部结束（见 analysis/research/visreg_all_attempts.md，
# 21 组实验）。结论：E 配方是最优——VISReg scale+shape 等权、no-center、w=1.83e-4、
# K=256、target=cls → COCO 24.06 / IN-1k 23.26（SIGReg 锚点 22.84/21.23）。
# 本脚本合并原 visreg_cc3m/sweep/magnitude/wsweep/slices/mech/stage2/xmodal 8 个脚本，
# 保留全部历史实验入口，并新增 gemma-dense 长文本配方（context_length=256）。
#
# 用法：
#   bash scripts/train/visreg.sh smoke                 # 冒烟（SIGReg/VISReg 各几步 + eval）
#   bash scripts/train/visreg.sh ab                    # A-E 组件消融（原 visreg_cc3m）
#   bash scripts/train/visreg.sh sweep                 # scale:shape 配比面（原 visreg_sweep）
#   bash scripts/train/visreg.sh magnitude             # 强度 0.5×/2×（原 visreg_magnitude）
#   bash scripts/train/visreg.sh wsweep                # 跨数量级强度（原 visreg_wsweep）
#   bash scripts/train/visreg.sh slices                # 切片数 K（原 visreg_slices）
#   bash scripts/train/visreg.sh mech                  # 机制改进（原 visreg_mech）
#   bash scripts/train/visreg.sh stage2                # 第二阶段（原 visreg_stage2）
#   bash scripts/train/visreg.sh xmodal                # 跨模态（原 visreg_xmodal）
#   bash scripts/train/visreg.sh gemma                 # ★ gemma-dense 长文本（context=256）
#   bash scripts/train/visreg.sh gemma-smoke           # gemma-dense 冒烟（1 epoch）
#   bash scripts/train/visreg.sh pcm                   # ★ Long-CLIP PCM 长短双分支
#   bash scripts/train/visreg.sh pcm-smoke             # PCM 冒烟（4 steps）
#   bash scripts/train/visreg.sh gt-std                # standard 口径对照（vs projective）
#
# 关键参数（全部可环境变量 override）：
#   GPUS PreGpuBS EPOCHS WARMUP VISREG_W VISREG_SLICES SIGREG_WEIGHT
#   DATA_VERSION=gt|short|dense|dense_256   # gemma 文本版本（默认 dense_256）
#   NEG_MODE=projective|standard            # 相似度几何；★评估须用同口径★
#   PCM_WEIGHT PCM_DIM                      # PCM 短分支权重 / PCA 维度（默认 1.0 / 32）

set -e
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:${PYTHONPATH}"
export TZ='Asia/Shanghai'

TS=$(date +%m%d_%H%M)
ROOT=/root/paddlejob/workspace/env_run/penghaotian
COCO="${ROOT}/datas/coco/annotations"
VAL="${COCO}/karpathy_5cap.tsv"
PROBE_TSV="${COCO}/karpathy_1cap.tsv"
IMNVAL="${ROOT}/datas/imagenet-val"
CC3M_TSV="${ROOT}/datas/cc3m-tsv/annotations/clip_train.tsv"
CC3M_N_TRAIN=2894191

# gemma-dense 长文本数据（build_gemma_tsv.py 产出，等待补跑完成后生成）
GEMMA_TSV_DIR="${GEMMA_TSV_DIR:-${ROOT}/datas/cc3m-tsv/annotations}"
DATA_VERSION="${DATA_VERSION:-dense_256}"
GEMMA_TSV="${GEMMA_TSV_DIR}/clip_train_${DATA_VERSION}.tsv"
# 文本塔窗口。默认 256 = 全部历史 run 的取值（改这个默认值会让历史 run 不可复现）。
# CTX_LEN=320 用于"dense 生产目标窗口 + 截断代替丢行"那条线（Task #21 / L1）。
# ⚠️ 评测端不要写死：eval_* 已改为按 ckpt 的 text.positional_embedding 自动探测，
#    写死 256 去加载 320 ckpt 会静默插值压回 256，不报错但指标变差。
CTX_LEN="${CTX_LEN:-256}"

GPUS=${GPUS:-8}; PreGpuBS=${PreGpuBS:-512}
GlobalBS=$((PreGpuBS * GPUS))
BASE_LR=3.4e-4
LR=$(python3 -c "import math; print(${BASE_LR} * math.sqrt(${GlobalBS}/(8*512)))")
MUON_LR=$(python3 -c "import math; print(0.01 * math.sqrt(${GlobalBS}/(8*512)))")
INIT_LS=$(python3 -c "import math; print(math.log(15))")
EPOCHS=${EPOCHS:-10}; WARMUP=${WARMUP:-512}

# 标定所得 VISReg 权重（gradient-match SIGReg 1e-4；可 override）
VISREG_W="${VISREG_W:-1.83e-4}"
SIGREG_W="${SIGREG_WEIGHT:-1e-4}"
VISREG_SLICES="${VISREG_SLICES:-256}"

# ═══════════════════════════════════════════════════════════════════════════
# 公共片段
# ═══════════════════════════════════════════════════════════════════════════
# 冠军配方（除正则项外全部固定）
# NEG_MODE 可 override（projective=E 配方默认；standard=与业界方案对齐的对照口径）
# ★ 评估时 --neg-mode 必须与训练一致，详见 analysis/research/eval_protocol.md ★
NEG_MODE="${NEG_MODE:-projective}"
CHAMPION="--siglip --neg-mode ${NEG_MODE} --init-logit-scale ${INIT_LS} \
    --sigreg-target cls \
    --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
    --probe-data ${PROBE_TSV}"

# E 配方（最优）：VISReg scale+shape 等权、no-center
VISREG_E="--reg-method visreg --sigreg-weight ${VISREG_W} --sigreg-slices ${VISREG_SLICES} \
    --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"

# gemma 长文本配方：同 E + 256 上下文（默认 PE-Core 已默认 256，显式传更稳）。
# csv loader 用实际行数作 num_samples（无需 --train-num-samples）。
# GEMMA_COMMON 在 run_gemma 内动态构造，保证 EPOCHS/WARMUP/PreGpuBS 的 override 生效。

run() {  # run TAG PORT DATA EXTRA
    local TAG=$1 PORT=$2 DATA=$3 EXTRA=$4
    local NAME="visreg_${TAG}_${TS}"
    # 动态构造，保证 EPOCHS/WARMUP/PreGpuBS 的 override 生效
    local BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
        --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --seed ${SEED:-0} \
        --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"
    local COMMON="--warmup ${WARMUP} ${BASE} --epochs ${EPOCHS} \
        --dataset-type csv --train-num-samples ${CC3M_N_TRAIN} \
        --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5 \
        --imagenet-val ${IMNVAL}"
    echo "======== [visreg] ${TAG} (${EXTRA}) => ${NAME} ========"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
            --model PE-Core-B-16-dinov3 --train-data ${DATA} --val-data ${VAL} \
            ${COMMON} ${CHAMPION} ${EXTRA} --name ${NAME}"
        return 0
    fi
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${DATA}" --val-data "${VAL}" \
        ${COMMON} ${CHAMPION} ${EXTRA} \
        --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 失败/崩溃（本身是信息，继续）"
}

run_gemma() {  # run_gemma TAG PORT EXTRA   (gemma dense 数据, 256 上下文)
    local TAG=$1 PORT=$2 EXTRA=$3
    local NAME="visreg_gemma_${DATA_VERSION}_${TAG}_${TS}"
    # 动态构造，保证 EPOCHS/WARMUP/PreGpuBS/GEMMA_N_TRAIN 的 override 生效
    local BASE="--precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
        --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 --seed ${SEED:-0} \
        --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1"
    # csv loader 用实际行数作 num_samples；GEMMA_N_TRAIN>0 时强制样本数（冒烟用）
    local NS=""
    [ -n "${GEMMA_N_TRAIN:-}" ] && [ "${GEMMA_N_TRAIN}" -gt 0 ] 2>/dev/null && NS="--train-num-samples ${GEMMA_N_TRAIN}"
    # PCM（Long-CLIP Primary Component Matching）：主分支长文本 + 短分支 PCA 图像×短文本
    local CAPKEY="${CSV_CAPTION_KEY:-caption}"
    local PCM=""
    if [ -n "${PCM_WEIGHT:-}" ]; then
        PCM="--pcm-weight ${PCM_WEIGHT} --pcm-dim ${PCM_DIM:-32} \
             --csv-caption2-key ${CSV_CAPTION2_KEY:-caption_short}"
    fi
    # 区域-短语对比（FG-CLIP 式）；坐标依赖 resize-only 变换
    local REGION=""
    if [ -n "${REGION_WEIGHT:-}" ]; then
        # 图像变换：默认 resize-only（框坐标要求）；REGION_CROP_AUG=1 时改用
        # 框随裁剪同步变换（RandomResizedCropWithBoxes），拿回随机裁剪的正则化收益
        local _IMGAUG="--image-resize-only"
        [ "${REGION_CROP_AUG:-0}" = "1" ] && _IMGAUG="--region-crop-aug"
        REGION="--region-weight ${REGION_WEIGHT} --csv-region-key ${CSV_REGION_KEY:-regions} \
                --max-region ${MAX_REGION:-12} --region-gather ${REGION_GATHER:-local} \
                --region-cc-weight ${REGION_CC_WEIGHT:-0.1} \
                ${REGION_SHARED_SCALE:+--region-shared-scale} \
                ${REGION_ROI_GRID:+--region-roi-grid ${REGION_ROI_GRID}} \
                ${REGION_ROI_AGG:+--region-roi-agg ${REGION_ROI_AGG}} \
                ${REGION_TEXT_CHUNK:+--region-text-chunk ${REGION_TEXT_CHUNK}} \
                ${REGION_NO_HEAD:+--region-no-boxtext-head} ${_IMGAUG}"
    fi
    # 无区域但要 resize-only 对照（A' 组）
    [ -z "${REGION_WEIGHT:-}" ] && [ "${RESIZE_ONLY:-0}" = "1" ] && REGION="--image-resize-only"
    local GEMMA_COMMON="--warmup ${WARMUP} ${BASE} --epochs ${EPOCHS} ${NS} \
        --dataset-type csv --force-context-length ${CTX_LEN} \
        --csv-img-key filepath --csv-caption-key ${CAPKEY} --val-num-captions-per-image 5 \
        ${PCM} ${REGION} --imagenet-val ${IMNVAL}"
    echo "======== [gemma] ${TAG} (data=${DATA_VERSION} ${EXTRA}) => ${NAME} ========"
    if [ ! -f "${GEMMA_TSV}" ]; then
        echo "!!!! 缺 ${GEMMA_TSV} —— 先跑 scripts/data/build_gemma_tsv.py 生成"
        return 1
    fi
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
            --model PE-Core-B-16-dinov3 --train-data ${GEMMA_TSV} --val-data ${VAL} \
            ${GEMMA_COMMON} ${CHAMPION} ${EXTRA} --name ${NAME}"
        return 0
    fi
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${GEMMA_TSV}" --val-data "${VAL}" \
        ${GEMMA_COMMON} ${CHAMPION} ${EXTRA} \
        --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 失败/崩溃（本身是信息，继续）"
}

run_dual() {  # run_dual TAG PORT   (DualTextCLIP 双文本塔, 双列 TSV)
    local TAG=$1 PORT=$2
    local NAME="visreg_dual_${TAG}_${TS}"
    # 双列数据：filepath, caption_short(gt), caption_dense；无 sigreg（双塔自带双 SigLIP）
    local DUAL_COMMON="--warmup ${WARMUP} --precision amp_bf16 --workers 32 --batch-size ${PreGpuBS} \
        --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
        --save-frequency 1 --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 \
        --epochs ${EPOCHS} --dataset-type csv --force-context-length ${CTX_LEN} \
        --csv-img-key filepath --csv-caption-key caption_short --csv-caption2-key caption_dense \
        --val-num-captions-per-image 5 --imagenet-val ${IMNVAL}"
    echo "======== [dual] ${TAG} => ${NAME} ========"
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "[dry-run] torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
            --model PE-Core-B-16-dinov3 --train-data ${DUAL_TSV} --val-data ${VAL} \
            ${DUAL_COMMON} --siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
            --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV} --dual-text --name ${NAME}"
        return 0
    fi
    torchrun --nproc_per_node=${GPUS} --master_port=${PORT} -m open_clip_train.main \
        --model "PE-Core-B-16-dinov3" \
        --train-data "${DUAL_TSV}" --val-data "${VAL}" \
        ${DUAL_COMMON} --siglip --neg-mode projective --init-logit-scale ${INIT_LS} \
        --opt muon --muon-lr ${MUON_LR} --probe-data ${PROBE_TSV} --dual-text \
        --name "${NAME}" < /dev/null || echo "!!!! ${TAG} 失败/崩溃（本身是信息，继续）"
}

# ═══════════════════════════════════════════════════════════════════════════
# 各模式
# ═══════════════════════════════════════════════════════════════════════════
case "${1:-usage}" in
  usage)
    echo "用法: bash scripts/train/visreg.sh {smoke|ab|sweep|magnitude|wsweep|slices|mech|stage2|xmodal|gemma|gemma-smoke}"
    ;;

  # ── 冒烟：SIGReg / VISReg 各跑几步 + 一次 COCO eval ─────────────────────
  smoke)
    SMOKE_N=$((GlobalBS * 4))
    SMOKE_BASE="--precision amp_bf16 --workers 8 --batch-size ${PreGpuBS} \
        --lr ${LR} --beta1 0.9 --beta2 0.95 --eps 1e-6 --wd 0.2 \
        --grad-checkpointing --log-every-n-steps 1 --val-frequency 1 --epochs 1 --warmup 2 \
        --dataset-type csv --train-num-samples ${SMOKE_N} \
        --csv-img-key filepath --csv-caption-key caption --val-num-captions-per-image 5"
    PASS=0; FAIL=0
    smoke() {
        local TAG=$1 PORT=$2 REG=$3
        local NAME="smoke_visreg_${TAG}"
        echo "======== [smoke] ${TAG} ========"
        if torchrun --nproc_per_node=${GPUS} --master_port=${PORT} \
            -m open_clip_train.main \
            --model "PE-Core-B-16-dinov3" \
            --train-data "${CC3M_TSV}" --val-data "${VAL}" \
            ${SMOKE_BASE} ${CHAMPION} ${REG} --name "${NAME}" < /dev/null 2>&1 | tail -40; then
            echo "[smoke] ${TAG} ... PASS"; PASS=$((PASS+1))
        else
            echo "[smoke] ${TAG} ... FAIL"; FAIL=$((FAIL+1))
        fi
    }
    smoke "sigreg" 29580 "--reg-method sigreg --sigreg-weight ${SIGREG_W}"
    smoke "visreg" 29581 "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 1.0"
    echo "======== PASSED=${PASS} FAILED=${FAIL} ========"
    [ "${FAIL}" -eq 0 ]
    ;;

  # ── A-E 组件消融（原 visreg_cc3m）────────────────────────────────────────
  ab)
    run "A_sigreg"        29570 "${CC3M_TSV}" "--reg-method sigreg --sigreg-weight ${SIGREG_W}"
    run "B_visreg_full"   29571 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 1.0"
    run "C_visreg_scale"  29572 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 0.0 --visreg-lambda-center 0.0"
    run "D_visreg_shape"  29573 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --visreg-lambda-scale 0.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "E_visreg_nocenter" 29574 "${CC3M_TSV}" "${VISREG_E}"
    ;;

  # ── scale:shape 配比面（原 visreg_sweep）─────────────────────────────────
  sweep)
    run "E_s1sh1"  29580 "${CC3M_TSV}" "${VISREG_E}"
    run "s2sh1"    29581 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --visreg-lambda-scale 2.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "s1sh2"    29582 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 2.0 --visreg-lambda-center 0.0"
    ;;

  # ── 强度 0.5×/2×（原 visreg_magnitude）───────────────────────────────────
  magnitude)
    run "w0p5x" 29585 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight $(python3 -c "print(${VISREG_W}*0.5)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "w2x"   29586 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight $(python3 -c "print(${VISREG_W}*2)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    ;;

  # ── 跨数量级强度（原 visreg_wsweep）──────────────────────────────────────
  wsweep)
    run "cls_1e2x"  29600 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight $(python3 -c "print(${VISREG_W}*100)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "cls_1e4x"  29601 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight $(python3 -c "print(${VISREG_W}*10000)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "proj_1e4x" 29603 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls_proj --sigreg-weight $(python3 -c "print(${VISREG_W}*10000)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "proj_1e6x" 29604 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls_proj --sigreg-weight $(python3 -c "print(${VISREG_W}*1000000)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    ;;

  # ── 切片数 K（原 visreg_slices）──────────────────────────────────────────
  slices)
    run "k32"  29590 "${CC3M_TSV}" "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --sigreg-slices 32 \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    ;;

  # ── 机制改进（原 visreg_mech）────────────────────────────────────────────
  mech)
    MIX="--visreg-mixture 5 --visreg-mixture-sep 2.0"
    run "mix5_1e4x" 29620 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight $(python3 -c "print(${VISREG_W}*10000)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0 ${MIX}"
    run "mix5_1e2x" 29621 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight $(python3 -c "print(${VISREG_W}*100)") \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0 ${MIX}"
    run "topk_1e2x" 29622 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight $(python3 -c "print(${VISREG_W}*100)") \
        --visreg-topk-pool 1024 \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    ;;

  # ── 第二阶段（原 visreg_stage2）──────────────────────────────────────────
  stage2)
    W_BEST="${W_BEST:-1.83e-2}"
    run "w1e3x"      29610 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight 1.83e-1 \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "topk"       29611 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight ${W_BEST} \
        --visreg-topk-pool 1024 \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "mix5"       29612 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight ${W_BEST} \
        --visreg-mixture 5 --visreg-mixture-sep 2.0 \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "mix5_1e4x"  29613 "${CC3M_TSV}" "--reg-method visreg --sigreg-target cls --sigreg-weight 1.83e0 \
        --visreg-mixture 5 --visreg-mixture-sep 2.0 \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    ;;

  # ── 跨模态（原 visreg_xmodal）────────────────────────────────────────────
  xmodal)
    BASE_REG="--reg-method visreg --sigreg-slices 256 --sigreg-weight 1.83e-4 \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    run "img_only"  29630 "${CC3M_TSV}" "${BASE_REG} --sigreg-target cls --reg-sides img"
    run "clip_base" 29631 "${CC3M_TSV}" "${BASE_REG} --sigreg-target clip"
    run "xm_pair"   29632 "${CC3M_TSV}" "${BASE_REG} --sigreg-target clip --xmatch-weight 2.3 --xmatch-mode pair"
    run "xm_dist"   29633 "${CC3M_TSV}" "${BASE_REG} --sigreg-target clip --xmatch-weight 7.8 --xmatch-mode dist"
    ;;

  # ── ★ gemma-dense 长文本（E 配方 + context_length=256）──────────────────
  gemma)
    if [ ! -f "${GEMMA_TSV}" ]; then
        echo "!!!! 缺 ${GEMMA_TSV}"
        echo "    先构建: python scripts/data/build_gemma_tsv.py"
        exit 1
    fi
    # 精确对齐 E 配方，仅数据换成 gemma-${DATA_VERSION}（256 上下文）
    run_gemma "E" 29640 "${VISREG_E}"
    # 对照：gt（原 CC3M caption）在 256 上下文下基线
    DATA_VERSION=gt GEMMA_TSV="${GEMMA_TSV_DIR}/clip_train_gt.tsv" run_gemma "gt_base" 29641 "${VISREG_E}"
    ;;

  # ── 噪声地板：同配置换 seed 重复跑，量 run-to-run 方差 ────────────────────
  #   动机：同配置同 seed 的两个 gt_base run（0806 / 0811）末期 i2t 相差 0.70 点，
  #   而超参扫描里不少差异只有 0.5 点级别 —— 没有方差界就无法定性。
  #   用法: bash scripts/train/visreg.sh seed-var        # 跑 seed 1 与 2
  #        SEED=3 bash scripts/train/visreg.sh gt-seed   # 单跑指定 seed
  seed-var)
    for s in 1 2; do
        SEED=$s DATA_VERSION="gt_s${s}" \
        GEMMA_TSV="${GEMMA_TSV_DIR}/clip_train_gt.tsv" \
          run_gemma "gt_base" $((29670 + s)) "${VISREG_E}"
    done
    ;;

  gt-seed)
    SEED="${SEED:-0}" DATA_VERSION="gt_s${SEED:-0}" \
    GEMMA_TSV="${GEMMA_TSV_DIR}/clip_train_gt.tsv" \
      run_gemma "gt_base" 29675 "${VISREG_E}"
    ;;

  # ── gemma-dense 冒烟（1 epoch 快速验证 256 上下文管线）──────────────────
  gemma-smoke)
    if [ ! -f "${GEMMA_TSV}" ]; then
        echo "!!!! 缺 ${GEMMA_TSV} —— 先构建"
        exit 1
    fi
    # 冒烟：1 epoch、小 batch、短 warmup、小样本（4 steps）
    EPOCHS=1 WARMUP=2 PreGpuBS=256
    GEMMA_N_TRAIN=$((PreGpuBS * GPUS * 4))
    run_gemma "smoke" 29642 "--reg-method visreg --sigreg-weight ${VISREG_W} \
        --visreg-lambda-scale 1.0 --visreg-lambda-shape 1.0 --visreg-lambda-center 0.0"
    ;;

  # ── ★ PCM：Long-CLIP Primary Component Matching（长短双分支）─────────────
  #   长分支：图像 × dense 长文本（学长文本能力）
  #   短分支：PCA_k(图像) × gt 短文本（保短模板 zero-shot 能力）
  #   数据：clip_train_dual.tsv（filepath / caption_short / caption_dense）
  #   口径：NEG_MODE=projective（默认，E 配方）| standard（业界对齐对照）
  #   用法: bash scripts/train/visreg.sh pcm
  #        NEG_MODE=standard bash scripts/train/visreg.sh pcm
  pcm)
    # DUAL_TSV_NAME=clip_train_dual_full.tsv → 全量 dense（超窗行截断，不丢行）
    DUAL_TSV="${GEMMA_TSV_DIR}/${DUAL_TSV_NAME:-clip_train_dual.tsv}"
    if [ ! -f "${DUAL_TSV}" ]; then
        echo "!!!! 缺 ${DUAL_TSV} —— 先跑 python scripts/data/build_dual_tsv.py"
        exit 1
    fi
    # 主文本=dense 长文，短分支=gt 短文
    DATA_VERSION="pcmw${PCM_WEIGHT:-1.0}d${PCM_DIM:-32}${DV_SUFFIX:-}_${NEG_MODE}" \
    GEMMA_TSV="${DUAL_TSV}" \
    CSV_CAPTION_KEY=caption_dense \
    CSV_CAPTION2_KEY=caption_short \
    PCM_WEIGHT="${PCM_WEIGHT:-1.0}" \
    PCM_DIM="${PCM_DIM:-32}" \
      run_gemma "E" "${PORT:-29660}" "${VISREG_E}"
    ;;

  # ── PCM 冒烟（4 steps 验证双分支管线 + 梯度）─────────────────────────────
  pcm-smoke)
    DUAL_TSV="${GEMMA_TSV_DIR}/clip_train_dual.tsv"
    if [ ! -f "${DUAL_TSV}" ]; then
        echo "!!!! 缺 ${DUAL_TSV}"
        exit 1
    fi
    EPOCHS=1 WARMUP=2 PreGpuBS=256
    GEMMA_N_TRAIN=$((PreGpuBS * GPUS * 4))
    DATA_VERSION="pcmsmoke_${NEG_MODE}" \
    GEMMA_TSV="${DUAL_TSV}" \
    CSV_CAPTION_KEY=caption_dense \
    CSV_CAPTION2_KEY=caption_short \
    PCM_WEIGHT="${PCM_WEIGHT:-1.0}" \
    PCM_DIM="${PCM_DIM:-32}" \
      run_gemma "smoke" 29661 "${VISREG_E}"
    ;;

  # ── ★ 区域-短语对比（FG-CLIP 式，ICML'25）─────────────────────────────────
  #   从倒数第二层 patch map RoIAlign 抠区域特征，与审核过的短语做对比。
  #   数据 clip_train_region.tsv（filepath/caption/regions），坐标已归一化。
  #   ★ 强制 resize-only 变换 ★ —— 区域坐标不能配 RandomResizedCrop。
  #   用法: bash scripts/train/visreg.sh region              # C 组：gt 短文 + 区域
  #        bash scripts/train/visreg.sh region-smoke        # 冒烟
  #        bash scripts/train/visreg.sh gt-resize           # A' 组：gt 短文 + resize-only 对照
  #        REGION_WEIGHT=0.2 MAX_REGION=6 bash ... region   # 超参 override
  region)
    # REGION_TSV_NAME 可换表：clip_train_region.tsv（建表期硬截 K=12，79.7% 的图卡在上限）
    # 或 clip_train_region_k24.tsv（K=24，18.71 区域/图，77.4% 的图超过 12）。
    # ⚠️ 换表必须同时传 MAX_REGION，否则训练侧仍按默认 12 截（按面积降序 → 等价于旧表）。
    REGION_TSV="${GEMMA_TSV_DIR}/${REGION_TSV_NAME:-clip_train_region.tsv}"
    if [ ! -f "${REGION_TSV}" ]; then
        echo "!!!! 缺 ${REGION_TSV} —— 先跑 python scripts/data/build_region_tsv.py"
        exit 1
    fi
    DATA_VERSION="regw${REGION_WEIGHT:-0.2}k${MAX_REGION:-12}${REGION_GATHER:+-}${REGION_GATHER:-}${REGION_SHARED_SCALE:+-sharedsc}${REGION_ROI_GRID:+-roi${REGION_ROI_GRID}${REGION_ROI_AGG:-mean}}${DV_SUFFIX:-}_${NEG_MODE}" \
    GEMMA_TSV="${REGION_TSV}" \
    CSV_CAPTION_KEY=caption \
    REGION_WEIGHT="${REGION_WEIGHT:-0.2}" \
    MAX_REGION="${MAX_REGION:-12}" \
      run_gemma "E" ${PORT:-29680} "${VISREG_E}"
    ;;

  # ── ★ H 组：C3 + 随机裁剪（框随裁剪同步变换）───────────────────────────────
  #   与 C3 的唯一变量：图像变换。C3 用 resize-only（框坐标的必要妥协），
  #   本组用 RandomResizedCropWithBoxes —— 框跟着裁剪走，裁出画面的丢弃。
  #   动机：关裁剪实测代价 COCO i2t −1.70 / IN-1k −0.70（gt_s1 vs A'，同 commit，均超 2σ）。
  #   取舍：单测 scale=(0.9,1.0) 下丢框率约 20~40%（贴边/大框易丢），
  #        「正则化收益」vs「区域样本减少」谁大，只能实测。
  #   用法: bash scripts/train/visreg.sh region-cropaug
  region-cropaug)
    REGION_TSV="${GEMMA_TSV_DIR}/clip_train_region.tsv"
    [ -f "${REGION_TSV}" ] || { echo "!!!! 缺 ${REGION_TSV}"; exit 1; }
    DATA_VERSION="regw${REGION_WEIGHT:-0.2}k${MAX_REGION:-12}-cropaug_${NEG_MODE}" \
    GEMMA_TSV="${REGION_TSV}" \
    CSV_CAPTION_KEY=caption \
    REGION_WEIGHT="${REGION_WEIGHT:-0.2}" \
    MAX_REGION="${MAX_REGION:-12}" \
    REGION_SHARED_SCALE=1 \
    REGION_CROP_AUG=1 \
      run_gemma "E" 29700 "${VISREG_E}"
    ;;

  region-cropaug-smoke)
    REGION_TSV="${GEMMA_TSV_DIR}/clip_train_region.tsv"
    [ -f "${REGION_TSV}" ] || { echo "!!!! 缺 ${REGION_TSV}"; exit 1; }
    EPOCHS=1 WARMUP=2 PreGpuBS=128
    GEMMA_N_TRAIN=$((PreGpuBS * GPUS * 4))
    DATA_VERSION="regcropaugsmoke_${NEG_MODE}" \
    GEMMA_TSV="${REGION_TSV}" \
    CSV_CAPTION_KEY=caption \
    REGION_WEIGHT="${REGION_WEIGHT:-0.2}" MAX_REGION="${MAX_REGION:-12}" \
    REGION_SHARED_SCALE=1 REGION_CROP_AUG=1 \
      run_gemma "smoke" 29701 "${VISREG_E}"
    ;;

  region-smoke)
    REGION_TSV="${GEMMA_TSV_DIR}/clip_train_region.tsv"
    [ -f "${REGION_TSV}" ] || { echo "!!!! 缺 ${REGION_TSV}"; exit 1; }
    EPOCHS=1 WARMUP=2 PreGpuBS=128
    GEMMA_N_TRAIN=$((PreGpuBS * GPUS * 4))
    DATA_VERSION="regsmoke_${NEG_MODE}" \
    GEMMA_TSV="${REGION_TSV}" \
    CSV_CAPTION_KEY=caption \
    REGION_WEIGHT="${REGION_WEIGHT:-0.2}" \
    MAX_REGION="${MAX_REGION:-12}" \
      run_gemma "smoke" 29681 "${VISREG_E}"
    ;;

  # ── ★ G 组：PCM + 区域监督叠加（本周两条最优线合并）────────────────────────
  #   主分支   SigLIP(img, caption_dense)          长文本
  #   PCM 短分支 SigLIP(PCA_k(img), caption_short)  保短模板（w=0.2 已调优）
  #   区域分支  SigLIP(RoIAlign(patch), 短语)       图像塔/分类（C3 配置）
  #   数据 clip_train_pcmregion.tsv（198.4 万，dense 覆盖所限）
  #   ⚠️ 样本量 198.4 万 ≠ C3 的 286.9 万，与 pcm_w0.2 可比、与 C3 不完全可比
  #   ⚠️ 首次运行（0823_2048）在 ep1 因 CUDA OOM 崩溃：三条监督分支叠加后
  #      allocated 58.3 GB、reserved-but-unallocated 15.9 GB —— 是**碎片**而非容量墙。
  #      故加 expandable_segments:True（仅换分配器策略，数值完全等价，
  #      不动 batch/lr，保住与所有历史组的可比性）。
  #   ⚠️ 第二次运行（0824_1618）1 秒即退：torchrun rendezvous 撞
  #      `EADDRINUSE port 29690`（同机另一进程占用），完全没进训练。
  #      故端口改为可覆盖 `PORT=`，队列侧先探测空闲端口再传入。
  #   用法: bash scripts/train/visreg.sh pcm-region   （可选 PORT=29691 前缀覆盖）
  #   ⚠️⚠️ 08-31：`clip_train_pcmregion.tsv` 只有 198.4 万行 = `clip_train_region.tsv`
  #      286.9 万的 **69.2%**，与 Task #21 查出的 `clip_train_dual.tsv` 丢行比例 69.34%
  #      几乎相同 —— 同一个 bug（建表时把 BPE>256 的行**整行丢掉**）。
  #      全量版 `clip_train_pcmregion_full.tsv`（286.9 万，与 region 表等量，超窗行运行时截断）
  #      已于 08-28 18:13 与 dual_full 同批建好。
  #   ✅✅ 08-31 §5.29/§5.30（C9F 判 X1、C9Fs1 判 Z1）后**默认表已改为 _full**：
  #      修丢行在三损失族上 13/13 口径同向为正（IN-1k +2.72、k-NN bb +2.02、
  #      DOCCI 71.70/74.98 刷新全项目最高），且 seed 双峰从 29.23 缩到 2.21。
  #      要复现 G / C9 / C9s1 / C11 的口径，显式传 `PR_TSV_NAME=clip_train_pcmregion.tsv`。
  pcm-region)
    PR_TSV="${GEMMA_TSV_DIR}/${PR_TSV_NAME:-clip_train_pcmregion_full.tsv}"
    [ -f "${PR_TSV}" ] || { echo "!!!! 缺 ${PR_TSV} —— 先跑 scripts/data/build_pcm_region_tsv.py"; exit 1; }
    DATA_VERSION="pcmregw${REGION_WEIGHT:-0.2}p${PCM_WEIGHT:-0.2}${REGION_ROI_GRID:+-roi${REGION_ROI_GRID}${REGION_ROI_AGG:-mean}}${DV_SUFFIX:-}_${NEG_MODE}" \
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
    GEMMA_TSV="${PR_TSV}" \
    CSV_CAPTION_KEY=caption_dense \
    CSV_CAPTION2_KEY=caption_short \
    PCM_WEIGHT="${PCM_WEIGHT:-0.2}" PCM_DIM="${PCM_DIM:-32}" \
    REGION_WEIGHT="${REGION_WEIGHT:-0.2}" MAX_REGION="${MAX_REGION:-12}" \
    REGION_SHARED_SCALE=1 \
      run_gemma "E" "${PORT:-29690}" "${VISREG_E}"
    ;;

  pcm-region-smoke)
    PR_TSV="${GEMMA_TSV_DIR}/clip_train_pcmregion.tsv"
    [ -f "${PR_TSV}" ] || { echo "!!!! 缺 ${PR_TSV}"; exit 1; }
    EPOCHS=1 WARMUP=2 PreGpuBS=128
    GEMMA_N_TRAIN=$((PreGpuBS * GPUS * 4))
    DATA_VERSION="pcmregsmoke_${NEG_MODE}" \
    GEMMA_TSV="${PR_TSV}" \
    CSV_CAPTION_KEY=caption_dense \
    CSV_CAPTION2_KEY=caption_short \
    PCM_WEIGHT="${PCM_WEIGHT:-0.2}" PCM_DIM="${PCM_DIM:-32}" \
    REGION_WEIGHT="${REGION_WEIGHT:-0.2}" MAX_REGION="${MAX_REGION:-12}" \
    REGION_SHARED_SCALE=1 \
      run_gemma "smoke" 29691 "${VISREG_E}"
    ;;

  # ── ★ E 组：verdict（gemma4 审核）的价值 ────────────────────────────────────
  #   三档对照，都用 C3 配置（共享 scale / local / K=12 / w=0.2 / cc=0.1）：
  #     firstbox : 只取首框，不用 verdict —— 对照基线（隔离"框选择策略"）
  #     soft     : YES 1.0 / NO 0.3 / null 1.0（MANIFEST 推荐）
  #     hard     : 只丢明确 NO，null 保留（丢 null 会砍六成数据）
  #   用法: VERDICT_MODE=soft bash scripts/train/visreg.sh region-verdict
  region-verdict)
    VM="${VERDICT_MODE:-soft}"
    V_TSV="${GEMMA_TSV_DIR}/clip_train_region_${VM}.tsv"
    [ -f "${V_TSV}" ] || { echo "!!!! 缺 ${V_TSV} —— 先跑 scripts/data/attach_verdict_sidecar.py --mode ${VM}"; exit 1; }
    DATA_VERSION="regvd-${VM}_${NEG_MODE}" \
    GEMMA_TSV="${V_TSV}" \
    CSV_CAPTION_KEY=caption \
    REGION_WEIGHT="${REGION_WEIGHT:-0.2}" MAX_REGION="${MAX_REGION:-12}" \
    REGION_SHARED_SCALE=1 \
      run_gemma "E" 29695 "${VISREG_E}"
    ;;

  # ── A' 组：gt 短文 + resize-only（隔离"关掉随机裁剪"的影响）──────────────
  gt-resize)
    DATA_VERSION="gt_resize" GEMMA_TSV="${GEMMA_TSV_DIR}/clip_train_gt.tsv" \
    RESIZE_ONLY=1 \
      run_gemma "gt_base" 29682 "${VISREG_E}"
    ;;

  # ── standard 口径对照（同 E 配方，仅 --neg-mode standard）────────────────
  #   与 projective 基线（gt_base 23.50/23.20）配对，量化两种几何的差异
  #   用法: bash scripts/train/visreg.sh gt-std
  gt-std)
    NEG_MODE=standard
    CHAMPION="--siglip --neg-mode standard --init-logit-scale ${INIT_LS} \
        --sigreg-target cls \
        --lr ${LR} --opt muon --muon-lr ${MUON_LR} \
        --probe-data ${PROBE_TSV}"
    DATA_VERSION=gt GEMMA_TSV="${GEMMA_TSV_DIR}/clip_train_gt.tsv" \
      run_gemma "gt_std" 29662 "${VISREG_E}"
    ;;

  # ── 噪声消融（cc3m gt 上验证 NOVIC 风格噪声）──────────────────────────
  #   usage: NOISE_SCHEME=gausselemuniformangle NOISE_VEC_NORM=0.5 \
  #          NOISE_ANGLE_MIN=45 NOISE_ANGLE_MAX=75 NOISE_MIX_RATIO=0.15 \
  #          bash scripts/train/visreg.sh noise TAG PORT
  noise)
    _tag="${2:-gauss}"; _port="${3:-29650}"
    _ns="${NOISE_SCHEME:-}"
    _extra=""
    if [ -n "$_ns" ]; then
        _extra="--noise-scheme ${_ns} --noise-vec-norm ${NOISE_VEC_NORM:-3.25} \
            --noise-angle-min ${NOISE_ANGLE_MIN:-45} --noise-angle-max ${NOISE_ANGLE_MAX:-75} \
            --noise-mix-ratio ${NOISE_MIX_RATIO:-0.15} --noise-sides ${NOISE_SIDES:-both}"
    fi
    run "noise_${_tag}" ${_port} "${CC3M_TSV}" "${VISREG_E} ${_extra}"
    ;;

  # ── 噪声消融全组（cc3m gt 串行，E 配方 + 各噪声方案）──────────────────
  #   baseline（无噪声）= 本次 cc3m gt 结果（COCO 24.10/IN 23.48），不重跑
  noise-ab)
    _p=29651
    run "n_g05"     $((_p)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme gausselem --noise-vec-norm 0.5"
    run "n_g325"    $((_p+1)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme gausselem --noise-vec-norm 3.25"
    run "n_uniform" $((_p+2)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme uniformangle --noise-angle-min 45 --noise-angle-max 75"
    run "n_mix05"   $((_p+3)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme gausselemuniformangle --noise-vec-norm 0.5 --noise-angle-min 45 --noise-angle-max 75 --noise-mix-ratio 0.15"
    run "n_mix325"  $((_p+4)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme gausselemuniformangle --noise-vec-norm 3.25 --noise-angle-min 45 --noise-angle-max 75 --noise-mix-ratio 0.15"
    ;;

  # ── 小角度噪声消融（验证"方向多样性 std"假设）─────────────────────────
  #   对照: n_g05(22.66/23.00)  n_mix05(21.66/22.94)
  noise-small)
    _p=29661
    run "n_s5_20"      $((_p)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme uniformangle --noise-angle-min 5 --noise-angle-max 20"
    run "n_s10_30"     $((_p+1)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme uniformangle --noise-angle-min 10 --noise-angle-max 30"
    run "n_g05_a5_20"  $((_p+2)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme gausselemuniformangle --noise-vec-norm 0.5 --noise-angle-min 5 --noise-angle-max 20 --noise-mix-ratio 0.15"
    run "n_g05_a10_30" $((_p+3)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme gausselemuniformangle --noise-vec-norm 0.5 --noise-angle-min 10 --noise-angle-max 30 --noise-mix-ratio 0.15"
    run "n_g05_a10_30r30" $((_p+4)) "${CC3M_TSV}" "${VISREG_E} --noise-scheme gausselemuniformangle --noise-vec-norm 0.5 --noise-angle-min 10 --noise-angle-max 30 --noise-mix-ratio 0.30"
    ;;

  # ── DualTextCLIP：双文本塔（短 gt + 长 dense），双 SigLIP 对齐 ──────────
  #   数据 = clip_train_dual.tsv（filepath, caption_short, caption_dense）
  dual-text)
    DUAL_TSV="${GEMMA_TSV_DIR}/clip_train_dual.tsv"
    if [ ! -f "${DUAL_TSV}" ]; then
        echo "!!!! 缺 ${DUAL_TSV} —— 先跑 scripts/data/build_dual_tsv.py"
        exit 1
    fi
    _tag="${2:-E}"; _port="${3:-29660}"
    # DualTextCLIP：无 sigreg（双塔自带双损失），E 配方超参 + 双列数据
    run_dual "${_tag}" ${_port}
    ;;

  *)
    echo "未知模式: ${1}"
    echo "用法: bash scripts/train/visreg.sh {smoke|ab|sweep|magnitude|wsweep|slices|mech|stage2|xmodal|gemma|gemma-smoke|noise|noise-ab|dual-text}"
    exit 1
    ;;
esac

echo "======== visreg ${1} done ($(date '+%F %T')) ========"
