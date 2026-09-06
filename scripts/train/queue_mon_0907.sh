#!/bin/bash
# ============================================================================
# 三臂驱动（2026-09-07）。判据见
# analysis/research/prereg/mon_0907_3arms_prereg.md（commit 7356dd7，起训前落盘）
#   C5s2 = C5 唯一换 --seed 2          → 无判决，只把 MIL 族地板补成 n=3
#   C16  = C15 唯一加 --region-crop-fix-align（配对错误 53.95% → 0%）
#   F3   = C5 唯一换 --region-select random-epoch（k24 表 + MAX_REGION=12）
#
# 教训落地三条：
#  (1) preflight **看显存不数进程**（09-03 C11F 被自己的收尾残留挤掉，队列停 11.5h）。
#  (2) **一次排够臂数** —— /tmp/c11f.sh 只排一臂，正常收尾后 8 卡空转 49h。
#      本脚本 3 臂 × ~6.5h ≈ 19.5h。
#  (3) **目录 glob 会撞同后缀旧臂** —— C5s2 的目录名与 C5 s0/s1 完全同形，
#      只靠 `ls -dt | head -1` 不够，gate 必须逐项核对 params.txt 身份。
# ----------------------------------------------------------------------------
set -u
LOG=/tmp/mon.log
OUT=/tmp/mon_eval.txt
ROOT=/root/paddlejob/workspace/env_run/penghaotian
CTRL_CK=logs/visreg_gemma_regw2.0k12_projective_E_0826_1738/checkpoints/epoch_10.pt
say() { echo "[mon $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
unset CUDA_VISIBLE_DEVICES

preflight() {   # $1=TAG $2=PORT ；显存判据，阈值 5000 MiB
    local T=$1 P=$2 MAXMEM
    MAXMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
    say "$T preflight: 单卡最大显存 ${MAXMEM} MiB（>5000 视为有人在训；我方 guard 只 626）"
    if [ "${MAXMEM:-0}" -gt 5000 ]; then
        say "!!! $T: 显存判据认为有训练在跑 → 按「先起者优先」退让，不起"; return 1
    fi
    if ss -ltn 2>/dev/null | grep -q ":$P "; then
        say "!!! $T: 端口 $P 被占（EADDRINUSE 会让训练 1 秒静默死掉）→ 跳过"; return 2
    fi
    return 0
}

gate() {        # $1=TAG $2=DIR $3=期望ckpt数 $4=身份串（`key: value` 用 ; 分隔，逐条精确匹配）
    local T=$1 D=$2 W=${3:-10} IDENT=${4:-} N kv IFS
    [ -n "$D" ] && [ -d "$D" ] || { say "  !!! $T 没找到输出目录（起训前就死了 → 查 /tmp/${T}_train.log 第一屏）"; return 1; }
    IFS=';'
    for kv in $IDENT; do
        [ -n "$kv" ] || continue
        grep -qxF "$kv" "$D/params.txt" 2>/dev/null || {
            say "  !!! $T 身份不符：params.txt 里没有「$kv」→ 大概率拿到了同后缀的旧臂目录，判失败"
            unset IFS; return 1; }
    done
    unset IFS
    N=$(ls "$D"/checkpoints/ 2>/dev/null | grep -c 'epoch_[0-9]*\.pt')
    say "  $T 目录 $D  ckpt=$N/$W  身份逐项已核对"
    grep -E '^(region_roi_grid|region_roi_agg|region_weight|max_region|region_select|region_text_chunk|region_crop_aug|region_crop_fix_align|region_keep_area_thr|image_resize_only|region_shared_scale|region_gather|region_cc_weight|seed|epochs|train_data):' \
        "$D/params.txt" 2>/dev/null | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"
    grep -m1 -E '删框判据|image_resize_only|region_select' "$D/out.log" 2>/dev/null | tee -a "$LOG"
    # 总样本闸：分母必须 == 2868984（--train-num-samples 对 csv/tsv 是静默 no-op）
    grep -oE 'Train Epoch: [0-9]+ \[[0-9 ]+/[0-9]+' "$D/out.log" 2>/dev/null | tail -1 | tee -a "$LOG"
    [ "$N" -eq "$W" ] || { say "  !!! $T ckpt 不足 $W → 不写结论，查 /tmp/${T}_train.log（OOM 堆栈只在那里）"; return 1; }
    return 0
}

evalarm() {     # $1=TAG $2=DIR
    local T=$1 D=$2
    say "  === $T 评测开始（全局五项 + 长文本 + OVSS 4 读出 ×2 epoch + 漂移校验）==="
    (
      source "$ROOT/envs/dino/bin/activate"
      export PYTHONPATH="./src:${PYTHONPATH:-}"
      export CUDA_VISIBLE_DEVICES=0
      export ADE_ROOT="$ROOT/datas/ade20k/ADEChallengeData2016"
      CK="$D/checkpoints/epoch_10.pt"; CK5="$D/checkpoints/epoch_5.pt"
      python scripts/eval/eval_standard.py  --ckpt "$CK" --tag "$T" --retrieval >> "$OUT" 2>&1
      python scripts/eval/eval_knn_probe.py --ckpt "$CK" --tag "$T"             >> "$OUT" 2>&1
      python scripts/eval/eval_urban1k.py   --ckpt "$CK" --tag "$T"             >> "$OUT" 2>&1
      python scripts/eval/eval_docci.py     --ckpt "$CK" --tag "$T"             >> "$OUT" 2>&1
      echo "[mon $(date '+%H:%M:%S')] $T 全局五项 + 长文本完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[mon $(date '+%H:%M:%S')] $T OVSS ep10 四读出完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK5" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}ep5_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[mon $(date '+%H:%M:%S')] $T OVSS ep5 四读出完" | tee -a "$LOG"
      python scripts/eval/eval_ovss.py --ckpt "$CTRL_CK" --dataset voc --dense-mode penult \
          --tag "DRIFT_after_${T}" >> "$OUT" 2>&1
    )
    say "  === $T 评测结束（subshell 退出，CVD 未泄漏）==="
    awk -v t="$T" '/^\[/ {keep = ($0 ~ ("\\[" t "]") || $0 ~ ("\\[" t "_") || $0 ~ ("\\[" t "ep5") || $0 ~ ("_" t "]")) } keep' "$OUT" \
        | grep -E "^\[|R@1=|top1=|k-NN (backbone|proj)|★.*mIoU=" | sed 's/^/[mon] /' | tee -a "$LOG"
}

TSVD="$ROOT/datas/cc3m-tsv/annotations"

# ── 臂 1：C5s2（C5 唯一换 seed 2；不传 chunk / select，resize-only）────────────
arm_c5s2() {
    local T=C5s2 P=29760 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（C5 逐位同配方，seed=2，端口 $P）—— 只为把 MIL 族地板补成 n=3 ==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 MAX_REGION=12 \
    SEED=2 EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    # ⚠️ 目录名与 C5 s0/s1 完全同形 → 靠 ls -dt 取最新 + gate 核对 seed: 2
    D=$(ls -dt logs/visreg_gemma_regw2.0k12-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "seed: 2;region_select: order;region_crop_aug: False;image_resize_only: True;max_region: 12;region_weight: 2.0;region_roi_grid: 2;region_roi_agg: mil;epochs: 10;train_data: $TSVD/clip_train_region.tsv"; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/mon_0907_3arms_prereg.md §3.1"
}

# ── 臂 2：C16（= C15 唯一加 --region-crop-fix-align；不传 chunk，因为 C15 没传）──
arm_c16() {
    local T=C16 P=29762 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（C15 + 修短语配对，thr=0.0，seed=0，端口 $P）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 MAX_REGION=12 \
    REGION_CROP_AUG=1 REGION_KEEP_AREA_THR=0.0 REGION_CROP_FIX_ALIGN=1 \
    SEED=0 DV_SUFFIX=-CROP00FIX EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D=$(ls -dt logs/visreg_gemma_regw2.0k12-roi2mil-CROP00FIX_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "seed: 0;region_crop_aug: True;region_crop_fix_align: True;region_keep_area_thr: 0.0;region_select: order;max_region: 12;region_weight: 2.0;region_roi_grid: 2;region_roi_agg: mil;epochs: 10;train_data: $TSVD/clip_train_region.tsv"; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/mon_0907_3arms_prereg.md §3.2"
}

# ── 臂 3：F3（k24 表 + MAX_REGION=12 + random-epoch；对照 = C5，mismatch 已实测 0）──
arm_f3() {
    local T=F3 P=29764 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（k24 表 + MAX_REGION=12 + random-epoch，seed=0，端口 $P）==="
    REGION_TSV_NAME=clip_train_region_k24.tsv \
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 MAX_REGION=12 \
    REGION_SELECT=random-epoch REGION_TEXT_CHUNK=2048 \
    SEED=0 DV_SUFFIX=-SELRE EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D=$(ls -dt logs/visreg_gemma_regw2.0k12-roi2mil-SELRE_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "seed: 0;region_select: random-epoch;region_text_chunk: 2048;region_crop_aug: False;image_resize_only: True;max_region: 12;region_weight: 2.0;region_roi_grid: 2;region_roi_agg: mil;epochs: 10;train_data: $TSVD/clip_train_region_k24.tsv"; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/mon_0907_3arms_prereg.md §3.3"
}

arm_c5s2
arm_c16
arm_f3

say "=== mon 队列走完（3 臂）。数字在 $OUT ==="
