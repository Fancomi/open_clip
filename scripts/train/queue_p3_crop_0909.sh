#!/bin/bash
# ============================================================================
# 三臂驱动（2026-09-09）。判据见
# analysis/research/prereg/thu_0909_3arms_prereg.md（起训前落盘）
#   P3   = P0 唯一换主分支 caption 列（caption_dense → caption_short，p=0 不变）
#          → 拆开 §5.40.1 那 −7.73 k-NN 的两个候选
#   C15′ = 重跑 09-07 起训前就死掉的 C15（crop-aug, thr=0.0, fix-align 关）
#          → 一臂解锁两个判决：自己的 H 组 + §5.38 那格 C16−C15
#   C14′ = 同上但 thr=0.6
#
# ⚠️ 日志用 /tmp/p3batch.log（不叫 thu.log —— 09-03 那批已经占了那个名字，
#    复用会把两批数字混进同一个文件，判决时分不清）。
#
# 教训落地：
#  (1) preflight **看显存不数进程**（第 6 条）。
#  (2) **一次排够臂数**（第 7 条）：3 臂 ≈ 19.3h ≥ 18h。
#  (3) gate 逐项核对 params.txt 身份；P3 额外查两条 PCM 已关的闸。
#  (4) ⚠️ 本批运行期间**不编辑 src/ 与 visreg.sh**（第 8 条）；非要改就走默认关的
#      开关 **且开关的开/关两条分支各跑一次 DRY_RUN** —— 09-07 C14/C15 两臂
#      就是死在「默认关那条分支没被测过」（§5.36.2.1）。
#  (5) 目录 glob：P3 与 P0 **同形**（都是 pcmregw0.2p0-roi2mil）→ ls -dt 取最新
#      + gate 核对 `csv_caption_key: caption_short` 区分；C14′/C15′ 用
#      CROP06/CROP00 后缀，与历史无碰撞（09-07 那两臂根本没建目录）。
# ----------------------------------------------------------------------------
set -u
LOG=/tmp/p3batch.log
OUT=/tmp/p3batch_eval.txt
ROOT=/root/paddlejob/workspace/env_run/penghaotian
CTRL_CK=logs/visreg_gemma_regw2.0k12_projective_E_0826_1738/checkpoints/epoch_10.pt
say() { echo "[p3b $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
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

gate() {        # $1=TAG $2=DIR $3=期望ckpt数 $4=身份串(`k: v` 用 ; 分隔) $5=PCM期望(off/none)
    local T=$1 D=$2 W=${3:-10} IDENT=${4:-} PCMEXP=${5:-none} N kv IFS NPCM NLOSS
    [ -n "$D" ] && [ -d "$D" ] || { say "  !!! $T 没找到输出目录（起训前就死了 → 查 /tmp/${T}_train.log 第一屏）"; return 1; }
    IFS=';'
    for kv in $IDENT; do
        [ -n "$kv" ] || continue
        grep -qxF "$kv" "$D/params.txt" 2>/dev/null || {
            say "  !!! $T 身份不符：params.txt 里没有「$kv」→ 判失败，不写结论"
            unset IFS; return 1; }
    done
    unset IFS
    N=$(ls "$D"/checkpoints/ 2>/dev/null | grep -c 'epoch_[0-9]*\.pt')
    say "  $T 目录 $D  ckpt=$N/$W  身份逐项已核对"
    grep -E '^(csv_caption_key|csv_caption2_key|pcm_weight|pcm_dim|region_roi_grid|region_roi_agg|region_weight|max_region|region_select|region_text_chunk|region_shared_scale|region_gather|region_cc_weight|region_crop_aug|region_crop_fix_align|region_keep_area_thr|image_resize_only|seed|epochs|train_data):' \
        "$D/params.txt" 2>/dev/null | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"
    # 闸 4：PCM 必须是关的（P3 传 off；C14′/C15′ 走 region 入口本就无 PCM 列）
    NPCM=$(grep -c '=> PCM enabled' "$D/out.log" 2>/dev/null || true)
    if [ "${NPCM:-0}" -ne 0 ]; then
        say "  !!! $T 闸4 失败：本批三臂都不该有 PCM，out.log 里却有 '=> PCM enabled'"; return 1; fi
    # 闸 5：损失项数（都不该出现 Pcm_loss）
    NLOSS=$(grep -m1 -oE 'Pcm_loss' "$D/out.log" 2>/dev/null | head -1)
    say "  $T 闸5 Pcm_loss: '${NLOSS:-<无>}'（本批三臂均期望 <无>）"
    if [ -n "${NLOSS:-}" ]; then
        say "  !!! $T 闸5 失败：不该有 PCM 却打印了 Pcm_loss"; return 1; fi
    # 闸 5b：启动行（crop-aug 臂必须有删框判据行；fix-align 关 ⇒ 不该出现「已修错配」）
    grep -m1 -E '删框判据|image_resize_only' "$D/out.log" 2>/dev/null | tee -a "$LOG"
    # 闸 3：总样本闸
    grep -oE 'Train Epoch: [0-9]+ \[[0-9 ]+/[0-9]+' "$D/out.log" 2>/dev/null | tail -1 | tee -a "$LOG"
    # 闸 7：有效吞吐 = 累计样本 ÷ 墙钟（不读日志瞬时值）
    say "  $T 墙钟(s)=$(( $(date -r "$D/checkpoints/epoch_10.pt" +%s 2>/dev/null || echo 0) - $(date -r "$D/params.txt" +%s 2>/dev/null || echo 0) ))  ⇒ 有效吞吐 = 28689840 ÷ 该值"
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
      echo "[p3b $(date '+%H:%M:%S')] $T 全局五项 + 长文本完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[p3b $(date '+%H:%M:%S')] $T OVSS ep10 四读出完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK5" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}ep5_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[p3b $(date '+%H:%M:%S')] $T OVSS ep5 四读出完" | tee -a "$LOG"
      python scripts/eval/eval_ovss.py --ckpt "$CTRL_CK" --dataset voc --dense-mode penult \
          --tag "DRIFT_after_${T}" >> "$OUT" 2>&1
    )
    say "  === $T 评测结束（subshell 退出，CVD 未泄漏）==="
    awk -v t="$T" '/^\[/ {keep = ($0 ~ ("\\[" t "]") || $0 ~ ("\\[" t "_") || $0 ~ ("\\[" t "ep5") || $0 ~ ("_" t "]")) } keep' "$OUT" \
        | grep -E "^\[|R@1=|top1=|k-NN (backbone|proj)|★.*mIoU=" | sed 's/^/[p3b] /' | tee -a "$LOG"
}

TSVD="$ROOT/datas/cc3m-tsv/annotations"
PRTSV="$TSVD/clip_train_pcmregion_full.tsv"
REGTSV="$TSVD/clip_train_region.tsv"
CROPID="seed: 0;region_crop_aug: True;region_crop_fix_align: False;region_select: order;max_region: 12;region_weight: 2.0;region_roi_grid: 2;region_roi_agg: mil;region_shared_scale: False;image_resize_only: False;epochs: 10;train_data: $REGTSV"

# ── 臂 1：P3（P0 唯一换主分支 caption 列）────────────────────────────────────
arm_p3() {
    local T=P3 P=29780 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（P0 逐位同配方，主分支 caption_short，p=0，端口 $P）—— 拆 −7.73 ==="
    PR_CAP_KEY=caption_short \
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=0.2 PCM_WEIGHT=0 PCM_DIM=32 \
    MAX_REGION=12 SEED=0 EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh pcm-region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    # ⚠️ 目录名与 P0 完全同形 → ls -dt 取最新 + gate 核对 csv_caption_key
    D=$(ls -dt logs/visreg_gemma_pcmregw0.2p0-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "csv_caption_key: caption_short;pcm_weight: 0.0;region_weight: 0.2;pcm_dim: 32;region_roi_grid: 2;region_roi_agg: mil;max_region: 12;region_cc_weight: 0.1;region_shared_scale: True;region_gather: local;region_select: order;region_text_chunk: 0;image_resize_only: True;seed: 0;epochs: 10;train_data: $PRTSV" off; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/thu_0909_3arms_prereg.md §3.1"
}

# ── 臂 2：C15′（crop-aug, thr=0.0）—— 解锁 H 组 + C16−C15 ─────────────────────
arm_c15p() {
    local T=C15p P=29782 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（C13 唯一换 thr → 0.0，fix-align 关，seed=0，端口 $P）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 MAX_REGION=12 \
    REGION_CROP_AUG=1 REGION_KEEP_AREA_THR=0.0 \
    SEED=0 DV_SUFFIX=-CROP00 EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D=$(ls -dt logs/visreg_gemma_regw2.0k12-roi2mil-CROP00_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "region_keep_area_thr: 0.0;$CROPID" none; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/thu_0909_3arms_prereg.md §3.2 + §3.3"
}

# ── 臂 3：C14′（crop-aug, thr=0.6）───────────────────────────────────────────
arm_c14p() {
    local T=C14p P=29784 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（C13 唯一换 thr → 0.6，fix-align 关，seed=0，端口 $P）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 MAX_REGION=12 \
    REGION_CROP_AUG=1 REGION_KEEP_AREA_THR=0.6 \
    SEED=0 DV_SUFFIX=-CROP06 EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D=$(ls -dt logs/visreg_gemma_regw2.0k12-roi2mil-CROP06_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "region_keep_area_thr: 0.6;$CROPID" none; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/thu_0909_3arms_prereg.md §3.4"
}

arm_p3
arm_c15p
arm_c14p

say "=== p3batch 队列走完（3 臂）。数字在 $OUT ==="
