#!/bin/bash
# ============================================================================
# 三臂驱动（2026-09-07/08）：混合族 `pcm_weight` 轴。判据见
# analysis/research/prereg/pcm_axis_0907_3arms_prereg.md（起训前落盘）
#   P0 = C9F 唯一换 --pcm-weight 0    → PCM 分支关掉（那个缺失的对照）
#   P1 = C9F 唯一换 --pcm-weight 0.1  → p 减半
#   P2 = C11F 唯一换 --pcm-weight 0.1 → 减半 p 能不能把 W=2.0 的崩点推回来
#
# 教训落地：
#  (1) preflight **看显存不数进程**（第 6 条：09-03 C11F 被自己收尾残留挤掉，停 11.5h）。
#  (2) **一次排够臂数**（第 7 条：单臂脚本正常收尾后 8 卡空转 49h）。本批 3 臂 ≈ 23.5h。
#  (3) gate 逐项核对 params.txt 身份；**P0 额外查两条 PCM 已关的闸**
#      —— out.log 不得有 `=> PCM enabled`、Loss 行只有 4 项。
#  (4) 三臂 DATA_VERSION 天然不同形（p0 / p0.1 / p0.2 各自后缀）⇒ 无 glob 撞旧臂风险；
#      `p0` 后紧跟 `-roi2mil`，不会前缀撞 `p0.1`。
# ----------------------------------------------------------------------------
set -u
LOG=/tmp/pcm.log
OUT=/tmp/pcm_eval.txt
ROOT=/root/paddlejob/workspace/env_run/penghaotian
CTRL_CK=logs/visreg_gemma_regw2.0k12_projective_E_0826_1738/checkpoints/epoch_10.pt
say() { echo "[pcm $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
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

gate() {        # $1=TAG $2=DIR $3=期望ckpt数 $4=身份串（`key: value` 用 ; 分隔）$5=PCM期望(on/off)
    local T=$1 D=$2 W=${3:-10} IDENT=${4:-} PCMEXP=${5:-on} N kv IFS NPCM NLOSS
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
    grep -E '^(pcm_weight|pcm_dim|csv_caption2_key|region_roi_grid|region_roi_agg|region_weight|max_region|region_select|region_text_chunk|region_shared_scale|region_gather|region_cc_weight|image_resize_only|seed|epochs|train_data):' \
        "$D/params.txt" 2>/dev/null | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"
    # 闸 4：PCM 开关（P0 必须没有这行；P1/P2 必须有）
    NPCM=$(grep -c '=> PCM enabled' "$D/out.log" 2>/dev/null || true)
    grep -m1 '=> PCM enabled' "$D/out.log" 2>/dev/null | tee -a "$LOG"
    if [ "$PCMEXP" = off ] && [ "${NPCM:-0}" -ne 0 ]; then
        say "  !!! $T 闸4 失败：期望 PCM 关，out.log 里却有 '=> PCM enabled'"; return 1
    fi
    if [ "$PCMEXP" = on ] && [ "${NPCM:-0}" -eq 0 ]; then
        say "  !!! $T 闸4 失败：期望 PCM 开，out.log 里没有 '=> PCM enabled'"; return 1
    fi
    # 闸 5：损失项数（P0 应 4 项无 Pcm；P1/P2 应 5 项）
    NLOSS=$(grep -m1 -oE 'Pcm_loss' "$D/out.log" 2>/dev/null | head -1)
    say "  $T 闸5 Pcm_loss 出现与否: '${NLOSS:-<无>}'（P0 期望 <无>，P1/P2 期望 Pcm_loss）"
    if [ "$PCMEXP" = off ] && [ -n "${NLOSS:-}" ]; then
        say "  !!! $T 闸5 失败：p=0 却打印了 Pcm_loss"; return 1
    fi
    # 闸 3：总样本闸，分母必须 == 2868984
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
      echo "[pcm $(date '+%H:%M:%S')] $T 全局五项 + 长文本完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[pcm $(date '+%H:%M:%S')] $T OVSS ep10 四读出完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK5" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}ep5_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[pcm $(date '+%H:%M:%S')] $T OVSS ep5 四读出完" | tee -a "$LOG"
      python scripts/eval/eval_ovss.py --ckpt "$CTRL_CK" --dataset voc --dense-mode penult \
          --tag "DRIFT_after_${T}" >> "$OUT" 2>&1
    )
    say "  === $T 评测结束（subshell 退出，CVD 未泄漏）==="
    awk -v t="$T" '/^\[/ {keep = ($0 ~ ("\\[" t "]") || $0 ~ ("\\[" t "_") || $0 ~ ("\\[" t "ep5") || $0 ~ ("_" t "]")) } keep' "$OUT" \
        | grep -E "^\[|R@1=|top1=|k-NN (backbone|proj)|★.*mIoU=" | sed 's/^/[pcm] /' | tee -a "$LOG"
}

TSVD="$ROOT/datas/cc3m-tsv/annotations"
PRTSV="$TSVD/clip_train_pcmregion_full.tsv"
# 三臂公共身份串（C9F 配方里不随本批变量变化的部分）
COMMON_ID="pcm_dim: 32;csv_caption2_key: caption_short;region_roi_grid: 2;region_roi_agg: mil;max_region: 12;region_cc_weight: 0.1;region_shared_scale: True;region_gather: local;region_select: order;region_text_chunk: 0;image_resize_only: True;seed: 0;epochs: 10;train_data: $PRTSV"

# ── 臂 1：P0（C9F 唯一换 pcm_weight 0 → PCM 分支关掉）───────────────────────────
arm_p0() {
    local T=P0 P=29770 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（C9F 逐位同配方，pcm_weight=0，端口 $P）—— 那个缺失的对照 ==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=0.2 PCM_WEIGHT=0 PCM_DIM=32 \
    MAX_REGION=12 SEED=0 EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh pcm-region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D=$(ls -dt logs/visreg_gemma_pcmregw0.2p0-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "pcm_weight: 0.0;region_weight: 0.2;$COMMON_ID" off; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/pcm_axis_0907_3arms_prereg.md §4.1"
}

# ── 臂 2：P1（C9F 唯一换 pcm_weight 0.1）──────────────────────────────────────
arm_p1() {
    local T=P1 P=29772 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（C9F 逐位同配方，pcm_weight=0.1，端口 $P）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=0.2 PCM_WEIGHT=0.1 PCM_DIM=32 \
    MAX_REGION=12 SEED=0 EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh pcm-region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D=$(ls -dt logs/visreg_gemma_pcmregw0.2p0.1-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "pcm_weight: 0.1;region_weight: 0.2;$COMMON_ID" on; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/pcm_axis_0907_3arms_prereg.md §4.2"
}

# ── 臂 3：P2（C11F 唯一换 pcm_weight 0.1；W=2.0）───────────────────────────────
arm_p2() {
    local T=P2 P=29774 D
    preflight "$T" "$P" || return 1
    say "=== $T 起训（C11F 逐位同配方，W=2.0，pcm_weight=0.1，端口 $P）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 PCM_WEIGHT=0.1 PCM_DIM=32 \
    MAX_REGION=12 SEED=0 EPOCHS=10 PORT="$P" \
        bash scripts/train/visreg.sh pcm-region >> "/tmp/${T}_train.log" 2>&1
    say "$T visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D=$(ls -dt logs/visreg_gemma_pcmregw2.0p0.1-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate "$T" "$D" 10 "pcm_weight: 0.1;region_weight: 2.0;$COMMON_ID" on; then
        evalarm "$T" "$D"
    fi
    say "$T 完。判据见 analysis/research/prereg/pcm_axis_0907_3arms_prereg.md §4.3"
}

arm_p0
arm_p1
arm_p2

say "=== pcm 队列走完（3 臂）。数字在 $OUT ==="
