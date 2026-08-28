#!/bin/bash
# ============================================================================
# 周末批次驱动（5 臂，2026-08-29 → 08-31）
# 用户指令：「后续任务一并排上必要实验一步干到周一吧」
# 预登记：analysis/research/prereg/weekend_0829_5arms_prereg.md（**五臂全部写在起训前**）
#
# 顺序即优先级，允许被截断；每臂独立训练 → 过闸 → 评测 → 落盘。
#   1 C5s1  grid2+mil W2.0 seed1   29720  ← 给 MIL 族实标 φ，Task #20，最高优先
#   2 C8    grid2+mil W0.5 seed0   29722  ← MIL 下重扫 W，最可能改结论
#   3 C7    grid3+mil W2.0 seed0   29724  ← 沿 MIL 轴推进，Task #22
#   4 C10   grid2+mil W4.0 seed0   29726  ← 把 MIL 下的 W 曲线补成三点
#   5 C9    G+MIL(pcm-region)      29728  ← 选型收口（局部一侧参照点双峰，判据已避开）
#
# ★ 结构约定（都是踩过的坑）
#   - 评测全部关在 subshell 里 → `export CUDA_VISIBLE_DEVICES=0` 结构上不可能泄漏到
#     下一臂的 `torchrun --nproc_per_node=8`。
#   - 训练与评测**串行**：旁挂评测会让 8 卡训练掉 15~20% 吞吐（35min 串行 < 1h 损耗）。
#   - 等的是**驱动脚本**消失，不是训练 wrapper —— l1l2 的 40 分钟评测段就在 wrapper 之后。
#   - pgrep 模式写 `open_clip_train[.]main` 防自匹配（08-26 白闲 11.5 小时）。
#   - 一律用字面绝对路径，不跨 `;` 传 shell 变量（Bash 工具会丢赋值）。
# ----------------------------------------------------------------------------
set -u
LOG=/tmp/weekend.log
OUT=/tmp/weekend_eval.txt
ROOT=/root/paddlejob/workspace/env_run/penghaotian
CTRL_CK=logs/visreg_gemma_regw2.0k12_projective_E_0826_1738/checkpoints/epoch_10.pt
say() { echo "[wk $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
unset CUDA_VISIBLE_DEVICES

# ── 0. 等前一条 lane（L1/L2：训练 + 评测 + 驱动脚本）整体退出 ────────────────
BUSY_PAT='open_clip_train[.]main|scripts/eval/eval_[a-z0-9_]*[.]py|bash /tmp/[a-z0-9_]*queue[.]sh|bash scripts/train/visreg[.]sh'
busy() { pgrep -af "$BUSY_PAT" | grep -v 'weekend_queue' | grep -v "^$$ "; }
say "等 L1/L2 整条 lane 退出（上限 20h）…"
for i in $(seq 1 1200); do
    busy >/dev/null || break
    [ $((i % 20)) -eq 1 ] && say "  仍忙：$(busy | head -2 | cut -c1-110 | tr '\n' ' ')"
    sleep 60
done
if busy >/dev/null; then say "!!! 20h 后仍忙，放弃（人工排查）"; busy | tee -a "$LOG"; exit 1; fi
say "前一条 lane 已退出。等 3 分钟让 NCCL / 显存释放干净。"
sleep 180

# ── 通用：起训前现查 ────────────────────────────────────────────────────────
preflight() {   # $1=TAG $2=PORT
    local N; N=$(pgrep -f 'open_clip_train[.]main' | wc -l)
    if [ "$N" -gt 0 ]; then
        say "!!! $1: 现查到 $N 个训练进程 → 按「先起者优先」退让，**整条队列停**"; return 1
    fi
    if ss -ltn 2>/dev/null | grep -q ":$2 "; then
        say "!!! $1: 端口 $2 被占（EADDRINUSE 会让训练 1 秒静默死掉）→ 跳过本臂"; return 2
    fi
    return 0
}

# ── 通用：过闸（口径读 params.txt，不猜目录名）────────────────────────────────
gate() {        # $1=TAG $2=DIR  → 0 通过
    local T=$1 D=$2 N
    [ -n "$D" ] && [ -d "$D" ] || { say "  !!! $T 没找到输出目录"; return 1; }
    N=$(ls "$D"/checkpoints/epoch_*.pt 2>/dev/null | wc -l)
    say "  $T 目录 $D  ckpt=$N/10"
    grep -E '^(region_roi_grid|region_roi_agg|region_weight|max_region|region_gather|region_cc_weight|seed|epochs|pcm_weight):' \
        "$D/params.txt" 2>/dev/null | tr '\n' ' ' | tee -a "$LOG"; echo | tee -a "$LOG"
    [ "$N" -eq 10 ] || { say "  !!! $T ckpt 不足 10 → 不写结论，先查 $D/out.log"; return 1; }
    return 0
}

# ── 通用：评测（★ 整段关在 subshell，CVD 结构上不可能泄漏）────────────────────
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
      echo "[wk $(date '+%H:%M:%S')] $T 全局五项 + 长文本完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[wk $(date '+%H:%M:%S')] $T OVSS ep10 四读出完" | tee -a "$LOG"
      for ds in voc ade; do for rd in penult last; do
          python scripts/eval/eval_ovss.py --ckpt "$CK5" --dataset "$ds" --dense-mode "$rd" \
              --tag "${T}ep5_${ds}_${rd}" >> "$OUT" 2>&1
      done; done
      echo "[wk $(date '+%H:%M:%S')] $T OVSS ep5 四读出完" | tee -a "$LOG"
      # 管线漂移校验：CTRL W2.0 的 VOC penult 必须精确复现 24.23
      python scripts/eval/eval_ovss.py --ckpt "$CTRL_CK" --dataset voc --dense-mode penult \
          --tag "DRIFT_after_${T}" >> "$OUT" 2>&1
    )
    say "  === $T 评测结束（subshell 退出，CVD 未泄漏）==="
    awk -v t="$T" '/^\[/ {keep = ($0 ~ ("\\[" t "]") || $0 ~ ("\\[" t "_") || $0 ~ ("\\[" t "ep5") || $0 ~ ("_" t "]")) } keep' "$OUT" \
        | grep -E "^\[|R@1=|top1=|k-NN (backbone|proj)|★.*mIoU=" \
        | sed 's/^/[wk] /' | tee -a "$LOG"
}

# ── 臂 1：C5s1 = grid2 + mil, W2.0, **seed 1**（Task #20，给 MIL 族实标 φ）────
if preflight C5s1 29720; then
    say "=== 臂1 C5s1 起训（grid2+mil W2.0 seed1）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 MAX_REGION=12 \
    SEED=1 DV_SUFFIX=-s1 EPOCHS=10 PORT=29720 \
        bash scripts/train/visreg.sh region >> /tmp/C5s1_train.log 2>&1
    say "臂1 visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"
    D1=$(ls -dt logs/visreg_gemma_regw2.0k12-roi2mil-s1_projective_E_* 2>/dev/null | head -1)
    if gate C5s1 "$D1"; then evalarm C5s1 "$D1"; fi
    say "臂1 完。判据在 prereg 的 R1/R2/R3/R4（主轴 VOC penult，参照 C5s0 47.23 / CTRL 24.23）"
else
    [ $? -eq 1 ] && { say "!!! 退让，整条队列停"; exit 0; }
fi

# ── 臂 2：C8 = grid2 + mil, **W0.5**（MIL 下重扫 W）──────────────────────────
if preflight C8 29722; then
    say "=== 臂2 C8 起训（grid2+mil W0.5 seed0）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=0.5 MAX_REGION=12 \
    SEED=0 EPOCHS=10 PORT=29722 \
        bash scripts/train/visreg.sh region >> /tmp/C8_train.log 2>&1
    say "臂2 visreg.sh 返回 $?"
    D2=$(ls -dt logs/visreg_gemma_regw0.5k12-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate C8 "$D2"; then evalarm C8 "$D2"; fi
    say "臂2 完。判据 S1/S2/S3（Δ_W = VOC penult − 47.23，2φ=7.76）"
else
    [ $? -eq 1 ] && { say "!!! 退让，整条队列停"; exit 0; }
fi

# ── 臂 3：C7 = **grid3** + mil, W2.0（Task #22，沿 MIL 轴推进）───────────────
if preflight C7 29724; then
    say "=== 臂3 C7 起训（grid3+mil W2.0 seed0）==="
    REGION_ROI_GRID=3 REGION_ROI_AGG=mil REGION_WEIGHT=2.0 MAX_REGION=12 \
    SEED=0 EPOCHS=10 PORT=29724 \
        bash scripts/train/visreg.sh region >> /tmp/C7_train.log 2>&1
    say "臂3 visreg.sh 返回 $?"
    D3=$(ls -dt logs/visreg_gemma_regw2.0k12-roi3mil_projective_E_* 2>/dev/null | head -1)
    if gate C7 "$D3"; then evalarm C7 "$D3"; fi
    say "臂3 完。判据 T1/T2/T3/T4（参照 C5 47.23 / C6 27.90）+ 长文本副判据 DOCCI t2i ≤17.16 / ≥20.04"
else
    [ $? -eq 1 ] && { say "!!! 退让，整条队列停"; exit 0; }
fi

# ── 臂 4：C10 = grid2 + mil, **W4.0**（把 MIL 下 W 曲线补成三点）─────────────
if preflight C10 29726; then
    say "=== 臂4 C10 起训（grid2+mil W4.0 seed0）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=4.0 MAX_REGION=12 \
    SEED=0 EPOCHS=10 PORT=29726 \
        bash scripts/train/visreg.sh region >> /tmp/C10_train.log 2>&1
    say "臂4 visreg.sh 返回 $?"
    D4=$(ls -dt logs/visreg_gemma_regw4.0k12-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate C10 "$D4"; then evalarm C10 "$D4"; fi
    say "臂4 完。判据 U1/U2/U3（参照 C5 47.23；全局对照 grid1+mean W4.0）"
else
    [ $? -eq 1 ] && { say "!!! 退让，整条队列停"; exit 0; }
fi

# ── 臂 5：C9 = G + MIL（pcm-region + grid2 mil，选型收口）────────────────────
if preflight C9 29728; then
    say "=== 臂5 C9 起训（G + MIL：pcm-region grid2+mil W0.2 p0.2）==="
    REGION_ROI_GRID=2 REGION_ROI_AGG=mil REGION_WEIGHT=0.2 MAX_REGION=12 \
    PCM_WEIGHT=0.2 PCM_DIM=32 SEED=0 EPOCHS=10 PORT=29728 \
        bash scripts/train/visreg.sh pcm-region >> /tmp/C9_train.log 2>&1
    say "臂5 visreg.sh 返回 $?"
    D5=$(ls -dt logs/visreg_gemma_pcmregw0.2p0.2-roi2mil_projective_E_* 2>/dev/null | head -1)
    if gate C9 "$D5"; then evalarm C9 "$D5"; fi
    say "臂5 完。判据 V1/V2/V3/V4（长文本 DOCCI ≥66.22/65.58 + 局部只与 C5 47.23 比）"
else
    [ $? -eq 1 ] && { say "!!! 退让，整条队列停"; exit 0; }
fi

say "=== 周末五臂全部走完。数字在 $OUT ==="
grep -E "R@1=|IN-1k top1|k-NN|mIoU|aAcc" "$OUT" | tail -80 | sed 's/^/[wk] /' | tee -a "$LOG"
say "逐臂：先报出口代号 → 写结果页小节 → 更 memory → commit && push。"

