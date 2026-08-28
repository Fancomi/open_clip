#!/bin/bash
# ============================================================================
# 任务 #21：长文本窗口 320 + 截断代替丢行（L1 / L2 两臂）
#
# 用户直接指令：「1. 长文本CLIP拉到320token，这是我们dense生产数据的目标；
#                 2. 以文本截断做输入训练，而不是整图丢弃」
# 预登记：analysis/research/prereg/l1_ctx320_prereg.md（起训前写，已 commit 41669bc）
#
# 三臂 / 三轴：
#   L0（已有）clip_train_dual.tsv      2,006,804 行  ctx256  超窗行丢整行
#   L2（新）  clip_train_dual_full.tsv 2,894,191 行  ctx256  超窗行**截断**
#   L1（新）  clip_train_dual_full.tsv 2,894,191 行  ctx320  超窗行**截断**  ← 交付物
#   A = L1−L0（端到端生产数字）  B = L1−L2（**只有窗口在动**，唯一无混淆的轴）
#   C = L2−L0（只有截断代替丢行在动）  一致性 |A−(B+C)| ≤ 0.91
#
# 判据（全部写在跑之前，三分带 + 符号）：
#   轴 A 主口径 IN-1k top1 全量，2φ=0.64，L0=21.53
#     A1 Δ≥+1.9  A2 +0.64≤Δ<+1.9  A3 |Δ|<0.64  A4 Δ≤−0.64
#   轴 B 主口径 DOCCI t2i，2φ=2.88（B-native 为主，B-matched 复核）
#     B1 Δ≥+2.88  B2 |Δ|<2.88  B3 Δ≤−2.88
#   轴 C 主口径 IN-1k top1，2φ=0.64
#     C1 Δ≥+0.64  C2 |Δ|<0.64  C3 Δ≤−0.64
#   ⚠️ 所有地板都是借 gt_base 4-run 的 → 只当量级读，不当显著性检验（§5.19）
#   ⚠️ Urban（φ=2.36）明示排除否决权
#   押注 A2+B2+C1（写在跑之前，**不影响判据**）
#
# 结构约定：
#   ★ 两臂训练段连排在前、评测段全部在后 —— 评测要 export CUDA_VISIBLE_DEVICES=0，
#     若插在两次训练之间，泄漏出来会让第二次 torchrun --nproc_per_node=8 只看到 1 卡。
#   ★ L1 训完先过闸（ckpt=10 + params.txt 的 context_length=320 + train-data 是 full）
#     不过闸就不起 L2，省掉 5 小时。
#   ★ 端口 L1=29716 / L2=29718（29660 是 pcm 默认值，不占）
# 成本：L2 ≈ 4h49m，L1 ≈ 5h10m~5h20m（文本塔窗口 ×1.25），评测 ~40min×2
# ----------------------------------------------------------------------------
set -u
LOG=/tmp/l1l2.log
say() { echo "[l1l2 $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
ANN=/root/paddlejob/workspace/env_run/penghaotian/datas/cc3m-tsv/annotations

# ── 0. 等 C6（Task #19）的**驱动脚本**退出 ────────────────────────────────
# 等驱动而非训练进程：C6 训完还要自己跑评测，臂间空窗若只看训练进程会被别人挤进去
say "等 C6 驱动脚本（bash visreg.sh region）退出…"
for i in $(seq 1 720); do          # 720 × 60s = 12h 上限
    pgrep -f 'bash scripts/train/visreg[.]sh region' >/dev/null || break
    sleep 60
done
if pgrep -f 'bash scripts/train/visreg[.]sh region' >/dev/null; then
    say "!!! 12 小时后 C6 驱动仍在，放弃（人工排查）"; exit 1
fi
say "C6 驱动已退出。再等 3 分钟让 NCCL / 显存释放干净。"
sleep 180

# ── 1. 起训前现查（铁律 1；08-26 撞车吞吐 −42%）──────────────────────────
OTHER=$(pgrep -f 'open_clip_train[.]main' | wc -l)
if [ "$OTHER" -gt 0 ]; then
    say "!!! 仍有 $OTHER 个 open_clip_train 进程，按「先起者优先」退让，不起训。"
    exit 0
fi
for P in 29716 29718; do
    if ss -ltn 2>/dev/null | grep -q ":${P} "; then
        say "!!! 端口 ${P} 被占 → EADDRINUSE 会让训练 1 秒内静默死掉。放弃。"; exit 1
    fi
done
[ -f "${ANN}/clip_train_dual_full.tsv" ] || { say "!!! 缺 clip_train_dual_full.tsv"; exit 1; }
say "现查通过：无训练进程、29716/29718 空闲、TSV 就位。"

# ── 2. L1 训练（ctx320 + 全量截断，交付物臂）─────────────────────────────
say "=== L1 起训：CTX_LEN=320 dual_full 2,894,191 行 PCM w0.2 d32 10ep seed0 ==="
DUAL_TSV_NAME=clip_train_dual_full.tsv DV_SUFFIX=ctx320full CTX_LEN=320 \
PORT=29716 PCM_WEIGHT=0.2 PCM_DIM=32 EPOCHS=10 \
    bash scripts/train/visreg.sh pcm >> /tmp/l1_train.log 2>&1
say "L1 visreg.sh 返回 $?（收尾 NCCL 报错不影响产物，只看 ckpt 数）"

L1DIR=$(ls -dt logs/visreg_gemma_pcmw0.2d32ctx320full_projective_E_* 2>/dev/null | head -1)
[ -n "$L1DIR" ] || { say "!!! 没找到 L1 输出目录。日志尾巴："; tail -30 /tmp/l1_train.log | tee -a "$LOG"; exit 1; }
N1=$(ls "$L1DIR"/checkpoints/epoch_*.pt 2>/dev/null | wc -l)
say "L1 目录 $L1DIR  ckpt=$N1/10"

# 闸门：口径读 params.txt，不猜目录名
# 键名是 force_context_length（params.txt:53），不是 context_length —— 别猜键名
grep -qE '^force_context_length: 320$'  "$L1DIR/params.txt" && say "  ✅ params.txt force_context_length=320" \
    || { say "  !!! params.txt 的 force_context_length 不是 320 —— 窗口没生效，整条轴 B 作废"; grep -E 'context_length|^train_data' "$L1DIR/params.txt" | tee -a "$LOG"; }
grep -q 'clip_train_dual_full.tsv' "$L1DIR/params.txt" && say "  ✅ params.txt train_data=dual_full" \
    || say "  !!! params.txt 的 train_data 不是 dual_full"
grep -oE '\[2[0-9]{6}/2894191' "$L1DIR/out.log" | tail -1 | tee -a "$LOG"
[ "$N1" -eq 10 ] || { say "!!! L1 ckpt 不足 10 个 → 不起 L2，先查 $L1DIR/out.log"; exit 1; }

# ── 3. L2 训练（ctx256 + 全量截断，只满足要求 2）─────────────────────────
say "=== L2 起训：CTX_LEN=256 dual_full 同数据同步数，与 L1 只差窗口 ==="
DUAL_TSV_NAME=clip_train_dual_full.tsv DV_SUFFIX=ctx256full CTX_LEN=256 \
PORT=29718 PCM_WEIGHT=0.2 PCM_DIM=32 EPOCHS=10 \
    bash scripts/train/visreg.sh pcm >> /tmp/l2_train.log 2>&1
say "L2 visreg.sh 返回 $?"

L2DIR=$(ls -dt logs/visreg_gemma_pcmw0.2d32ctx256full_projective_E_* 2>/dev/null | head -1)
[ -n "$L2DIR" ] || { say "!!! 没找到 L2 输出目录。日志尾巴："; tail -30 /tmp/l2_train.log | tee -a "$LOG"; }
N2=$(ls "$L2DIR"/checkpoints/epoch_*.pt 2>/dev/null | wc -l)
say "L2 目录 $L2DIR  ckpt=$N2/10"
grep -qE '^force_context_length: 256$' "$L2DIR/params.txt" && say "  ✅ params.txt force_context_length=256" || say "  !!! L2 窗口不是 256"

# ── 4. 评测段（★ 所有 CVD 导出都关在 subshell 里，结构上不可能泄漏）────────
say "=== 评测：两臂 × 全局五项 + 轴 B 的 B-matched 双读法 ==="
OUT=/tmp/l1l2_eval.txt
(
  source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
  export PYTHONPATH="./src:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES=0
  for SPEC in "L1:$L1DIR" "L2:$L2DIR"; do
      TAG=${SPEC%%:*}; D=${SPEC#*:}
      [ -f "$D/checkpoints/epoch_10.pt" ] || { echo "[skip] $TAG 无 epoch_10.pt" >> "$OUT"; continue; }
      CK="$D/checkpoints/epoch_10.pt"
      python scripts/eval/eval_standard.py  --ckpt "$CK" --tag "$TAG" --retrieval >> "$OUT" 2>&1
      python scripts/eval/eval_knn_probe.py --ckpt "$CK" --tag "$TAG"             >> "$OUT" 2>&1
      python scripts/eval/eval_urban1k.py   --ckpt "$CK" --tag "$TAG"             >> "$OUT" 2>&1
      python scripts/eval/eval_docci.py     --ckpt "$CK" --tag "$TAG"             >> "$OUT" 2>&1
      echo "[l1l2 $(date '+%H:%M:%S')] $TAG 全局五项完（native 读法）" | tee -a "$LOG"
  done
  # B-matched：L1 按 256 分词，去掉"评测文本少被截"这一项，只留模型侧差异
  # （encode_text 是**切**位置编码 → 320 模型喂 256 长 token 精确等价，无插值）
  if [ -f "$L1DIR/checkpoints/epoch_10.pt" ]; then
      python scripts/eval/eval_docci.py   --ckpt "$L1DIR/checkpoints/epoch_10.pt" \
          --tag L1_tok256 --tok-context-length 256 >> "$OUT" 2>&1
      python scripts/eval/eval_urban1k.py --ckpt "$L1DIR/checkpoints/epoch_10.pt" \
          --tag L1_tok256 --tok-context-length 256 >> "$OUT" 2>&1
      echo "[l1l2 $(date '+%H:%M:%S')] B-matched 完" | tee -a "$LOG"
  fi
  # 管线漂移校验：L0 自己重跑 DOCCI，期望精确复现 i2t 62.00 / t2i 63.00
  python scripts/eval/eval_docci.py \
      --ckpt logs/visreg_gemma_pcmw0.2d32_projective_E_0818_1857/checkpoints/epoch_10.pt \
      --tag DRIFT_L0 >> "$OUT" 2>&1
)
say "评测段结束（subshell 退出，CVD 未泄漏到本 shell）"

say "--- 数字汇总 ---"
grep -E "R@1=|IN-1k top1|k-NN|^\[L1|^\[L2|^\[DRIFT" "$OUT" | sed 's/^/[l1l2] /' | tee -a "$LOG"
say "L0 参照（pcmw0.2d32_0818_1857 ep10）：IN-1k 21.53 / k-NN bb 41.44 proj 41.38 /"
say "  COCO i2t 28.60 t2i 17.33 / Urban 59.50/59.30 / DOCCI i2t 62.00 t2i 63.00"
say "判据 A/B/C 在本脚本头与 analysis/research/prereg/l1_ctx320_prereg.md，**先报判定再解释**。"
say "写 §5.22（轴 A/B/C 三张表 + 一致性检查 |A−(B+C)|≤0.91）→ 更两个 memory 文件 → commit + push。"
