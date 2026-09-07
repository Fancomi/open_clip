#!/bin/bash
# ============================================================================
# 接力器：/tmp/mon.sh（C5s2→C16→F3）走完后自动起 /tmp/pcm.sh（P0→P1→P2）。
#
# 这是 memory/train-queue-silent-failures.md 第 7 条（8 卡空转 49h）的兜底实现，
# 与 09-06 的 /tmp/chain.sh 同一套三条设计约束：
#  · **全程等产物不等进程** —— 主触发是 /tmp/mon.log 的收尾行（一个 grep），
#    **没有 pgrep** ⇒ 结构上躲开第 5 条的自匹配死锁。
#  · **等驱动脚本不等训练进程** —— mon.sh 的臂间评测段只占卡 0，若按显存判会在空窗
#    被挤进去、两批互相抢卡（实测吞吐掉 15~20%）。显存只作兜底。
#  · **有上限**（26h）而不是 while true；成功用显式 GO=1 判，不能用 `[ "$i" -ge N ]`
#    —— 后者在「恰好末次轮询命中」时会把成功当超时。
#
# 时点：C16 09-07 09:48 起训 → 约 16:20 完；F3 约 23:00 完 ⇒ pcm.sh 预计 09-07 23:00 起。
# ----------------------------------------------------------------------------
set -u
LOG=/tmp/chain_mon2pcm.log
say() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
unset CUDA_VISIBLE_DEVICES

IDLE=0
GO=0
say "接力器起。等 /tmp/mon.log 出现「mon 队列走完」，或 8 卡连续 20 分钟全部 <1500 MiB。上限 26h。"

for i in $(seq 1 1560); do
    if grep -q 'mon 队列走完' /tmp/mon.log 2>/dev/null; then
        say "mon.sh 已打出收尾行（第 $i 次轮询）→ 起 pcm.sh"
        GO=1; break
    fi
    MAXMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    if [ "${MAXMEM:-999999}" -lt 1500 ]; then
        IDLE=$((IDLE + 1))
        [ "$IDLE" -eq 1 ] && say "8 卡最大显存 ${MAXMEM} MiB < 1500 → 开始计空闲分钟数"
        if [ "$IDLE" -ge 20 ]; then
            say "!!! 8 卡已连续 20 分钟全空（最大 ${MAXMEM} MiB）而 mon.log 没有收尾行"
            say "!!! 判定 mon.sh 异常终止 → 兜底起 pcm.sh（第 7 条那 49 小时不能再来一次）"
            GO=1; break
        fi
    else
        [ "$IDLE" -gt 0 ] && say "显存回到 ${MAXMEM} MiB → 空闲计数清零（mon.sh 的臂间/评测窗口）"
        IDLE=0
    fi
    sleep 60
done

if [ "$GO" -ne 1 ]; then
    say "!!! 26h 上限到，两个触发条件都没满足 → 不起 pcm.sh，等人工介入（宁可空转也不抢卡）"
    exit 1
fi

say "=== 起 pcm.sh（3 臂 ≈ 23.5h；pcm.sh 自己的 preflight 仍会做显存 + 端口二次确认）==="
exec bash /tmp/pcm.sh
