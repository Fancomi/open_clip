#!/bin/bash
# ============================================================================
# 接力器：sun.sh（C13s1→C14→C15）走完后自动起 /tmp/mon.sh（C5s2→C16→F3）。
#
# 这是 memory/train-queue-silent-failures.md 第 7 条（8 卡空转 49h）的兜底实现。
# 设计上同时躲开前面几条教训：
#  · **不用 pgrep** —— 第 5 条自匹配死锁的根因是「等进程」；这里全程**等产物**。
#  · **有上限** —— `for i in $(seq 1 1560)` ≈ 26h，不是 `while true`（教训 C）。
#  · **等驱动脚本而不是等训练进程** —— sun.sh 的臂间空窗（评测段只占卡 0）
#    如果只看显存会被挤进去、把两批变成互相抢卡；所以主判据是 sun.log 的收尾行。
#  · **兜底触发** —— 若 sun.sh 异常死掉、永远不打收尾行，则「8 卡全部 <1500 MiB
#    连续 20 分钟」也算它没了（这正是第 7 条要求的那个可检测量）。
# ----------------------------------------------------------------------------
set -u
LOG=/tmp/chain_sun2mon.log   # ⚠️ 不要用 /tmp/chain.log —— 07-29 的旧脚本占着它（4.8 MB 训练输出）
say() { echo "[chain $(date '+%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
unset CUDA_VISIBLE_DEVICES

IDLE=0
GO=0
say "接力器起。等 /tmp/sun.log 出现「sun 队列走完」，或 8 卡连续 20 分钟全部 <1500 MiB。上限 26h。"

for i in $(seq 1 1560); do
    if grep -q 'sun 队列走完' /tmp/sun.log 2>/dev/null; then
        say "sun.sh 已打出收尾行（第 $i 次轮询）→ 起 mon.sh"
        GO=1; break
    fi
    MAXMEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | sort -n | tail -1)
    if [ "${MAXMEM:-999999}" -lt 1500 ]; then
        IDLE=$((IDLE + 1))
        [ "$IDLE" -eq 1 ] && say "8 卡最大显存 ${MAXMEM} MiB < 1500 → 开始计空闲分钟数"
        if [ "$IDLE" -ge 20 ]; then
            say "!!! 8 卡已连续 20 分钟全空（最大 ${MAXMEM} MiB）而 sun.log 没有收尾行"
            say "!!! 判定 sun.sh 异常终止 → 兜底起 mon.sh（第 7 条那 49 小时不能再来一次）"
            GO=1; break
        fi
    else
        [ "$IDLE" -gt 0 ] && say "显存回到 ${MAXMEM} MiB → 空闲计数清零（sun.sh 的臂间/评测窗口）"
        IDLE=0
    fi
    sleep 60
done

# ⚠️ 用显式 GO 标志而不是 `[ "$i" -ge 1560 ]` —— 后者在「恰好第 1560 次轮询命中触发条件」
#    时会把成功当成超时（1/1560 的概率，但这脚本是无人值守的，不留这种坑）
if [ "$GO" -ne 1 ]; then
    say "!!! 26h 上限到，两个触发条件都没满足 → 不起 mon.sh，等人工介入（宁可空转也不抢卡）"
    exit 1
fi

say "=== 起 mon.sh（3 臂 ≈ 19.5h；mon.sh 自己的 preflight 仍会做显存 + 端口二次确认）==="
exec bash /tmp/mon.sh
