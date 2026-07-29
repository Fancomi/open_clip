#!/bin/bash
# run_chain.sh — 串行执行：等 k32 跑完 → 权重大范围扫描
#
# k32 的 driver 已在运行（visreg_slices.sh），本脚本等它退出后接上 wsweep。
# 全部串行，8 卡独占，互不抢资源。

cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
export PATH="$HOME/.local/bin:$PATH"

echo "[chain] $(date '+%F %T') 等待 visreg_slices.sh (k32) 结束…"
while pgrep -f "bash scripts/train/visreg_slices.sh" >/dev/null; do sleep 60; done
echo "[chain] $(date '+%F %T') k32 结束，启动 visreg_wsweep.sh"

bash scripts/train/visreg_wsweep.sh
echo "[chain] $(date '+%F %T') 全部完成"
