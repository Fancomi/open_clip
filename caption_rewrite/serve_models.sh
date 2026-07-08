#!/bin/bash
# tmux 起 student 端点: gemma×4(8001-8004) + qwen×4(8005-8008), 用 sglang。
# 用法: bash caption_rewrite/serve_models.sh [gemma|qwen|all]  (默认 all)
# 停止: tmux kill-session -t crw_serve
set -e
WHICH="${1:-all}"
SESSION=crw_serve
GEMMA=/dev/shm/models/gemma-4-26B-A4B-it
QWEN=/dev/shm/models/Qwen3.6-35B-A3B-FP8
ACT="source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate"

launch() {  # name model port gpu
    local name=$1 model=$2 port=$3 gpu=$4
    tmux new-window -t "$SESSION" -n "$name" \
        "$ACT; CUDA_VISIBLE_DEVICES=$gpu python -m sglang.launch_server \
         --model-path $model --port $port --mem-fraction-static 0.85 \
         2>&1 | tee /tmp/${name}.log"
}

tmux has-session -t "$SESSION" 2>/dev/null && { echo "session $SESSION 已存在"; exit 0; }
tmux new-session -d -s "$SESSION" -n init "sleep 1"

gpu=0
if [ "$WHICH" = "gemma" ] || [ "$WHICH" = "all" ]; then
    for p in 8001 8002 8003 8004; do launch "gemma_$p" "$GEMMA" $p $gpu; gpu=$((gpu+1)); done
fi
if [ "$WHICH" = "qwen" ] || [ "$WHICH" = "all" ]; then
    for p in 8005 8006 8007 8008; do launch "qwen_$p" "$QWEN" $p $gpu; gpu=$((gpu+1)); done
fi
echo "已在 tmux session '$SESSION' 起服务。curl http://127.0.0.1:8001/v1/models 探活。"
echo "停止: tmux kill-session -t $SESSION"
