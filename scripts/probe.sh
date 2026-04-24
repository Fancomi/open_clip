#!/bin/bash
# Feature space analysis — 与 quick.sh 训练流程完全分离
#
# Exp1 pretrained (COCO, 随时可跑，npz 有缓存则跳过推理):
#   bash probe.sh pretrained
#
# Exp1 pretrained (CC3M 全量 wds，100k subsample):
#   bash probe.sh cc3m
#
# Exp2 epoch 演化 (需先完成训练):
#   bash probe.sh epochs <probe_dir>
#   例: bash probe.sh epochs logs/cc3m_pe_dinov3_leproj_probe_XXXX/checkpoints/probe

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
set -e
export PYTHONPATH="./src:${PYTHONPATH}"

MODE="${1:-}"
SCRIPT="python3 scripts/feature_probe.py"

CC3M_WDS='/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds/{00000..00280}.tar'
CC3M_OUT='/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/feature_probe'

case "$MODE" in
    pretrained)
        echo "=== [probe] Exp1: COCO pretrained analysis ==="
        $SCRIPT --mode pretrained
        ;;
    cc3m)
        echo "=== [probe] Exp1: CC3M pretrained analysis (wds, 100k subsample) ==="
        $SCRIPT --mode pretrained --data-type wds \
            --data "${CC3M_WDS}" --out-dir "${CC3M_OUT}"
        ;;
    epochs)
        PROBE_DIR="${2:?Usage: probe.sh epochs <probe_dir>}"
        echo "=== [probe] Exp2: epoch evolution  probe_dir=${PROBE_DIR} ==="
        $SCRIPT --mode epochs --probe-dir "$PROBE_DIR"
        ;;
    *)
        echo "Usage:"
        echo "  bash probe.sh pretrained"
        echo "  bash probe.sh cc3m"
        echo "  bash probe.sh epochs <probe_dir>"
        exit 1
        ;;
esac

echo "=== [probe] done ==="
