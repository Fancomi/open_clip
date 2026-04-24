#!/bin/bash
# Feature space analysis — 与 quick.sh 训练流程完全分离
#
# pretrained (COCO, 随时可跑，npz 有缓存则跳过推理):
#   bash probe.sh pretrained
#
# pretrained (CC3M 全量 wds，100k subsample):
#   bash probe.sh cc3m
#
# epoch 演化 (需先完成训练):
#   bash probe.sh epochs <probe_dir>
#
# COCO vs CC3M 分布重合 (需两边 cache 均已生成):
#   bash probe.sh overlap
#
# 各向异性指标 (从已有 cache 快速计算):
#   bash probe.sh anisotropy [coco|cc3m]

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
set -e
export PYTHONPATH="./src:${PYTHONPATH}"

MODE="${1:-}"
SCRIPT="python3 scripts/feature_probe.py"

COCO_OUT='/root/paddlejob/workspace/env_run/penghaotian/datas/coco/feature_probe/pretrained'
CC3M_WDS='/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds/{00000..00280}.tar'
CC3M_OUT='/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/feature_probe'
CC3M_PRE="${CC3M_OUT}/pretrained"

case "$MODE" in
    pretrained)
        echo "=== [probe] COCO pretrained analysis ==="
        $SCRIPT --mode pretrained
        ;;
    cc3m)
        echo "=== [probe] CC3M pretrained analysis (wds, 100k subsample) ==="
        $SCRIPT --mode pretrained --data-type wds \
            --data "${CC3M_WDS}" --out-dir "${CC3M_OUT}"
        ;;
    epochs)
        PROBE_DIR="${2:?Usage: probe.sh epochs <probe_dir>}"
        echo "=== [probe] epoch evolution  probe_dir=${PROBE_DIR} ==="
        $SCRIPT --mode epochs --probe-dir "$PROBE_DIR"
        ;;
    overlap)
        echo "=== [probe] COCO vs CC3M overlap ==="
        $SCRIPT --mode overlap \
            --coco-dir "${COCO_OUT}" \
            --cc3m-dir "${CC3M_PRE}"
        ;;
    anisotropy)
        TARGET="${2:-coco}"
        if [ "$TARGET" = "cc3m" ]; then
            ANISO_DIR="${CC3M_PRE}"
        else
            ANISO_DIR="${COCO_OUT}"
        fi
        echo "=== [probe] anisotropy  dir=${ANISO_DIR} ==="
        $SCRIPT --mode anisotropy --aniso-dir "${ANISO_DIR}"
        ;;
    *)
        echo "Usage:"
        echo "  bash probe.sh pretrained"
        echo "  bash probe.sh cc3m"
        echo "  bash probe.sh epochs <probe_dir>"
        echo "  bash probe.sh overlap"
        echo "  bash probe.sh anisotropy [coco|cc3m]"
        exit 1
        ;;
esac

echo "=== [probe] done ==="
