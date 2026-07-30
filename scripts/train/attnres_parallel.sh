#!/bin/bash
# attnres_parallel.sh -- run the AttnRes arms concurrently, one GPU each.
#
# The 8 H800s in this box are shared with another job (~19 GB / 100% util each),
# so arms are pinned to distinct GPUs and run at the same time: contention hits
# all arms equally, which keeps the comparison fair even though absolute
# throughput is depressed.
set -e
cd "$(dirname "$0")/../.."

SAMPLES=${SAMPLES:-800000}
BS=${BS:-256}
LR=${LR:-1e-3}
TAG=${TAG:-ar800k}
OUT=${OUT:-/tmp/attnres_runs}
mkdir -p "${OUT}"

# arm:gpu:port
JOBS=${JOBS:-"base:4:29810 attnres4:5:29811 attnres2:6:29812 k3init:7:29813"}

PIDS=()
for J in ${JOBS}; do
    ARM="${J%%:*}"; REST="${J#*:}"; GPU="${REST%%:*}"; PORT="${REST#*:}"
    echo "[launch] ${ARM} on GPU ${GPU} port ${PORT} -> ${OUT}/${TAG}_${ARM}.log"
    CUDA_VISIBLE_DEVICES="${GPU}" GPUS=1 BS="${BS}" SAMPLES="${SAMPLES}" \
        LR="${LR}" PORT="${PORT}" TAG="${TAG}" ARMS="${ARM}" \
        bash scripts/train/attnres_cc3m.sh > "${OUT}/${TAG}_${ARM}.log" 2>&1 &
    PIDS+=($!)
    sleep 5   # stagger so the shared subset TSV is written once
done

echo "[wait] ${#PIDS[@]} arms running: ${PIDS[*]}"
FAIL=0
for P in "${PIDS[@]}"; do wait "${P}" || FAIL=$((FAIL + 1)); done
echo "[done] failures=${FAIL}"
exit "${FAIL}"
