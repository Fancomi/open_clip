#!/bin/bash
# Feature space analysis — run from repo root.
# All cache/skip logic lives in Python (--force bypasses sentinels).
#
# Usage:
#   bash analysis/probe.sh coco
#   bash analysis/probe.sh cc3m
#   bash analysis/probe.sh overlap
#   bash analysis/probe.sh anisotropy [coco|cc3m]
#   bash analysis/probe.sh layers <model>                     # dinov3|pe_core|siglip2|eupe
#   bash analysis/probe.sh probe_full <probe_dir|logs_root> [--rerun] [extra python args...]
#   bash analysis/probe.sh log_parse [prefix] [--logs-dir DIR] [--plot-dir DIR] [--no-plot] [--no-md]
#
# --rerun  → passes --force to all Python calls (bypasses all sentinels)

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
set -e
export PYTHONPATH="./src:${PYTHONPATH}"

_BASE='/root/paddlejob/workspace/env_run/penghaotian'
CC3M_WDS="${_BASE}/datas/LLaVA-ReCap-CC3M/wds/{00000..00280}.tar"
CC3M_OUT="${_BASE}/datas/LLaVA-ReCap-CC3M/feature_probe"

# ── Arg parsing ───────────────────────────────────────────────────────────────
MODE="${1:-}"; shift || true

# Separate --rerun from other pass-through args.
# FORCE is passed to every Python call; PY_EXTRA holds additional Python flags.
FORCE=()
PY_EXTRA=()
for arg in "$@"; do
    [[ "$arg" == "--rerun" ]] && FORCE=("--force") || PY_EXTRA+=("$arg")
done

# ── Helpers ───────────────────────────────────────────────────────────────────

# True if dir contains .npz files (is a probe dir, not a logs root)
_is_probe_dir() { compgen -G "${1}/*.npz" >/dev/null 2>&1; }

# Find probe npz directory under a logdir (checkpoints/probe > probe > none)
_find_probe_dir() {
    local logdir="$1"
    for sub in "checkpoints/probe" "probe"; do
        if [[ -d "${logdir}/${sub}" ]] && _is_probe_dir "${logdir}/${sub}"; then
            echo "${logdir}/${sub}"; return 0
        fi
    done
    return 1
}

# Run all three analyses for one experiment dir.
# Accepts either a probe_dir (…/checkpoints/probe) or a logdir.
# Extra Python args passed via global FORCE and PY_EXTRA arrays.
_probe_one() {
    local logdir probe_dir
    if _is_probe_dir "$1"; then
        probe_dir="$1"
        logdir="$(realpath "${probe_dir}/../..")"
    else
        logdir="$(realpath "$1")"
        probe_dir="$(_find_probe_dir "$logdir")" || probe_dir=""
    fi

    # (1) training metrics CSV — needs out.log
    if [[ -f "${logdir}/out.log" ]]; then
        python3 -m analysis.log_parser --single "${logdir}" "${FORCE[@]}"
    fi

    # (2) epoch/step evolution GIF + UMAP + aniso; (3) PC alignment
    if [[ -n "${probe_dir}" ]]; then
        python3 -m analysis.run "${FORCE[@]}" --mode epochs       --probe-dir "${probe_dir}" "${PY_EXTRA[@]}"
        python3 -m analysis.run "${FORCE[@]}" --mode pc_alignment --probe-dir "${probe_dir}" "${PY_EXTRA[@]}"
    fi
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
case "$MODE" in
    coco|pretrained)
        python3 -m analysis.run "${FORCE[@]}" --mode pretrained --fps-model DINOv3 "${PY_EXTRA[@]}"
        ;;
    cc3m)
        python3 -m analysis.run "${FORCE[@]}" --mode pretrained --data-type wds --fps-model DINOv3 \
             --data "$CC3M_WDS" --out-dir "$CC3M_OUT" "${PY_EXTRA[@]}"
        ;;
    overlap)
        python3 -m analysis.run "${FORCE[@]}" --mode overlap "${PY_EXTRA[@]}"
        ;;
    anisotropy)
        TARGET="${PY_EXTRA[0]:-coco}"
        ANISO_DIR="$( [[ "$TARGET" == "cc3m" ]] && \
            echo "${CC3M_OUT}/pretrained" || \
            echo "${_BASE}/datas/coco/feature_probe/pretrained" )"
        python3 -m analysis.run "${FORCE[@]}" --mode anisotropy --aniso-dir "${ANISO_DIR}"
        ;;
    layers)
        MODEL="${PY_EXTRA[0]:?Usage: probe.sh layers <model>  (dinov3|pe_core|siglip2|eupe)}"
        OUT_DIR="${PY_EXTRA[1]:-analysis/layer_probe_out}"
        python3 -m analysis.layer_probe --model "${MODEL}" --out-dir "${OUT_DIR}"
        ;;
    probe_full|probe)
        # TARGET is the first non-flag arg; the rest are extra Python flags (e.g. --n-pcs 30)
        TARGET="${PY_EXTRA[0]:?Usage: probe.sh probe_full <probe_dir|logs_root> [--rerun] [--n-pcs N ...]}"
        PY_EXTRA=("${PY_EXTRA[@]:1}")   # consume TARGET; remaining = real Python extra args
        if _is_probe_dir "$TARGET"; then
            _probe_one "$TARGET"
        else
            for logdir in "${TARGET}"/*/; do
                [[ -d "$logdir" ]] || continue
                _probe_one "$logdir"
            done
        fi
        ;;
    log_parse)
        # First non-flag arg is an optional prefix
        if [[ "${PY_EXTRA[0]:-}" != --* && -n "${PY_EXTRA[0]:-}" ]]; then
            PREFIX="${PY_EXTRA[0]}"
            python3 -m analysis.log_parser "${FORCE[@]}" --prefix "${PREFIX}" "${PY_EXTRA[@]:1}"
        else
            python3 -m analysis.log_parser "${FORCE[@]}" "${PY_EXTRA[@]}"
        fi
        ;;
    eval_pretrained|eval)
        MODEL="${PY_EXTRA[0]:?Usage: probe.sh eval_pretrained <model>  (pe_core|siglip2)}"
        PY_EXTRA=("${PY_EXTRA[@]:1}")
        python3 -m analysis.run "${FORCE[@]}" --mode eval_pretrained \
            --eval-model "${MODEL}" --max-samples 5000 "${PY_EXTRA[@]}"
        ;;
    *)
        echo "Usage:"
        echo "  bash analysis/probe.sh coco"
        echo "  bash analysis/probe.sh cc3m"
        echo "  bash analysis/probe.sh overlap"
        echo "  bash analysis/probe.sh anisotropy [coco|cc3m]"
        echo "  bash analysis/probe.sh layers <model>  (dinov3|pe_core|siglip2|eupe)"
        echo "  bash analysis/probe.sh probe_full <probe_dir|logs_root> [--rerun] [--n-pcs N]"
        echo "  bash analysis/probe.sh eval_pretrained <model>  (pe_core|siglip2)"
        echo "  bash analysis/probe.sh log_parse [prefix] [--logs-dir DIR] [--plot-dir DIR] [--no-plot] [--no-md]"
        echo ""
        echo "  --rerun  bypass all sentinels (re-generate all outputs)"
        exit 1
        ;;
esac

echo "=== [probe] done ==="
