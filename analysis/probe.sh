#!/bin/bash
# Feature space analysis — lives in analysis/, runs from repo root
#
# ═══════════════════════════════════════════════════════════════════════════════
# 输出文件（coco / cc3m 模式）
# ═══════════════════════════════════════════════════════════════════════════════
#   pe_core_modality_gap.png    PE-Core 图文模态鸿沟
#   siglip2_modality_gap.png    SigLIP2 图文模态鸿沟
#   tips_modality_gap.png       TIPSv2  图文模态鸿沟
#   image_allmodels.png         6 模型图像特征对比 + FPS 跨模型锚点追踪
#   anisotropy.png              各向异性 + 秩 + 多峰性指标对比
#
# ═══════════════════════════════════════════════════════════════════════════════
# 各向异性指标详解
# ═══════════════════════════════════════════════════════════════════════════════
#
#   ┌───────────────────────┬──────────────────────────────────────┬──────────┐
#   │ 指标                  │ 定义                                 │ ↑=各向同性│
#   ├───────────────────────┼──────────────────────────────────────┼──────────┤
#   │ Effective Rank        │ exp(H(λ/Σλ))  ∈ [1, D]             │ ↑        │
#   │ Participation Ratio   │ 1/(D·Σλ²)     ∈ (0,1]              │ ↑        │
#   │ Stable Rank           │ 1/λ_max  (= Σλ/λ_max)              │ ↑        │
#   │ Numerical Rank        │ #{s_i ≥ 1%·s_max}                  │ ↑        │
#   │ Avg Cosine Sim        │ 均值余弦相似度                       │ ↓        │
#   │ Std Cosine Sim        │ 标准差余弦 → 多峰性(simplex)检测     │ ↑多峰    │
#   │ pct_var_top{k}        │ 前 k 个 PC 累计方差%                │ ↓        │
#   └───────────────────────┴──────────────────────────────────────┴──────────┘
#
#   std_cos 是新增指标，用于区分 DINOv3-style simplex（多峰）和
#   RADIO-style 平滑低秩流形（单峰）——两者 effective_rank 可能接近，
#   但 std_cos 会明显不同。
#
# ═══════════════════════════════════════════════════════════════════════════════
# COCO vs CC3M 分布区分现象解读（overlap 模式）
# ═══════════════════════════════════════════════════════════════════════════════
#
#   A. DINOv3 / PE-Core / TIPSv2  — COCO 与 CC3M 沿 PC2 有明显分离
#      这些模型保留了数据集"指纹"（COCO 精选有标注，CC3M 爬虫图文对），
#      体现对域偏移更敏感。
#
#   B. RADIO / EUPE / SigLIP2     — COCO 与 CC3M 在 PC1-PC2 完全重合
#      多教师蒸馏/大规模多样训练 → 域不变语义表示。
#      对跨域迁移学习 B 更好；A 类仅沿方差较小的 PC2 区分，PC1 语义轴
#      仍域不变，实际影响有限。
#
# ═══════════════════════════════════════════════════════════════════════════════

source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
set -e
export PYTHONPATH="./src:${PYTHONPATH}"

MODE="${1:-}"
SCRIPT="python3 -m analysis.run"

COCO_OUT='/root/paddlejob/workspace/env_run/penghaotian/datas/coco/feature_probe/pretrained'
CC3M_WDS='/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/wds/{00000..00280}.tar'
CC3M_OUT='/root/paddlejob/workspace/env_run/penghaotian/datas/LLaVA-ReCap-CC3M/feature_probe'
CC3M_PRE="${CC3M_OUT}/pretrained"

# ── helpers ──────────────────────────────────────────────────────────────────

# _run_epochs <probe_dir> <rerun:0|1>
_run_epochs() {
    local probe_dir="$1" rerun="${2:-0}"
    local plots_dir
    plots_dir="$(realpath "${probe_dir}/../../probe/plots" 2>/dev/null || echo "${probe_dir}/../../probe/plots")"
    local sentinel="${plots_dir}/aniso_evolution.png"
    if [[ "$rerun" -eq 0 && -f "$sentinel" ]]; then
        echo "=== [probe] epochs  SKIP (already done)  ${probe_dir} ==="
        return
    fi
    echo "=== [probe] epoch evolution  probe_dir=${probe_dir} ==="
    $SCRIPT --mode epochs --probe-dir "$probe_dir"
}

# _is_probe_dir <path> — true if path contains .npz files (is a probe dir, not a logs root)
_is_probe_dir() {
    compgen -G "${1}/*.npz" > /dev/null 2>&1
}

_run_log_metrics() {
    # $1 可以是 probe_dir (checkpoints/probe) 或 logdir (实验根目录)
    local input="$1" rerun="${2:-0}"
    local logdir plots_dir sentinel
    # 如果传入的是 probe_dir，向上两级得到 logdir；否则直接用
    if [[ "$(basename "$(dirname "$input")")" == "checkpoints" ]]; then
        logdir="$(realpath "${input}/../.." 2>/dev/null || echo "${input}/../..")"
    else
        logdir="$(realpath "$input" 2>/dev/null || echo "$input")"
    fi
    plots_dir="${logdir}/probe/plots"
    sentinel="${plots_dir}/training_metrics.csv"
    if [[ "$rerun" -eq 0 && -f "$sentinel" ]]; then
        echo "=== [probe] log_metrics  SKIP (already done)  ${logdir} ==="
        return
    fi
    [[ -f "${logdir}/out.log" ]] || { echo "=== [probe] log_metrics  SKIP (no out.log)  ${logdir} ==="; return; }
    echo "=== [probe] log_metrics  logdir=${logdir} ==="
    python3 -m analysis.log_parser --single "${logdir}" --out "${sentinel}"
}

_run_pc_alignment() {
    local probe_dir="$1" n_pcs="${2:-20}" rerun="${3:-0}"
    local plots_dir
    plots_dir="$(realpath "${probe_dir}/../../probe/plots" 2>/dev/null || echo "${probe_dir}/../../probe/plots")"
    local sentinel="${plots_dir}/pc_alignment_grassmann.png"
    if [[ "$rerun" -eq 0 && -f "$sentinel" ]]; then
        echo "=== [probe] pc_alignment  SKIP (already done)  ${probe_dir} ==="
        return
    fi
    echo "=== [probe] PC alignment  probe_dir=${probe_dir}  n_pcs=${n_pcs} ==="
    $SCRIPT --mode pc_alignment --probe-dir "$probe_dir" --n-pcs "${n_pcs}"
}

# ── dispatch ──────────────────────────────────────────────────────────────────

case "$MODE" in
    coco|pretrained)
        echo "=== [probe] COCO analysis (cache-first) ==="
        $SCRIPT --mode pretrained --fps-model DINOv3
        ;;
    cc3m)
        echo "=== [probe] CC3M analysis (cache-first, wds 100k) ==="
        $SCRIPT --mode pretrained --data-type wds --fps-model DINOv3 \
            --data "${CC3M_WDS}" --out-dir "${CC3M_OUT}"
        ;;
    epochs)
        # epochs <probe_dir|logs_root> [--rerun]
        # If $2 is a logs/ root, iterate all subdirs; else treat as single probe_dir.
        ARG2="${2:?Usage: probe.sh epochs <probe_dir|logs_root> [--rerun]}"
        RERUN=0; [[ "${3:-}" == "--rerun" || "${2:-}" == "--rerun" ]] && RERUN=1
        if _is_probe_dir "$ARG2"; then
            # single probe_dir
            _run_epochs "$ARG2" "$RERUN"
        else
            # logs root — iterate subdirs
            for logdir in "${ARG2}"/*/; do
                probe_dir="${logdir}checkpoints/probe"
                [[ -d "$probe_dir" ]] || continue
                _run_epochs "$probe_dir" "$RERUN"
            done
        fi
        ;;
    overlap)
        echo "=== [probe] COCO vs CC3M overlap ==="
        $SCRIPT --mode overlap \
            --coco-dir "${COCO_OUT}" \
            --cc3m-dir "${CC3M_PRE}"
        ;;
    anisotropy)
        TARGET="${2:-coco}"
        ANISO_DIR="$( [ "$TARGET" = "cc3m" ] && echo "${CC3M_PRE}" || echo "${COCO_OUT}" )"
        echo "=== [probe] anisotropy  dir=${ANISO_DIR} ==="
        $SCRIPT --mode anisotropy --aniso-dir "${ANISO_DIR}"
        ;;
    layers)
        MODEL="${2:?Usage: probe.sh layers <model>  (dinov3|pe_core|siglip2|eupe)}"
        OUT_DIR="${3:-analysis/layer_probe_out}"
        echo "=== [probe] layer-wise feature probe  model=${MODEL} ==="
        python3 -m analysis.layer_probe --model "${MODEL}" --out-dir "${OUT_DIR}"
        ;;
    pc_alignment)
        # pc_alignment <probe_dir|logs_root> [n_pcs=16] [--rerun]
        ARG2="${2:?Usage: probe.sh pc_alignment <probe_dir|logs_root> [n_pcs=16] [--rerun]}"
        N_PCS=20; RERUN=0
        for arg in "${@:3}"; do
            [[ "$arg" == "--rerun" ]] && RERUN=1 || N_PCS="$arg"
        done
        if _is_probe_dir "$ARG2"; then
            # single probe_dir
            _run_pc_alignment "$ARG2" "$N_PCS" "$RERUN"
        else
            # logs root — iterate subdirs
            for logdir in "${ARG2}"/*/; do
                probe_dir="${logdir}checkpoints/probe"
                [[ -d "$probe_dir" ]] || continue
                _run_pc_alignment "$probe_dir" "$N_PCS" "$RERUN"
            done
        fi
        ;;
    probe_full|probe)
        # probe_full <probe_dir|logs_root> [n_pcs=20] [--rerun]
        # Runs epochs + pc_alignment in one pass.
        ARG2="${2:?Usage: probe.sh probe_full <probe_dir|logs_root> [n_pcs=20] [--rerun]}"
        N_PCS=20; RERUN=0
        for arg in "${@:3}"; do
            [[ "$arg" == "--rerun" ]] && RERUN=1 || N_PCS="$arg"
        done
        if _is_probe_dir "$ARG2"; then
            _run_log_metrics   "$ARG2" "$RERUN"
            _run_epochs        "$ARG2" "$RERUN"
            _run_pc_alignment  "$ARG2" "$N_PCS" "$RERUN"
        else
            for logdir in "${ARG2}"/*/; do
                [[ -d "$logdir" ]] || continue
                # log_metrics: only needs out.log, always run
                _run_log_metrics "${logdir}" "$RERUN"
                # epochs + pc_alignment: need probe npz — skip if not present
                probe_dir="${logdir}checkpoints/probe"
                if [[ -d "$probe_dir" ]]; then
                    _run_epochs       "$probe_dir" "$RERUN"
                    _run_pc_alignment "$probe_dir" "$N_PCS" "$RERUN"
                fi
            done
        fi
        ;;
    crop_probe)
        # crop_probe requires individual image files on disk (tsv/COCO mode).
        # wds/CC3M images live inside tar archives and cannot be opened by PIL.
        COCO_FEAT_DIR="$(dirname "${COCO_OUT}")"   # .../datas/coco/feature_probe
        OUT_DIR="${2:-${COCO_FEAT_DIR}}"
        echo "=== [probe] crop_probe  out_dir=${OUT_DIR} ==="
        $SCRIPT --mode crop_probe --out-dir "${OUT_DIR}"
        ;;
    log_parse)
        # log_parse [prefix_or_dir] [--logs-dir DIR] [--plot-dir DIR] [--no-plot] [--no-md]
        #
        # Examples:
        #   bash analysis/probe.sh log_parse ft_
        #     → prefix=ft_, plots → analysis/research/plots/ft
        #   bash analysis/probe.sh log_parse --logs-dir logs/20260508_0_ft_book
        #     → all experiments in that dir, plots → logs/20260508_0_ft_book/plots
        #   bash analysis/probe.sh log_parse ft_ --logs-dir logs/20260508_0_ft_book
        #     → prefix=ft_ inside that dir

        # $2 is prefix if it doesn't start with '--'; pass everything through
        if [[ "${2:-}" != --* && -n "${2:-}" ]]; then
            PREFIX="${2}"
            shift 2 2>/dev/null || true
            echo "=== [probe] log_parse  prefix=${PREFIX} ==="
            python3 -m analysis.log_parser --prefix "${PREFIX}" "$@"
        else
            shift 1 2>/dev/null || true
            echo "=== [probe] log_parse ==="
            python3 -m analysis.log_parser "$@"
        fi
        ;;
    *)
        echo "Usage:"
        echo "  bash analysis/probe.sh coco"
        echo "  bash analysis/probe.sh cc3m"
        echo "  bash analysis/probe.sh epochs <probe_dir|logs_root> [--rerun]"
        echo "  bash analysis/probe.sh overlap"
        echo "  bash analysis/probe.sh anisotropy [coco|cc3m]"
        echo "  bash analysis/probe.sh layers <model>  (dinov3|pe_core|siglip2|eupe)"
        echo "  bash analysis/probe.sh pc_alignment <probe_dir|logs_root> [n_pcs=20] [--rerun]"
        echo "  bash analysis/probe.sh probe_full   <probe_dir|logs_root> [n_pcs=20] [--rerun]"
        echo "  bash analysis/probe.sh crop_probe [out_dir=CC3M_OUT]"
        echo "  bash analysis/probe.sh log_parse [prefix]  [--logs-dir DIR] [--plot-dir DIR] [--no-plot] [--no-md]"
        exit 1
        ;;
esac

echo "=== [probe] done ==="
