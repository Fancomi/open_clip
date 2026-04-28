#!/usr/bin/env bash
# =============================================================================
# run_experiments.sh  ─  Run the full toy experiment suite for PCA regularization
#
# Usage
# ─────
#   cd experiments/pca_drop_toy
#   bash scripts/run_experiments.sh            # all experiments
#   bash scripts/run_experiments.sh smoke      # quick smoke test (1 run, 5 epochs)
#   bash scripts/run_experiments.sh baseline   # only baseline configs
#   bash scripts/run_experiments.sh sweep      # hyperparameter sweep
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${SCRIPT_DIR}/.."
cd "${ROOT}"

MODE="${1:-all}"
SEEDS=(42 1 2)          # run each config with 3 seeds for variance estimation
SMOKE_EPOCHS=5

echo "================================================================"
echo "  PCA Regularization Toy Experiment Suite"
echo "  Mode    : ${MODE}"
echo "  CWD     : $(pwd)"
echo "  Python  : $(python3 --version)"
echo "================================================================"
echo ""

# ─── helpers ──────────────────────────────────────────────────────────────────
run_one() {
    local CFG="$1"
    local SEED="$2"
    local EXTRA="${3:-}"
    echo "──────────────────────────────────────────────────────────"
    echo "  Config : ${CFG}"
    echo "  Seed   : ${SEED}"
    echo "  Extra  : ${EXTRA}"
    echo "──────────────────────────────────────────────────────────"
    python3 train.py --config "${CFG}" --seed "${SEED}" ${EXTRA}
}

# ─── smoke test ───────────────────────────────────────────────────────────────
if [[ "${MODE}" == "smoke" ]]; then
    echo "[smoke] Running 1 seed × 1 config × ${SMOKE_EPOCHS} epochs ..."
    run_one configs/toy_baseline.yaml 42 "--epochs ${SMOKE_EPOCHS}"
    echo "[smoke] Done."
    exit 0
fi

# ─── unit tests ───────────────────────────────────────────────────────────────
if [[ "${MODE}" == "test" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> Running unit tests ..."
    cd tests && python3 -m pytest test_pca_regularizer.py -v --tb=short 2>&1 | tee /tmp/toy_test_results.txt
    cd ..
    echo ""
fi

if [[ "${MODE}" == "test" ]]; then exit 0; fi

# ─── baseline ─────────────────────────────────────────────────────────────────
echo ""
echo ">>> [1/4] Baseline experiments (mode=none) ..."
for SEED in "${SEEDS[@]}"; do
    run_one configs/toy_baseline.yaml "${SEED}"
done

if [[ "${MODE}" == "baseline" ]]; then exit 0; fi

# ─── nuisance attenuate ────────────────────────────────────────────────────────
echo ""
echo ">>> [2/4] Nuisance high-variance dataset + attenuate_topk ..."
for SEED in "${SEEDS[@]}"; do
    run_one configs/toy_nuisance_high_variance.yaml "${SEED}"
done

# ─── signal attenuate (risk test) ─────────────────────────────────────────────
echo ""
echo ">>> [3/4] Signal in top-PCs (risk test: should HURT) ..."
for SEED in "${SEEDS[@]}"; do
    run_one configs/toy_signal_high_variance.yaml "${SEED}"
done

# ─── mixed dataset ────────────────────────────────────────────────────────────
echo ""
echo ">>> [4/4] Mixed dataset (spurious correlation) ..."
for SEED in "${SEEDS[@]}"; do
    run_one configs/toy_mixed.yaml "${SEED}"
done

# ─── drop_topk ─────────────────────────────────────────────────────────────────
if [[ "${MODE}" == "all" || "${MODE}" == "sweep" ]]; then
    echo ""
    echo ">>> [5] drop_topk mode ..."
    for SEED in "${SEEDS[@]}"; do
        run_one configs/toy_drop_topk.yaml "${SEED}"
    done

    echo ""
    echo ">>> [6] drop_all_pc_weighted mode ..."
    for SEED in "${SEEDS[@]}"; do
        run_one configs/toy_drop_all_weighted.yaml "${SEED}"
    done
fi

# ─── momentum ablation sweep ──────────────────────────────────────────────────
if [[ "${MODE}" == "sweep" || "${MODE}" == "all" ]]; then
    echo ""
    echo ">>> [7] Momentum ablation sweep ..."
    for MOM in 0.9 0.99 0.995; do
        for SEED in 42 1; do
            run_one configs/toy_momentum_ablation.yaml "${SEED}" "--pca_momentum ${MOM} --out_dir outputs/mom_ablation_m${MOM}"
        done
    done

    echo ""
    echo ">>> [8] top_k sweep (Dataset B + attenuate) ..."
    for K in 2 4 8 16; do
        for SEED in 42 1; do
            run_one configs/toy_nuisance_high_variance.yaml "${SEED}" \
                "--pca_top_k ${K} --out_dir outputs/topk_sweep_k${K}"
        done
    done

    echo ""
    echo ">>> [9] alpha sweep (Dataset B + attenuate) ..."
    for ALPHA in 0.1 0.3 0.5 1.0; do
        for SEED in 42 1; do
            run_one configs/toy_nuisance_high_variance.yaml "${SEED}" \
                "--pca_alpha ${ALPHA} --out_dir outputs/alpha_sweep_a${ALPHA}"
        done
    done

    echo ""
    echo ">>> [10] drop_prob sweep (Dataset B + drop_topk) ..."
    for P in 0.05 0.1 0.2 0.5; do
        for SEED in 42 1; do
            run_one configs/toy_drop_topk.yaml "${SEED}" \
                "--pca_drop_prob ${P} --out_dir outputs/drop_prob_sweep_p${P}"
        done
    done
fi

# ─── summary ──────────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  All experiments done."
echo "  Results in: $(pwd)/outputs/"
echo ""
echo "  Quick summary of all runs:"
python3 - <<'PYEOF'
import json, glob, os
rows = []
for f in sorted(glob.glob("outputs/**/summary.json", recursive=True)):
    try:
        d = json.load(open(f))
        rows.append({
            "name":         d.get("name", "?")[:40],
            "test_acc":     f"{d.get('test_acc', 0):.4f}",
            "best_val_acc": f"{d.get('best_val_acc', 0):.4f}",
            "config":       os.path.dirname(f).split("outputs/")[-1],
        })
    except Exception:
        pass
if rows:
    w = max(len(r["name"]) for r in rows) + 2
    header = f"{'Name':<{w}} {'Test Acc':>10} {'BestVal':>10}  Path"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<{w}} {r['test_acc']:>10} {r['best_val_acc']:>10}  {r['config']}")
else:
    print("  No summary.json files found yet.")
PYEOF
echo "================================================================"
