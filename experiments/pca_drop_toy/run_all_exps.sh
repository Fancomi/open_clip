#!/bin/bash
# 全量实验脚本：核心对比 + 消融 sweep
# 直接用绝对路径，无需 source activate

PY=/root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/python3
WORK=/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip/experiments/pca_drop_toy
cd ${WORK}

run() {
    local TAG=$1 CFG=$2 SEED=$3 EXTRA=$4 OUTDIR=$5
    ${PY} train.py --config configs/${CFG}.yaml \
        --seed ${SEED} --epochs 80 \
        --out_dir outputs/${OUTDIR} \
        ${EXTRA} 2>&1 \
        | grep -E "^(Ep 80|Test|===)" \
        | sed "s/^/[${TAG} s${SEED}] /"
}

echo "======= CORE EXPERIMENTS ======="
for SEED in 42 1 2; do
  run "b2_base"     b2_baseline       ${SEED} "" b2_baseline
  run "b2_att"      b2_attenuate      ${SEED} "" b2_attenuate
  run "b2_drop"     b2_drop_topk      ${SEED} "" b2_drop_topk
  run "b2_regdrop"  b2_regular_dropout ${SEED} "" b2_regular_dropout
  run "c_base"      c_baseline        ${SEED} "" c_baseline
  run "c_att"       c_attenuate       ${SEED} "" c_attenuate
  run "c_drop"      c_drop_topk       ${SEED} "" c_drop_topk
  run "e_base"      e_baseline        ${SEED} "" e_baseline
  run "e_att"       e_attenuate       ${SEED} "" e_attenuate
  run "a_base"      a_baseline        ${SEED} "" a_baseline
  run "a_att"       a_attenuate       ${SEED} "" a_attenuate
done
echo "======= CORE DONE ======="

echo "======= MOMENTUM ABLATION ======="
for MOM in 0.9 0.99 0.995; do
  for SEED in 42 1 2; do
    run "mom${MOM}" b2_attenuate ${SEED} "--pca_momentum ${MOM}" ablation_momentum_m${MOM}
  done
done
echo "======= MOMENTUM DONE ======="

echo "======= TOP_K SWEEP ======="
for K in 2 4 8 12 16; do
  for SEED in 42 1 2; do
    run "k${K}" b2_attenuate ${SEED} "--pca_top_k ${K}" ablation_topk_k${K}
  done
done
echo "======= TOP_K DONE ======="

echo "======= ALPHA SWEEP ======="
for ALPHA in 0.1 0.3 0.5 0.7 1.0; do
  for SEED in 42 1 2; do
    run "a${ALPHA}" b2_attenuate ${SEED} "--pca_alpha ${ALPHA}" ablation_alpha_a${ALPHA}
  done
done
echo "======= ALPHA DONE ======="

echo "======= DROP_PROB SWEEP ======="
for P in 0.05 0.1 0.2 0.3 0.5; do
  for SEED in 42 1 2; do
    run "p${P}" b2_drop_topk ${SEED} "--pca_drop_prob ${P}" ablation_droprob_p${P}
  done
done
echo "======= DROP_PROB DONE ======="

echo "ALL EXPERIMENTS COMPLETE"
