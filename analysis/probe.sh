#!/bin/bash
# Feature space analysis — lives in analysis/, runs from repo root
#
# ═══════════════════════════════════════════════════════════════════════════════
# 用法
# ═══════════════════════════════════════════════════════════════════════════════
#   bash analysis/probe.sh coco                    # COCO — 有缓存则直接出图
#   bash analysis/probe.sh cc3m                    # CC3M — 有缓存则直接出图
#   bash analysis/probe.sh epochs <probe_dir>      # epoch 演化
#   bash analysis/probe.sh overlap                 # COCO vs CC3M 分布重合
#   bash analysis/probe.sh anisotropy [coco|cc3m]  # 各向异性指标（秒级）
#
#   bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_leproj_probe_0424_0119/checkpoints/probe
#   bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_dinov3_probe_0424_1400/checkpoints/probe
#   bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_dinov3_probe_clip_0427_0146/checkpoints/probe
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
    *)
        echo "Usage:"
        echo "  bash analysis/probe.sh coco"
        echo "  bash analysis/probe.sh cc3m"
        echo "  bash analysis/probe.sh epochs <probe_dir>"
        echo "  bash analysis/probe.sh overlap"
        echo "  bash analysis/probe.sh anisotropy [coco|cc3m]"
        echo "  bash analysis/probe.sh layers <model>  (dinov3|pe_core|siglip2|eupe)"
        exit 1
        ;;
esac

echo "=== [probe] done ==="
