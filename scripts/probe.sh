#!/bin/bash
# Feature space analysis — 与 quick.sh 训练流程完全分离
#
# ═══════════════════════════════════════════════════════════════════════════════
# 用法
# ═══════════════════════════════════════════════════════════════════════════════
#   bash probe.sh pretrained              # COCO TSV，随时可跑，npz 缓存则跳过推理
#   bash probe.sh cc3m                    # CC3M WDS，100k subsample
#   bash probe.sh epochs <probe_dir>      # epoch 演化（需先跑训练）
#   bash probe.sh overlap                 # COCO vs CC3M 分布重合（需两边 cache）
#   bash probe.sh anisotropy [coco|cc3m]  # 各向异性指标（纯 cache 计算，秒级）
#
#   bash scripts/probe.sh epochs logs/cc3m_pe_dinov3_leproj_probe_0424_0119/checkpoints/probe
#   bash scripts/probe.sh epochs logs/cc3m_pe_dinov3_dinov3_probe_0424_1400/checkpoints/probe
#
# ═══════════════════════════════════════════════════════════════════════════════
# 输出文件说明（pretrained 模式）
# ═══════════════════════════════════════════════════════════════════════════════
#   pe_core_modality_gap.png    PE-Core 图文模态鸿沟 (image vs text)
#   siglip2_modality_gap.png    SigLIP2 图文模态鸿沟
#   tips_modality_gap.png       TIPSv2  图文模态鸿沟（TIPSv2 唯一含文本塔的新模型）
#                               注：RADIO / EUPE 为纯视觉模型，无文本塔
#   image_allmodels.png         6 模型图像特征对比 + FPS 锚点跨模型追踪
#                               - 最右一列 "FPS anchors"：5 个在 PE 空间中
#                                 最远距离采样的样本，用 ★◆▲■✚ 标记
#                               - 同一符号在每行代表同一张图像，跨模型
#                                 的位置变化揭示各模型对"多样性"的不同理解
#   anisotropy.png              各向异性指标对比图
#
# ═══════════════════════════════════════════════════════════════════════════════
# 各向异性指标详解
# ═══════════════════════════════════════════════════════════════════════════════
#
# 背景：l2 归一化特征分布在超球面上，各向异性 (anisotropy) 描述特征
#       是否集中在少数方向（锥形）还是均匀散布（球形）。
#       各向异性越强 → 特征空间"塌陷"到低维子空间 → 检索/分类区分度下降。
#
#   ┌─────────────────────────┬────────────────────────────────────────────────┐
#   │ 指标                    │ 定义与解读                                      │
#   ├─────────────────────────┼────────────────────────────────────────────────┤
#   │ Effective Rank (eff_r)  │ exp(H(λ/Σλ))，λ 为协方差矩阵特征值            │
#   │                         │ = 特征值分布的指数熵，值域 [1, D]               │
#   │                         │ 越高 → 更多方向被均匀使用 → 越 isotropic       │
#   │                         │ 来源：Roy & Vetterli, 2007                      │
#   ├─────────────────────────┼────────────────────────────────────────────────┤
#   │ Participation Ratio(PR) │ (Σλ)² / (D · Σλ²)，值域 (0, 1]                │
#   │                         │ 直觉：若只有 k 个方向等强，PR ≈ k/D             │
#   │                         │ 越高 → 越 isotropic；< 0.1 通常已严重塌陷      │
#   ├─────────────────────────┼────────────────────────────────────────────────┤
#   │ Avg Cosine Sim (cos)    │ 随机 2000 对特征的均值余弦相似度                │
#   │                         │ 越低（趋近 0）→ 特征越分散 → 越 isotropic      │
#   │                         │ 接近 1 说明特征几乎全在一个方向（退化）         │
#   ├─────────────────────────┼────────────────────────────────────────────────┤
#   │ pct_var_top{k}          │ 前 k 个 PC 解释的方差百分比                    │
#   │                         │ top4 / top10 / top50 / top100                   │
#   │                         │ 若 top4 > 80% → 特征几乎降至 4 维 → 高各向异性 │
#   │                         │ 正常模型 top4 约 20-40%，top100 约 60-80%       │
#   ├─────────────────────────┼────────────────────────────────────────────────┤
#   │ Eigenvalue spectrum     │ 归一化特征值（λ_i/Σλ）随 PC 序号的衰减曲线     │
#   │                         │ 快速衰减 → 各向异性强；缓慢衰减 → isotropic    │
#   └─────────────────────────┴────────────────────────────────────────────────┘
#
# ═══════════════════════════════════════════════════════════════════════════════
# COCO vs CC3M 分布区分现象解读（overlap 模式）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 现象：overlap 图中可见两类模型行为不同：
#
#   A. DINOv3 / PE-Core / TIPSv2 — COCO 与 CC3M 沿 PC2 有明显分离
#      原因：这些模型的特征空间中编码了一定的"数据集指纹"——
#            COCO 是精选有标注的中心物体图，CC3M 是网络爬取的图文对，
#            内容分布差异被模型保留在低频主成分方向。
#            PE-Core 以 EVA-CLIP 蒸馏为主，DINOv3 以 SSL 判别为主，
#            TIPSv2 对图像细粒度信息保留更多，因而对域偏移更敏感。
#
#   B. RADIO / EUPE / SigLIP2 — COCO 与 CC3M 在 PC1-PC2 几乎完全重合
#      原因：
#        · SigLIP2 以 sigmoid 损失在超大规模多样数据上训练，
#          学到了域不变的语义表示。
#        · RADIO 蒸馏自 SigLIP2-g + DINOv3-7B + SAM3，
#          多教师蒸馏的平均效应使域偏移被抵消。
#        · EUPE 再次从 RADIO 蒸馏，继承了其域不变性。
#
#   哪种现象更好？
#   ► 对于迁移学习、零样本检索、跨域泛化：B 更好。
#     特征不依赖数据来源 → 训练在 COCO 上的下游头可无缝泛化到 CC3M 或其他域。
#   ► 但两类模型的 COCO-CC3M 间距远小于图文模态鸿沟（~10-50x），
#     说明所有模型都能在域间"桥接"，A 类只是保留了更细粒度的域信息。
#   ► A 类模型的域区分若主要在 PC2（解释方差较小的方向），
#     说明 PC1（语义主轴）仍是域不变的，实际影响有限。
#
# ═══════════════════════════════════════════════════════════════════════════════

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
