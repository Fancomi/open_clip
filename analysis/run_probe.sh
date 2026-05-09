# bash analysis/probe.sh coco                    # COCO — 有缓存则直接出图
# bash analysis/probe.sh cc3m                    # CC3M — 有缓存则直接出图
# bash analysis/probe.sh overlap                 # COCO vs CC3M 分布重合

# # crop_probe: 仅支持 COCO（tsv 模式，图片文件可直接访问）
# # wds/CC3M 图片存于 tar，PIL 无法打开，不支持
# bash analysis/probe.sh crop_probe              # 默认用 COCO feature cache

# bash analysis/probe.sh anisotropy coco  # 各向异性指标（秒级）
# bash analysis/probe.sh anisotropy cc3m

# bash analysis/probe.sh layers dinov3
# bash analysis/probe.sh layers pe_core
# bash analysis/probe.sh layers siglip2
# bash analysis/probe.sh layers eupe


# bash analysis/probe.sh probe_full logs  # --rerun  (旧 wmc_ 系列)
# bash analysis/probe.sh log_parse wmc_

bash analysis/probe.sh probe_full logs/20260508_0_ft_book
bash analysis/probe.sh log_parse ft_ --logs-dir logs/20260508_0_ft_book --plot-dir analysis/research/plots/book_run

bash analysis/probe.sh probe_full logs/20260508_0_ft_cc3m
bash analysis/probe.sh log_parse ft_ --logs-dir logs/20260508_0_ft_cc3m --plot-dir analysis/research/plots/cc3m_run


# python3 analysis/modality_gap.py \
#     --probe logs/pe_dinov3_sigreg_cls_probe/probe/step_001740.npz \
#     --split proj_features \
#     --out   analysis/research/modality_gap_baseline.json
