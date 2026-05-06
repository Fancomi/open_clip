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





for logdir in \
    "wmc_baseline_0506_1337"
    # "mgap_wm01_0505_1419"
    # "mgap_center_v2_m9_0504_1228"
    # "mgap_center_v2_0504_1228"
    # "mgap_center_0503_1729"
    # "mgap_gap001_0503_1729" 
    # "cc3m_pe_dinov3_dinov3_clip_probe_0430_1929" \
    # "cc3m_pe_dinov3_dinov3_probe_0430_1929" \
    # "cc3m_vit_probe_0430_1218" \
    # "cc3m_pe_dinov3_leproj_probe_0424_0119" \
    # "cc3m_pe_dinov3_leproj_muon_lr001_0429_1821" \
    # "cc3m_pe_dinov3_dinov3_muon_probe_0501_1042"
do
    bash analysis/probe.sh epochs "logs/${logdir}/checkpoints/probe"
    bash analysis/probe.sh pc_alignment "logs/${logdir}/checkpoints/probe"
done

# python3 analysis/modality_gap.py \
#     --probe logs/pe_dinov3_sigreg_cls_probe/probe/step_001740.npz \
#     --split proj_features \
#     --out   analysis/research/modality_gap_baseline.json  