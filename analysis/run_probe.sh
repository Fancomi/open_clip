# bash analysis/probe.sh coco                    # COCO — 有缓存则直接出图
# bash analysis/probe.sh cc3m                    # CC3M — 有缓存则直接出图
# bash analysis/probe.sh overlap                 # COCO vs CC3M 分布重合

# bash analysis/probe.sh anisotropy coco  # 各向异性指标（秒级）
# bash analysis/probe.sh anisotropy cc3m

# bash analysis/probe.sh layers dinov3
# bash analysis/probe.sh layers pe_core
# bash analysis/probe.sh layers siglip2
# bash analysis/probe.sh layers eupe

# bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_leproj_probe_0424_0119/checkpoints/probe
# bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_dinov3_probe_0424_1400/checkpoints/probe

# bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_dinov3_probe_clip_0427_0146/checkpoints/probe 
# bash analysis/probe.sh pc_alignment logs/cc3m_pe_dinov3_dinov3_probe_clip_0427_0146/checkpoints/probe 

# bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_leproj_muon_lr002_0428_1149/checkpoints/probe
# bash analysis/probe.sh pc_alignment logs/cc3m_pe_dinov3_leproj_muon_lr002_0428_1149/checkpoints/probe

# bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_leproj_probe_0424_0119/checkpoints/probe
# bash analysis/probe.sh pc_alignment logs/cc3m_pe_dinov3_leproj_probe_0424_0119/checkpoints/probe

bash analysis/probe.sh epochs logs/cc3m_pe_dinov3_leproj_muon_lr0005_0429_1354/checkpoints/probe
bash analysis/probe.sh pc_alignment logs/cc3m_pe_dinov3_leproj_muon_lr0005_0429_1354/checkpoints/probe

