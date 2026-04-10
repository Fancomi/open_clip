source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

LOG_DIR=/root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs

# # SigLIP对比图
# python scripts/plot_logs.py --logs \
# $LOG_DIR/quick_vit_siglip_0404_1514/out.log \
# $LOG_DIR/quick_pe_cls_siglip_0404_1514/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_0404_1514/out.log \
# $LOG_DIR/quick_dinov3_siglip_0404_1514/out.log \
# --names ViT PE-CLS PE-DinoV3 DinoV3  --output logs/compare_siglip_202604031148

# # CLIP对比图
# python scripts/plot_logs.py --logs \
# $LOG_DIR/quick_vit_clip_0404_1514/out.log \
# $LOG_DIR/quick_pe_cls_clip_0404_1514/out.log \
# $LOG_DIR/quick_pe_dinov3_clip_0404_1514/out.log \
# $LOG_DIR/quick_dinov3_clip_0404_1514/out.log \
# --names ViT PE-CLS PE-DinoV3 DinoV3  --output logs/compare_clip_202604031148


# # 不同lejepa的wight （1e-3到1e-4最好）
# python scripts/plot_logs.py --logs \
# $LOG_DIR/quick_dinov3_siglip_lejepa_0406_1307/out.log \
# $LOG_DIR/quick_dinov3_siglip_lejepa_proj_0406_1307/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_0406_1246/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_e-2_0406_2059/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_e-3_0406_2059/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_proj_0406_1307/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_proj_e-2_0406_2059/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_proj_e-3_0406_2059/out.log \
# --names dino_lejepa dino_lejepa_proj pe_dinov3_lejepa pe_dinov3_lejepa_e-2 pe_dinov3_lejepa_e-3 pe_dinov3_lejepa_proj pe_dinov3_lejepa_proj_e-2 pe_dinov3_lejepa_proj_e-3    --output logs/compare_siglip_202604070956


python scripts/plot_logs.py --logs \
$LOG_DIR/LR2e-4_epoch30/quick_dinov3_siglip_0403_1253/out.log \
$LOG_DIR/LR2e-4_epoch30/quick_pe_dinov3_siglip_0403_0046/out.log \
$LOG_DIR/quick_dinov3_siglip_lejepa_0406_1307/out.log \
$LOG_DIR/quick_dinov3_siglip_lejepa_proj_0406_1307/out.log \
$LOG_DIR/quick_pe_dinov3_siglip_lejepa_0406_1246/out.log \
$LOG_DIR/quick_pe_dinov3_siglip_lejepa_proj_0406_1307/out.log \
--names dinov3 pe_dinov3 dino_lejepa dino_lejepa_proj pe_dinov3_lejepa pe_dinov3_lejepa_proj --output logs/compare_siglip_202604070956

