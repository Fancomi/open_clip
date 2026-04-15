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

# # 对比【部分】的le 和le proj
# python scripts/plot_logs.py --logs \
# $LOG_DIR/LR2e-4_epoch30/quick_dinov3_siglip_0403_1253/out.log \
# $LOG_DIR/LR2e-4_epoch30/quick_pe_dinov3_siglip_0403_0046/out.log \
# $LOG_DIR/quick_dinov3_siglip_lejepa_0406_1307/out.log \
# $LOG_DIR/quick_dinov3_siglip_lejepa_proj_0406_1307/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_0406_1246/out.log \
# $LOG_DIR/quick_pe_dinov3_siglip_lejepa_proj_0406_1307/out.log \
# --names dinov3 pe_dinov3 dino_lejepa dino_lejepa_proj pe_dinov3_lejepa pe_dinov3_lejepa_proj --output logs/compare_siglip_202604070956

# # 对比全部的le 和le proj
# python scripts/plot_logs.py --logs \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_dinov3_le_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_dinov3_leproj_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_cls_le_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_cls_leproj_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_dinov3_le_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_dinov3_leproj_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_vit_le_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_vit_leproj_0409_2205/out.log \
# --names dinov3_le dinov3_leproj pe_cls_le pe_cls_leproj pe_dinov3_le pe_dinov3_leproj vit_le vit_leproj --output logs/compare_siglip_202604070956

# # 对比全部的无le和有le
# python scripts/plot_logs.py --logs \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30/quick_vit_siglip_0403_0046/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30_lejepa_all/quick_vit_leproj_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30/quick_pe_cls_siglip_0403_0046/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30_lejepa_all/quick_pe_cls_le_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30/quick_pe_dinov3_siglip_0403_0046/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30_lejepa_all/quick_pe_dinov3_leproj_0409_2205/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30/quick_dinov3_siglip_0403_1253/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/LR2e-4_epoch30_lejepa_all/quick_dinov3_le_0409_2205/out.log \
#  --names vit vit_le pe_cls pe_cls_le pe_dinov3 pe_dinov3_le dinov3 dinov3_le --output logs/compare_siglip_202604101404

python scripts/plot_logs.py --logs \
/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip/logs/cc3m_20260414/cc3m_vit_le_0414_2322/out.log \
/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip/logs/cc3m_20260414/cc3m_vit_leproj_0414_2322/out.log \
/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip/logs/cc3m_20260414/cc3m_pe_dinov3_le_0414_2322/out.log \
/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip/logs/cc3m_20260414/cc3m_pe_dinov3_leproj_0414_2322/out.log \
/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip/logs/cc3m_20260414/cc3m_dinov3_le_0414_2322/out.log \
/root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip/logs/cc3m_20260414/cc3m_dinov3_leproj_0414_2322/out.log \
--names vit vit_le pe_dinov3 pe_dinov3_le dinov3 dinov3_le --output logs/compare_siglip_202604151801



# # 对比全部的无attnres和有attnres
# python scripts/plot_logs.py --logs \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_vit_0411_1146/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_vit_attnres2_0412_1802/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_cls_0411_1146/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_cls_attnres2_0412_1802/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_dinov3_0411_1146/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_pe_dinov3_attnres2_0412_1802/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_dinov3_0411_1146/out.log \
# /root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs/quick_dinov3_attnres2_0412_1802/out.log \
#   --names vit vit_attnres pe_cls pe_cls_attnres pe_dinov3 pe_dinov3_attnres dinov3 dinov3_attnres --output logs/compare_siglip_202604121720
