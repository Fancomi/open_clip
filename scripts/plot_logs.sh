source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate

LOG_DIR=/root/paddlejob/workspace/env_run/penghaotian/vision_encoders/open_clip/logs

# SigLIP对比图
python scripts/plot_logs.py --logs \
$LOG_DIR/quick_vit_siglip_0401_2231/out.log \
$LOG_DIR/quick_pe_cls_siglip_0401_2231/out.log \
$LOG_DIR/quick_pe_dinov3_siglip_0401_2231/out.log \
--names ViT PE-CLS PE-DinoV3 --output logs/compare_siglip_2231

# CLIP对比图
python scripts/plot_logs.py --logs \
$LOG_DIR/quick_vit_clip_0401_2231/out.log \
$LOG_DIR/quick_pe_cls_clip_0401_2231/out.log \
$LOG_DIR/quick_pe_dinov3_clip_0401_2231/out.log \
--names ViT PE-CLS PE-DinoV3 --output logs/compare_clip_2231