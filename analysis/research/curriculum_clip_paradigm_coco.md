# COCO CLIP-paradigm FPS Curriculum

## 背景

CC3M sample-level 实验中，`fps + pe_core` 是当前最强配置；而 COCO 早期实验中 `fps_reverse + dinov3` 曾表现最好。这说明 FPS curriculum 的收益可能强依赖特征几何来源与数据规模。

本轮实验专门回答两个问题：
1. PE-Core FPS 的收益来自 epoch0 的一次性排序，还是需要每个 epoch 都使用 frozen PE-Core 几何排序？
2. 参考 `analysis/clip_fps_probe.py`，把不同 CLIP 预训练权重当作几何先验，比较同一采样策略在不同权重下是否出现共性规律。

## 方法

训练配置与 COCO curriculum 历史实验对齐：
- Train: COCO `clip_train_dedup.tsv`
- Val: COCO Karpathy 5-caption
- Model: `PE-Core-B-16-dinov3`
- Loss: SigLIP + SIGReg(cls, 1e-4)
- Optimizer: Muon, lr=3.4e-4, muon_lr=0.01
- Epochs: 20, batch size: 512×8=4096

新增实现：
- `--curriculum-epochs`: 控制 curriculum 只作用前 N 个 epoch；`0` 表示全程。
- `pe_core_always`: 每个 active epoch 都用 frozen PE-Core 特征排序。
- 更多 external CLIP init: `siglip2`, `datacomp`, `dfn2b`, `eva02`, `laion2b`, `metaclip`, `random_init`。
- epoch0-only 后恢复原始随机/DistributedSampler，避免 ordered sampler 残留。

## 实验矩阵

| Tag | Strategy | Init | Curriculum epochs | 目的 |
|-----|----------|------|-------------------|------|
| baseline | - | - | - | 随机顺序基线 |
| fps_pe_e0_random | fps | pe_core | 1 | 只测 epoch0 PE FPS 冲击 |
| fps_pe_frozen_all | fps | pe_core_always | all | 测 frozen PE FPS 全程排序 |
| fpsrev_pe_e0_random | fps_reverse | pe_core | 1 | PE 反序 epoch0 冲击 |
| fpsrev_pe_frozen_all | fps_reverse | pe_core_always | all | PE 反序全程排序 |
| fps_siglip2 / fpsrev_siglip2 | fps / fps_reverse | siglip2 | all | SigLIP2 几何正反序 |
| fps_datacomp / fpsrev_datacomp | fps / fps_reverse | datacomp | all | DataComp 几何正反序 |
| fps_dfn2b / fpsrev_dfn2b | fps / fps_reverse | dfn2b | all | DFN2B 几何正反序 |
| fps_eva02 / fpsrev_eva02 | fps / fps_reverse | eva02 | all | EVA02 几何正反序 |
| fps_laion2b / fpsrev_laion2b | fps / fps_reverse | laion2b | all | LAION2B 几何正反序 |
| fps_metaclip / fpsrev_metaclip | fps / fps_reverse | metaclip | all | MetaCLIP 几何正反序 |
| fps_randominit / fpsrev_randominit | fps / fps_reverse | random_init | all | 随机视觉几何正反序 |

## Smoke 测试

所有正式配置先用 `SMOKE=1 experiments/curriculum_coco_clipparadigm.sh` 跑 COCO Karpathy 1cap 训练 + Karpathy 5cap eval。

额外需要验证：`fps_pe_e0_random` 在 epoch1+ 会恢复随机 sampler。

## 效果

待实验完成后填写。

| Tag | best i2t R@1 | final i2t R@1 | final t2i R@1 | 备注 |
|-----|--------------|---------------|---------------|------|
| baseline | | | | |
| fps_pe_e0_random | | | | |
| fps_pe_frozen_all | | | | |
| fpsrev_pe_e0_random | | | | |
| fpsrev_pe_frozen_all | | | | |
| fps_siglip2 | | | | |
| fpsrev_siglip2 | | | | |
| fps_datacomp | | | | |
| fpsrev_datacomp | | | | |
| fps_dfn2b | | | | |
| fpsrev_dfn2b | | | | |
| fps_eva02 | | | | |
| fpsrev_eva02 | | | | |
| fps_laion2b | | | | |
| fpsrev_laion2b | | | | |
| fps_metaclip | | | | |
| fpsrev_metaclip | | | | |
| fps_randominit | | | | |
| fpsrev_randominit | | | | |

## 分析

待实验完成后填写。
