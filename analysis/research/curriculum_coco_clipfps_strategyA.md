# COCO FPS Curriculum 策略A扫描：epoch0 teacher-init, 后续随机

> 日期：2026-05-24  
> 实验脚本：`experiments/curriculum_coco_clipfps_A.sh`  
> 对比实验：`experiments/curriculum_coco_clipparadigm.sh`（策略B）、`analysis/research/curriculum_learning.md`（策略C）

---

## 背景

在 COCO CLIP-paradigm FPS curriculum 实验（`curriculum_clip_paradigm_coco.md`）中，我们完成了**策略B**（每epoch都使用同一frozen外部teacher进行FPS排序）的全部扫描，覆盖7个预训练模型（siglip2, datacomp, dfn2b, eva02, laion2b, metaclip, pe_core_always）。

此前已对 pe_core teacher 测试了三种策略：

| 策略 | 说明 | Tag（pe_core teacher） | i2t R@1 |
|------|------|----------------------|---------|
| **A** | epoch0 用外部teacher FPS排序，epoch1+ 恢复**随机采样** | `fps_pe_e0_random` | 0.0174 |
| **B** | 每epoch都使用同一**frozen** teacher排序 | `fps_pe_frozen_all` | 0.0152 |
| **C** | epoch0 用外部teacher，epoch1+ 用**当前模型自身**特征排序 | 历史实验（curriculum_learning.md） | — |

baseline 为 0.0136。对 pe_core teacher，**策略A (+28%) > 策略B (+12%)**，说明epoch0 teacher初始化本身提供了关键的几何先验，而持续使用frozen teacher并不优于一次性冲击后恢复随机。

**本实验的问题：** 策略B已对所有CLIP-paradigm teacher完成扫描，但策略A（epoch0-only）仅测了pe_core一个teacher。策略A在其他teacher上是否也一致优于策略B？ 能否找到跨teacher一致的最优策略？

## 目的

1. 对所有CLIP-paradigm teacher（siglip2, datacomp, dfn2b, eva02, laion2b, metaclip）补充策略A扫描
2. 对比同一teacher下策略A vs 策略B，验证"一次性teacher冲击 > 持续frozen"这一规律是否普适
3. 确认fps正序 vs fpsrev反序在策略A下的方向性规律是否与策略B一致

## 方法

### 三种策略的代码映射

策略差异由 `--curriculum-init` 和 `--curriculum-epochs` 两个参数共同决定：

```python
# curriculum.py: _extract_feature_block
def _extract_feature_block(model, paths, preprocess, device, args, epoch):
    init = args.curriculum_init
    if init == 'pe_core' and epoch == 0:
        return _extract_with_pe_core(...)        # 策略A/C: epoch0用pe_core
    if init in _EXTERNAL_CLIPS or init == 'random_init':
        return _extract_with_open_clip(init, ...)  # 策略B: 每epoch frozen外部
    return _extract_with_self(...)               # 策略C的epoch1+: 用自身特征

# main.py
if cur_epochs == 0 or epoch < cur_epochs:
    apply_curriculum(...)     # 策略A: cur_epochs=1 → 只epoch0有效
else:
    restore_default_order(...)  # 策略A在epoch1+恢复随机
```

**策略A**（本实验）：`--curriculum-init {teacher} --curriculum-epochs 1`
- epoch 0：加载外部teacher模型提取全量特征，计算FPS排序
- epoch 1+：恢复原始随机DistributedSampler（curriculum不再介入）
- 效果：teacher几何先验提供一次性冲击，之后训练自由演化

**策略B**（对比基准，已完成）：`--curriculum-init {teacher_always or external_name}`（`curriculum-epochs` 默认0=全程）
- 每epoch都重新加载frozen teacher提取特征并排序
- 效果：每epoch都在teacher定义的几何上进行FPS采样

### 训练配置

与 `curriculum_clip_paradigm_coco.md` 对齐：

| 配置项 | 值 |
|--------|---|
| Model | PE-Core-B-16-dinov3 |
| Train | COCO `clip_train_dedup.tsv` (~82K samples) |
| Val | COCO Karpathy 5-caption |
| Loss | SigLIP + SIGReg(cls, w=1e-4) |
| Optimizer | Muon, lr=3.4e-4, muon_lr=0.01 |
| Epochs | 20, warmup 42 steps |
| Batch | 512×8=4096 |

### Smoke 测试

执行 `SMOKE=1 bash experiments/curriculum_coco_clipfps_A.sh`，在 karpathy_1cap (~5000 samples, ~1 step) 上验证所有12个run的 train+eval 流程，**全部通过**（12/12 eval 输出正常，curriculum 排序日志、probe hook、retrieval metrics 均正常）。

## 实验矩阵

策略A × 6 teacher × 2 方向 = **12 个 run**

| Tag | Direction | Teacher Init | 对比策略B run |
|-----|-----------|-------------|--------------|
| fps_siglip2_e0 | fps | siglip2 | fps_siglip2 (0.0186, +37%) |
| fpsrev_siglip2_e0 | fps_reverse | siglip2 | fpsrev_siglip2 (0.0174, +28%) |
| fps_datacomp_e0 | fps | datacomp | fps_datacomp (0.0170, +25%) |
| fpsrev_datacomp_e0 | fps_reverse | datacomp | fpsrev_datacomp (0.0164, +21%) |
| fps_dfn2b_e0 | fps | dfn2b | fps_dfn2b (0.0138, +1%) |
| fpsrev_dfn2b_e0 | fps_reverse | dfn2b | fpsrev_dfn2b (0.0174, +28%) |
| fps_eva02_e0 | fps | eva02 | fps_eva02 (0.0148, +9%) |
| fpsrev_eva02_e0 | fps_reverse | eva02 | fpsrev_eva02 (0.0166, +22%) |
| fps_laion2b_e0 | fps | laion2b | fps_laion2b (0.0174, +28%) |
| fpsrev_laion2b_e0 | fps_reverse | laion2b | fpsrev_laion2b (0.0168, +24%) |
| fps_metaclip_e0 | fps | metaclip | fps_metaclip (0.0162, +19%) |
| fpsrev_metaclip_e0 | fps_reverse | metaclip | fpsrev_metaclip (0.0178, +31%) |

**baseline (random):** i2t R@1 = 0.0136

已知策略A对pe_core teacher：fps_pe_e0_random = 0.0174 (+28%) > fps_pe_frozen_all = 0.0152 (+12%)

## 效果

策略A × 6 teacher × 2方向全部完成（20 epoch），对比策略B（来自 `curriculum_clip_paradigm_coco.md`）：

| Tag | i2t R@1 | t2i R@1 | vs baseline | 策略B同配置 | A vs B |
|-----|---------|---------|-------------|------------|--------|
| **fps_eva02_e0** | **0.0210** | 0.0144 | **+54%** | 0.0148 (+9%) | **A +62pp** |
| fpsrev_eva02_e0 | 0.0176 | 0.0154 | +29% | 0.0166 (+22%) | A +10pp |
| fps_metaclip_e0 | 0.0166 | 0.0144 | +22% | 0.0162 (+19%) | A +4pp |
| fpsrev_dfn2b_e0 | 0.0164 | 0.0143 | +21% | 0.0174 (+28%) | B +10pp |
| fpsrev_siglip2_e0 | 0.0158 | 0.0146 | +16% | 0.0174 (+28%) | B +16pp |
| fps_laion2b_e0 | 0.0158 | 0.0141 | +16% | 0.0174 (+28%) | B +16pp |
| fps_dfn2b_e0 | 0.0156 | 0.0128 | +15% | 0.0138 (+1%) | A +18pp |
| fpsrev_datacomp_e0 | 0.0156 | 0.0135 | +15% | 0.0164 (+21%) | B +8pp |
| fpsrev_laion2b_e0 | 0.0154 | 0.0135 | +13% | 0.0168 (+24%) | B +14pp |
| fpsrev_metaclip_e0 | 0.0160 | 0.0144 | +18% | 0.0178 (+31%) | B +18pp |
| fps_siglip2_e0 | 0.0148 | 0.0149 | +9% | 0.0186 (+37%) | B +38pp |
| fps_datacomp_e0 | 0.0120 | 0.0143 | -12% | 0.0170 (+25%) | B +50pp |

**baseline:** i2t R@1 = 0.0136，t2i R@1 = （策略B实验中约 0.0100）

A > B 的情况：eva02 fps/fpsrev、dfn2b fps、metaclip fps（4/12）  
B > A 的情况：siglip2 fps/fpsrev、datacomp fps/fpsrev、dfn2b fpsrev、laion2b fps/fpsrev、metaclip fpsrev（8/12）

## 分析

### 1. 策略A不普遍优于策略B

pe_core teacher 下 A (+28%) > B (+12%) 的规律**未能推广**到其他 teacher。整体计分 A:B = 4:8，策略B在多数teacher+方向组合下仍更强。

**pe_core 的特殊性**：pe_core_always 和 pe_core_e0_random 的差距（+12% vs +28%）可能与 pe_core 特征空间的"洋葱结构"（`fps_convergence_pecore.md`中记录的独特几何：中心密集、密度动态范围5.6x）有关——在这种结构下，持续排序带来的重复几何偏差会消耗收益，一次性冲击反而最优。其他 CLIP 模型的特征分布更均匀（密度动态范围仅2x），持续排序可以稳定地提供有效信号。

### 2. fps_eva02_e0 是跨所有实验的最强配置

| 实验范围 | 最强单 run | i2t R@1 | vs baseline |
|---------|-----------|---------|-------------|
| 本实验（策略A） | fps_eva02_e0 | **0.0210** | **+54%** |
| 策略B扫描 | fps_siglip2 | 0.0186 | +37% |
| pe_core 对照 | fps_pe_e0_random | 0.0174 | +28% |
| 历史 COCO 实验（策略C） | fps_reverse_dinov3 | 0.0202 | +16% |

eva02 + fps + 策略A 的组合跑出了所有 curriculum 实验中最高分。EVA02 与策略A 的协同效应显著：策略B下 eva02_fps 仅 +9%，而策略A下爆发至 +54%（差异 62pp）。这说明 EVA02 特征空间的几何结构非常适合做**一次性**FPS冲击，但不适合反复使用——反复排序可能在其均匀分布+边缘簇的结构上产生负向归纳偏差。

### 3. dfn2b 的 fps 方向异常被部分修复

策略B 下 dfn2b 有显著的方向不对称：fps +1%（几乎无效），fpsrev +28%。本实验中：
- fps_dfn2b_e0：+15%（策略A显著好于策略B的 +1%）
- fpsrev_dfn2b_e0：+21%（策略A略低于策略B的 +28%）

说明 dfn2b fps 方向在策略B下的近零收益**与持续排序机制耦合**，而非 dfn2b 特征空间的内在性质。一次性冲击下，dfn2b fps 方向同样有效。

### 4. datacomp fps 方向在策略A下跌负

fps_datacomp_e0 = 0.0120（-12% vs baseline），是所有非 random_init 配置中唯一跌破 baseline 的。策略B下 fps_datacomp = 0.0170（+25%）。这一反转说明：持续的 datacomp fps 几何约束对训练有正向效果，而仅 epoch0 一次性冲击不仅无益反而有害，可能因为 datacomp 特征空间的 FPS 排序与随机训练的预热节奏产生了干扰。

### 5. 三策略总结

| 策略 | 机制 | 整体表现 | 最优配置 |
|------|------|---------|---------|
| **A**（epoch0冲击+随机） | teacher 几何一次性初始化采样顺序 | 结果分化大，有高峰也有低谷 | eva02+fps = **0.0210** |
| **B**（全程frozen） | 每epoch持续在teacher几何上采样 | 稳定正向，整体胜率高 | siglip2+fps = 0.0186 |
| **C**（epoch0冲击+自身） | teacher初始化，后续随训练自适应 | COCO小实验下 dinov3 最优，CC3M下 pe_core 最优 | dinov3+fpsrev = 0.0202（COCO）|

**结论：没有跨teacher普适的最优策略**。策略的优劣与 teacher 特征空间的几何性质紧密耦合：
- pe_core（洋葱结构，5.6x密度动态范围）：策略A最优
- eva02（均匀+边缘簇，但FPS冲击协同强）：策略A爆发
- siglip2/laion2b/metaclip（均匀分布）：策略B稳健更优
- datacomp（fps方向对策略A敏感）：策略B安全，策略A有风险

**最优绝对性能：** `fps_eva02_e0`（策略A）= 0.0210，是目前所有 COCO curriculum 实验中的新高分。
