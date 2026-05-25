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

待实验完成后填写（20 epoch × 12 run，运行中）。

| Tag | i2t R@1 (ep19) | t2i R@1 (ep19) | vs baseline | vs 策略B同teacher同方向 |
|-----|----------------|----------------|-------------|----------------------|
| fps_siglip2_e0 | | | | vs 0.0186 |
| fpsrev_siglip2_e0 | | | | vs 0.0174 |
| fps_datacomp_e0 | | | | vs 0.0170 |
| fpsrev_datacomp_e0 | | | | vs 0.0164 |
| fps_dfn2b_e0 | | | | vs 0.0138 |
| fpsrev_dfn2b_e0 | | | | vs 0.0174 |
| fps_eva02_e0 | | | | vs 0.0148 |
| fpsrev_eva02_e0 | | | | vs 0.0166 |
| fps_laion2b_e0 | | | | vs 0.0174 |
| fpsrev_laion2b_e0 | | | | vs 0.0168 |
| fps_metaclip_e0 | | | | vs 0.0162 |
| fpsrev_metaclip_e0 | | | | vs 0.0178 |

## 分析

待实验完成后填写。

**核心问题（待验证）：**

1. **策略A是否普遍优于策略B？** pe_core teacher下A>B (+28% vs +12%)，若此规律在多数teacher下成立，说明epoch0 teacher冲击是关键机制，持续排序带来的是不必要的归纳偏差。

2. **哪个teacher配策略A最强？** 策略B下siglip2是最强teacher（fps_siglip2 +37%）。策略A下siglip2是否仍领先？还是某些teacher（如dfn2b，策略B的fps方向异常弱，fpsrev却很强）在策略A下会反转？

3. **fps vs fpsrev方向一致性：** 策略B下发现dfn2b在fps方向几乎无效（+1%），但fpsrev方向强（+28%）。这是因为dfn2b特征空间的FPS排序结构不同，还是与策略B的持续排序机制耦合？策略A下是否改变这一方向偏好？

4. **三策略总结：** 结合策略C（历史实验，`curriculum_learning.md`，只在COCO小实验数据上有结果）、策略B（本次全扫），策略A（本次扫），可以对三种teacher介入方式做完整结论。
