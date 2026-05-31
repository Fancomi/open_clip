# COCO FPS Curriculum 综合实验报告

> 整合自：`curriculum_learning.md`（策略C原始实验）、`curriculum_clip_paradigm_coco.md`（策略A/B pe_core & 外部CLIP）、`curriculum_coco_clipfps_strategyA.md`（策略A外部CLIP全扫）  
> 补充实验：`experiments/curriculum_coco_missing.sh`（策略C pe_core/dinov3/外部CLIP，策略A dinov3）  
> 最终更新：2026-05-27

---

## 一、背景

CLIP-style 对比学习中训练顺序通常完全随机。本系列实验探索：**在 PE-Core-B-16-dinov3 + SigLIP + SIGReg + Muon 最优配置上，通过控制样本呈现顺序（FPS curriculum）能否提升 COCO retrieval 性能。**

核心机制：每 epoch 开始前，用某个模型的特征空间对全量训练集做 FPS（Farthest Point Sampling）排序，然后按此顺序遍历。FPS 正序 = 多样性优先（每批覆盖尽量远的点），FPS 反序 = 冗余/中心优先。

---

## 二、三种 Teacher 介入策略

| 策略 | 参数 | epoch0 特征来源 | epoch1+ 特征来源 |
|------|------|----------------|----------------|
| **A** | `--curriculum-init {T} --curriculum-epochs 1` | 外部 teacher T | 恢复**随机**采样 |
| **B** | `--curriculum-init {T_always 或 external_T}` | 外部 teacher T | 继续使用同一 frozen T |
| **C** | `--curriculum-init {T}` 或 `--curriculum-init {T}_c` | 外部 teacher T | 切换为**当前模型自身**特征 |

代码实现（`curriculum.py: _extract_feature_block`）：
- pe_core / dinov3：天然支持策略A（加 `--curriculum-epochs 1`）和策略C（不加，epoch0后自动切self）
- pe_core_always：在 `_EXTERNAL_CLIPS` 中，总是用 frozen pe_core = 策略B
- siglip2 / datacomp / dfn2b / eva02 / laion2b / metaclip：默认在 `_EXTERNAL_CLIPS` 中 = 策略B；加 `_c` 后缀（如 `siglip2_c`）= 策略C

---

## 三、训练配置

| 项目 | 值 |
|------|----|
| Model | PE-Core-B-16-dinov3 |
| Train | COCO `clip_train_dedup.tsv` (~82K) |
| Val | COCO Karpathy 5-caption split (5K×5) |
| Loss | SigLIP + SIGReg(cls, w=1e-4) |
| Optimizer | Muon, lr=3.4e-4, muon_lr=0.01 |
| Epochs | 20, warmup 42 steps |
| Batch | 512×8=4096 |
| val_frequency | 1（新实验）/ 2（旧实验，最后 eval 在 ep18）|

> **注意两批实验的 baseline 差异**：  
> - 新实验 baseline（ep19）= **0.0136**（实验脚本 `curriculum_coco_clipparadigm.sh` / `_A.sh` / `_missing.sh`）  
> - 旧实验 baseline（ep18）= **0.0162**（实验脚本 `curriculum_coco.sh`，val_frequency=2）  
> 两批数据在同批内部可做比较；跨批比较时使用相对 % vs 各自 baseline。

---

## 四、实验矩阵与结果

### 4.1 按策略汇总

#### 策略 A：epoch0 用外部 teacher，epoch1+ 恢复随机

**新实验 baseline = 0.0136**

| Teacher | fps i2t R@1 | vs base | fpsrev i2t R@1 | vs base |
|---------|------------|---------|---------------|---------|
| pe_core | 0.0174 | +28% | 0.0144 | +6% |
| dinov3 | 0.0170 | +25% | **0.0182** | **+34%** |
| siglip2 | 0.0148 | +9% | 0.0158 | +16% |
| datacomp | 0.0120 | -12% | 0.0156 | +15% |
| dfn2b | 0.0156 | +15% | 0.0164 | +21% |
| **eva02** | **0.0210** | **+54%** | 0.0176 | +29% |
| laion2b | 0.0158 | +16% | 0.0154 | +13% |
| metaclip | 0.0166 | +22% | 0.0160 | +18% |

#### 策略 B：每 epoch 都使用同一 frozen teacher

**新实验 baseline = 0.0136**

| Teacher | fps i2t R@1 | vs base | fpsrev i2t R@1 | vs base |
|---------|------------|---------|---------------|---------|
| pe_core | 0.0152 | +12% | 0.0140 | +3% |
| dinov3 | N/A¹ | — | N/A¹ | — |
| **siglip2** | **0.0186** | **+37%** | 0.0174 | +28% |
| datacomp | 0.0170 | +25% | 0.0164 | +21% |
| dfn2b | 0.0138 | +1% | 0.0174 | +28% |
| eva02 | 0.0148 | +9% | 0.0166 | +22% |
| laion2b | 0.0174 | +28% | 0.0168 | +24% |
| metaclip | 0.0162 | +19% | 0.0178 | +31% |
| random_init | 0.0010 | -93% | 0.0012 | -91% |

¹ dinov3 不在 `_EXTERNAL_CLIPS`，无法直接做策略B（需添加 `dinov3_always`，未实现）

#### 策略 C：epoch0 用外部 teacher，epoch1+ 用自身特征

**旧实验（val_freq=2）baseline ep18 = 0.0162（i2t R@1 best ep10 = 0.0174）**

| Teacher | fps i2t R@1 (ep18) | vs base | fpsrev i2t R@1 (ep18) | vs base |
|---------|-------------------|---------|----------------------|---------|
| pe_core（旧） | 0.0152 | -6% | 0.0122 | -25% |
| dinov3（旧） | 0.0156 | -4% | 0.0156 | -4% |
| self（旧） | 0.0180 | +11% | 0.0174 | +7% |

**新实验 baseline = 0.0136（直接可与策略A/B比较）**

| Teacher | fps i2t R@1 | vs base | fpsrev i2t R@1 | vs base |
|---------|------------|---------|---------------|---------|
| pe_core | 0.0148 | +9% | 0.0142 | +4% |
| dinov3 | 0.0144 | +6% | 0.0162 | +19% |
| **siglip2** | **0.0194** | **+43%** | 0.0116 | -15% |
| datacomp | 0.0178 | +31% | 0.0178 | +31% |
| **dfn2b** | 0.0156 | +15% | **0.0194** | **+43%** |
| eva02 | 0.0152 | +12% | 0.0160 | +18% |
| laion2b | 0.0146 | +7% | 0.0154 | +13% |
| metaclip | 0.0180 | +32% | 0.0152 | +12% |

---

### 4.2 按 Teacher 汇总（仅新实验，baseline=0.0136）

#### pe_core teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | **0.0174** (+28%) | 0.0152 (+12%) | 0.0148 (+9%) |
| fpsrev | 0.0144 (+6%) | 0.0140 (+3%) | 0.0142 (+4%) |

#### dinov3 teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | 0.0170 (+25%) | N/A | 0.0144 (+6%) |
| fpsrev | **0.0182 (+34%)** | N/A | 0.0162 (+19%) |

#### siglip2 teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | 0.0148 (+9%) | **0.0186 (+37%)** | **0.0194 (+43%)** |
| fpsrev | 0.0158 (+16%) | 0.0174 (+28%) | 0.0116 (-15%) |

#### datacomp teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | 0.0120 (-12%) | **0.0170 (+25%)** | 0.0178 (+31%) |
| fpsrev | 0.0156 (+15%) | 0.0164 (+21%) | 0.0178 (+31%) |

#### dfn2b teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | 0.0156 (+15%) | 0.0138 (+1%) | 0.0156 (+15%) |
| fpsrev | 0.0164 (+21%) | **0.0174 (+28%)** | **0.0194 (+43%)** |

#### eva02 teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | **0.0210 (+54%)** | 0.0148 (+9%) | 0.0152 (+12%) |
| fpsrev | 0.0176 (+29%) | 0.0166 (+22%) | 0.0160 (+18%) |

#### laion2b teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | 0.0158 (+16%) | **0.0174 (+28%)** | 0.0146 (+7%) |
| fpsrev | 0.0154 (+13%) | 0.0168 (+24%) | 0.0154 (+13%) |

#### metaclip teacher

| 方向 | 策略A | 策略B | 策略C |
|------|-------|-------|-------|
| fps | 0.0166 (+22%) | 0.0162 (+19%) | **0.0180 (+32%)** |
| fpsrev | 0.0160 (+18%) | **0.0178 (+31%)** | 0.0152 (+12%) |

---

### 4.3 全局排行（新实验，baseline=0.0136）

| Rank | 策略 | Teacher | 方向 | i2t R@1 | vs baseline |
|------|------|---------|------|---------|-------------|
| 1 | **A** | **eva02** | **fps** | **0.0210** | **+54%** |
| 2 | B | siglip2 | fps | 0.0186 | +37% |
| 3 | **C** | **siglip2** | **fps** | **0.0194** | **+43%** |
| 3 | **C** | **dfn2b** | **fpsrev** | **0.0194** | **+43%** |
| 5 | A | eva02 | fpsrev | 0.0176 | +29% |
| 5 | B | siglip2 | fpsrev | 0.0174 | +28% |
| 5 | A | pe_core | fps | 0.0174 | +28% |
| 5 | B | laion2b | fps | 0.0174 | +28% |
| 5 | B | dfn2b | fpsrev | 0.0174 | +28% |
| 10 | A | dinov3 | fpsrev | 0.0182 | +34% |
| 11 | B | metaclip | fpsrev | 0.0178 | +31% |
| 11 | C | datacomp | fps | 0.0178 | +31% |
| 11 | C | datacomp | fpsrev | 0.0178 | +31% |
| — | baseline | — | — | 0.0136 | — |

---

## 五、分析

### 5.1 策略间规律

**没有跨 teacher 普适的最优策略。** A vs B 的优劣强依赖 teacher 选择：

| Teacher | 最优策略 | 最优 i2t R@1 | 特征空间特性 |
|---------|---------|-------------|------------|
| eva02 | **A (fps)** | 0.0210 (+54%) | 均匀分布+边缘簇，一次性冲击协同效应强 |
| pe_core | **A (fps)** | 0.0174 (+28%) | 洋葱结构（5.6x密度动态范围），持续排序带来负向归纳偏差 |
| siglip2 | **B (fps)** | 0.0186 (+37%) | 均匀分布，持续 frozen 排序稳定正向 |
| dfn2b | fps方向A优，fpsrev方向B优 | 0.0174 / 0.0164 | 策略B下 fps 方向近零是持续排序的副作用 |
| datacomp | **B** | 0.0170 (+25%) | 策略A fps 方向跌负（-12%），需持续约束 |
| laion2b / metaclip | **B** | 0.0174 / 0.0178 | 均匀分布，持续排序稳健 |

### 5.2 FPS 方向规律（fps vs fpsrev）

| Teacher | 策略B fps vs fpsrev | 策略A fps vs fpsrev |
|---------|--------------------|--------------------|
| pe_core | fps (0.0152) > fpsrev (0.0140) | fps (0.0174) > fpsrev (0.0144) |
| siglip2 | fps (0.0186) > fpsrev (0.0174) | fps (0.0148) < fpsrev (0.0158) |
| dfn2b | **fps (0.0138) << fpsrev (0.0174)** | fps (0.0156) ≈ fpsrev (0.0164) |
| eva02 | fps (0.0148) < fpsrev (0.0166) | **fps (0.0210) >> fpsrev (0.0176)** |
| laion2b | fps (0.0174) > fpsrev (0.0168) | fps (0.0158) > fpsrev (0.0154) |
| metaclip | fps (0.0162) < fpsrev (0.0178) | fps (0.0166) > fpsrev (0.0160) |

**dfn2b 方向反转现象**：策略B下 fps 方向几乎无效（+1%），策略A下恢复正常（+15%），证明 dfn2b fps 在策略B下的问题是持续排序机制的副作用，不是特征空间本身的性质。

**eva02 方向大反转**：策略B下 fpsrev > fps，策略A下 fps 爆发（+54%），远超 fpsrev（+29%）。EVA02 的特征几何在一次性 fps 冲击下有特殊协同效应。

### 5.3 策略C分析

策略C（epoch0 外部 teacher 冲击，epoch1+ 自身特征自适应）完整新实验结果如下（baseline=0.0136）：

**有效的组合：**

| Teacher | fps | fpsrev | 规律 |
|---------|-----|--------|------|
| siglip2 | **+43% (0.0194)** | -15% (0.0116) | 强方向依赖，fps方向超越策略B |
| dfn2b | +15% (0.0156) | **+43% (0.0194)** | 强方向依赖，fpsrev方向大幅提升 |
| datacomp | +31% (0.0178) | +31% (0.0178) | 双向稳健，两方向对称 |
| metaclip | +32% (0.0180) | +12% (0.0152) | fps方向有效，超过策略A/B的fps表现 |
| dinov3 | +6% (0.0144) | +19% (0.0162) | 轻微正向 |

**表现较差的组合：**

| Teacher | fps | fpsrev | 分析 |
|---------|-----|--------|------|
| pe_core | +9% (0.0148) | +4% (0.0142) | 低于策略A的 +28%，初始冲击被自适应抵消 |
| eva02 | +12% (0.0152) | +18% (0.0160) | 远低于策略A的 +54%，eva02 的协同效应在持续自适应中消失 |
| laion2b | +7% (0.0146) | +13% (0.0154) | 低于策略B的 +28%，持续 frozen 更有效 |

**跨策略对比结论：**

1. **策略C不是万能的"折中"**：siglip2-fps 和 dfn2b-fpsrev 在策略C下达到 +43%，实际上超过了对应策略A/B的最优值（siglip2-B-fps=+37%，dfn2b-B-fpsrev=+28%），说明策略C的"初始冲击+自适应"组合对特定 teacher 有额外增益。

2. **eva02 是策略A的专属**：eva02 在策略A-fps 下爆发 +54%，但策略C下仅 +12%，自适应过程破坏了 EVA02 特征几何与训练动态的一次性协同效应。

3. **siglip2 的方向大反转**：策略C下 fps(+43%) vs fpsrev(-15%) 形成极端对比；而策略B下 fps(+37%) vs fpsrev(+28%) 相对均衡。说明自适应排序放大了 siglip2 特征空间中的方向不对称性。

4. **datacomp 的自适应优势**：策略A-fps=-12%（最差），策略B-fps=+25%，策略C-fps=+31%。随着训练推进，当前模型自身特征比 frozen datacomp 更适合作为 datacomp 类特征的排序基准。

### 5.4 关键结论

1. **全实验最高分**：`fps_eva02_e0`（策略A）= 0.0210（+54% vs baseline），是所有 COCO curriculum 实验的新高
2. **最稳健策略**：策略B + siglip2/laion2b/metaclip 等均匀分布 teacher，对 fps/fpsrev 方向均有显著提升
3. **策略A适合**：pe_core（洋葱结构）和 eva02（边缘簇+一次性冲击协同）
4. **策略A风险**：datacomp fps 方向跌负（-12%），使用前需smoke验证
5. **策略C的惊喜**：siglip2-fps(+43%) 和 dfn2b-fpsrev(+43%) 在策略C下超越各自策略A/B最优值，是三策略矩阵中并列第2的成绩（仅次于 eva02-A-fps）
6. **策略选择指南**：eva02→策略A-fps；siglip2→策略C-fps 或 策略B-fps；dfn2b→策略C-fpsrev；datacomp/metaclip→策略C；pe_core/laion2b→策略B

---

## 六、实验脚本索引

| 脚本 | 内容 |
|------|------|
| `experiments/curriculum_coco.sh` | 策略C原始实验（旧config，val_freq=2）|
| `experiments/curriculum_coco_clipparadigm.sh` | 策略A pe_core + 策略B all teachers |
| `experiments/curriculum_coco_clipfps_A.sh` | 策略A 外部CLIP teacher扫描 |
| `experiments/curriculum_coco_missing.sh` | 策略C pe_core/dinov3/外部CLIP + 策略A dinov3 |

## 七、其他策略实验参考

原始 COCO 实验（旧config）还测试了 density 和 curvature 排序策略（结果见 `curriculum_learning.md`），整体结论：

- **density_high**（高密度/简单样本优先）：一致负向，所有 init 均低于 baseline
- **density_low**（低密度/困难样本优先）：轻微正向，self init 下 +3.7%（ep18）
- **curvature_high/low**：中性到轻微正向，pe_core init 下 chi +3.7%，clo_self +8.6%

FPS 策略整体优于 density/curvature，成为本系列实验的核心。
