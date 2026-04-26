# 分析工具现状与后续方向

> 更新日期：2026-04-26  
> 文档范围：`analysis/` 包的当前能力、接口说明、实验结果摘要、后续方向

---

## 一、当前能力全景

```
analysis/
├── models.py          # 6 模型统一加载接口
├── extractors.py      # CLS 特征提取（cache-first）
├── metrics.py         # 各向异性指标计算
├── viz.py             # 所有可视化函数
├── pipeline.py        # 5 种分析 pipeline 模式
├── layer_probe.py     # 逐层特征 hook 提取 + 各向异性 + PCA 图
├── pca_demo.py        # patch-level PCA 可视化
├── run.py             # CLI 入口（analysis/probe.sh 调用）
└── probe.sh           # 一键 shell 入口
```

### 1.1 probe.sh 支持的模式

| 命令 | 功能 |
|------|------|
| `bash analysis/probe.sh coco` | COCO 全模型 CLS 特征提取 + 散点图 + 各向异性 |
| `bash analysis/probe.sh cc3m` | CC3M(WDS) 同上 |
| `bash analysis/probe.sh overlap` | COCO vs CC3M 分布重合，双层绘图（COCO on top / CC3M on top） |
| `bash analysis/probe.sh epochs <probe_dir>` | 读取 `step_*.npz` 或 `epoch_*.npz` → GIF 动画 + trajectory.png + aniso_evolution.png |
| `bash analysis/probe.sh anisotropy [coco\|cc3m]` | 快速出各向异性指标 bar chart + 特征值谱 |
| `bash analysis/probe.sh layers <model>` | 逐层 hook 探针（dinov3/pe_core/siglip2/eupe） |

### 1.2 训练侧 step-granularity probe

在 `src/open_clip_train/` 中新增/修改：

- `params.py`：`--probe-freq-steps N` — 每 N 个 optimizer step 触发一次 probe
- `probe_hook.py`：`run_probe(..., step=N)` — 按 step 命名为 `step_XXXXXX.npz`
- `train.py`：在训练内层循环末尾检查 `(step+1) % probe_freq_steps == 0`

**CC3M 推荐配置**（每 epoch 4 次 probe）：
```bash
# CC3M epoch = 2,883,584 samples / batch_size / world_size
# 例：batch=256, 8 GPU → steps_per_epoch ≈ 1407
--probe-freq-steps 352   # 约 1407 // 4
```

---

## 二、逐层特征分析实验结果（DINOv3-B/16, COCO 500 样本）

### 2.1 各层 CLS 特征各向异性变化

| Layer | EffRank | AvgCos | top4%  | 解读 |
|-------|---------|--------|--------|------|
| 0     | 8.1     | 0.996  | 77.4%  | patch embed 输出：极度各向异性，几乎 1D |
| 1     | 16.9    | 0.990  | 52.7%  | 早期 attn：维度快速展开 |
| 4     | 35.5    | 0.947  | 42.7%  | 注意力开始捕获局部结构 |
| 7     | 49.0    | 0.908  | 35.2%  | 中层：平台期 |
| 10    | 80.4    | 0.777  | 24.8%  | 语义跳变：EffRank 大幅上升 |
| **11**| **171.6**| **0.243**| **10.8%**| 最终层：Simplex 结构完成 |

**核心发现**：DINOv3 的 Simplex 结构主要在**最后 1-2 层形成**（layer 10→11 的 EffRank 从 80→172，AvgCos 从 0.78→0.24）。前 10 层特征空间是"逐渐解缠绕的低秩流形"，并非 Simplex。这说明 DINO 的 centering+sharpening 损失只在最终层输出上施加约束，深层才形成语义原型结构。

### 2.2 意义

1. **微调友好层**：layer 7-9（EffRank 平台期）是语义信息最"稳定"的层，适合作为下游任务特征提取点
2. **蒸馏目标层选择**：若目标是复现 Simplex 结构，应以 layer 11 特征为蒸馏目标；若目标是保留中间语义表示，应用 layer 8-10
3. **与 RADIO 对比**：RADIO 的 top4%=40% 说明其特征结构更像 DINOv3 layer 3-5 的状态，多教师蒸馏将"深层 Simplex"抹平回到"中层低秩流形"

---

## 三、Epoch/Step Anisotropy 演化（CC3M 训练，10 epoch）

`bash analysis/probe.sh epochs <probe_dir>` 现在输出：

1. **`step_evolution.gif` / `epoch_evolution.gif`**：PCA 散点动画，每帧一个 checkpoint
2. **`trajectory.png`**：100 个随机样本的轨迹图（o=起点，*=终点）
3. **`aniso_evolution.png`**：6 个关键指标折线图（EffRank / StableR / AvgCos / StdCos / top4% / top10%）

---

## 四、后续方向

### 方向 1：Step-granularity probe 验证 Simplex 形成时机

**问题**：Simplex 结构（高 EffRank, 低 AvgCos）是在训练的哪个阶段形成的？是平滑过渡还是突然跳变？

**实验**：启动新训练时加 `--probe-freq-steps <steps_per_epoch//4>`，用 `bash analysis/probe.sh epochs <probe_dir>` 生成 aniso_evolution.gif，观察 AvgCos 从高到低的跳变点。

**预期**：结合已知的 DINOv3 论文中 teacher temp warmup（前 30 epoch 从低温到高温），预期 AvgCos 下降（Simplex 成型）发生在 teacher temp 达到稳定值附近。

### 方向 2：逐层 anisotropy 对比多个模型

当前只测了 DINOv3。扩展到 PE-Core / EUPE / RADIO，对比"哪层开始出现语义结构"：

```bash
bash analysis/probe.sh layers pe_core
bash analysis/probe.sh layers eupe
```

预期 RADIO 没有最终的 Simplex 跳变，EUPE 跳变幅度介于 DINOv3 和 RADIO 之间。

### 方向 3：patch token 逐层 PCA 可视化

`layer_probe.py` 当前提取 CLS token。`--token patch_mean` 可切换为 patch 均值特征，但更有价值的是保留完整 patch grid 并做 PCA RGB 可视化（类似 `pca_demo.py`）——每层一张 PCA 图，观察语义分割边界在哪层出现。

实现路径：在 `_extract_patch_grid` 中返回 `(B, H, W, D)` 然后在 `plot_layer_pca_grid` 中对每层 patches 做 PCA RGB，类似 `pca_demo.py` 的 `_pca_rgb`。

### 方向 4：Anisotropy 与线性探针精度的相关性

收集多个 checkpoint 的 (EffRank, top4%, AvgCos) 三元组，与对应 epoch 的 ImageNet linear probe Top-1 精度做散点图，量化各向同性指标对下游精度的预测力。

**假设**：EffRank 与 linear probe 精度正相关；但存在阈值效应——EffRank > 200 后精度不再提升（信息已充分展开）。

### 方向 5：RADIO top-4 PC 语义解释

RADIO top4% = 40%，对每个 PC 取 COCO 中投影值最高/最低的各 20 张图，验证 PC 是否对应可解释语义轴（前景/背景、室内/室外、人物/场景...）。

实现：在 `run_anisotropy` 中加 `--explain-pcs` 开关，读取 COCO 图像路径，取 top/bottom 各 N 张拼接保存。

### 方向 6：CC3M 上重测逐层 anisotropy

当前 layer_probe 使用 COCO（5000 样本，80 类）。CC3M 语义更多样，预期 DINOv3 的 StdCos 在最终层会显著高于 COCO 上的 0.066（Simplex 顶点更多、更均匀分布）。

```bash
bash analysis/probe.sh layers dinov3 --data <cc3m_tsv> --out-dir analysis/layer_probe_cc3m
```

---

## 五、接口速查

### layer_probe.py

```bash
python3 -m analysis.layer_probe \
    --model dinov3|pe_core|siglip2|eupe \
    --data  <tsv_path>              # filepath/caption TSV
    --out-dir <dir>                 # 输出目录
    --max-samples 2000              # 默认 2000（各向异性已足够）
    --token cls|patch_mean          # 特征类型
    --batch-size 64
    --force                         # 重新提取（忽略缓存）
# 输出：
#   <model>_layers.npz             # 缓存的逐层特征
#   <model>_layer_anisotropy.png   # 各层各向异性折线图
#   <model>_layer_pca.png          # 各层 PCA 散点图
```

### 训练侧 step probe

```bash
python -m open_clip_train.main \
    ... \
    --probe-data  /path/to/karpathy_1cap.tsv \
    --probe-dir   logs/<run>/checkpoints/probe \
    --probe-freq-steps 352   # CC3M: steps_per_epoch // 4
```

### 分析 step probe 结果

```bash
bash analysis/probe.sh epochs logs/<run>/checkpoints/probe
# 输出：
#   plots/step_evolution.gif       # PCA 散点动画
#   plots/trajectory.png           # 样本轨迹
#   plots/aniso_evolution.png      # 各向异性时序曲线
```
