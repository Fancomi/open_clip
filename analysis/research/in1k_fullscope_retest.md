# IN-1k 全量口径重测：所有历史组归一到标准协议

*创建: 2026-08-25 | 更新: 2026-08-26（补 regw0.5 / regw1.0 / regw2.0，全项目 IN-1k 新高 34.00）
| 起因: 历史所有 IN-1k 数字都是 100 类 × 20 图 = 2000 图子集，与 open_clip 标准协议（1000 类 × 50 图 = 50000 图）不可混比*

## 0. 结论速览

1. **旧 IN-1k 列全部作废**。子集口径系统性虚高（E_firstbox 子集 25.50% → 全量 27.15%，
   方向甚至不一致，因为子集只用前 100 个 wnid，类别难度分布有偏）。
2. **region 监督是目前唯一在 IN-1k 上稳定为正的改动**：8 组 `region_weight=0.2`
   全部落在 27.13~27.66，相对 gt_base 23.41 是 **+3.7~+4.2 点**，约 12σ，铁实。
   **补充（08-26）：把 `region_weight` 开到 0.5 是 30.45（+7.04，22σ），
   开到 1.0 是 **32.07（+8.66，27σ）**，全项目最高。**
   这个旋钮此前从未被扫过，见 [[region_01_supervision]] §5.7。
3. **C3（sharedsc）的 "+7.20" 是子集伪影**。全量下 C1 27.50 / C2 27.44 / C3 27.29 / C4 27.13，
   四者极差 0.37 点 ≈ 1 个 2σ，**互相不可分辨**。此前"共享 scale 更优"的结论撤回。
4. **PCM 全系在 IN-1k 上是负的**（18.54~21.53，即 −1.9~−4.9 点）。此前 PCM 只报了
   COCO/Urban 两轴，IN-1k 的代价一直没被记账。`pcm_weight=0.2` 仍是 PCM 内最优（21.53），
   但它对短模板分类的损害是真实的、超噪声地板 6σ 的。
5. **纯 dense 长描述训练直接摧毁短模板 zero-shot**：dense_256 全量 top1 **0.96%**
   （随机基线 0.10%，即仅 10× 随机）。这不是评测 bug，口径已核对为 projective（训练一致）。
6. **projective 与 standard 在 IN-1k 上打平**：gt_std（standard 口径评测）23.34 vs
   gt_base（projective）23.41，差 0.07 < 2σ。几何模式的收益不在分类轴上。
7. **H 组 box 对齐裁剪增强无效**：cropaug 27.38 vs 无 cropaug 的 C1 27.50，差 −0.12 < 2σ。

## 1. 全量口径噪声地板

同配方（gt_base，projective，`clip_train_gt.tsv`，10 epoch）四次独立运行，epoch_10：

| 运行 | top1 |
|---|---|
| gt_base_0806 | 23.41 |
| gt_base_0811 | 23.27 |
| gt_s1 | 23.64 |
| gt_s2 | 23.55 |

均值 23.47，样本 σ = 0.162 → **2σ = 0.32 点**。

判读规则：**IN-1k 全量差值 < 0.32 点视为不可分辨**，不得写成结论。
（旧子集口径下测得的 2σ = 0.35 点数值相近纯属巧合，两者不可互换使用。）

## 2. 全量总表（全部 epoch_10，50000 图 / 1000 类 / 80 官方模板）

Δ 列相对 gt_base_0806（23.41）。评测口径已逐个与训练 `params.txt` 的 `neg_mode` 对齐。

### 2.1 基线与数据配方

| 模型 | 口径 | top1 | top5 | Δtop1 |
|---|---|---|---|---|
| gt_base_0806 | projective | **23.41** | 42.20 | — |
| gt_base_0811 | projective | 23.27 | 41.91 | −0.14 |
| gt_s1 | projective | 23.64 | 42.41 | +0.23 |
| gt_s2 | projective | 23.55 | 42.45 | +0.14 |
| gt_std | standard | 23.34 | 41.81 | −0.07 |
| gt_resize (A′) | projective | 22.52 | 40.95 | −0.89 |
| mix50 | projective | 19.06 | 36.41 | −4.35 |
| dense_256 | projective | 0.96 | 2.51 | −22.45 |

### 2.2 PCM（Long-CLIP 主成分匹配）

| 模型 | 口径 | top1 | top5 | Δtop1 |
|---|---|---|---|---|
| pcmw0.2d32 | projective | **21.53** | 41.41 | −1.88 |
| pcmw1.0d64 | projective | 21.44 | 40.58 | −1.97 |
| pcmw0.5d32 | projective | 21.20 | 40.08 | −2.21 |
| pcmw0.3d32 | projective | 21.11 | 40.70 | −2.30 |
| pcmw0.1d32 | projective | 20.88 | 40.41 | −2.53 |
| pcm32_projective | projective | 20.20 | 38.83 | −3.21 |
| pcmw1.0d16 | projective | 18.54 | 37.36 | −4.87 |
| pcm32_standard | standard | 18.23 | 35.49 | −5.18 |

### 2.3 Region 监督（FG-CLIP 风格）

| 模型 | 口径 | top1 | top5 | Δtop1 |
|---|---|---|---|---|
| **regw2.0k12（W=2.0）** | projective | **34.00** | **59.07** | **+10.59** |
| **regw1.0k12（W=1.0）** | projective | **32.07** | **56.82** | **+8.66** |
| **regw0.5k12（W=0.5）** | projective | **30.45** | **53.78** | **+7.04** |
| regvd-hard (E3) | projective | **27.66** | 49.60 | **+4.25** |
| regw0.2k12 (C1) | projective | 27.50 | 49.94 | +4.09 |
| regw0.2k12-gather (C2) | projective | 27.44 | 50.01 | +4.03 |
| regw0.2k12-cropaug (H) | projective | 27.38 | 49.57 | +3.97 |
| regw0.2k12-sharedsc (C3) | projective | 27.29 | 49.67 | +3.88 |
| regvd-soft (E2) | projective | 27.18 | 49.78 | +3.77 |
| regvd-firstbox (E1) | projective | 27.15 | 49.54 | +3.74 |
| regw0.2k4 (C4) | projective | 27.13 | 49.18 | +3.72 |

8 个 `region_weight=0.2` 变体的组内极差 0.53 点（E3 27.66 − C4 27.13），略超 2σ=0.32，
只支持"E3 ≳ C4"这一条弱结论；其余两两之差均在噪声内。
**region 这一族的价值在"是否加 region"，不在这些变体。**

⚠️ **补充（2026-08-26）**：上面那句话在 8 个 W=0.2 变体内部成立，但**不能推广到
`region_weight` 本身**。把权重从 0.2 开到 0.5 拿到 **30.45（+2.95 相对 C1，9.2× 2σ）**、
再开到 1.0 拿到 **32.07（+4.57 相对 C1，14.3× 2σ）**、再开到 2.0 拿到
**34.00（+6.50 相对 C1，20.3× 2σ）** —— 单调上升、每跳都过 2σ，
是全项目 IN-1k 最高值，也是本表任何两个变体之差的 12 倍以上。
9 个区域组里前 8 个都用 0.2 是因为照搬 FG-CLIP 2，**这个旋钮此前从未被扫过**。
按等倍跨度（每次 ×2）看，IN-1k 的增量是 **+1.62（0.5→1.0）→ +1.93（1.0→2.0）**，
**在加速而非减速** → 最优点在 2.0 之外，W=4.0 在跑（见 [[region_01_supervision]] §6.2）。
详见 [[region_01_supervision]] §5.7。

## 3. 复现命令

```bash
source /root/paddlejob/workspace/env_run/penghaotian/envs/dino/bin/activate
export PYTHONPATH="./src:$PYTHONPATH"
# 全量已是默认值，不要再传 --in1k-classes / --in1k-per-class
CUDA_VISIBLE_DEVICES=0 python scripts/eval/eval_standard.py \
    --ckpt logs/<run>/checkpoints/epoch_10.pt --tag <run> \
    --neg-mode <与 logs/<run>/params.txt 的 neg_mode 一致> --num-workers 14
```

单模型 98 秒（H800，含建 classifier）。批处理驱动脚本见 `/tmp/in1k_full_retest.sh`。

## 4. 本次为跑通全量而做的两处工程改动

1. `eval_standard.py` 的 IN-1k 前向改用 `DataLoader(shuffle=False, num_workers=N)`。
   单线程 PIL 解码实测 90.7 img/s，全量 5 万张要 9.2 分钟且 GPU 空转；改后 98 秒。
   顺序与 transform 未变，逐样本结果等价（gt_base_0806 两次独立运行同为 23.41/42.20，
   说明评测是确定性的，可用于精确复现）。
2. `--in1k-classes` / `--in1k-per-class` 默认值改为 1000 / 50，并在输出行打
   `★全量★` 或 `⚠️子集(不可与全量混比)` 标记，防止子集数字再次流入对比表。

## 5. 本次踩到的两个新坑（已修，记录备查）

1. **口径错配又犯了一次**：驱动脚本只对 `pcm32_standard` 特判 standard 口径，漏了
   `gt_std`，导致它以 projective 评出 8.54%（真值 23.34%）。**教训：口径不要靠文件名
   猜，直接读 `logs/<run>/params.txt` 的 `neg_mode` 字段。**
2. **标签碰撞导致结果静默覆盖**：`gt_base_0806` 与 `gt_base_0811` 经 tag 裁剪后同名
   `gt_gt_base`，日志互相覆盖，23 个模型只落地 22 份结果。已用显式区分标签补测。

## 6. 相关文档

- `eval_protocol.md` — 四条铁律与口径绑定规则（本页是其第 4 节"样本量陷阱"的落地整改）
- `longclip_01_pcm.md` — PCM 路线结果主页，其 IN-1k 列已按本页作废重写
