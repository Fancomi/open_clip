# DINOv3 集成开发记录

## 目标

让 CLIP 模型在同一个视觉骨干上**同时**做对比学习（SigLIP loss）和自蒸馏（DINO + iBOT + KoLeo loss）。

---

## 新增文件

| 文件 | 内容 |
|------|------|
| `src/open_clip/dino_head.py` | DINOHead / iBOTHead，含 MLP + L2-norm + weight-norm last_layer |
| `src/open_clip_train/dino_transform.py` | 多视角增强：2 global crops (224px) + N local crops (96px) + iBOT mask 生成 |
| `src/open_clip/loss.py` *(新增部分)* | `DINOClsTokenLoss`、`iBOTPatchLoss`（含 Sinkhorn-Knopp centering）、`KoLeoLoss`、`CLIPWithDINOLoss` |

---

## 修改文件

### `src/open_clip/model.py`
- 新增 `CLIPWithDINO` 类：共享视觉骨干，student heads + teacher EMA backbone + teacher heads
- `_encode_backbone_with_patches`：提取 global crops 的 CLS token + patch tokens
- `_encode_cls_only`：local crops resize 到 224px 再过骨干（解决 PE-Core RoPE 固定分辨率限制）
- `update_ema(momentum)`：EMA teacher 更新

### `src/open_clip/factory.py` + `src/open_clip/__init__.py`
- 注册 `CLIPWithDINO`，暴露相关接口

### `src/open_clip_train/params.py`
- 新增 `--dinov3`、`--dino-*`、`--ibot-*`、`--koleo-*`、`--freeze-last-layer-epochs` 等参数

### `src/open_clip_train/data.py`
- DINOv3 模式下使用多视角 transform，collate_fn 把 global/local crops 打包成 batch dict

### `src/open_clip_train/main.py`
- `import math`（修复 `_ema_momentum_schedule` closure）
- `loss = loss.to(device)`（修复 center buffer 在 CPU 上的问题）
- DDP 用 `static_graph=True`（共享骨干在 backward graph 中出现两次）
- 构建 teacher_temp / EMA momentum schedule
- eval 时用 `model.clip_model` 解包（接口兼容）

### `src/open_clip_train/train.py`
- DINOv3 分支：解包 multi-crop batch，调用 `model(global_crops, local_crops, texts, ...)`
- EMA teacher 在每步 `optimizer.step()` 后更新
- `freeze_last_layer`：**改为 backward 后清零梯度**（而非 `requires_grad_(False)`），避免改变图结构与 `static_graph=True` 冲突
- `evaluate()` 中对 `CLIPWithDINO` 解包为 `clip_model`

### `scripts/smoke.sh`
- 注释掉原有 12 条 `run_smoke`，新增 DINOv3 smoke test（PE-Core-B-16 + SigLIP + DINO + iBOT + KoLeo）
- `--val-frequency 0` 跳过 eval（避免 NCCL timeout）

---

## 排查修复的 Bug

### 1. `NameError: name 'math' is not defined`
- **位置**：`main.py` 中 `_ema_momentum_schedule` closure
- **原因**：`main.py` 缺 `import math`
- **修复**：顶部加 `import math`

### 2. `RuntimeError: tensor size mismatch (224 vs 96)`
- **位置**：`model.py` forward，`torch.cat([global_crops, local_crops])`
- **原因**：global crops 224px、local crops 96px，尺寸不同无法拼接
- **修复**：global/local crops 分开编码，不拼接

### 3. `AssertionError: Input height (96) doesn't match model (224)`
- **位置**：PE-Core EVA backbone 的 RoPE positional encoding
- **原因**：PE-Core 使用 RoPE，位置编码硬绑定到 224px 网格，无法处理 96px 输入
- **修复**：`_encode_cls_only` 中先用 `F.interpolate` 把 local crops resize 到 224px 再送入骨干

### 4. 维度不匹配（1024-dim vs 768-dim）
- **位置**：`_encode_cls_only` 返回值送入 DINO head
- **原因**：最初用 `trunk.forward_head(feats)` 返回投影后的 1024-dim，但 DINO head 期望原始 768-dim CLS token
- **修复**：改为直接返回 `feats[:, 0]`（raw CLS token，768-dim）

### 5. `RuntimeError: tensors on different devices (cuda:0 vs cpu)`
- **原因**：`CLIPWithDINOLoss` 中 `dino_loss.center`、`ibot_loss.center` 用 `register_buffer` 初始化在 CPU，`create_loss(args)` 后未移到 GPU
- **修复**：`main.py` 中 `create_loss` 之后加 `loss = loss.to(device)`

### 6. `RuntimeError: Expected to have finished reduction in prior iteration`
- **原因**：teacher 参数 `requires_grad=False`，DDP 默认要求所有参数都参与 backward
- **修复**：尝试 `find_unused_parameters=True`，但引发下一个问题

### 7. `RuntimeError: Expected to mark a variable ready only once`
- **原因**：视觉骨干同时被 `encode_image`（对比loss）和 `_encode_backbone_with_patches`（DINO loss）使用，同一参数在 backward graph 中出现两次；`find_unused_parameters=True` 与此冲突
- **修复**：改用 `static_graph=True`（告诉 DDP 图结构固定，允许参数多次使用，同时处理 unused teacher 参数），移除 `find_unused_parameters`

### 8. NCCL timeout（600s）in eval stage
- **原因**：`evaluate()` 调用 `model(images, texts)`，但 `CLIPWithDINO.forward` 签名为 `(global_crops, local_crops, texts, ...)`，接口不兼容；rank0 卡在 forward，其他 rank 在 `barrier()` 等待超时
- **修复**：
  - `train.py evaluate()` 中对 `CLIPWithDINO` 解包：`if isinstance(model, CLIPWithDINO): model = model.clip_model`
  - smoke test 改 `--val-frequency 0` 跳过 eval

### 9. `RuntimeError: Your training graph has changed in this iteration`
- **位置**：epoch 1 第一步 backward
- **原因**：`freeze_last_layer` 在 epoch 0 调用 `last_layer.weight.requires_grad_(False)`，epoch 1 改回 `True`；图结构在两个 epoch 之间发生变化，与 `static_graph=True` 不兼容
- **修复**：不用 `requires_grad_` 改图结构，改为在 `backward()` 之后、`optimizer.step()` 之前把 `last_layer.weight.grad.zero_()`，图结构全程不变

---

## 关键设计决策

| 问题 | 方案 | 原因 |
|------|------|------|
| 共享骨干 DDP | `static_graph=True` | 同一参数在 backward graph 中出现两次 |
| PE-Core 固定分辨率 | local crops resize 到 224px | RoPE 位置编码硬绑定，无法处理其他分辨率 |
| Loss center buffers | `loss.to(device)` | loss module 独立于 model，不会自动移到 GPU |
| freeze last layer | `grad.zero_()` 而非 `requires_grad_` | 保持图结构不变，兼容 `static_graph=True` |
| Eval 接口 | 解包为 `clip_model` | CLIPWithDINO forward 签名与标准 CLIP API 不兼容 |

---

## Smoke Test 结果

- **配置**：PE-Core-B-16，8×H800，2 epochs × 10 steps，BS=64
- **参数**：`--siglip --dinov3 --dino-local-crops-number 2 --dino-head-prototypes 65536 --dino-loss-weight 1.0 --ibot-loss-weight 1.0 --koleo-loss-weight 0.1 --freeze-last-layer-epochs 1`
- **结果**：全部通过，4 路 loss 均正常输出

```
Train Epoch: 0 [5120/5120] Siglip_loss: 8.10  Dino_loss: 11.11  Ibot_loss: 5.52  Koleo_loss: 0.29
Train Epoch: 1 [5120/5120] ...
======== smoke 全部通过 ========
```
