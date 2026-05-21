# Reverse-LiT 实验设计

## 研究问题

LiT (Locked-image Tuning) 验证了"预训练图像塔作为锚点，从头训文本塔"的有效性。反过来行不行？
- **预训练文本塔能否作为语义锚点，引导从头训练的图像塔学习视觉-语言对齐？**
- 在frozen text encoder上加trainable MLP head能否改善效果？
- Tu-Ti (tune text + from-scratch image) 配置是否优于纯frozen？

## 背景

| 配置名 | 出处 | Image | Text | 效果 |
|--------|------|-------|------|------|
| Tu-Tu | Standard CLIP | scratch | scratch | baseline |
| Li-Tu | LiT (Google 2022) | pretrained locked | scratch tuned | **best** |
| Li-Ti | LiT ablation | pretrained locked | pretrained tuned | 接近Li-Tu |
| Tu-Ti | LiT ablation | scratch tuned | pretrained tuned | 比Tu-Tu好但远不如Li-Tu |

**LiT的core insight:** 预训练图像模型已有判别性语义聚类，文本只需学对齐——任务简单。
**反向的挑战:** 预训练文本空间几何未必适合视觉对齐；image从scratch学负担更重。

## 本实验贡献

1. 在OpenCLIP框架下系统复现reverse-LiT全配置矩阵
2. 引入MLP bridge (非线性text projection)，缓解frozen text空间几何不友好问题
3. 用PE-Core和SigLIP2两个模型族对比，验证结论泛化性

## 实验配置

| ID | 名称 | Text Tower | Image Tower | 训练参数 |
|----|------|-----------|-------------|----------|
| D1 | Scratch Baseline | random, tuned | random, tuned | 全部 |
| A | Reverse-LiT | pretrained, frozen | random, tuned | image + logit |
| B | R-LiT + MLP | pretrained, frozen + MLP head | random, tuned | image + MLP + logit |
| C | Tu-Ti | pretrained, tuned | random, tuned | 全部 |
| C2 | Partial R-LiT | pretrained, last 4 layers tuned | random, tuned | image + text top4 + logit |
| D2 | Standard LiT | pretrained, tuned | pretrained, frozen | text + logit |

## 代码实现

### 新增CLI参数

| 参数 | 作用 |
|------|------|
| `--pretrained-text-path PATH` | 加载text tower权重（支持full CLIP checkpoint，自动strip `text.` prefix） |
| `--text-proj-type mlp` | 将text projection替换为2层MLP (width→4×width→output_dim) |

### MLP Bridge 架构

```
text_encoder(frozen) → [1024] → Linear(1024,4096) → GELU → LN(4096) → Linear(4096,1024) → [1024]
                                                                                              ↕ contrastive
image_encoder(scratch) → [1024] ─────────────────────────────────────────────────────────────→ [1024]
```

PE-Core MLP: ~8.4M params | SigLIP2 MLP: ~4.7M params

### 工作流

1. `create_model()` 随机初始化整个模型（不传`--pretrained`）
2. `pretrained_text_path` 从full checkpoint提取text权重加载到`model.text`
3. 如果`--text-proj-type mlp`：删除原始text_projection，注册MLP Sequential
4. `--lock-text`冻结text tower全部参数
5. 如果有MLP：显式unfreeze MLP参数（lock会冻结它）
6. Optimizer只包含requires_grad=True的参数

## 训练设置

| 项目 | 值 |
|------|-----|
| 数据 | CC3M WebDataset (2.9M pairs) |
| Eval | COCO karpathy 5-cap retrieval |
| GPU | 8 × A100, BS=512/GPU (GlobalBS=4096) |
| LR | 3.4e-4 (sqrt-scaled) |
| Optimizer | AdamW (β1=0.9, β2=0.95, ε=1e-6, wd=0.2) |
| Schedule | cosine, warmup=512 steps |
| Epochs | 10 |
| Precision | amp_bf16 |
| Loss | SigLIP (sigmoid contrastive) |

## 预期结果与假设

| 配置 | 预期 T2I R@1 | 理由 |
|------|:----------:|------|
| D2 (LiT) | **最高** | 预训练image features已经很好 |
| C (Tu-Ti) | 中等偏高 | pretrained text提供好的初始化，但双方都在动 |
| B (R-LiT+MLP) | 中等 | MLP让text空间更灵活，但仍受限于frozen text质量 |
| A (R-LiT) | 中等偏低 | frozen linear proj可能限制对齐空间 |
| D1 (Scratch) | 低 | 从零开始，CC3M数据量不够 |
| C2 (Partial) | 介于A和C之间 | 折中方案 |

## 运行

```bash
cd /root/paddlejob/workspace/env_run/penghaotian/vision_encoder/open_clip
bash experiments/reverse_lit.sh
```

## 文件改动清单

| 文件 | 改动 |
|------|------|
| `src/open_clip_train/params.py` | +`--pretrained-text-path`, +`--text-proj-type` |
| `src/open_clip_train/main.py` | 传参 + MLP替换 + unfreeze逻辑 |
| `src/open_clip/factory.py` | auto-strip `text.` prefix |
| `src/open_clip/transformer.py` | `isinstance(nn.Module)` dispatch + 移除assert |
| `src/open_clip/model.py` | `isinstance(nn.Module)` dispatch (CLIP类) |
| `experiments/reverse_lit.sh` | 训练脚本 |
