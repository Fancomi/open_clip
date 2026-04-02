# PE-CLS vs DINOv3 Base 架构对比

## 共同配置（Base尺寸）

| 配置项 | PE-CLS | DINOv3 Base |
|--------|--------|-------------|
| patch_size | 16 | 16 |
| embed_dim | 768 | 768 |
| depth | 12 | 12 |
| num_heads | 12 | 12 |
| mlp_ratio | 4.0 | 4.0 (ffn_ratio) |
| Params | ~86.6M | ~86.6M |

## 关键架构差异

### 1. Normalization 位置

| 差异点 | PE-CLS (EvaBlock) | DINOv3 (SelfAttentionBlock) |
|--------|-------------------|------------------------------|
| Block norm顺序 | Pre-LN | Pre-LN |
| 实现 | `norm1 → attn → residual + ls1`<br>`norm2 → mlp → residual + ls2` | `norm1 → attn → residual + ls1`<br>`norm2 → mlp → residual + ls2` |
| **结论** | ✅ **一致** | ✅ **一致** |

### 2. Normalization 类型

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| Default norm | LayerNorm(eps=1e-5) | LayerNorm(eps=1e-6) |
| 可选norm | - | RMSNorm (via `norm_layer`参数) |
| **差异** | eps不同 | 支持RMSNorm |

### 3. LayerScale

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| LayerScale | ❌ 无 | ✅ 有（`init_values`参数控制） |
| 默认值 | - | None（默认不启用） |
| **差异** | **缺少LayerScale** | 可选启用 |

### 4. FFN 类型

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| Default FFN | Mlp (GELU activation) | Mlp (GELU activation) |
| 可选FFN | - | SwiGLU, SwiGLU32, SwiGLU64, SwiGLU128 |
| **差异** | 仅Mlp | 支持SwiGLU |

### 5. RoPE 实现

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| RoPE class | `RotaryEmbeddingCat` (timm) | `RopePositionEmbedding` (自定义) |
| grid_offset | 1.0 | - |
| grid_indexing | 'xy' | - |
| base | - | 100.0 (默认) |
| normalize_coords | - | "separate" |
| **差异** | timm实现，参数不同 | 自定义实现，更灵活 |

### 6. Pre-Transformer Norm

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| norm_pre | ✅ 有（`use_pre_transformer_norm=True`） | ❌ 无 |
| 位置 | patch_embed之后，blocks之前 | - |
| **差异** | **额外的前置norm** | 无 |

### 7. CLS Token

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| CLS token | ✅ 有 | ✅ 有 |
| Storage tokens | ❌ 无 | ✅ 可选（`n_storage_tokens`） |
| Untie norms | ❌ 无 | ✅ `untie_cls_and_patch_norms` |
| **差异** | 标准CLS | 支持registers和解耦norm |

### 8. Position Embedding

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| Type | RoPE only | RoPE only |
| ref_feat_shape | (14, 14) | 动态计算 |
| **差异** | 固定参考形状 | 动态 |

### 9. 初始化

| 差异点 | PE-CLS | DINOv3 |
|--------|--------|--------|
| init方法 | `use_deep_norm_init=False` | `init_weights_vit` |
| CLS token | - | `nn.init.normal_(std=0.02)` |
| **差异** | 默认初始化 | 显式trunc_normal初始化 |

## 总结：关键差异点

**DINOv3 相比 PE-CLS 的改进**：

1. **LayerScale**: 可选的layer scaling，提升训练稳定性
2. **SwiGLU FFN**: 更强的FFN变体（可选）
3. **RMSNorm**: 更高效的normalization（可选）
4. **Storage Tokens/Registers**: 支持额外的storage tokens
5. **Untie Norms**: CLS和patch tokens可以使用不同的norm
6. **动态RoPE**: 不依赖固定的ref_feat_shape
7. **初始化**: 显式的trunc_normal初始化

**PE-CLS 相比 DINOv3 的差异**：

1. **Pre-Transformer Norm**: 额外的norm_pre层
2. **RoPE参数**: grid_offset=1.0, grid_indexing='xy'
3. **Norm eps**: 1e-5 vs 1e-6

## 已实现：PE-Core-B-16-dinov3

**配置文件**: `src/open_clip/model_configs/PE-Core-B-16-dinov3.json`

**已实现特性**:
- ✅ **RMSNorm** (eps=1e-6) 替代 LayerNorm (eps=1e-5)
- ✅ **Pre-Transformer Norm** (use_pre_transformer_norm=True) - 保留以提升训练稳定性
- ✅ **LayerScale** (init_values=1e-4) - 更大的初始值加速收敛
- ✅ **标准 MLP FFN** (与 DINOv3 默认一致)
- ✅ **Storage tokens/Registers** (num_reg_tokens=4)

**未实现**:
- ⚠️ **Untie norms**: timm Eva 不支持，需要修改 forward 逻辑（优先级低）

**参数量对比**:
| 模型 | Visual Params | 差异 |
|------|---------------|------|
| PE-CLS | 86.6M | - |
| PE-DINOv3 | 86.6M | +0.0M |

**使用方式**:
```python
import open_clip
model, _, _ = open_clip.create_model_and_transforms('PE-Core-B-16-dinov3', pretrained=None)
```
