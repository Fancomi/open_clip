# Multi-Teacher Knowledge Distillation

## Background

Dual-teacher experiments (E1-E5) explored aligning one trainable image encoder with two frozen text encoders (PE-Core + SigLIP2). Key results on CC3M (10 epochs):

| ID | Config | i2t R@1 | t2i R@1 | Notes |
|----|--------|---------|---------|-------|
| E1 | 2T, scratch, 1CLS | **27.6%** | **40.1%** | Surpasses rlit_pe (24.7%) by +2.9% |
| E2 | 2T, scratch, 2CLS | 27.5% | 40.2% | On par with E1 |
| E3 | 2T, pretrained, 1CLS | 42.2% | 61.4% | Below rlit_sig2 (49.2%) |
| E4 | 2T, pretrained, 2CLS | 42.9% | 62.0% | Still degraded |
| E5 | 2T, pretrained(sig), 1CLS | TBD | TBD | |

**Conclusion**: Multi-teacher supervision improves from-scratch training (E1 > rlit_pe) but hurts pretrained initialization. This motivates expanding the from-scratch direction with more teachers.

## Purpose

1. **Scaling curve**: Does performance improve monotonically as we add more teachers (2→3→5→7)?
2. **SigLIP2 ablation**: Is the unique SigLIP2 teacher (different tokenizer, architecture) important, or do homogeneous CLIP teachers suffice?
3. **Combination sensitivity**: Does the specific choice of teachers matter at fixed N?

## Method

### Architecture: MultiTeacherCLIP

- **Image encoder**: PE-Core-B-16 backbone (ViT-B/16, 768-dim), trained from scratch
- **Text teachers**: N frozen pretrained text encoders, each with independent projection head and logit scale
- **Pooling**: Shared single CLS pool across all teachers
- **Loss**: Sum of N independent SigLIP losses (one per teacher)
- **Data pipeline**: N-way tokenization — each sample is tokenized by all N tokenizers in parallel

### Available Teachers (all B/16, 224px)

| Model | embed_dim | Tokenizer | Text Architecture |
|-------|-----------|-----------|-------------------|
| PE-Core-B-16 | 1024 | SimpleTokenizer (ctx=32) | 24L-1024W |
| ViT-B-16-SigLIP2 | 768 | GemmaTokenizer (ctx=64) | 12L-768W |
| DataComp-XL-B-16 | 512 | CLIPTokenizer (ctx=77) | 12L-512W |
| DFN2B-ViT-B-16 | 512 | CLIPTokenizer (ctx=77) | 12L-512W |
| EVA02-B-16 | 512 | CLIPTokenizer (ctx=77) | 12L-512W |
| LAION2B-B-16 | 512 | CLIPTokenizer (ctx=77) | 12L-512W |
| MetaCLIP-FullCC-B-16 | 512 | CLIPTokenizer (ctx=77) | 12L-512W (QuickGELU) |

Three tokenizer families: SimpleTokenizer (PE-Core), GemmaTokenizer (SigLIP2), CLIPTokenizer (5 standard CLIP models). The 5 standard CLIP models share the same tokenizer but were trained on different data distributions.

### Training Setup

- Data: CC3M-wds (2.9M samples)
- Epochs: 10
- Batch size: 512 × 8 GPUs = 4096
- LR: 3.4e-4 (scaled by sqrt(GlobalBS / 4096))
- Optimizer: AdamW (β1=0.9, β2=0.95)
- Precision: amp_bf16
- Warmup: 512 steps

## Experiments

All from-scratch, single CLS, same hyperparameters as E1.

| ID | Teachers | N | Purpose |
|----|----------|---|---------|
| M1 | PE + SigLIP2 + DataComp | 3 | Incremental +1 teacher over E1 |
| M2 | PE + SigLIP2 + DataComp + LAION2B + DFN2B | 5 | Scale to 5 |
| M3 | PE + SigLIP2 + DataComp + DFN2B + EVA02 + LAION2B + MetaCLIP | 7 | All 7 teachers |
| M4 | PE + DataComp + DFN2B + LAION2B + MetaCLIP | 5 | 5T without SigLIP2 (same-family tokenizer ablation) |
| M5 | PE + SigLIP2 + EVA02 | 3 | Alternative 3T combination |

Evaluation: teacher[0] (PE-Core) used for all retrieval metrics to ensure fair comparison with E1/E2 baselines.

### Comparisons

- **Scaling**: E1 (2T) → M1 (3T) → M2 (5T) → M3 (7T)
- **SigLIP2 ablation**: M2 (5T with SigLIP2) vs M4 (5T without SigLIP2)
- **Combination**: M1 (PE+Sig+DC) vs M5 (PE+Sig+EVA)

### Memory Budget

7-teacher worst case (fp16 text encoders):
- PE-Core text: ~600MB
- SigLIP2 text: ~170MB
- 5 × CLIP-512 text: ~130MB each = ~650MB
- Total: ~1.4GB extra (well within A100 80GB)

## Results

| ID | N | i2t R@1 | i2t R@5 | i2t R@10 | t2i R@1 | t2i R@5 | t2i R@10 |
|----|---|---------|---------|----------|---------|---------|----------|
| E1 (baseline) | 2 | 27.6% | | | 40.1% | | |
| M1 | 3 | | | | | | |
| M2 | 5 | | | | | | |
| M3 | 7 | | | | | | |
| M4 | 5 | | | | | | |
| M5 | 3 | | | | | | |

*(Results pending — experiments running)*

## Analysis

*(To be filled after experiments complete)*

Key questions to answer:
1. Does the scaling curve (2→3→5→7) show monotonic improvement or diminishing returns?
2. How much does SigLIP2's unique architecture/tokenizer contribute (M2 vs M4)?
3. At fixed N=3, does teacher diversity matter (M1 vs M5)?
4. Per-teacher loss breakdown: which teachers dominate the gradient signal?
