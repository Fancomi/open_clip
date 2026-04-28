"""
metrics.py  ─  Feature analysis metrics for PCA regularization experiments.

All functions accept torch.Tensor or numpy arrays; they always return Python
scalars or lists so they are safe to serialize in JSON / JSONL.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Core spectral statistics
# ─────────────────────────────────────────────────────────────────────────────

def _to_float32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().float().cpu()


def singular_values(features: torch.Tensor, top_k: Optional[int] = None) -> torch.Tensor:
    """
    Return singular values of the centered feature matrix in descending order.
    σ_i = sqrt(λ_i) where λ_i are eigenvalues of the sample covariance.
    """
    x = _to_float32(features)
    xc = x - x.mean(0)
    try:
        sv = torch.linalg.svdvals(xc)
    except Exception:
        sv = torch.zeros(min(x.shape))
    if top_k is not None:
        sv = sv[:top_k]
    return sv


def eigenvalue_spectrum(features: torch.Tensor, top_k: int = 16) -> List[float]:
    """Top-k eigenvalues (= σ² / (n-1)) as a Python list."""
    sv = singular_values(features, top_k=top_k)
    n = max(features.shape[0] - 1, 1)
    return (sv ** 2 / n).tolist()


def explained_variance_ratio(features: torch.Tensor, k: int = 4) -> float:
    """Fraction of total variance captured by the top-k principal components."""
    sv = singular_values(features)
    sv2 = sv ** 2
    total = sv2.sum().item()
    if total < 1e-10:
        return float("nan")
    return (sv2[:k].sum() / total).item()


def effective_rank(features: torch.Tensor) -> float:
    """
    Effective rank = exp(H) where H is the Shannon entropy of the
    normalized squared-singular-value distribution.

    Reference: Roy & Vetterli, "The effective rank: A measure of
    effective dimensionality", EUSIPCO 2007.
    """
    sv = singular_values(features)
    sv2 = (sv ** 2).clamp(min=0.0)
    total = sv2.sum()
    if total < 1e-10:
        return float("nan")
    p = sv2 / total
    p = p[p > 1e-10]
    return torch.exp(-(p * p.log()).sum()).item()


def stable_rank(features: torch.Tensor) -> float:
    """
    Stable rank = ||A||_F^2 / ||A||_2^2  (ratio of Frobenius to spectral norm squared).
    Always <= matrix rank.  Faster to compute than effective rank.
    """
    sv = singular_values(features)
    if sv[0].item() < 1e-10:
        return float("nan")
    return (sv ** 2).sum().item() / (sv[0] ** 2).item()


def feature_norm_stats(features: torch.Tensor) -> dict:
    """Mean, std, min, max of per-sample L2 norms."""
    norms = features.float().norm(dim=-1)
    return {
        "norm_mean": norms.mean().item(),
        "norm_std":  norms.std().item(),
        "norm_min":  norms.min().item(),
        "norm_max":  norms.max().item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Classification metrics
# ─────────────────────────────────────────────────────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return (pred == labels).float().mean().item()


def top_k_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int = 5) -> float:
    _, top_k_preds = logits.topk(min(k, logits.shape[-1]), dim=-1)
    correct = top_k_preds.eq(labels.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Spurious correlation analysis
# ─────────────────────────────────────────────────────────────────────────────

def spurious_alignment(
    features: torch.Tensor,
    spurious_dim: int = 0,
    top_k: int = 4,
) -> float:
    """
    Measure how much of the top-k PCA variance is explained by a known
    spurious direction (e.g., dim 0 in Dataset C).

    Returns the cosine similarity squared between the first PC and the
    spurious dimension unit vector, averaged over top-k PCs.
    """
    x = _to_float32(features)
    xc = x - x.mean(0)
    try:
        _, _, Vh = torch.linalg.svd(xc, full_matrices=False)
        top_pcs = Vh[:top_k]            # [k, d]
    except Exception:
        return float("nan")

    # Unit vector for spurious dimension
    e = torch.zeros(x.shape[1])
    e[spurious_dim] = 1.0
    cos2 = (top_pcs @ e) ** 2          # [k]
    return cos2.mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Gradient norm utility
# ─────────────────────────────────────────────────────────────────────────────

def gradient_norm(model: torch.nn.Module) -> float:
    """Total L2 norm of all parameter gradients (after backward)."""
    total = 0.0
    count = 0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
            count += 1
    return total ** 0.5 if count > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Full metric bundle  (call once per eval round)
# ─────────────────────────────────────────────────────────────────────────────

def compute_feature_metrics(
    features: torch.Tensor,
    spectrum_top_k: int = 16,
    evr_k: int = 4,
    spurious_dim: Optional[int] = None,
) -> dict:
    """
    Compute all spectral / feature metrics in one call.

    Args:
        features     : [N, d] tensor of activations or embeddings
        spectrum_top_k: how many eigenvalues to log
        evr_k        : number of PCs for explained_variance_ratio
        spurious_dim : if provided, also compute spurious_alignment

    Returns dict suitable for JSON serialization.
    """
    m: dict = {}
    m["effective_rank"]         = effective_rank(features)
    m["stable_rank"]            = stable_rank(features)
    m["explained_var_ratio"]    = explained_variance_ratio(features, k=evr_k)
    m[f"explained_var_ratio_k{evr_k}"] = m["explained_var_ratio"]
    m["eigenvalue_spectrum"]    = eigenvalue_spectrum(features, top_k=spectrum_top_k)
    m.update(feature_norm_stats(features))

    if spurious_dim is not None:
        m["spurious_alignment"]  = spurious_alignment(features, spurious_dim=spurious_dim)

    return m
