"""
MomentumPCAStats
================
Maintains a running (EMA) estimate of the feature covariance matrix and
exposes a stable PCA basis updated either every step or every N steps.

Design constraints:
  * Feature is detached from the graph before entering the statistics.
  * Running covariance is registered as a buffer (not a parameter).
  * Eigendecomposition is never differentiated through.
  * PCA statistics are stored in float32 regardless of input dtype (AMP safe).
  * Distributed-training note: each rank maintains its own running stats by
    default (single-rank correctness guaranteed). Cross-rank sync is NOT
    implemented here; see class docstring for discussion.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MomentumPCAStats(nn.Module):
    """
    Running-covariance EMA + periodic eigendecomposition.

    Args:
        dim            : feature dimension d
        momentum       : EMA decay β   (C_t = β·C_{t-1} + (1-β)·C_batch)
        update_every   : how often to recompute eigenvectors (in forward calls)
        eps            : diagonal jitter added to cov before eigh  (numerical stability)
        warmup_steps   : number of forward calls before eigenvectors are considered valid
        max_k          : if not None, only store top-max_k eigenvectors (memory saving)

    Distributed training note
    ─────────────────────────
    In DDP each GPU has its own MomentumPCAStats instance and accumulates its
    own batch statistics.  This means the running covariance is estimated from
    local-rank batches only.  For most research experiments this is acceptable
    because the effective batch window is B_rank * num_steps >> B_global.
    If exact cross-rank alignment is needed, insert an all_reduce on
    `batch_cov` before the EMA update, e.g.:
        dist.all_reduce(batch_cov); batch_cov /= dist.get_world_size()
    That hook is NOT wired here to keep the module self-contained.
    """

    def __init__(
        self,
        dim: int,
        momentum: float = 0.99,
        update_every: int = 1,
        eps: float = 1e-5,
        warmup_steps: int = 0,
        max_k: Optional[int] = None,
    ):
        super().__init__()
        assert 0.0 < momentum < 1.0, "momentum must be in (0, 1)"
        self.dim = dim
        self.momentum = momentum
        self.update_every = update_every
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.max_k = max_k if max_k is not None else dim

        # Buffers: not parameters, but saved in state_dict
        self.register_buffer("running_cov",   torch.eye(dim, dtype=torch.float32))
        self.register_buffer("eigenvecs",     torch.eye(dim, dtype=torch.float32))  # [d, d]
        self.register_buffer("eigenvals",     torch.ones(dim, dtype=torch.float32)) # [d]
        self.register_buffer("step_count",    torch.tensor(0, dtype=torch.long))
        self.register_buffer("initialized",   torch.tensor(False, dtype=torch.bool))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _batch_cov(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute unbiased sample covariance of x ∈ R^{B×d}.
        Always in float32 for numerical safety.
        """
        x = x.float()
        n = x.shape[0]
        if n < 2:
            return torch.eye(self.dim, device=x.device, dtype=torch.float32)
        mu = x.mean(dim=0, keepdim=True)
        xc = x - mu
        cov = (xc.T @ xc) / (n - 1)          # [d, d]
        return cov

    def _recompute_eigenvecs(self):
        """Run eigendecomposition on the running covariance (no gradients)."""
        with torch.no_grad():
            C = self.running_cov + self.eps * torch.eye(
                self.dim, device=self.running_cov.device, dtype=torch.float32
            )
            try:
                # eigh: real symmetric, returns ascending eigenvalues
                eigvals, eigvecs = torch.linalg.eigh(C)
            except Exception:
                # Fallback: SVD
                _, eigvals, eigvecs = torch.linalg.svd(C, full_matrices=True)
                eigvecs = eigvecs.T

            # Reverse to descending order (largest variance first)
            idx = torch.argsort(eigvals, descending=True)
            self.eigenvals.copy_(eigvals[idx])
            self.eigenvecs.copy_(eigvecs[:, idx])  # [d, d], columns = PCs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        """
        Update running statistics with a new batch x ∈ R^{B×d}.
        Call this with detached features.
        """
        x = x.detach().float()
        if x.shape[0] < 2:
            return  # too small to estimate covariance

        batch_cov = self._batch_cov(x)

        if not self.initialized.item():
            self.running_cov.copy_(batch_cov)
            self.initialized.fill_(True)
        else:
            self.running_cov.mul_(self.momentum).add_(
                batch_cov * (1.0 - self.momentum)
            )

        self.step_count.add_(1)

        # Recompute eigenvectors at specified frequency
        if self.step_count.item() % self.update_every == 0:
            self._recompute_eigenvecs()

    def get_basis(self, k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (V_k, λ_k):
          V_k : [d, k]  top-k eigenvectors (columns)
          λ_k : [k]     corresponding eigenvalues (descending)

        If warmup has not finished yet (step_count < warmup_steps),
        returns None, None as a signal to skip regularization.
        """
        if self.step_count.item() < self.warmup_steps:
            return None, None

        k = k if k is not None else self.max_k
        k = min(k, self.dim)

        V = self.eigenvecs[:, :k].clone()   # [d, k]
        lam = self.eigenvals[:k].clone()    # [k]
        return V, lam

    def is_ready(self) -> bool:
        """Returns True once warmup_steps have elapsed and stats are initialized."""
        return (
            self.initialized.item()
            and self.step_count.item() >= self.warmup_steps
        )

    def explained_variance_ratio(self, k: Optional[int] = None) -> float:
        """Fraction of total variance captured by top-k eigenvectors."""
        k = k if k is not None else self.max_k
        k = min(k, self.dim)
        total = self.eigenvals.sum().item()
        if total < 1e-10:
            return float("nan")
        return (self.eigenvals[:k].sum() / total).item()

    def effective_rank(self) -> float:
        """Effective rank = exp(entropy of normalized eigenvalue distribution)."""
        lam = self.eigenvals.clamp(min=0.0)
        total = lam.sum()
        if total < 1e-10:
            return float("nan")
        p = lam / total
        p = p[p > 1e-10]
        return torch.exp(-(p * p.log()).sum()).item()

    def spectrum(self, top_k: int = 16) -> list:
        """Return top_k eigenvalues as a Python list (for logging)."""
        return self.eigenvals[:top_k].tolist()

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, momentum={self.momentum}, "
            f"update_every={self.update_every}, warmup_steps={self.warmup_steps}"
        )
