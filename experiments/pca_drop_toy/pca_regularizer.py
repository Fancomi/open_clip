"""
PCARegularizer
==============
A structured regularization module that operates on batch feature tensors
using a stable PCA basis maintained by MomentumPCAStats.

Modes
─────
  none               : identity pass-through  (baseline)
  attenuate_topk     : H' = H_c - α · H_c V_k V_k^T + μ
  drop_topk          : zero out / mask PC coordinates of top-k components
  drop_all_pc_weighted: eigenvalue-proportional drop probability for all PCs

All modes
  * identity at eval() time by default (train_only=True)
  * PCA basis never receives gradients (detach_basis=True)
  * PCA statistics updated from detached features
  * NaN guard: falls back to identity on numerical failure
  * float32 internally for PCA; output dtype matches input
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from momentum_pca import MomentumPCAStats


# ────────────────────────────────────────────────────────────────────────────
# Helper
# ────────────────────────────────────────────────────────────────────────────

def _safe_center(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (x - mean, mean).  Robust to batch-size=1."""
    mu = x.mean(dim=0, keepdim=True)
    return x - mu, mu


# ────────────────────────────────────────────────────────────────────────────
# Main module
# ────────────────────────────────────────────────────────────────────────────

class PCARegularizer(nn.Module):
    """
    Args
    ────
    dim            : feature dimension d
    mode           : 'none' | 'attenuate_topk' | 'drop_topk' |
                     'drop_all_pc_weighted'
    top_k          : number of top PCs to act on
    alpha          : attenuation strength in [0,1]  (attenuate_topk)
    drop_prob      : base drop probability for PC coordinates
    per_sample_mask: if True, each sample gets an independent mask;
                     else the whole batch shares one mask per step
    inverted_scale : rescale surviving coordinates by 1/(1-p) like inverted dropout
    max_drop_prob  : cap for eigenvalue-weighted drop prob  (drop_all_pc_weighted)
    min_drop_prob  : floor for eigenvalue-weighted drop prob (drop_all_pc_weighted)
    momentum       : EMA decay for running covariance
    update_every   : recompute eigenvectors every N forward calls
    warmup_steps   : forward calls before regularization is active
    eps            : jitter added to covariance diagonal
    train_only     : if True, identity at eval() time (strongly recommended)
    detach_basis   : if True, no gradients flow through PCA basis (always True)
    use_fp32       : compute PCA statistics and basis in float32
    """

    MODES = ("none", "attenuate_topk", "drop_topk", "drop_all_pc_weighted")

    def __init__(
        self,
        dim: int,
        mode: str = "none",
        top_k: int = 4,
        alpha: float = 0.3,
        drop_prob: float = 0.1,
        per_sample_mask: bool = False,
        inverted_scale: bool = True,
        max_drop_prob: float = 0.8,
        min_drop_prob: float = 0.01,
        momentum: float = 0.99,
        update_every: int = 1,
        warmup_steps: int = 0,
        eps: float = 1e-5,
        train_only: bool = True,
        detach_basis: bool = True,
        use_fp32: bool = True,
    ):
        super().__init__()
        assert mode in self.MODES, f"Unknown mode '{mode}'. Choose from {self.MODES}"
        self.dim = dim
        self.mode = mode
        self.top_k = top_k
        self.alpha = alpha
        self.drop_prob = drop_prob
        self.per_sample_mask = per_sample_mask
        self.inverted_scale = inverted_scale
        self.max_drop_prob = max_drop_prob
        self.min_drop_prob = min_drop_prob
        self.train_only = train_only
        self.detach_basis = detach_basis  # always True in practice
        self.use_fp32 = use_fp32

        self.pca_stats = MomentumPCAStats(
            dim=dim,
            momentum=momentum,
            update_every=update_every,
            eps=eps,
            warmup_steps=warmup_steps,
        )

    # ──────────────────────────────────────────────────────────────────
    # Mode implementations  (work in float32 internally)
    # ──────────────────────────────────────────────────────────────────

    def _attenuate_topk(
        self,
        xc: torch.Tensor,   # [B, d]  centered, float32
        Vk: torch.Tensor,   # [d, k]  top-k eigenvectors
        mu: torch.Tensor,   # [1, d]  batch mean
    ) -> torch.Tensor:
        """
        H' = H_c - α · H_c V_k V_k^T + μ
        α = 0  → identity;  α = 1  → fully removes top-k variance
        """
        proj = xc @ Vk @ Vk.T          # [B, d]
        xc_out = xc - self.alpha * proj
        return xc_out + mu

    def _drop_topk(
        self,
        xc: torch.Tensor,   # [B, d]
        Vk: torch.Tensor,   # [d, k]
        mu: torch.Tensor,   # [1, d]
    ) -> torch.Tensor:
        """
        Project onto top-k basis, apply random mask, project back.
        Unaffected (bottom) subspace is preserved exactly.
        """
        B, d = xc.shape
        k = Vk.shape[1]
        coords = xc @ Vk                # [B, k]

        if self.per_sample_mask:
            p_keep = 1.0 - self.drop_prob
            mask = torch.bernoulli(
                torch.full((B, k), p_keep, device=xc.device, dtype=xc.dtype)
            )
        else:
            p_keep = 1.0 - self.drop_prob
            mask = torch.bernoulli(
                torch.full((k,), p_keep, device=xc.device, dtype=xc.dtype)
            )

        if self.inverted_scale and self.drop_prob < 1.0:
            mask = mask / (1.0 - self.drop_prob)

        coords_masked = coords * mask   # [B, k]

        # Reconstruct: remove original top-k contribution, add masked version
        # xc_out = xc - xc·Vk·Vk^T  +  coords_masked·Vk^T
        proj_orig   = coords @ Vk.T     # [B, d]
        proj_masked = coords_masked @ Vk.T
        xc_out = xc - proj_orig + proj_masked
        return xc_out + mu

    def _drop_all_pc_weighted(
        self,
        xc: torch.Tensor,   # [B, d]
        V: torch.Tensor,    # [d, d]  all eigenvectors
        lam: torch.Tensor,  # [d]     all eigenvalues
        mu: torch.Tensor,   # [1, d]
    ) -> torch.Tensor:
        """
        Drop each PC with probability proportional to its eigenvalue.
        p_i = clip(lam_i / lam_max · max_drop_prob, min_drop_prob, max_drop_prob)
        """
        lam = lam.clamp(min=0.0)
        lam_max = lam.max().clamp(min=1e-8)
        # Scale proportionally, then clip
        drop_probs = (lam / lam_max * self.max_drop_prob).clamp(
            self.min_drop_prob, self.max_drop_prob
        )                               # [d]
        keep_probs = 1.0 - drop_probs  # [d]

        B, d = xc.shape
        coords = xc @ V                 # [B, d]

        if self.per_sample_mask:
            # [B, d] independent
            mask = torch.bernoulli(
                keep_probs.unsqueeze(0).expand(B, -1).to(xc.device)
            )
        else:
            mask = torch.bernoulli(keep_probs.to(xc.device))  # [d]

        if self.inverted_scale:
            # avoid division by zero when keep_prob = 0
            scale = torch.where(keep_probs > 1e-6, 1.0 / keep_probs, torch.zeros_like(keep_probs))
            mask = mask * scale.to(xc.device)

        coords_masked = coords * mask
        xc_out = coords_masked @ V.T    # project back
        return xc_out + mu

    # ──────────────────────────────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, d]  or  [B, T, d]

        Returns x unchanged when:
          - mode == 'none'
          - train_only=True and model is in eval mode
          - warmup not finished
          - numerical failure (NaN guard)
        """
        # ── quick-exit conditions ──────────────────────────────────────
        if self.mode == "none":
            return x
        if self.train_only and not self.training:
            return x

        orig_shape = x.shape
        orig_dtype = x.dtype

        # Flatten sequence dimension if present
        if x.dim() == 3:
            B, T, d = x.shape
            x_flat = x.reshape(B * T, d)
        elif x.dim() == 2:
            x_flat = x
        else:
            raise ValueError(f"PCARegularizer expects 2D or 3D input, got {x.dim()}D")

        # ── update running statistics (no grad) ───────────────────────
        self.pca_stats.update(x_flat)

        # ── check warmup ──────────────────────────────────────────────
        if not self.pca_stats.is_ready():
            return x

        # ── get PCA basis ─────────────────────────────────────────────
        k = min(self.top_k, self.dim)

        if self.mode == "drop_all_pc_weighted":
            V_all, lam_all = self.pca_stats.get_basis(k=self.dim)
        else:
            V_all, lam_all = self.pca_stats.get_basis(k=k)

        if V_all is None:
            return x  # not ready

        # Cast to float32 for computation
        x_f32 = x_flat.float()
        V_f32 = V_all.to(x_flat.device)
        lam_f32 = lam_all.to(x_flat.device)

        # ── center ────────────────────────────────────────────────────
        xc, mu = _safe_center(x_f32)

        # ── apply mode ────────────────────────────────────────────────
        try:
            if self.mode == "attenuate_topk":
                out_f32 = self._attenuate_topk(xc, V_f32, mu)

            elif self.mode == "drop_topk":
                out_f32 = self._drop_topk(xc, V_f32, mu)

            elif self.mode == "drop_all_pc_weighted":
                out_f32 = self._drop_all_pc_weighted(xc, V_f32, lam_f32, mu)

            else:  # should not happen
                return x

        except Exception:
            return x  # NaN guard: fallback to identity

        # ── NaN guard ─────────────────────────────────────────────────
        if torch.isnan(out_f32).any() or torch.isinf(out_f32).any():
            return x

        out = out_f32.to(orig_dtype).reshape(orig_shape)
        return out

    # ──────────────────────────────────────────────────────────────────
    # Convenience / logging
    # ──────────────────────────────────────────────────────────────────

    def log_stats(self) -> dict:
        """Return a dict of loggable statistics (no tensors)."""
        return {
            "pca/effective_rank":   self.pca_stats.effective_rank(),
            "pca/expl_var_ratio":   self.pca_stats.explained_variance_ratio(self.top_k),
            "pca/step_count":       self.pca_stats.step_count.item(),
            "pca/is_ready":         int(self.pca_stats.is_ready()),
            "pca/spectrum_top8":    self.pca_stats.spectrum(8),
        }

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, mode={self.mode}, top_k={self.top_k}, "
            f"alpha={self.alpha}, drop_prob={self.drop_prob}, "
            f"train_only={self.train_only}"
        )
