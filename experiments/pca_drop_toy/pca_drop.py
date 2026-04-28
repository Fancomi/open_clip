"""
PCADrop: PCA-based feature suppression and dropout module.

Two modes:
  suppress  -- subtract alpha * projection onto top-k PCA directions
               H' = H_c - alpha * H_c V_k V_k^T  + mu
  dropout   -- randomly zero out PCA basis components (train only)
               H' = H_c V M V^T + mu   where M is a diagonal mask

Design choices:
  * PCA basis always computed with float32 (safe under AMP).
  * PCA basis detached from computation graph by default.
  * eval() -> identity pass-through.
  * NaN guard: falls back to identity if SVD produces NaN.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCADrop(nn.Module):
    """
    Args:
        dim         : feature dimension d
        k           : number of top principal components to act on
        alpha       : suppression strength in [0, 1]  (suppress mode)
        drop_prob   : probability of dropping each top-k PC  (dropout mode)
        mode        : 'suppress' | 'dropout' | 'mixed'
        only_topk   : if True, act only on top-k PCs; else act on all PCs
        detach_basis: if True, stop gradient through PCA basis
        train_only  : if True, identity at eval time (always recommended)
        use_fp32    : compute PCA in float32 even under AMP
    """

    def __init__(
        self,
        dim: int,
        k: int = 4,
        alpha: float = 0.2,
        drop_prob: float = 0.1,
        mode: str = "suppress",
        only_topk: bool = True,
        detach_basis: bool = True,
        train_only: bool = True,
        use_fp32: bool = True,
    ):
        super().__init__()
        assert mode in ("suppress", "dropout", "mixed"), f"Unknown mode: {mode}"
        self.dim = dim
        self.k = k
        self.alpha = alpha
        self.drop_prob = drop_prob
        self.mode = mode
        self.only_topk = only_topk
        self.detach_basis = detach_basis
        self.train_only = train_only
        self.use_fp32 = use_fp32

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_pca_basis(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (V, singular_values) where V has shape [d, d] (columns = PCs,
        ordered from largest to smallest singular value).

        x : [B, d], already centered.
        """
        orig_dtype = x.dtype
        if self.use_fp32:
            x = x.float()

        try:
            # SVD on centered features: x = U S V^T
            # V columns are principal components
            _, S, Vh = torch.linalg.svd(x, full_matrices=False)
            V = Vh.T  # [d, min(B,d)]
        except Exception:
            # Fallback: covariance + eigh
            cov = (x.T @ x) / max(x.shape[0] - 1, 1)
            eigvals, eigvecs = torch.linalg.eigh(cov)
            # eigh returns ascending order; reverse for descending
            idx = torch.argsort(eigvals, descending=True)
            S = eigvals[idx].sqrt().clamp(min=0)
            V = eigvecs[:, idx]  # [d, d]

        if self.detach_basis:
            V = V.detach()
            S = S.detach()

        V = V.to(orig_dtype)
        S = S.to(orig_dtype)
        return V, S

    # ------------------------------------------------------------------

    def _suppress(self, xc: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Suppress top-k PCs with strength alpha."""
        k = min(self.k, V.shape[1])
        Vk = V[:, :k]  # [d, k]
        # projection onto top-k subspace
        proj = xc @ Vk @ Vk.T  # [B, d]
        return xc - self.alpha * proj

    def _dropout(self, xc: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Random dropout in PCA basis."""
        k = min(self.k, V.shape[1]) if self.only_topk else V.shape[1]
        Vk = V[:, :k]  # [d, k]
        # Project to PCA coordinates
        coords = xc @ Vk  # [B, k]
        # Mask
        mask = torch.bernoulli(
            torch.full((k,), 1 - self.drop_prob, device=xc.device, dtype=xc.dtype)
        )
        coords = coords * mask  # [B, k]
        # Reconstruct perturbation
        perturbed = xc - (xc @ Vk) @ Vk.T + coords @ Vk.T
        return perturbed

    def _mixed(self, xc: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """First suppress, then dropout."""
        xc = self._suppress(xc, V)
        xc = self._dropout(xc, V)
        return xc

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, d]  or  [B, T, d]

        For [B, T, d] input, PCA is computed over the flattened (B*T, d) view
        and the same transform is applied to all tokens.
        """
        if self.train_only and not self.training:
            return x

        orig_shape = x.shape
        orig_dtype = x.dtype

        # Flatten seq dim if present
        if x.dim() == 3:
            B, T, d = x.shape
            x = x.reshape(B * T, d)
        elif x.dim() == 2:
            pass
        else:
            raise ValueError(f"PCADrop expects 2D or 3D input, got {x.dim()}D")

        # Center
        mu = x.mean(dim=0, keepdim=True)
        xc = x - mu

        # Compute PCA basis
        V, S = self._compute_pca_basis(xc)

        # Apply transformation
        if self.mode == "suppress":
            xc_out = self._suppress(xc, V)
        elif self.mode == "dropout":
            xc_out = self._dropout(xc, V)
        else:  # mixed
            xc_out = self._mixed(xc, V)

        # NaN guard
        if torch.isnan(xc_out).any():
            return x.reshape(orig_shape)

        out = (xc_out + mu).to(orig_dtype)
        return out.reshape(orig_shape)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, k={self.k}, alpha={self.alpha}, "
            f"drop_prob={self.drop_prob}, mode={self.mode}, "
            f"detach_basis={self.detach_basis}"
        )


# ===========================================================================
# Unit tests
# ===========================================================================

def _run_tests():
    import sys

    torch.manual_seed(0)
    B, d = 64, 32
    k = 4

    def make(mode, **kw):
        return PCADrop(dim=d, k=k, mode=mode, **kw)

    # --- shape preservation ---
    for mode in ("suppress", "dropout", "mixed"):
        m = make(mode)
        m.train()
        x = torch.randn(B, d)
        y = m(x)
        assert y.shape == x.shape, f"Shape mismatch in {mode}: {y.shape}"

    # --- 3D input ---
    m = make("suppress")
    m.train()
    x3 = torch.randn(8, 16, d)
    y3 = m(x3)
    assert y3.shape == x3.shape, f"3D shape mismatch: {y3.shape}"

    # --- eval mode is identity ---
    m = make("suppress", train_only=True)
    m.eval()
    x = torch.randn(B, d)
    y = m(x)
    assert torch.equal(y, x), "eval should be identity"

    # --- alpha=0 is identity for suppress ---
    m = PCADrop(dim=d, k=k, mode="suppress", alpha=0.0)
    m.train()
    x = torch.randn(B, d)
    y = m(x)
    assert torch.allclose(y, x, atol=1e-5), f"alpha=0 suppress should be identity, max diff={( y-x).abs().max()}"

    # --- suppress top-k lowers variance along top-k directions ---
    torch.manual_seed(42)
    # Create data with clear top PC
    x = torch.zeros(B, d)
    x[:, 0] = torch.randn(B) * 10  # dominant direction
    x[:, 1:] = torch.randn(B, d - 1) * 0.1

    m = PCADrop(dim=d, k=1, alpha=1.0, mode="suppress", detach_basis=True)
    m.train()
    y = m(x)
    # Variance along original top direction should decrease
    var_before = x[:, 0].var().item()
    # Project y onto original top direction (dim 0)
    var_after = y[:, 0].var().item()
    assert var_after < var_before * 0.1, (
        f"Suppression should reduce variance: before={var_before:.3f}, after={var_after:.3f}"
    )

    # --- no NaN ---
    m = make("suppress")
    m.train()
    x = torch.randn(B, d)
    y = m(x)
    assert not torch.isnan(y).any(), "NaN detected in output"

    # --- dtype preservation ---
    m = make("suppress")
    m.train()
    x_fp16 = torch.randn(B, d, dtype=torch.float16)
    y_fp16 = m(x_fp16)
    assert y_fp16.dtype == torch.float16, f"dtype mismatch: {y_fp16.dtype}"

    print("All PCADrop unit tests passed.")
    return True


if __name__ == "__main__":
    _run_tests()
