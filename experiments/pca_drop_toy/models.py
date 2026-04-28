"""
Models for PCA-drop toy / MLP experiments.

Supported architectures:
  linear      -- single linear layer
  mlp2        -- 2-layer MLP
  mlp4        -- 4-layer MLP
  mlp2_bn     -- 2-layer MLP + BatchNorm
  mlp4_ln     -- 4-layer MLP + LayerNorm

PCADrop can be inserted after any specified layer via `pca_insert_after`.
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn

from pca_drop import PCADrop
from pca_regularizer import PCARegularizer


# ---------------------------------------------------------------------------
# Helper: build a block
# ---------------------------------------------------------------------------

def _linear_block(in_dim: int, out_dim: int, norm: Optional[str] = None,
                   act: bool = True, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    if norm == "bn":
        layers.append(nn.BatchNorm1d(out_dim))
    elif norm == "ln":
        layers.append(nn.LayerNorm(out_dim))
    if act:
        layers.append(nn.GELU())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Classifier with optional PCADrop insertion
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    """
    Configurable MLP for classification experiments.

    Args:
        in_dim       : input feature dimension
        n_classes    : number of output classes
        hidden_dims  : list of hidden layer widths ([] = linear classifier)
        norm         : None | 'bn' | 'ln'
        dropout      : standard dropout probability
        pca_cfg      : PCADrop config dict (None = disabled)
        pca_insert_after : list of layer indices (0-based) after which to insert PCADrop
                           e.g. [0] = after first hidden layer
                           'input' = before any hidden layer
    """

    def __init__(
        self,
        in_dim: int,
        n_classes: int,
        hidden_dims: List[int] = (256, 256),
        norm: Optional[str] = None,
        dropout: float = 0.0,
        pca_cfg: Optional[dict] = None,
        pca_insert_after: Optional[List] = None,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        pca_insert_after = pca_insert_after or []

        layers: list[nn.Module] = []

        # Optional PCADrop at input
        if "input" in pca_insert_after and pca_cfg is not None:
            layers.append(self._make_pca(in_dim, pca_cfg))

        dims = [in_dim] + list(hidden_dims)
        for i, (d_in, d_out) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(_linear_block(d_in, d_out, norm=norm,
                                        act=True, dropout=dropout))
            if i in pca_insert_after and pca_cfg is not None:
                layers.append(self._make_pca(d_out, pca_cfg))

        # classifier head
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _make_pca(dim: int, cfg: dict) -> nn.Module:
        """
        Factory: returns PCARegularizer (new, momentum-based) or legacy PCADrop.
        Determined by cfg['backend']:
          'momentum' (default)  -> PCARegularizer
          'batch'               -> PCADrop  (single-batch, no running stats)
        """
        backend = cfg.get("backend", "momentum")
        if backend == "batch":
            # Legacy single-batch PCADrop
            # Map new mode names to old ones
            mode_map = {
                "attenuate_topk": "suppress",
                "drop_topk": "dropout",
                "none": "suppress",  # will use alpha=0
            }
            old_mode = mode_map.get(cfg.get("mode", "suppress"), cfg.get("mode", "suppress"))
            return PCADrop(
                dim=dim,
                k=cfg.get("k", cfg.get("top_k", 4)),
                alpha=cfg.get("alpha", 0.2),
                drop_prob=cfg.get("drop_prob", 0.1),
                mode=old_mode,
                only_topk=cfg.get("only_topk", True),
                detach_basis=cfg.get("detach_basis", True),
                train_only=cfg.get("train_only", True),
                use_fp32=cfg.get("use_fp32", True),
            )
        else:
            # New momentum-based PCARegularizer
            return PCARegularizer(
                dim=dim,
                mode=cfg.get("mode", "none"),
                top_k=cfg.get("top_k", cfg.get("k", 4)),
                alpha=cfg.get("alpha", 0.3),
                drop_prob=cfg.get("drop_prob", 0.1),
                per_sample_mask=cfg.get("per_sample_mask", False),
                inverted_scale=cfg.get("inverted_scale", True),
                max_drop_prob=cfg.get("max_drop_prob", 0.8),
                min_drop_prob=cfg.get("min_drop_prob", 0.01),
                momentum=cfg.get("momentum", 0.99),
                update_every=cfg.get("update_every", 1),
                warmup_steps=cfg.get("warmup_steps", 0),
                eps=cfg.get("eps", 1e-5),
                train_only=cfg.get("train_only", True),
                spectral_gamma=cfg.get("spectral_gamma", 0.5),
                spectral_w_min=cfg.get("spectral_w_min", 0.5),
                spectral_w_max=cfg.get("spectral_w_max", 2.0),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Named factory
# ---------------------------------------------------------------------------

def build_model(
    arch: str,
    in_dim: int,
    n_classes: int,
    dropout: float = 0.0,
    pca_cfg: Optional[dict] = None,
    pca_insert_after: Optional[List] = None,
) -> MLPClassifier:
    """
    arch options:
      linear, mlp2, mlp4, mlp2_bn, mlp4_ln
    """
    ARCH = {
        "linear":  dict(hidden_dims=[], norm=None),
        "mlp2":    dict(hidden_dims=[256, 256], norm=None),
        "mlp4":    dict(hidden_dims=[256, 256, 256, 256], norm=None),
        "mlp2_bn": dict(hidden_dims=[256, 256], norm="bn"),
        "mlp4_ln": dict(hidden_dims=[256, 256, 256, 256], norm="ln"),
    }
    if arch not in ARCH:
        raise ValueError(f"Unknown arch '{arch}'. Options: {list(ARCH)}")
    kwargs = ARCH[arch]
    return MLPClassifier(
        in_dim=in_dim,
        n_classes=n_classes,
        dropout=dropout,
        pca_cfg=pca_cfg,
        pca_insert_after=pca_insert_after,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    x = torch.randn(32, 64)
    for arch in ("linear", "mlp2", "mlp4", "mlp2_bn", "mlp4_ln"):
        m = build_model(arch, in_dim=64, n_classes=4,
                        pca_cfg=dict(k=4, alpha=0.2, mode="suppress"),
                        pca_insert_after=[0])
        m.train()
        logits = m(x)
        assert logits.shape == (32, 4), f"{arch}: bad shape {logits.shape}"
        print(f"  {arch}: ok  params={sum(p.numel() for p in m.parameters()):,}")
    print("models.py smoke test passed.")
