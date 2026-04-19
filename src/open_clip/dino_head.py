"""DINOHead: MLP projection head used for DINO/iBOT self-distillation.

Ported from DINOv3 (Meta AI), adapted to remove dinov3-library dependencies.
Architecture: Linear(in) -> GELU -> ... -> Linear(bottleneck) -> L2-norm -> Linear(out, no bias)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_


def _build_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    """Build the MLP trunk of DINOHead."""
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
    if use_bn:
        layers.append(nn.BatchNorm1d(hidden_dim))
    layers.append(nn.GELU())
    for _ in range(nlayers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
    layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
    return nn.Sequential(*layers)


class DINOHead(nn.Module):
    """Projection head for DINO/iBOT self-distillation.

    Structure:
        MLP: in_dim -> hidden_dim(×nlayers-1) -> bottleneck_dim
        L2-normalize (bottleneck_dim)
        Linear: bottleneck_dim -> out_dim  (no bias, weight-normalized prototype layer)

    Args:
        in_dim:         Input feature dimension (backbone output dim).
        out_dim:        Number of prototypes (e.g. 65536).
        use_bn:         Whether to use BatchNorm in the MLP trunk.
        nlayers:        Number of MLP layers (>=1).
        hidden_dim:     Hidden dim of MLP trunk (default: 2048).
        bottleneck_dim: Bottleneck dim before L2-norm (default: 256).
        mlp_bias:       Whether to use bias in MLP linear layers.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        use_bn: bool = False,
        nlayers: int = 3,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        mlp_bias: bool = True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp(
            nlayers,
            in_dim,
            bottleneck_dim,
            hidden_dim=hidden_dim,
            use_bn=use_bn,
            bias=mlp_bias,
        )
        self.last_layer = nn.Linear(bottleneck_dim, out_dim, bias=False)

    def init_weights(self) -> None:
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [N, in_dim] or [N*crops, in_dim].

        Returns:
            Prototype logits of shape [N, out_dim].
        """
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = F.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x
