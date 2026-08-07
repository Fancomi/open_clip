"""CLIP 空间 embedding 噪声注入（NOVIC 风格，用于对比训练增强）。

NOVIC (WACV'25) 在训练中向 CLIP embedding 加噪声：
  - GaussElemNoise:  元素级高斯（范数 vec_norm），85% 样本
  - UniformAngleNoise: 绕原向量旋转 angle∈[min,max]，15% 样本
  - 混合 GaussElemUniformAngle: 按 mix_ratio 混合

在 CLIP 对比训练中，对 L2-normalized 的 image/text features 加噪后
再算对比损失，迫使模型对特征扰动鲁棒、学到更本质的语义结构
（类比 VAE：重建变差、生成/表征变强）。
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingNoise(nn.Module):
    """embedding 噪声基类：输入 [B, D] L2-normalized 向量，输出加噪后仍单位范数。"""

    @staticmethod
    def create(
        scheme: str,
        embed_dim: int,
        vec_norm: float = 3.25,
        angle_min: float = 45.0,
        angle_max: float = 75.0,
        mix_ratio: float = 0.15,
    ) -> Optional["EmbeddingNoise"]:
        if not scheme:
            return None
        s = scheme.lower()
        if s == "gausselem":
            return GaussElemNoise(embed_dim, vec_norm)
        if s == "uniformangle":
            return UniformAngleNoise(embed_dim, angle_min, angle_max)
        if s == "gausselemuniformangle":
            return GaussElemUniformAngle(embed_dim, vec_norm, angle_min, angle_max, mix_ratio)
        raise ValueError(f"Unsupported embedding noise scheme: {scheme}")

    def __init__(self, scheme: str, embed_dim: int):
        super().__init__()
        self.scheme = scheme
        self.embed_dim = embed_dim

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class GaussElemNoise(EmbeddingNoise):
    """元素级高斯噪声：embed += N(0, vec_norm/√D)，再 L2-normalize。"""

    def __init__(self, embed_dim: int, vec_norm: float):
        super().__init__("GaussElem", embed_dim)
        assert vec_norm > 0, f"vec_norm must be positive: {vec_norm}"
        self.vec_norm = float(vec_norm)
        self.elem_std = self.vec_norm / math.sqrt(self.embed_dim)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        return F.normalize(embed + torch.randn_like(embed) * self.elem_std, dim=-1)


class UniformAngleNoise(EmbeddingNoise):
    """均匀角度噪声：绕原向量旋转 angle∈[min,max]°（保持单位范数）。"""

    def __init__(self, embed_dim: int, angle_min: float, angle_max: float):
        super().__init__("UniformAngle", embed_dim)
        assert 0 <= angle_min <= angle_max <= 180, f"bad angle range: {angle_min}-{angle_max}"
        self.angle_min = math.radians(angle_min)
        self.angle_max = math.radians(angle_max)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        # 在垂直于 embed 的子空间取随机方向，绕该方向旋转随机角度
        noise_dirn = torch.randn_like(embed)
        noise_dirn = noise_dirn - (noise_dirn * embed).sum(-1, keepdim=True) * embed
        noise_dirn = F.normalize(noise_dirn, dim=-1)
        angle = embed.new_empty(embed.shape[0], 1).uniform_(self.angle_min, self.angle_max)
        out = embed * angle.cos() + noise_dirn * angle.sin()
        return F.normalize(out, dim=-1)


class GaussElemUniformAngle(EmbeddingNoise):
    """混合噪声：mix_ratio 概率用 UniformAngle，否则用 GaussElem。"""

    def __init__(self, embed_dim: int, vec_norm: float, angle_min: float, angle_max: float, mix_ratio: float):
        super().__init__("GaussElemUniformAngle", embed_dim)
        assert 0 <= mix_ratio <= 1, f"mix_ratio must be in [0,1]: {mix_ratio}"
        self.mix_ratio = float(mix_ratio)
        self.gauss = GaussElemNoise(embed_dim, vec_norm)
        self.angle = UniformAngleNoise(embed_dim, angle_min, angle_max)

    def forward(self, embed: torch.Tensor) -> torch.Tensor:
        use_angle = torch.rand(embed.shape[0], 1, device=embed.device) < self.mix_ratio
        out_angle = self.angle(embed)
        out_gauss = self.gauss(embed)
        return torch.where(use_angle, out_angle, out_gauss)
