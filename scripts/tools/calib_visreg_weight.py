#!/usr/bin/env python3
"""标定 VISReg 权重，使其正则强度 ≈ 冠军 SIGReg(weight=1e-4)。

思路（对齐 spec §5）：
  正则器作用在 backbone-output 特征 z=[B, D]（cls raw, unnorm）上。要让两种正则对
  backbone 施加的"力度"可比，最直接、最稳健的代理量是**正则项对特征 z 的梯度范数**
  ‖∂L/∂z‖——它正是 backbone 反传时在其输出端收到的信号，且与 backbone 权重的随机
  初始化无关（无需真实/训练好的模型即可标定）。

  匹配权重：w_vis = w_sig × ‖∂L_sig/∂z‖ / ‖∂L_vis/∂z‖
  （分子分母均为 weight=1 时的裸梯度范数，w_sig=1e-4）

关键：SIGReg 损失 ∝ N×world_size（显式相乘），VISReg batch-invariant。故 w_vis 随
训练 batch 增大而增大——必须在**实际全局 batch（默认 4096）**下标定。跨卡时 VISReg
靠 all-gather 达到全局 N，单机模拟：直接令 B = global batch、world_size = 目标卡数。

特征分布用 N(0, feat_std²) 模拟 backbone 输出；对量级比值不敏感（scale/shape 项对
特征整体缩放近似同阶），默认 feat_std=1，可 --feat-std 扫描确认稳定性。CPU 秒级完成。

用法：
  python scripts/tools/calib_visreg_weight.py \
      --global-batch 4096 --dim 768 --world-size 8 --slices 256 \
      --steps 30 --sigreg-weight 1e-4
"""
import argparse
import math
import warnings

warnings.filterwarnings("ignore")

import torch

from open_clip.loss import SIGReg, VISReg


def _feat_grad_norm(reg, z_batches, weight, world_size):
    """正则项对特征 z 的平均梯度范数（weight 已乘入）。

    SIGReg 内部按 dist 未初始化取 world_size=1；标定时手动 ×world_size 还原全局量级。
    VISReg batch-invariant，标定时 z 直接给全局 batch（模拟 all-gather 后），不额外缩放。
    """
    is_sig = isinstance(reg, SIGReg)
    total = 0.0
    for z0 in z_batches:
        z = z0.clone().requires_grad_(True)
        loss = weight * reg(z)
        if is_sig:
            loss = loss * world_size          # 还原多卡全局量级（forward 内 world_size=1）
        loss.backward()
        total += float(z.grad.detach().norm())
    return total / len(z_batches)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--global-batch", type=int, default=4096,
                    help="全局 batch N（= per_gpu_bs × world_size），SIGReg/VISReg 量级都随它变")
    ap.add_argument("--dim", type=int, default=768,
                    help="backbone cls raw 维度（PE-Core-B-16 trunk num_features）")
    ap.add_argument("--world-size", type=int, default=8)
    ap.add_argument("--slices", type=int, default=256)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--sigreg-weight", type=float, default=1e-4)
    ap.add_argument("--feat-std", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    N, D = args.global_batch, args.dim
    print(f"=> calibrating at global_batch={N}, dim={D}, world_size={args.world_size}, "
          f"slices={args.slices}, feat_std={args.feat_std}")

    z_batches = [torch.randn(N, D) * args.feat_std for _ in range(args.steps)]

    sig = SIGReg(num_slices=args.slices)
    vis = VISReg(num_slices=args.slices, gather=False)  # 标定时 z 已是全局 batch，不再 gather

    g_sig = _feat_grad_norm(sig, z_batches, weight=args.sigreg_weight, world_size=args.world_size)
    g_vis = _feat_grad_norm(vis, z_batches, weight=1.0, world_size=args.world_size)

    w_vis = args.sigreg_weight * g_sig / max(g_vis, 1e-30)

    print("\n================ CALIBRATION RESULT ================")
    print(f"  SIGReg (w={args.sigreg_weight:.1e}) ‖∂L/∂z‖ : {g_sig:.6e}")
    print(f"  VISReg (w=1.0)          ‖∂L/∂z‖ : {g_vis:.6e}")
    print(f"  => matched VISReg --sigreg-weight ≈ {w_vis:.4e}")
    print(f"     sweep suggestion: {0.5*w_vis:.4e} / {w_vis:.4e} / {2*w_vis:.4e}")
    print("====================================================")
    print(f"VISREG_WEIGHT={w_vis:.6e}")


if __name__ == "__main__":
    main()

