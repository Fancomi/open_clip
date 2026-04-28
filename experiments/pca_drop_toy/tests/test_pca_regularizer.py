"""
tests/test_pca_regularizer.py
Unit tests for MomentumPCAStats and PCARegularizer.

Run with:
    cd experiments/pca_drop_toy
    python -m pytest tests/ -v
or
    python tests/test_pca_regularizer.py
"""

from __future__ import annotations

import sys
import os
# Allow imports from parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Minimal stub so the file is importable without pytest installed
    class _pytest_stub:
        @staticmethod
        def fixture(fn=None, **kw):
            return fn if fn else (lambda f: f)
        @staticmethod
        def mark():
            pass
        @staticmethod
        def skipif(cond, reason=""):
            def deco(fn): return fn
            return deco
    class _mark_stub:
        @staticmethod
        def parametrize(names, vals):
            def deco(fn):
                fn._parametrize = (names, vals)
                return fn
            return deco
        @staticmethod
        def skipif(cond, reason=""):
            def deco(fn): return fn
            return deco
    class pytest:
        fixture = _pytest_stub.fixture
        mark = _mark_stub
        skipif = _pytest_stub.skipif

import torch
import torch.nn as nn

from momentum_pca import MomentumPCAStats
from pca_regularizer import PCARegularizer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def dim():
    return 32

@pytest.fixture
def batch():
    return 64

@pytest.fixture
def small_batch():
    """Batch smaller than dim – edge case."""
    return 8

@pytest.fixture
def x(batch, dim):
    torch.manual_seed(0)
    return torch.randn(batch, dim)

@pytest.fixture
def x_3d(batch, dim):
    torch.manual_seed(1)
    return torch.randn(batch, 10, dim)  # [B, T, d]


# ─────────────────────────────────────────────────────────────────────────────
# MomentumPCAStats tests
# ─────────────────────────────────────────────────────────────────────────────

class TestMomentumPCAStats:

    def test_init_shape(self, dim):
        stats = MomentumPCAStats(dim=dim)
        assert stats.running_cov.shape == (dim, dim)
        assert stats.eigenvecs.shape == (dim, dim)
        assert stats.eigenvals.shape == (dim,)

    def test_update_initializes(self, dim, x):
        stats = MomentumPCAStats(dim=dim, warmup_steps=0)
        assert not stats.initialized.item()
        stats.update(x)
        assert stats.initialized.item()
        assert stats.step_count.item() == 1

    def test_running_cov_updates(self, dim, x):
        stats = MomentumPCAStats(dim=dim, momentum=0.9)
        cov_before = stats.running_cov.clone()
        stats.update(x)
        assert not torch.allclose(stats.running_cov, cov_before)

    def test_running_cov_shape(self, dim, x):
        stats = MomentumPCAStats(dim=dim)
        stats.update(x)
        assert stats.running_cov.shape == (dim, dim)

    def test_warmup_blocks_basis(self, dim, x):
        stats = MomentumPCAStats(dim=dim, warmup_steps=5)
        for _ in range(3):
            stats.update(x)
        V, lam = stats.get_basis(k=4)
        assert V is None and lam is None

    def test_warmup_passes_after_enough_steps(self, dim, x):
        stats = MomentumPCAStats(dim=dim, warmup_steps=3)
        for _ in range(4):
            stats.update(x)
        V, lam = stats.get_basis(k=4)
        assert V is not None and lam is not None

    def test_eigenvecs_orthonormal(self, dim, x):
        stats = MomentumPCAStats(dim=dim)
        for _ in range(5):
            stats.update(x)
        V = stats.eigenvecs  # [d, d]
        # V^T V should be close to I
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(dim), atol=1e-4), \
            f"Eigenvectors not orthonormal; max diff={( VtV - torch.eye(dim)).abs().max():.6f}"

    def test_eigenvals_descending(self, dim, x):
        stats = MomentumPCAStats(dim=dim)
        for _ in range(5):
            stats.update(x)
        lam = stats.eigenvals
        assert (lam[:-1] >= lam[1:] - 1e-5).all(), "Eigenvalues should be descending"

    def test_get_basis_top_k(self, dim, x):
        stats = MomentumPCAStats(dim=dim, warmup_steps=0)
        for _ in range(3):
            stats.update(x)
        k = 4
        V, lam = stats.get_basis(k=k)
        assert V.shape == (dim, k)
        assert lam.shape == (k,)

    def test_batch_too_small_skip(self, dim):
        stats = MomentumPCAStats(dim=dim)
        x_tiny = torch.randn(1, dim)  # B=1
        stats.update(x_tiny)          # should not crash
        assert stats.step_count.item() == 0  # skipped

    def test_top_k_clamp_at_dim(self, dim, x):
        stats = MomentumPCAStats(dim=dim, warmup_steps=0)
        stats.update(x)
        V, lam = stats.get_basis(k=dim + 100)  # larger than dim
        assert V.shape[1] == dim

    def test_effective_rank_finite(self, dim, x):
        stats = MomentumPCAStats(dim=dim)
        for _ in range(5):
            stats.update(x)
        er = stats.effective_rank()
        assert math.isfinite(er), f"effective_rank is not finite: {er}"
        assert er > 0

    def test_explained_variance_ratio(self, dim, x):
        stats = MomentumPCAStats(dim=dim)
        for _ in range(5):
            stats.update(x)
        evr = stats.explained_variance_ratio(k=4)
        assert 0.0 <= evr <= 1.0, f"evr={evr}"

    def test_is_ready(self, dim, x):
        stats = MomentumPCAStats(dim=dim, warmup_steps=3)
        assert not stats.is_ready()
        for _ in range(4):
            stats.update(x)
        assert stats.is_ready()

    def test_float32_internal(self, dim):
        """Running covariance must stay float32 even with float16 input."""
        stats = MomentumPCAStats(dim=dim)
        x_fp16 = torch.randn(64, dim, dtype=torch.float16)
        stats.update(x_fp16)
        assert stats.running_cov.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# PCARegularizer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPCARegularizerShape:

    @pytest.mark.parametrize("mode", ["none", "attenuate_topk", "drop_topk", "drop_all_pc_weighted"])
    def test_2d_shape_preserved(self, mode, x, dim):
        reg = PCARegularizer(dim=dim, mode=mode, warmup_steps=0)
        reg.train()
        for _ in range(3):
            reg(x)
        y = reg(x)
        assert y.shape == x.shape, f"Shape mismatch in mode={mode}: {y.shape} vs {x.shape}"

    @pytest.mark.parametrize("mode", ["attenuate_topk", "drop_topk"])
    def test_3d_shape_preserved(self, mode, x_3d, dim):
        reg = PCARegularizer(dim=dim, mode=mode, warmup_steps=0)
        reg.train()
        for _ in range(3):
            reg(x_3d)
        y = reg(x_3d)
        assert y.shape == x_3d.shape, f"3D shape mismatch: {y.shape}"


class TestPCARegularizerMode:

    def test_mode_none_is_identity(self, x, dim):
        reg = PCARegularizer(dim=dim, mode="none")
        reg.train()
        y = reg(x)
        assert torch.equal(y, x)

    def test_eval_mode_is_identity(self, x, dim):
        reg = PCARegularizer(dim=dim, mode="attenuate_topk", train_only=True, warmup_steps=0)
        reg.train()
        for _ in range(5):
            reg(x)
        reg.eval()
        y = reg(x)
        assert torch.equal(y, x), "eval() should produce identity"

    def test_eval_mode_is_identity_all_modes(self, x, dim):
        for mode in ["attenuate_topk", "drop_topk", "drop_all_pc_weighted"]:
            reg = PCARegularizer(dim=dim, mode=mode, train_only=True, warmup_steps=0)
            reg.train()
            for _ in range(5):
                reg(x)
            reg.eval()
            y = reg(x)
            assert torch.equal(y, x), f"eval should be identity for mode={mode}"

    def test_attenuate_alpha0_identity(self, x, dim):
        """alpha=0 attenuate should be identity (no suppression)."""
        reg = PCARegularizer(dim=dim, mode="attenuate_topk", alpha=0.0, warmup_steps=0)
        reg.train()
        for _ in range(5):
            reg(x)
        y = reg(x)
        assert torch.allclose(y, x, atol=1e-5), \
            f"alpha=0 should be identity, max diff={(y - x).abs().max():.6f}"

    def test_attenuate_suppresses_top_variance(self, dim):
        """alpha=1 on a dominant direction should greatly reduce that direction's variance."""
        torch.manual_seed(42)
        B = 128
        x = torch.zeros(B, dim)
        x[:, 0] = torch.randn(B) * 10.0   # dominant direction
        x[:, 1:] = torch.randn(B, dim - 1) * 0.1

        reg = PCARegularizer(
            dim=dim, mode="attenuate_topk", top_k=1, alpha=1.0,
            warmup_steps=0, momentum=0.9
        )
        reg.train()
        # Warm up stats
        for _ in range(20):
            reg(x)
        y = reg(x)

        var_before = x[:, 0].var().item()
        var_after  = y[:, 0].var().item()
        assert var_after < var_before * 0.2, \
            f"alpha=1 should reduce variance: before={var_before:.3f}, after={var_after:.3f}"

    def test_warmup_returns_identity(self, x, dim):
        reg = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=100)
        reg.train()
        y = reg(x)   # step_count=1, still in warmup
        assert torch.equal(y, x), "Should be identity during warmup"


class TestPCARegularizerNumerics:

    def test_no_nan(self, x, dim):
        for mode in ["attenuate_topk", "drop_topk", "drop_all_pc_weighted"]:
            reg = PCARegularizer(dim=dim, mode=mode, warmup_steps=0)
            reg.train()
            for _ in range(10):
                y = reg(x)
            assert not torch.isnan(y).any(), f"NaN in mode={mode}"

    def test_dtype_preserved_fp16(self, dim):
        """Output dtype should match input dtype (float16)."""
        x_fp16 = torch.randn(64, dim, dtype=torch.float16)
        reg = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=0, use_fp32=True)
        reg.train()
        for _ in range(5):
            y = reg(x_fp16)
        assert y.dtype == torch.float16, f"dtype mismatch: expected fp16, got {y.dtype}"

    def test_dtype_preserved_fp32(self, x, dim):
        for mode in ["attenuate_topk", "drop_topk", "drop_all_pc_weighted"]:
            reg = PCARegularizer(dim=dim, mode=mode, warmup_steps=0)
            reg.train()
            for _ in range(3):
                y = reg(x)
            assert y.dtype == torch.float32


class TestPCARegularizerGradients:

    def test_no_grad_through_pca_basis(self, x, dim):
        """
        The PCA basis should not carry gradients.
        Loss gradient w.r.t. input should still exist (path through feature transform).
        """
        reg = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=0, top_k=4)
        reg.train()
        # warm up
        with torch.no_grad():
            for _ in range(10):
                reg(x)

        x_req = x.clone().requires_grad_(True)
        y = reg(x_req)
        loss = y.sum()
        loss.backward()

        # x gradient should exist (feature transform participates in graph)
        assert x_req.grad is not None, "Input should receive gradients"

        # PCA stats buffers should NOT have gradients
        for name, buf in reg.pca_stats.named_buffers():
            assert buf.grad is None, f"Buffer {name} should not have grad"

    def test_pca_stats_not_in_parameters(self, dim):
        reg = PCARegularizer(dim=dim, mode="attenuate_topk")
        param_names = {n for n, _ in reg.named_parameters()}
        for buf_name in ["running_cov", "eigenvecs", "eigenvals"]:
            full_name = f"pca_stats.{buf_name}"
            assert full_name not in param_names, \
                f"{full_name} should be a buffer, not a parameter"


class TestPCARegularizerEdgeCases:

    def test_top_k_larger_than_dim_safe(self, x, dim):
        reg = PCARegularizer(dim=dim, mode="attenuate_topk", top_k=dim + 10, warmup_steps=0)
        reg.train()
        for _ in range(5):
            reg(x)
        y = reg(x)
        assert y.shape == x.shape

    def test_small_batch_fallback(self, dim):
        """Batch size 1 should not crash."""
        x_tiny = torch.randn(1, dim)
        reg = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=0)
        reg.train()
        y = reg(x_tiny)
        assert y.shape == x_tiny.shape

    def test_all_modes_consistent_shape(self, x, dim):
        for mode in PCARegularizer.MODES:
            reg = PCARegularizer(dim=dim, mode=mode, warmup_steps=0)
            reg.train()
            for _ in range(3):
                reg(x)
            y = reg(x)
            assert y.shape == x.shape, f"mode={mode}: shape mismatch"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_cpu_consistency(self, dim):
        """CPU and CUDA should produce near-identical results (same seeds)."""
        torch.manual_seed(99)
        x_cpu = torch.randn(64, dim)
        x_gpu = x_cpu.cuda()

        reg_cpu = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=0, top_k=4)
        reg_gpu = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=0, top_k=4).cuda()

        reg_cpu.train(); reg_gpu.train()
        for _ in range(10):
            reg_cpu(x_cpu)
            reg_gpu(x_gpu)

        # Force sync of running_cov
        reg_gpu.pca_stats.running_cov.copy_(reg_cpu.pca_stats.running_cov.cuda())
        reg_gpu.pca_stats.eigenvecs.copy_(reg_cpu.pca_stats.eigenvecs.cuda())
        reg_gpu.pca_stats.eigenvals.copy_(reg_cpu.pca_stats.eigenvals.cuda())

        # Since random dropout may differ, only test deterministic mode
        reg_cpu_det = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=0)
        reg_gpu_det = PCARegularizer(dim=dim, mode="attenuate_topk", warmup_steps=0).cuda()
        # Copy same stats
        for r_gpu, r_cpu in [(reg_gpu_det, reg_cpu_det)]:
            r_gpu.pca_stats.running_cov.copy_(reg_cpu.pca_stats.running_cov.cuda())
            r_gpu.pca_stats.eigenvecs.copy_(reg_cpu.pca_stats.eigenvecs.cuda())
            r_gpu.pca_stats.eigenvals.copy_(reg_cpu.pca_stats.eigenvals.cuda())
            r_gpu.pca_stats.initialized.fill_(True)
            r_gpu.pca_stats.step_count.fill_(99)
            r_cpu.pca_stats.running_cov.copy_(reg_cpu.pca_stats.running_cov)
            r_cpu.pca_stats.eigenvecs.copy_(reg_cpu.pca_stats.eigenvecs)
            r_cpu.pca_stats.eigenvals.copy_(reg_cpu.pca_stats.eigenvals)
            r_cpu.pca_stats.initialized.fill_(True)
            r_cpu.pca_stats.step_count.fill_(99)

        y_cpu = reg_cpu_det(x_cpu)
        y_gpu = reg_gpu_det(x_gpu).cpu()
        assert torch.allclose(y_cpu, y_gpu, atol=1e-4), \
            f"CPU/CUDA mismatch, max diff={(y_cpu - y_gpu).abs().max():.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner (no pytest)
# ─────────────────────────────────────────────────────────────────────────────

def run_all():
    """
    Self-contained test runner that does NOT use pytest fixtures.
    Each test is a plain function called directly.
    """
    import traceback
    import math

    torch.manual_seed(0)
    DIM, BATCH = 32, 64
    x = torch.randn(BATCH, DIM)
    x_3d = torch.randn(BATCH, 10, DIM)

    passed = failed = 0

    def ok(name):
        nonlocal passed
        print(f"  PASS  {name}")
        passed += 1

    def fail(name, err):
        nonlocal failed
        print(f"  FAIL  {name}: {err}")
        traceback.print_exc()
        failed += 1

    # ── MomentumPCAStats ─────────────────────────────────────────────

    def t_init_shape():
        s = MomentumPCAStats(dim=DIM)
        assert s.running_cov.shape == (DIM, DIM)
        assert s.eigenvecs.shape == (DIM, DIM)
        assert s.eigenvals.shape == (DIM,)

    def t_update_initializes():
        s = MomentumPCAStats(dim=DIM, warmup_steps=0)
        assert not s.initialized.item()
        s.update(x)
        assert s.initialized.item() and s.step_count.item() == 1

    def t_running_cov_updates():
        s = MomentumPCAStats(dim=DIM, momentum=0.9)
        cov0 = s.running_cov.clone()
        s.update(x)
        assert not torch.allclose(s.running_cov, cov0)

    def t_running_cov_shape():
        s = MomentumPCAStats(dim=DIM)
        s.update(x)
        assert s.running_cov.shape == (DIM, DIM)

    def t_warmup_blocks_basis():
        s = MomentumPCAStats(dim=DIM, warmup_steps=5)
        for _ in range(3): s.update(x)
        V, lam = s.get_basis(k=4)
        assert V is None and lam is None

    def t_warmup_passes():
        s = MomentumPCAStats(dim=DIM, warmup_steps=3)
        for _ in range(4): s.update(x)
        V, lam = s.get_basis(k=4)
        assert V is not None and lam is not None

    def t_eigenvecs_orthonormal():
        s = MomentumPCAStats(dim=DIM)
        for _ in range(5): s.update(x)
        V = s.eigenvecs
        VtV = V.T @ V
        assert torch.allclose(VtV, torch.eye(DIM), atol=1e-4), \
            f"Not orthonormal, max={(VtV - torch.eye(DIM)).abs().max():.6f}"

    def t_eigenvals_descending():
        s = MomentumPCAStats(dim=DIM)
        for _ in range(5): s.update(x)
        lam = s.eigenvals
        assert (lam[:-1] >= lam[1:] - 1e-5).all()

    def t_get_basis_top_k():
        s = MomentumPCAStats(dim=DIM, warmup_steps=0)
        for _ in range(3): s.update(x)
        V, lam = s.get_basis(k=4)
        assert V.shape == (DIM, 4) and lam.shape == (4,)

    def t_batch_too_small_skip():
        s = MomentumPCAStats(dim=DIM)
        s.update(torch.randn(1, DIM))   # B=1, should be skipped
        assert s.step_count.item() == 0

    def t_top_k_clamp():
        s = MomentumPCAStats(dim=DIM, warmup_steps=0)
        s.update(x)
        V, lam = s.get_basis(k=DIM + 100)
        assert V.shape[1] == DIM

    def t_effective_rank_finite():
        s = MomentumPCAStats(dim=DIM)
        for _ in range(5): s.update(x)
        er = s.effective_rank()
        assert math.isfinite(er) and er > 0, f"er={er}"

    def t_expl_var_ratio():
        s = MomentumPCAStats(dim=DIM)
        for _ in range(5): s.update(x)
        evr = s.explained_variance_ratio(k=4)
        assert 0.0 <= evr <= 1.0, f"evr={evr}"

    def t_is_ready():
        s = MomentumPCAStats(dim=DIM, warmup_steps=3)
        assert not s.is_ready()
        for _ in range(4): s.update(x)
        assert s.is_ready()

    def t_fp32_internal():
        s = MomentumPCAStats(dim=DIM)
        s.update(torch.randn(BATCH, DIM, dtype=torch.float16))
        assert s.running_cov.dtype == torch.float32

    # ── PCARegularizer ───────────────────────────────────────────────

    def t_shape_all_modes():
        for mode in PCARegularizer.MODES:
            reg = PCARegularizer(dim=DIM, mode=mode, warmup_steps=0)
            reg.train()
            for _ in range(3): reg(x)
            y = reg(x)
            assert y.shape == x.shape, f"mode={mode}: {y.shape}"

    def t_3d_shape():
        for mode in ["attenuate_topk", "drop_topk"]:
            reg = PCARegularizer(dim=DIM, mode=mode, warmup_steps=0)
            reg.train()
            for _ in range(3): reg(x_3d)
            y = reg(x_3d)
            assert y.shape == x_3d.shape, f"3D mode={mode}: {y.shape}"

    def t_mode_none_identity():
        reg = PCARegularizer(dim=DIM, mode="none")
        reg.train()
        y = reg(x)
        assert torch.equal(y, x)

    def t_eval_identity():
        for mode in ["attenuate_topk", "drop_topk", "drop_all_pc_weighted"]:
            reg = PCARegularizer(dim=DIM, mode=mode, train_only=True, warmup_steps=0)
            reg.train()
            for _ in range(5): reg(x)
            reg.eval()
            assert torch.equal(reg(x), x), f"eval identity failed for {mode}"

    def t_alpha0_identity():
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk", alpha=0.0, warmup_steps=0)
        reg.train()
        for _ in range(10): reg(x)
        y = reg(x)
        assert torch.allclose(y, x, atol=1e-5), f"max diff={(y-x).abs().max():.8f}"

    def t_alpha1_suppresses_variance():
        torch.manual_seed(42)
        xd = torch.zeros(128, DIM)
        xd[:, 0] = torch.randn(128) * 10.0
        xd[:, 1:] = torch.randn(128, DIM - 1) * 0.1
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk", top_k=1, alpha=1.0,
                             warmup_steps=0, momentum=0.9)
        reg.train()
        for _ in range(20): reg(xd)
        y = reg(xd)
        vb, va = xd[:, 0].var().item(), y[:, 0].var().item()
        assert va < vb * 0.2, f"before={vb:.3f} after={va:.3f}"

    def t_warmup_identity():
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk", warmup_steps=100)
        reg.train()
        y = reg(x)
        assert torch.equal(y, x)

    def t_no_nan():
        for mode in ["attenuate_topk", "drop_topk", "drop_all_pc_weighted"]:
            reg = PCARegularizer(dim=DIM, mode=mode, warmup_steps=0)
            reg.train()
            for _ in range(10): y = reg(x)
            assert not torch.isnan(y).any(), f"NaN in mode={mode}"

    def t_dtype_fp16():
        xf16 = torch.randn(BATCH, DIM, dtype=torch.float16)
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk", warmup_steps=0, use_fp32=True)
        reg.train()
        for _ in range(5): y = reg(xf16)
        assert y.dtype == torch.float16, f"got {y.dtype}"

    def t_dtype_fp32():
        for mode in ["attenuate_topk", "drop_topk", "drop_all_pc_weighted"]:
            reg = PCARegularizer(dim=DIM, mode=mode, warmup_steps=0)
            reg.train()
            for _ in range(3): y = reg(x)
            assert y.dtype == torch.float32, f"mode={mode} got {y.dtype}"

    def t_no_grad_basis():
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk", warmup_steps=0, top_k=4)
        reg.train()
        with torch.no_grad():
            for _ in range(10): reg(x)
        x_req = x.clone().requires_grad_(True)
        reg(x_req).sum().backward()
        assert x_req.grad is not None
        for name, buf in reg.pca_stats.named_buffers():
            assert buf.grad is None, f"Buffer {name} has grad"

    def t_stats_not_parameters():
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk")
        params = {n for n, _ in reg.named_parameters()}
        for bname in ["running_cov", "eigenvecs", "eigenvals"]:
            assert f"pca_stats.{bname}" not in params

    def t_top_k_larger_than_dim():
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk", top_k=DIM + 100, warmup_steps=0)
        reg.train()
        for _ in range(5): reg(x)
        y = reg(x)
        assert y.shape == x.shape

    def t_small_batch():
        x1 = torch.randn(1, DIM)
        reg = PCARegularizer(dim=DIM, mode="attenuate_topk", warmup_steps=0)
        reg.train()
        y = reg(x1)
        assert y.shape == x1.shape

    # ── dispatch ─────────────────────────────────────────────────────

    all_tests = [
        t_init_shape, t_update_initializes, t_running_cov_updates, t_running_cov_shape,
        t_warmup_blocks_basis, t_warmup_passes, t_eigenvecs_orthonormal,
        t_eigenvals_descending, t_get_basis_top_k, t_batch_too_small_skip,
        t_top_k_clamp, t_effective_rank_finite, t_expl_var_ratio, t_is_ready,
        t_fp32_internal,
        t_shape_all_modes, t_3d_shape, t_mode_none_identity, t_eval_identity,
        t_alpha0_identity, t_alpha1_suppresses_variance, t_warmup_identity,
        t_no_nan, t_dtype_fp16, t_dtype_fp32,
        t_no_grad_basis, t_stats_not_parameters,
        t_top_k_larger_than_dim, t_small_batch,
    ]

    for fn in all_tests:
        name = fn.__name__
        try:
            fn()
            ok(name)
        except Exception as e:
            fail(name, e)

    print(f"\n{'─'*50}")
    print(f"Results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)

