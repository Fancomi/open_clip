"""
Synthetic datasets for PCA-drop toy experiments.

Dataset A  : top PCs = useful signal  (suppressing them hurts)
Dataset B  : top PCs = nuisance noise  (suppressing them helps)
Dataset C  : spurious correlation in train, reversed in test
Dataset D  : ordinary iid data (control)

Each dataset class returns (X_train, y_train, X_val, y_val, X_test, y_test)
and a metadata dict for later analysis.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, Any

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split(X: np.ndarray, y: np.ndarray, val_frac=0.15, test_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    i_test = idx[:n_test]
    i_val  = idx[n_test:n_test + n_val]
    i_train = idx[n_test + n_val:]
    return (X[i_train], y[i_train],
            X[i_val],   y[i_val],
            X[i_test],  y[i_test])


def _to_tensors(*arrays):
    result = []
    for a in arrays:
        if a.dtype in (np.float32, np.float64):
            result.append(torch.tensor(a, dtype=torch.float32))
        else:
            result.append(torch.tensor(a, dtype=torch.long))
    return result


@dataclass
class DatasetBundle:
    name: str
    description: str
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val:   torch.Tensor
    y_val:   torch.Tensor
    X_test:  torch.Tensor
    y_test:  torch.Tensor
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self):
        return self.X_train.shape[1]

    @property
    def n_classes(self):
        return int(self.y_train.max().item()) + 1

    def loaders(self, batch_size=256, num_workers=0):
        train_ds = TensorDataset(self.X_train, self.y_train)
        val_ds   = TensorDataset(self.X_val,   self.y_val)
        test_ds  = TensorDataset(self.X_test,  self.y_test)
        kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=False)
        return (
            DataLoader(train_ds, shuffle=True,  **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw),
        )


# ---------------------------------------------------------------------------
# Dataset A  ─  top PCs carry the label signal
# ---------------------------------------------------------------------------

def make_dataset_a(
    n: int = 4000,
    dim: int = 32,
    n_classes: int = 4,
    signal_dims: int = 4,   # label signal lives in first `signal_dims` dims
    signal_scale: float = 5.0,
    noise_scale: float = 0.5,
    seed: int = 0,
) -> DatasetBundle:
    """
    High-variance directions = label signal.
    PCA suppression should HURT performance here.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    # One-hot-like class centres in first `signal_dims` dims
    centres = rng.standard_normal((n_classes, signal_dims)) * signal_scale
    X = np.zeros((n, dim), dtype=np.float32)
    for c in range(n_classes):
        mask = y == c
        X[mask, :signal_dims] = centres[c] + rng.standard_normal(
            (mask.sum(), signal_dims)) * noise_scale
    # Low-variance noise in remaining dims
    X[:, signal_dims:] = rng.standard_normal((n, dim - signal_dims)) * noise_scale

    Xtr, ytr, Xv, yv, Xt, yt = _split(X, y, seed=seed)
    return DatasetBundle(
        name="dataset_a_signal_in_top_pcs",
        description="Top PCs = label signal. Suppressing top PCs should HURT.",
        *map(lambda a: torch.tensor(a) if not isinstance(a, np.ndarray) else a, []),
        X_train=torch.tensor(Xtr), y_train=torch.tensor(ytr),
        X_val=torch.tensor(Xv),   y_val=torch.tensor(yv),
        X_test=torch.tensor(Xt),  y_test=torch.tensor(yt),
        meta=dict(signal_dims=signal_dims, signal_scale=signal_scale,
                  noise_scale=noise_scale, n_classes=n_classes),
    )


# ---------------------------------------------------------------------------
# Dataset B  ─  top PCs are nuisance; label is in low-variance dims
# ---------------------------------------------------------------------------

def make_dataset_b(
    n: int = 4000,
    dim: int = 32,
    n_classes: int = 4,
    signal_dims_start: int = 12,  # label signal in dims [signal_dims_start:]
    nuisance_scale: float = 8.0,
    signal_scale: float = 1.5,
    noise_scale: float = 0.3,
    seed: int = 1,
) -> DatasetBundle:
    """
    Top PCs = nuisance variance (e.g., lighting, domain shift).
    Label info lives in low-variance dims.
    PCA suppression should HELP here.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    X = np.zeros((n, dim), dtype=np.float32)
    # Nuisance: huge variance, class-independent
    X[:, :signal_dims_start] = rng.standard_normal((n, signal_dims_start)) * nuisance_scale
    # Signal: small variance, class-dependent
    n_signal = dim - signal_dims_start
    centres = rng.standard_normal((n_classes, n_signal)) * signal_scale
    for c in range(n_classes):
        mask = y == c
        X[mask, signal_dims_start:] = centres[c] + rng.standard_normal(
            (mask.sum(), n_signal)) * noise_scale

    Xtr, ytr, Xv, yv, Xt, yt = _split(X, y, seed=seed)
    return DatasetBundle(
        name="dataset_b_nuisance_in_top_pcs",
        description="Top PCs = nuisance. Label in low-var dims. Suppressing top PCs should HELP.",
        X_train=torch.tensor(Xtr), y_train=torch.tensor(ytr),
        X_val=torch.tensor(Xv),   y_val=torch.tensor(yv),
        X_test=torch.tensor(Xt),  y_test=torch.tensor(yt),
        meta=dict(nuisance_scale=nuisance_scale, signal_scale=signal_scale,
                  signal_dims_start=signal_dims_start, n_classes=n_classes),
    )


# ---------------------------------------------------------------------------
# Dataset C  ─  spurious correlation in train, reversed in test
# ---------------------------------------------------------------------------

def make_dataset_c(
    n_train: int = 3000,
    n_test: int = 1000,
    dim: int = 32,
    n_classes: int = 2,
    spurious_scale: float = 6.0,   # strength of spurious feature
    signal_scale: float = 1.5,     # strength of true signal
    noise_scale: float = 0.5,
    spurious_corr: float = 0.9,    # P(spurious aligns with label) in train
    seed: int = 2,
) -> DatasetBundle:
    """
    Train: spurious high-variance feature correlated with label (P=spurious_corr).
    Test:  spurious feature anti-correlated or random (P=0.5).
    PCA suppression may reduce reliance on spurious shortcut.
    """
    rng = np.random.default_rng(seed)

    def _make_split(n, corr):
        y = rng.integers(0, n_classes, size=n)
        X = np.zeros((n, dim), dtype=np.float32)
        # True signal: dim 1 onwards (low variance)
        n_signal = dim // 2
        centres = rng.standard_normal((n_classes, n_signal)) * signal_scale
        for c in range(n_classes):
            mask = y == c
            X[mask, 1:1 + n_signal] = centres[c] + rng.standard_normal(
                (mask.sum(), n_signal)) * noise_scale
        # Spurious feature: dim 0 (huge variance, correlated with label)
        spurious_sign = np.where(y == 0, 1.0, -1.0) * spurious_scale
        flip = rng.random(n) > corr
        spurious_sign[flip] *= -1
        X[:, 0] = spurious_sign + rng.standard_normal(n) * noise_scale
        return X.astype(np.float32), y

    X_train_full, y_train = _make_split(n_train, spurious_corr)
    X_test,  y_test  = _make_split(n_test,  0.5)  # spurious random in test

    # Use last 15% of train as val
    n_val = int(n_train * 0.15)
    X_val, y_val = X_train_full[-n_val:], y_train[-n_val:]
    X_train, y_train = X_train_full[:-n_val], y_train[:-n_val]

    return DatasetBundle(
        name="dataset_c_spurious_correlation",
        description=(
            f"Train: spurious corr={spurious_corr}. Test: spurious corr=0.5. "
            "PCA suppression may reduce shortcut dependency."
        ),
        X_train=torch.tensor(X_train), y_train=torch.tensor(y_train),
        X_val=torch.tensor(X_val),     y_val=torch.tensor(y_val),
        X_test=torch.tensor(X_test),   y_test=torch.tensor(y_test),
        meta=dict(spurious_scale=spurious_scale, signal_scale=signal_scale,
                  spurious_corr=spurious_corr, n_classes=n_classes),
    )


# ---------------------------------------------------------------------------
# Dataset D  ─  ordinary iid Gaussian classification (control)
# ---------------------------------------------------------------------------

def make_dataset_d(
    n: int = 4000,
    dim: int = 32,
    n_classes: int = 4,
    signal_scale: float = 2.0,
    noise_scale: float = 1.0,
    seed: int = 3,
) -> DatasetBundle:
    """
    Isotropic Gaussian clusters. No spurious structure.
    PCA-drop should behave similarly to or slightly worse than dropout.
    """
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((n_classes, dim)) * signal_scale
    y = rng.integers(0, n_classes, size=n)
    X = np.zeros((n, dim), dtype=np.float32)
    for c in range(n_classes):
        mask = y == c
        X[mask] = centres[c] + rng.standard_normal((mask.sum(), dim)) * noise_scale

    Xtr, ytr, Xv, yv, Xt, yt = _split(X, y, seed=seed)
    return DatasetBundle(
        name="dataset_d_iid_control",
        description="Isotropic Gaussian clusters. Control: no spurious structure.",
        X_train=torch.tensor(Xtr), y_train=torch.tensor(ytr),
        X_val=torch.tensor(Xv),   y_val=torch.tensor(yv),
        X_test=torch.tensor(Xt),  y_test=torch.tensor(yt),
        meta=dict(signal_scale=signal_scale, noise_scale=noise_scale,
                  n_classes=n_classes),
    )


# ---------------------------------------------------------------------------
# Dataset B2  ─  nuisance in top PCs + domain shift in test
# ---------------------------------------------------------------------------

def make_dataset_b2(
    n_train: int = 4000,
    n_test: int = 1000,
    dim: int = 64,
    n_classes: int = 4,
    signal_dims_start: int = 16,
    nuisance_scale_train: float = 8.0,
    nuisance_scale_test: float = 20.0,   # SHIFT: test nuisance is 2.5x larger
    signal_scale: float = 1.5,
    noise_scale: float = 0.3,
    seed: int = 10,
) -> DatasetBundle:
    """
    Dataset B with explicit domain shift:
      Train: nuisance ~ N(0, nuisance_scale_train²)
      Test:  nuisance ~ N(0, nuisance_scale_test²)   <-- OOD domain

    A model that memorizes nuisance will fail at test time.
    PCA suppression of top PCs should help generalize.

    Signal direction is identical in train and test (only nuisance shifts).
    """
    rng = np.random.default_rng(seed)
    n_signal = dim - signal_dims_start
    centres = rng.standard_normal((n_classes, n_signal)) * signal_scale

    def _make(n: int, nuisance_scale: float):
        y = rng.integers(0, n_classes, size=n)
        X = np.zeros((n, dim), dtype=np.float32)
        X[:, :signal_dims_start] = rng.standard_normal(
            (n, signal_dims_start)) * nuisance_scale
        for c in range(n_classes):
            mask = y == c
            X[mask, signal_dims_start:] = centres[c] + rng.standard_normal(
                (mask.sum(), n_signal)) * noise_scale
        return X, y

    X_full, y_full = _make(n_train, nuisance_scale_train)
    X_test, y_test = _make(n_test, nuisance_scale_test)

    n_val = int(n_train * 0.15)
    X_val, y_val = X_full[-n_val:], y_full[-n_val:]
    X_tr,  y_tr  = X_full[:-n_val], y_full[:-n_val]

    return DatasetBundle(
        name="dataset_b2_nuisance_domain_shift",
        description=(
            f"Train nuisance scale={nuisance_scale_train}, "
            f"Test nuisance scale={nuisance_scale_test} (OOD shift). "
            "PCA suppression should HELP generalization."
        ),
        X_train=torch.tensor(X_tr),   y_train=torch.tensor(y_tr),
        X_val=torch.tensor(X_val),    y_val=torch.tensor(y_val),
        X_test=torch.tensor(X_test),  y_test=torch.tensor(y_test),
        meta=dict(nuisance_scale_train=nuisance_scale_train,
                  nuisance_scale_test=nuisance_scale_test,
                  signal_scale=signal_scale, signal_dims_start=signal_dims_start,
                  n_classes=n_classes, ood_type="nuisance_scale_shift"),
    )


# ---------------------------------------------------------------------------
# Dataset E  ─  LINEAR model + correlated nuisance (harder separation)
# ---------------------------------------------------------------------------

def make_dataset_e(
    n_train: int = 4000,
    n_test: int = 1000,
    dim: int = 64,
    n_classes: int = 4,
    signal_dims: int = 8,
    spurious_dims: int = 8,
    signal_scale: float = 2.0,
    spurious_scale_train: float = 10.0,
    spurious_scale_test: float = 0.1,   # spurious nearly vanishes in test
    noise_scale: float = 0.5,
    spurious_corr: float = 0.85,
    seed: int = 20,
) -> DatasetBundle:
    """
    Multi-class spurious correlation with scale shift:
      - Dims [0:signal_dims]: true label signal
      - Dims [signal_dims:signal_dims+spurious_dims]: high-var spurious (train-only)
      - Dims [signal_dims+spurious_dims:]: background noise

    Train: spurious dims are correlated with label AND have large variance.
    Test:  spurious dims scale is tiny → model must rely on signal dims.

    This is a harder version of Dataset C for multi-class.
    """
    rng = np.random.default_rng(seed)
    s_start = signal_dims
    s_end   = signal_dims + spurious_dims

    signal_centres   = rng.standard_normal((n_classes, signal_dims)) * signal_scale
    spurious_centres = rng.standard_normal((n_classes, spurious_dims)) * spurious_scale_train

    def _make(n: int, s_scale: float, s_corr: float):
        y = rng.integers(0, n_classes, size=n)
        X = np.zeros((n, dim), dtype=np.float32)
        # True signal
        for c in range(n_classes):
            mask = y == c
            X[mask, :s_start] = signal_centres[c] + rng.standard_normal(
                (mask.sum(), signal_dims)) * noise_scale
        # Spurious (correlated with label in train, random in test)
        for c in range(n_classes):
            mask = np.where(y == c)[0]
            n_c = len(mask)
            aligned_idx   = mask[rng.random(n_c) < s_corr]
            unaligned_idx = mask[rng.random(n_c) >= s_corr]
            if len(aligned_idx) > 0:
                X[aligned_idx, s_start:s_end] = (
                    spurious_centres[c] * (s_scale / spurious_scale_train)
                    + rng.standard_normal((len(aligned_idx), spurious_dims)) * noise_scale
                )
            if len(unaligned_idx) > 0:
                X[unaligned_idx, s_start:s_end] = (
                    rng.standard_normal((len(unaligned_idx), spurious_dims)) * s_scale * 0.5
                )
        # Background
        X[:, s_end:] = rng.standard_normal((n, dim - s_end)) * noise_scale
        return X, y

    X_full, y_full = _make(n_train, spurious_scale_train, spurious_corr)
    X_test, y_test = _make(n_test,  spurious_scale_test,  0.5)  # near-random in test

    n_val = int(n_train * 0.15)
    X_val, y_val = X_full[-n_val:], y_full[-n_val:]
    X_tr,  y_tr  = X_full[:-n_val], y_full[:-n_val]

    return DatasetBundle(
        name="dataset_e_multiclass_spurious",
        description=(
            f"Multi-class spurious corr={spurious_corr} train, random test. "
            "Spurious dims scale shrinks in test. PCA suppression should HELP."
        ),
        X_train=torch.tensor(X_tr),   y_train=torch.tensor(y_tr),
        X_val=torch.tensor(X_val),    y_val=torch.tensor(y_val),
        X_test=torch.tensor(X_test),  y_test=torch.tensor(y_test),
        meta=dict(spurious_corr=spurious_corr, signal_dims=signal_dims,
                  spurious_dims=spurious_dims, ood_type="spurious_scale_shift",
                  n_classes=n_classes),
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATASET_REGISTRY = {
    "a":  make_dataset_a,
    "b":  make_dataset_b,
    "b2": make_dataset_b2,
    "c":  make_dataset_c,
    "d":  make_dataset_d,
    "e":  make_dataset_e,
}


def get_dataset(name: str, **kwargs) -> DatasetBundle:
    key = name.lower().replace("dataset_", "").strip()
    if key not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {list(DATASET_REGISTRY)}")
    return DATASET_REGISTRY[key](**kwargs)


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for k, fn in DATASET_REGISTRY.items():
        db = fn()
        print(
            f"Dataset {k.upper()}: {db.name}\n"
            f"  train={db.X_train.shape}  val={db.X_val.shape}  "
            f"test={db.X_test.shape}  classes={db.n_classes}\n"
        )
    print("datasets.py smoke test passed.")
