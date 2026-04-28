"""
train.py  ─  Training script for PCA regularization toy / MLP experiments.

Usage
─────
    # Single run
    python train.py --config configs/toy_baseline.yaml

    # Override specific hyperparameters
    python train.py --config configs/toy_nuisance_high_variance.yaml \\
        --seed 1 --epochs 100 --pca_momentum 0.995 --pca_top_k 4

    # Run all baseline configs in one shot (see scripts/run_experiments.sh)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from datasets import get_dataset, DatasetBundle
from models import build_model
from metrics import (
    compute_feature_metrics,
    gradient_norm,
    accuracy,
)
from pca_regularizer import PCARegularizer


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    collect_features: bool = False,
) -> Dict[str, Any]:
    model.eval()
    correct = total = 0
    losses: list = []
    loss_fn = nn.CrossEntropyLoss()
    all_feats: list = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        losses.append(loss_fn(logits, y).item())
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += len(y)
        if collect_features:
            all_feats.append(x.cpu())

    result: Dict[str, Any] = {
        "loss": float(np.mean(losses)),
        "acc": correct / total if total > 0 else 0.0,
    }
    if collect_features and all_feats:
        result["features"] = torch.cat(all_feats, 0)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Fetch all PCARegularizer modules from a model
# ─────────────────────────────────────────────────────────────────────────────

def _collect_pca_regularizers(model: nn.Module) -> list:
    return [m for m in model.modules() if isinstance(m, PCARegularizer)]


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, cfg: Dict[str, Any], out_dir: Path):
        self.cfg = cfg
        self.out_dir = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device(
            cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        set_seed(cfg.get("seed", 42))

        # ── dataset ───────────────────────────────────────────────────
        dataset_kwargs = cfg.get("dataset_kwargs", {})
        dataset_kwargs["seed"] = cfg.get("seed", 42)
        self.bundle: DatasetBundle = get_dataset(
            cfg["dataset"], **dataset_kwargs
        )
        self.train_loader, self.val_loader, self.test_loader = self.bundle.loaders(
            batch_size=cfg.get("batch_size", 256)
        )
        # OOD loader: for dataset C we swap test as OOD;
        # for others, test == OOD (same distribution, held-out split)
        self.ood_loader = self.test_loader

        # ── model ─────────────────────────────────────────────────────
        model_cfg = cfg.get("model", {})
        pca_cfg_raw = cfg.get("pca_drop", {})
        pca_cfg = pca_cfg_raw if pca_cfg_raw.get("enabled", False) else None

        self.model = build_model(
            arch=model_cfg.get("arch", "mlp2"),
            in_dim=self.bundle.dim,
            n_classes=self.bundle.n_classes,
            dropout=model_cfg.get("dropout", 0.0),
            pca_cfg=pca_cfg,
            pca_insert_after=model_cfg.get("pca_insert_after", [0]),
        ).to(self.device)

        # ── optimizer ─────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.get("lr", 1e-3),
            weight_decay=cfg.get("weight_decay", 1e-4),
        )
        self.loss_fn = nn.CrossEntropyLoss()
        self.epochs = cfg.get("epochs", 80)
        self.log_every = max(1, self.epochs // 20)

        # ── logging ───────────────────────────────────────────────────
        self.jsonl_path = out_dir / "train_log.jsonl"
        self.summary: Dict[str, Any] = {
            "name": cfg.get("name", "unnamed"),
            "dataset": self.bundle.name,
            "config": cfg,
        }

    # ──────────────────────────────────────────────────────────────────
    # One training epoch
    # ──────────────────────────────────────────────────────────────────

    def _train_epoch(self) -> Dict[str, float]:
        self.model.train()
        correct = total = 0
        losses: list = []
        grad_norms: list = []

        for x, y in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.loss_fn(logits, y)
            loss.backward()
            gn = gradient_norm(self.model)
            self.optimizer.step()

            losses.append(loss.item())
            grad_norms.append(gn)
            correct += (logits.argmax(1) == y).sum().item()
            total += len(y)

        return {
            "train_loss": float(np.mean(losses)),
            "train_acc": correct / total,
            "grad_norm": float(np.mean([g for g in grad_norms if np.isfinite(g)])) if grad_norms else float("nan"),
        }

    # ──────────────────────────────────────────────────────────────────
    # Collect feature statistics from PCARegularizer modules
    # ──────────────────────────────────────────────────────────────────

    def _pca_stats_log(self) -> dict:
        regs = _collect_pca_regularizers(self.model)
        if not regs:
            return {}
        # Take first regularizer's stats (most experiments have one)
        reg = regs[0]
        return reg.log_stats()

    # ──────────────────────────────────────────────────────────────────
    # Feature metrics on input data
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _input_feature_metrics(self) -> dict:
        """Spectral stats of raw train inputs (for tracking spurious structure)."""
        X = self.bundle.X_train.to(self.device)
        meta = self.bundle.meta
        spurious_dim = 0 if "spurious_corr" in meta else None
        m = compute_feature_metrics(X, evr_k=min(4, self.bundle.dim), spurious_dim=spurious_dim)
        return {f"input/{k}": v for k, v in m.items()}

    # ──────────────────────────────────────────────────────────────────
    # Full training loop
    # ──────────────────────────────────────────────────────────────────

    def train(self) -> Dict[str, Any]:
        print(f"\n{'='*64}")
        print(f"  Experiment : {self.cfg.get('name', 'unnamed')}")
        print(f"  Dataset    : {self.bundle.name}  (dim={self.bundle.dim}, "
              f"classes={self.bundle.n_classes})")
        pca_raw = self.cfg.get("pca_drop", {})
        print(f"  PCA mode   : {pca_raw.get('mode', 'none') if pca_raw.get('enabled') else 'none'}")
        print(f"  Device     : {self.device}")
        print(f"{'='*64}\n")

        t0 = time.time()
        best_val_acc = 0.0
        epoch_rows: list = []

        for epoch in range(1, self.epochs + 1):
            row = {"epoch": epoch}

            # Train
            tr = self._train_epoch()
            row.update(tr)

            # Validation
            val = evaluate_loader(self.model, self.val_loader, self.device)
            row["val_loss"] = val["loss"]
            row["val_acc"]  = val["acc"]

            # OOD / test
            ood = evaluate_loader(self.model, self.ood_loader, self.device)
            row["ood_loss"] = ood["loss"]
            row["ood_acc"]  = ood["acc"]

            # PCA regularizer stats
            if epoch % self.log_every == 0:
                row.update(self._pca_stats_log())
                row.update(self._input_feature_metrics())

            row["elapsed"] = round(time.time() - t0, 2)

            # Save best model
            if val["acc"] > best_val_acc:
                best_val_acc = val["acc"]
                torch.save(self.model.state_dict(), self.out_dir / "best_model.pt")

            # Append to JSONL
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps(row) + "\n")

            epoch_rows.append(row)

            # Console print at intervals
            if epoch % self.log_every == 0 or epoch == 1:
                pca_evr = row.get("pca/expl_var_ratio", float("nan"))
                pca_er  = row.get("pca/effective_rank", float("nan"))
                print(
                    f"Ep {epoch:3d}/{self.epochs}  "
                    f"tr={tr['train_loss']:.4f}/{tr['train_acc']:.3f}  "
                    f"val={val['loss']:.4f}/{val['acc']:.3f}  "
                    f"ood={ood['acc']:.3f}  "
                    f"evr@k={pca_evr:.3f}  er={pca_er:.1f}  "
                    f"t={row['elapsed']:.1f}s"
                )

        # ── final evaluation ──────────────────────────────────────────
        best_ckpt = self.out_dir / "best_model.pt"
        if best_ckpt.exists():
            self.model.load_state_dict(
                torch.load(best_ckpt, map_location=self.device, weights_only=True)
            )
        test_metrics = evaluate_loader(
            self.model, self.test_loader, self.device, collect_features=True
        )
        test_feats = test_metrics.pop("features", None)
        test_feat_m = {}
        if test_feats is not None:
            test_feat_m = compute_feature_metrics(
                test_feats,
                evr_k=min(4, self.bundle.dim),
                spurious_dim=0 if "spurious_corr" in self.bundle.meta else None,
            )

        print(f"\nTest acc : {test_metrics['acc']:.4f}   (best val: {best_val_acc:.4f})")

        summary = {
            **self.summary,
            "best_val_acc":   best_val_acc,
            "test_acc":       test_metrics["acc"],
            "test_loss":      test_metrics["loss"],
            "test_feature_metrics": test_feat_m,
            "total_epochs":   self.epochs,
            "total_time_s":   round(time.time() - t0, 2),
        }
        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Logs    : {self.jsonl_path}")
        print(f"Summary : {self.out_dir / 'summary.json'}")
        return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Toy MLP + PCA regularization experiment runner"
    )
    p.add_argument("--config",      required=True, help="Path to YAML config file")
    p.add_argument("--seed",        type=int,   default=None, help="Override seed")
    p.add_argument("--epochs",      type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--out_dir",     type=str,   default=None, help="Override output directory")
    # PCA overrides
    p.add_argument("--pca_mode",    type=str,   default=None,
                   choices=["none", "attenuate_topk", "drop_topk", "drop_all_pc_weighted"])
    p.add_argument("--pca_top_k",   type=int,   default=None)
    p.add_argument("--pca_alpha",   type=float, default=None)
    p.add_argument("--pca_drop_prob", type=float, default=None)
    p.add_argument("--pca_momentum", type=float, default=None)
    p.add_argument("--pca_warmup",  type=int,   default=None)
    return p.parse_args()


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    """Patch cfg dict with CLI overrides."""
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["lr"] = args.lr

    pca = cfg.setdefault("pca_drop", {})
    if args.pca_mode is not None:
        pca["mode"] = args.pca_mode
        pca["enabled"] = args.pca_mode != "none"
    if args.pca_top_k is not None:
        pca["top_k"] = args.pca_top_k
    if args.pca_alpha is not None:
        pca["alpha"] = args.pca_alpha
    if args.pca_drop_prob is not None:
        pca["drop_prob"] = args.pca_drop_prob
    if args.pca_momentum is not None:
        pca["momentum"] = args.pca_momentum
    if args.pca_warmup is not None:
        pca["warmup_steps"] = args.pca_warmup

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cfg = apply_overrides(cfg, args)

    run_name = cfg.get("name", Path(args.config).stem)
    seed = cfg.get("seed", 42)

    if args.out_dir:
        out_dir = Path(args.out_dir) / f"seed{seed}"
    else:
        out_dir = Path("outputs") / run_name / f"seed{seed}"

    trainer = Trainer(cfg, out_dir)
    trainer.train()


if __name__ == "__main__":
    main()
