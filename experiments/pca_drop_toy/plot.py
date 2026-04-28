"""
Plot utilities for PCA-drop toy experiments.

Generates:
  - loss / accuracy curves
  - effective rank over time
  - eigenvalue spectrum (bar chart)
  - ablation summary bar chart
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Single-run plots
# ---------------------------------------------------------------------------

def plot_training_curves(results: dict, out_path: str):
    hist = results["history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(hist["train_loss"]) + 1)

    ax = axes[0]
    ax.plot(epochs, hist["train_loss"], label="train loss")
    ax.plot(epochs, hist["val_loss"],   label="val loss")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title(f"{results.get('name', '')}  loss")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, hist["train_acc"], label="train acc")
    ax.plot(epochs, hist["val_acc"],   label="val acc")
    ax.axhline(results.get("test_acc", 0), color="red", linestyle="--",
               label=f"test acc={results.get('test_acc', 0):.3f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
    ax.set_title(f"{results.get('name', '')}  accuracy")
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_eigenvalue_spectrum(eigenvalues: List[float], title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    k = len(eigenvalues)
    ax.bar(range(1, k + 1), eigenvalues, color="steelblue")
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Singular value")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_effective_rank(eff_ranks: List[float], title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(eff_ranks, marker="o")
    ax.set_xlabel("Checkpoint (×10 epochs)")
    ax.set_ylabel("Effective rank")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-run ablation plot
# ---------------------------------------------------------------------------

def plot_ablation(
    results_list: List[dict],
    x_key: str,
    y_key: str = "test_acc",
    title: str = "Ablation",
    out_path: str = "ablation.png",
):
    """
    results_list: list of result dicts, each with a 'name', 'test_acc', etc.
    x_key: key in config (or result) to use as x-axis label.
    """
    names, ys = [], []
    for r in results_list:
        # Try to extract x_key from config
        val = r.get("config", {}).get(x_key, r.get(x_key, r.get("name", "?")))
        names.append(str(val))
        ys.append(r.get(y_key, 0))

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.2), 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.bar(names, ys, color=colors)
    for bar, y in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{y:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(title)
    ax.set_ylim(0, max(ys) * 1.15 if ys else 1.0)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved ablation plot: {out_path}")


# ---------------------------------------------------------------------------
# Multi-run comparison (overlay curves)
# ---------------------------------------------------------------------------

def plot_comparison_curves(
    results_list: List[dict],
    metric: str = "val_acc",
    title: str = "Comparison",
    out_path: str = "comparison.png",
):
    fig, ax = plt.subplots(figsize=(10, 5))
    for r in results_list:
        hist = r.get("history", {})
        vals = hist.get(metric, [])
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, label=r.get("name", "?"))
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Saved comparison plot: {out_path}")


# ---------------------------------------------------------------------------
# Auto-plot from results directory
# ---------------------------------------------------------------------------

def auto_plot_dir(results_dir: str):
    """Scan a results directory and generate all plots."""
    root = Path(results_dir)
    json_files = list(root.rglob("results.json"))
    if not json_files:
        print(f"No results.json found under {root}")
        return

    all_results = []
    for jf in json_files:
        with open(jf) as f:
            r = json.load(f)
        out_base = jf.parent
        all_results.append(r)

        # Training curves
        plot_training_curves(r, str(out_base / "training_curves.png"))

        # Eigenvalue spectrum (last checkpoint)
        if r["history"].get("eigenvalues"):
            last_eigs = r["history"]["eigenvalues"][-1]
            plot_eigenvalue_spectrum(
                last_eigs,
                title=f"{r.get('name','')}  eigenvalue spectrum (final)",
                out_path=str(out_base / "eigenvalue_spectrum.png"),
            )

        # Effective rank
        if r["history"].get("eff_rank"):
            plot_effective_rank(
                r["history"]["eff_rank"],
                title=f"{r.get('name','')}  effective rank",
                out_path=str(out_base / "effective_rank.png"),
            )

    # Ablation bar chart across all runs
    if len(all_results) > 1:
        plot_ablation(
            all_results,
            x_key="name",
            y_key="test_acc",
            title="Test Accuracy Comparison",
            out_path=str(root / "ablation_test_acc.png"),
        )
        plot_comparison_curves(
            all_results,
            metric="val_acc",
            title="Val Accuracy Comparison",
            out_path=str(root / "comparison_val_acc.png"),
        )

    print(f"\nPlots generated under {root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", help="Directory containing results.json files")
    args = p.parse_args()
    auto_plot_dir(args.results_dir)
