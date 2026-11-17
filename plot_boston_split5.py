"""Holdout scatter for Boston split 5, matching the simulation and S&P 100 plots."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import run_boston as rb

ROOT = Path(__file__).resolve().parent
# Reported chains live in the adaptive archive. The scatter is a diagnostic figure.
JSON_DIR = Path(os.environ.get("BOSTON_RESULTS", ROOT / "results" / "boston_adaptive"))
FIG_DIR = JSON_DIR
OUT_SCATTER = FIG_DIR / "boston_split5_pred_vs_actual.png"
OUT_S9 = FIG_DIR / "figure_s9.png"


def split_predictions(split_id: int = 5):
    data = rb.load_boston()
    built = rb.build_networks(data)
    record = json.loads((JSON_DIR / f"split_{split_id:02d}.json").read_text(encoding="utf-8"))
    seed = int(record["seed"])
    part = np.random.default_rng([seed, 1])
    train = np.sort(part.choice(rb.N, size=rb.N_TRAIN, replace=False))
    test = np.sort(np.setdiff1d(np.arange(rb.N), train))
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = raw[train].std(axis=0, ddof=0)
    spread = np.where(spread < 1e-12, 1.0, spread)
    scaled = (raw - center) / spread
    x_train = np.column_stack([np.ones(rb.N_TRAIN), scaled[train]])
    x_test = np.column_stack([np.ones(rb.N_TEST), scaled[test]])
    y_train = data["y"][train]
    y_test = data["y"][test]
    beta, _, _ = rb.ols(y_train, x_train)
    ols_hat = x_test @ beta
    names = record["multi"]["names"]
    multi_hat = rb.precision_kriging(
        y_train,
        x_train,
        x_test,
        train,
        test,
        np.asarray(record["multi"]["lambda"], dtype=float),
        np.asarray(record["multi"]["beta"], dtype=float),
        [built["adjacency"][name] for name in names],
    )
    return y_test, ols_hat, multi_hat, record, test


def write_figure_s9(data, y_test, yhat, test):
    """Supplementary Figure S9: municipal centroids and the split-5 holdout residual."""
    xy = data["xy"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    axes[0].scatter(xy[:, 0], xy[:, 1], c=data["town_id"], s=12, cmap="nipy_spectral", linewidths=0)
    axes[0].set_title("Tract centroids by municipality")
    axes[0].set_xlabel("UTM easting (m)")
    axes[0].set_ylabel("UTM northing (m)")
    axes[0].set_aspect("equal")
    residual = y_test - yhat
    scale = max(float(np.max(np.abs(residual))), 1e-6)
    sc = axes[1].scatter(
        xy[test, 0],
        xy[test, 1],
        c=residual,
        s=28,
        cmap="coolwarm",
        vmin=-scale,
        vmax=scale,
        linewidths=0,
    )
    axes[1].set_title("Holdout residual, split 5")
    axes[1].set_xlabel("UTM easting (m)")
    axes[1].set_aspect("equal")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="log value")
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    OUT_S9.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_S9, dpi=160)
    plt.close(fig)
    print(f"wrote {OUT_S9}")


def main():
    rmses = []
    for path in sorted(JSON_DIR.glob("split_[0-9][0-9].json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        rmses.append((int(row["split"]), float(row["multi"]["RMSE"])))
    ordered = sorted(rmses, key=lambda item: item[1])
    median_split, median_rmse = ordered[len(ordered) // 2]
    y, ols_hat, multi_hat, record, test = split_predictions(5)
    m_ols = rb.metrics_rmse_mae(y, ols_hat)
    m_multi = rb.metrics_rmse_mae(y, multi_hat)
    if abs(m_ols["RMSE"] - record["ols"]["RMSE"]) > 1e-8:
        raise SystemExit(f"OLS RMSE {m_ols['RMSE']} != stored {record['ols']['RMSE']}")
    if abs(m_multi["RMSE"] - record["multi"]["RMSE"]) > 5e-4:
        raise SystemExit(f"Multi RMSE {m_multi['RMSE']} != stored {record['multi']['RMSE']}")

    panels = [
        (ols_hat, "OLS (no network)", m_ols),
        (multi_hat, "Multi-SEM (precision kriging)", m_multi),
    ]
    data_lo = min(y.min(), min(pred.min() for pred, _, _ in panels))
    data_hi = max(y.max(), max(pred.max() for pred, _, _ in panels))
    lo, hi = 2.0, 4.3
    if data_lo < lo or data_hi > hi:
        raise SystemExit(f"points outside [2.0, 4.3]: {data_lo:.3f}, {data_hi:.3f}")
    fig, axes = plt.subplots(1, 2, figsize=(6.26, 3.45), sharex=True, sharey=True)
    for ax, (pred, name, met) in zip(axes, panels):
        ax.scatter(y, pred, s=16, c="#4C6A92", alpha=0.85, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="#B85C38", lw=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([2.0, 2.5, 3.0, 3.5, 4.0])
        ax.set_yticks([2.0, 2.5, 3.0, 3.5, 4.0])
        ax.tick_params(labelsize=8)
        ax.set_title(name, fontsize=9)
        ax.set_xlabel("Actual log(cmedv)", fontsize=9)
        ax.set_ylabel("Predicted log(cmedv)", fontsize=9)
        ax.text(
            0.04,
            0.96,
            f"RMSE={met['RMSE']:.4f}\nMAE={met['MAE']:.4f}\ncorr={met['corr']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
        )
    fig.suptitle("Boston housing holdout, split 5 (n* = 101)", fontsize=10)
    fig.tight_layout()
    OUT_SCATTER.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_SCATTER, dpi=200)
    print(f"wrote {OUT_SCATTER}")
    plt.close(fig)
    write_figure_s9(rb.load_boston(), y, multi_hat, test)
    print(f"median multi RMSE split {median_split} value {median_rmse:.6f}")
    print(f"split 5 OLS {m_ols['RMSE']:.6f} corr {m_ols['corr']:.6f}")
    print(f"split 5 multi {m_multi['RMSE']:.6f} corr {m_multi['corr']:.6f}")
    print(f"sd actual {y.std(ddof=0):.4f} sd OLS {ols_hat.std(ddof=0):.4f} sd kriging {multi_hat.std(ddof=0):.4f}")
    print(f"range {data_lo:.4f} {data_hi:.4f}")
    print(f"rmse reduction {(m_ols['RMSE'] - m_multi['RMSE']) / m_ols['RMSE']:.6f}")
    print(f"mae reduction {(m_ols['MAE'] - m_multi['MAE']) / m_ols['MAE']:.6f}")


if __name__ == "__main__":
    main()
