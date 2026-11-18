"""Paired radial deletion on the 28 screened Boston splits.

Uses the archived four-layer posterior means. The radial coefficient is removed
from the saved precision-kriging predictor. The other coefficients are not
refitted, and the Gibbs sampler is not run.
"""

from __future__ import annotations

import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
import pandas as pd

import run_boston as rb
from run_revision_tasks import nadeau_bengio

ROOT = Path(__file__).resolve().parent
JSON_DIR = ROOT / "results" / "boston_adaptive"
OUT = JSON_DIR / "radial_paired_ablation.json"


def paired_radial() -> dict:
    data = rb.load_boston()
    built = rb.build_networks(data)
    raw = data["raw"]
    rows = []
    for path in sorted(JSON_DIR.glob("split_[0-9][0-9].json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        names = list(record["multi"]["names"])
        if names != ["geo", "resolvent", "town", "radial"]:
            continue
        seed = int(record["seed"])
        part = np.random.default_rng([seed, 1])
        train = np.sort(part.choice(rb.N, size=rb.N_TRAIN, replace=False))
        test = np.sort(np.setdiff1d(np.arange(rb.N), train))
        center = raw[train].mean(axis=0)
        spread = raw[train].std(axis=0, ddof=0)
        spread = np.where(spread < 1e-12, 1.0, spread)
        scaled = (raw - center) / spread
        x_train = np.column_stack([np.ones(rb.N_TRAIN), scaled[train]])
        x_test = np.column_stack([np.ones(rb.N_TEST), scaled[test]])
        y_train = data["y"][train]
        y_test = data["y"][test]
        lam = np.asarray(record["multi"]["lambda"], dtype=float)
        beta = np.asarray(record["multi"]["beta"], dtype=float)
        nets = [built["adjacency"][name] for name in names]
        full = rb.precision_kriging(y_train, x_train, x_test, train, test, lam, beta, nets)
        reduced = rb.precision_kriging(
            y_train, x_train, x_test, train, test, lam[:3], beta, nets[:3]
        )
        rows.append(
            {
                "split": int(record["split"]),
                "stored_rmse": float(record["multi"]["RMSE"]),
                "rmse_four": float(np.sqrt(np.mean((full - y_test) ** 2))),
                "rmse_three": float(np.sqrt(np.mean((reduced - y_test) ** 2))),
                "mse_four": float(np.mean((full - y_test) ** 2)),
                "mse_three": float(np.mean((reduced - y_test) ** 2)),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 28:
        raise SystemExit(f"expected 28 radial splits, found {len(frame)}")
    statistic = nadeau_bengio(
        frame["mse_three"].to_numpy(),
        frame["mse_four"].to_numpy(),
        405,
        101,
        "four-layer minus radial-deleted predictor",
    )
    payload = {
        "n_splits": int(len(frame)),
        "mean_stored_rmse": float(frame["stored_rmse"].mean()),
        "mean_rmse_four": float(frame["rmse_four"].mean()),
        "mean_rmse_three_operator": float(frame["rmse_three"].mean()),
        "mean_paired_rmse_diff_four_minus_three": float(
            (frame["rmse_four"] - frame["rmse_three"]).mean()
        ),
        "max_abs_four_vs_stored": float((frame["rmse_four"] - frame["stored_rmse"]).abs().max()),
        "nadeau_bengio": statistic,
        "note": (
            "The three-operator predictor uses the archived four-layer posterior means "
            "with the radial coefficient removed. It is not a refitted three-layer posterior."
        ),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = paired_radial()
    nb = result["nadeau_bengio"]
    print(
        f"splits {result['n_splits']}  "
        f"mean RMSE diff {result['mean_paired_rmse_diff_four_minus_three']:.6e}  "
        f"t_{nb['df']} {nb['t']:.2f}  p {nb['p']:.2f}"
    )
