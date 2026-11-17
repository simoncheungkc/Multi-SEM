"""Prior sensitivity for Table S4 on the n=200 synthetic design."""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from mnreg.model import GibbsConfig, gibbs_multi_sem
from run_simulation import TRUE_BETA, TRUE_LAMBDA, make_X, make_networks, simulate_y

OUT = Path(__file__).resolve().parent / "results" / "unified" / "table_s4_prior.json"

SETTINGS = [
    ("baseline", {"xi": 1.0, "alpha0": 0.001, "gamma0": 0.001, "lambda_prior": "gaussian"}),
    ("wider", {"xi": 0.1, "alpha0": 0.001, "gamma0": 0.001, "lambda_prior": "gaussian"}),
    ("ridge", {"xi": 5.0, "alpha0": 0.001, "gamma0": 0.001, "lambda_prior": "gaussian"}),
    ("informative", {"xi": 1.0, "alpha0": 2.0, "gamma0": 1.0, "lambda_prior": "gaussian"}),
    ("beta", {"xi": 1.0, "alpha0": 0.001, "gamma0": 0.001, "lambda_prior": "beta_l1"}),
]


def _design():
    rng = np.random.default_rng(2026)
    As, _, names, _ = make_networks(200, np.random.default_rng(int(rng.integers(1, 2**31 - 1))))
    X = make_X(200, rng)
    Y = simulate_y(X, TRUE_BETA, TRUE_LAMBDA, As, 1.0, rng)
    return Y, X, As, names


def _fit(task):
    label, kwargs, Y, X, As, names = task
    cfg = GibbsConfig(
        n_iter=6000,
        burn=3000,
        n_chains=4,
        seed=2026,
        target_accept=0.30,
        adapt_covariance=True,
        **kwargs,
    )
    result = gibbs_multi_sem(Y, X, As, names=names, config=cfg)
    lam = result.samples["lambda"].reshape(-1, result.samples["lambda"].shape[-1])
    sig = result.samples["sigma2"].reshape(-1)
    mean = lam.mean(axis=0)
    lo = np.quantile(lam, 0.025, axis=0)
    hi = np.quantile(lam, 0.975, axis=0)
    total = np.abs(lam).sum(axis=1)
    return {
        "label": label,
        "lambda_mean": mean.tolist(),
        "lambda_lo": lo.tolist(),
        "lambda_hi": hi.tolist(),
        "inactive_max_abs_mean": float(np.max(np.abs(mean[2:]))),
        "sigma2_mean": float(sig.mean()),
        "sigma2_lo": float(np.quantile(sig, 0.025)),
        "sigma2_hi": float(np.quantile(sig, 0.975)),
        "mean_L": float(total.mean()),
        "p_L_gt_095": float(np.mean(total > 0.95)),
    }


def main():
    Y, X, As, names = _design()
    tasks = [(label, kwargs, Y, X, As, names) for label, kwargs in SETTINGS]
    rows = []
    with ProcessPoolExecutor(max_workers=5) as pool:
        for row in pool.map(_fit, tasks):
            rows.append(row)
            print(row["label"], [round(v, 3) for v in row["lambda_mean"][:2]], row["mean_L"], flush=True)
    OUT.write_text(json.dumps({"names": names, "rows": rows, "adapt_covariance": True}, indent=2), encoding="utf-8")
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
