"""Table 4a with a predefined split-R-hat extension rule.

Every replication starts at 4 chains of 4,000 iterations (2,000 burn-in).
If the maximum split-R-hat exceeds 1.05, the chain length and burn-in are
doubled, and that check is repeated, until the replication is at or below
1.05 or the length reaches 32,000 iterations.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from run_revision_diagnostics import TRUE_BETA, TRUE_LAMBDA, _one_recovery

OUT = Path(__file__).resolve().parent / "results" / "unified" / "table4a.json"
RHAT_LIMIT = 1.05
MAX_ITER = 32_000


def _extended_recovery(task):
    seed, n_iter, burn, n_chains = task
    used_iter, used_burn = n_iter, burn
    out = _one_recovery((seed, used_iter, used_burn, n_chains))
    passes = 1
    while out["rhat_max"] > RHAT_LIMIT and used_iter < MAX_ITER:
        used_iter *= 2
        used_burn *= 2
        out = _one_recovery((seed, used_iter, used_burn, n_chains))
        passes += 1
    out["seed"] = seed
    out["n_iter_used"] = used_iter
    out["burn_used"] = used_burn
    out["passes"] = passes
    return out


def _summarize(err, cover, truth, labels):
    rows = []
    count = float(len(err))
    for index, label in enumerate(labels):
        coverage = float(cover[:, index].mean())
        squared = err[:, index] ** 2
        rmse = float(np.sqrt(np.mean(squared)))
        rmse_mcse = float(np.std(squared, ddof=1) / (2.0 * rmse * np.sqrt(count))) if rmse > 0 else 0.0
        rows.append({
            "parameter": label,
            "truth": float(truth[index]),
            "bias": float(err[:, index].mean()),
            "rmse": rmse,
            "rmse_mcse": rmse_mcse,
            "coverage": coverage,
            "coverage_mcse": float(np.sqrt(coverage * (1.0 - coverage) / count)),
        })
    return rows


def main() -> None:
    tasks = [(50_000 + rep, 4000, 2000, 4) for rep in range(200)]
    started = time.time()
    rows = []
    workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, out in enumerate(pool.map(_extended_recovery, tasks, chunksize=1)):
            rows.append(out)
            if (index + 1) % 10 == 0:
                worst = max(row["rhat_max"] for row in rows)
                print(f"recovery {index + 1}/200 max R-hat {worst:.4f}", flush=True)
    lam = np.array([row["lambda"]["bias_draw"] for row in rows])
    bet = np.array([row["beta"]["bias_draw"] for row in rows])
    lam_c = np.array([row["lambda"]["cover"] for row in rows])
    bet_c = np.array([row["beta"]["cover"] for row in rows])
    rhat = np.array([row["rhat_max"] for row in rows], dtype=float)
    payload = {
        "R": 200,
        "n": 200,
        "n_iter": 4000,
        "burn": 2000,
        "n_chains": 4,
        "extension_rule": "double length and burn-in while max split-R-hat > 1.05, up to 32000 iterations",
        "rhat_limit": RHAT_LIMIT,
        "elapsed_sec": time.time() - started,
        "lambda": _summarize(lam, lam_c, TRUE_LAMBDA, rows[0]["names"]),
        "beta": _summarize(bet, bet_c, TRUE_BETA, ["intercept", "x1", "x2", "x3", "x4", "x5"]),
        "rhat_max": float(rhat.max()),
        "rhat_median": float(np.median(rhat)),
        "rhat_p95": float(np.quantile(rhat, 0.95)),
        "rhat_by_rep": rhat.tolist(),
        "n_iter_by_rep": [int(row["n_iter_used"]) for row in rows],
        "extended_reps": int(sum(row["n_iter_used"] > 4000 for row in rows)),
        "min_network_ess": float(min(row["min_network_ess"] for row in rows)),
        "adapt_covariance": True,
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"wrote {OUT} rhat_max={payload['rhat_max']:.4f} "
        f"median={payload['rhat_median']:.4f} extended={payload['extended_reps']} "
        f"min_ess={payload['min_network_ess']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
