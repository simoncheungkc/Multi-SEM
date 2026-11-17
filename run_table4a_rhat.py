"""Save the per-replication split-R-hat distribution for Table 4a.

Same design as run_table4a_long.py: 200 replications, 4 chains, 4,000 iterations,
2,000 burn-in, adapt_covariance=True, seeds 50000+rep.
"""
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

from run_revision_diagnostics import _one_recovery

OUT = Path("results/unified/table4a_rhat.json")


def main():
    tasks = [(50_000 + rep, 4000, 2000, 4) for rep in range(200)]
    started = time.time()
    rows = []
    workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for index, out in enumerate(pool.map(_one_recovery, tasks, chunksize=1)):
            rows.append(float(out["rhat_max"]))
            if (index + 1) % 20 == 0:
                print(f"rhat {index + 1}/200 max {max(rows):.4f}", flush=True)
    values = np.asarray(rows, dtype=float)
    payload = {
        "R": 200,
        "n_iter": 4000,
        "burn": 2000,
        "n_chains": 4,
        "adapt_covariance": True,
        "seed0": 50000,
        "rhat_max": float(values.max()),
        "rhat_median": float(np.median(values)),
        "rhat_p95": float(np.quantile(values, 0.95)),
        "rhat_mean": float(values.mean()),
        "elapsed_sec": time.time() - started,
        "rhat_by_rep": values.tolist(),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT, payload["rhat_median"], payload["rhat_p95"], payload["rhat_max"], flush=True)


if __name__ == "__main__":
    main()
