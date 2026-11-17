"""Run the unified sampler refresh and the four new experiments.

Archives the current S&P and simulation CSVs, then writes results/unified/.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import shutil
import time
from pathlib import Path

import numpy as np

from mnreg.model import GibbsConfig
from run_revision_diagnostics import monte_carlo
from run_simulation import run_gibbs_recovery, run_oos
from run_unified_revision import OUT, run_boundary, run_collinear, run_loto, run_schur

ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "results" / "archive_isotropic"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _dump(name, payload):
    path = OUT / name
    path.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def archive():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in (
        "gibbs_recovery_posterior.csv",
        "simulation_oos.csv",
        "simulation_meta.json",
        "sp100_posterior.csv",
        "sp100_running_oos.csv",
        "sp100_showcase.json",
        "sp100_meta.json",
    ):
        source = ROOT / "results" / name
        if source.exists() and not (ARCHIVE / name).exists():
            shutil.copy2(source, ARCHIVE / name)
            print(f"archived {name}", flush=True)


def table4():
    print("=== Table 4 ===", flush=True)
    t0 = time.time()
    cfg = GibbsConfig(n_iter=6000, burn=3000, n_chains=4, seed=2026, target_accept=0.30, adapt_covariance=True)
    recovery = run_gibbs_recovery(200, np.random.default_rng(2026), cfg)
    table = recovery["summary"]["table"]
    rhats = [row["Rhat"] for row in table]
    _dump("table4.json", {"table": table, "rhat_max": max(rhats), "elapsed_sec": time.time() - t0, "adapt_covariance": True})


def table4a():
    print("=== Table 4a ===", flush=True)
    payload = monte_carlo()
    _dump("table4a.json", payload)


def table5():
    print("=== Table 5 ===", flush=True)
    t0 = time.time()
    cfg = GibbsConfig(n_iter=2500, burn=1200, n_chains=2, seed=7, target_accept=0.30, adapt_covariance=True)
    frame, _showcase = run_oos(200, 30, np.random.default_rng(2026), cfg)
    means = {column: float(frame[column].mean()) for column in frame.columns if "RMSE" in column or "MAE" in column}
    _dump("table5.json", {"means": means, "elapsed_sec": time.time() - t0, "adapt_covariance": True})


def sp100():
    print("=== S&P 100 ===", flush=True)
    import run_sp100
    run_sp100.main()
    showcase = json.loads((ROOT / "results" / "sp100_showcase.json").read_text(encoding="utf-8"))
    posterior = (ROOT / "results" / "sp100_posterior.csv").read_text(encoding="utf-8")
    oos = (ROOT / "results" / "sp100_running_oos.csv").read_text(encoding="utf-8")
    (OUT / "sp100_showcase.json").write_text(json.dumps(showcase, indent=2), encoding="utf-8")
    (OUT / "sp100_posterior.csv").write_text(posterior, encoding="utf-8")
    (OUT / "sp100_running_oos.csv").write_text(oos, encoding="utf-8")


def main():
    archive()
    smoke = OUT / "boundary_0999.json"
    if smoke.exists():
        smoke.unlink()
    table4()
    table4a()
    run_schur()
    run_collinear()
    run_loto()
    table5()
    sp100()
    run_boundary()
    print("ALL UNIFIED JOBS FINISHED", flush=True)


if __name__ == "__main__":
    main()
