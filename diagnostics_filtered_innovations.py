"""Moran's I for OLS residuals and Multi-SEM filtered innovations.

The innovations are epsilon_hat = V_lambda_hat (Y_t - X_t beta_hat) on each
training block. The filtered statistic uses an intercept-only residual maker.
"""

from __future__ import annotations

import json
from pathlib import Path

import run_revision_tasks as revision


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "boston_adaptive" / "filtered_innovations.json"


def main() -> dict:
    records = revision._load_adaptive_records()
    revision._prepare({"adaptive": True, "n_iter": 8000, "burn": 4000, "n_chains": 4})
    result = revision.task6(records)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    geo = result["summary"]["geo"]
    checks = {
        "ols_mean_I": (round(geo["ols_mean_I"], 3), 0.442),
        "filtered_mean_I": (round(geo["filtered_mean_I"], 3), -0.021),
        "split5_I": (round(geo["filtered_split5_I"], 3), -0.012),
        "split5_p": (round(geo["filtered_split5_p"], 3), 0.755),
    }
    for name, (got, target) in checks.items():
        if got != target:
            raise RuntimeError(f"geographic {name} {got} != {target}")
    shares = {name: result["summary"][name]["filtered_frac_p_gt_030"] for name in result["summary"]}
    if any(share < 0.80 for share in shares.values()):
        raise RuntimeError(f"a layer has fewer than 80% of splits with p>0.30: {shares}")
    print(f"filtered-innovation Moran statistics match Table S11 -> {OUT}", flush=True)
    print(json.dumps(result["summary"], indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
