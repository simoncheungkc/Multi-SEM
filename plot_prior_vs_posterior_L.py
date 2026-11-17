"""Prior shell p(L) proportional to L^{k-1} against the Boston Split 5 posterior.

The Gaussian-prior and Beta(1, 2) draws are the retained MCMC samples already
stored in results/revision. This script redraws Figure S10 from those samples.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_revision_tasks as revision


ROOT = Path(__file__).resolve().parent
REVISION = ROOT / "results" / "revision"


def main() -> dict:
    summary = revision.make_figure(REVISION)
    (REVISION / "figure_s10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    gauss = summary["gaussian_prior_posterior"]
    beta = summary["beta_prior_posterior"]
    prior_mean = summary["prior_uniform_ball_mean_L"]
    if round(prior_mean, 3) != 0.750:
        raise RuntimeError(f"uniform-ball prior mean {prior_mean} is not 0.750")
    if round(gauss["mean_L"], 3) != 0.852 or round(gauss["mode_L"], 3) != 0.844:
        raise RuntimeError(f"paper-prior posterior {gauss['mean_L']}, {gauss['mode_L']}")
    if round(beta["mean_L"], 3) != 0.836 or round(beta["mode_L"], 3) != 0.844:
        raise RuntimeError(f"Beta(1, 2) posterior {beta['mean_L']}, {beta['mode_L']}")
    print(f"Figure S10 written to {summary['figure']}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


if __name__ == "__main__":
    main()
