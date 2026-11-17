"""August 2026 industry-layer conditional Gaussian Monte Carlo p-value.

The training block is the same 80/20 split used by the S&P showcase
(seed 2026). The Monte Carlo seed is 2026 as well, so the count is fixed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_sp100 as sp
from mnreg.model import moran_i_test, ols
from run_sp100_geo import binary_kappa


def mc_counts(resid, design, adjacency, rng, draws=499):
    observed = moran_i_test(resid, adjacency, design)
    n = design.shape[0]
    gram_inv = np.linalg.inv(design.T @ design)
    projector = np.eye(n) - design @ gram_inv @ design.T
    exceed_i = 0
    exceed_z = 0
    for _ in range(draws):
        simulated = projector @ rng.normal(size=n)
        draw = moran_i_test(simulated, adjacency, design)
        if abs(draw["I"]) >= abs(observed["I"]) - 1e-12:
            exceed_i += 1
        if abs(draw["z"]) >= abs(observed["z"]) - 1e-12:
            exceed_z += 1
    return {
        "I": observed["I"],
        "z": observed["z"],
        "gaussian_p": observed["pvalue"],
        "exceed_abs_I": exceed_i,
        "exceed_abs_z": exceed_z,
        "p_mc_I": (1 + exceed_i) / (draws + 1),
        "p_mc_z": (1 + exceed_z) / (draws + 1),
        "draws": draws,
    }


def main():
    wiki = sp.load_constituents()
    tickers = [t for t in wiki["Symbol"].dropna().unique().tolist()]
    px = sp.download_prices(tickers)
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    fund = sp.fetch_fundamentals(tickers, wiki)
    panel = sp.monthly_panel(px)
    dates = panel["ret"].dropna(how="all").index
    august = [d for d in dates if d.year == 2026 and d.month == 8]
    if not august:
        raise RuntimeError("no August 2026 month-end in the frozen panel")
    date = august[-1]
    cs = sp.align_cross_section(panel, fund, date)
    if cs is None:
        raise RuntimeError("August cross-section could not be built")
    industry = cs["names"].index("Industry (related companies)")
    n = len(cs["Y"])
    rng_split = np.random.default_rng(2026)
    order = rng_split.permutation(n)
    n_test = max(int(0.2 * n), 15)
    test = np.sort(order[:n_test])
    train = np.sort(order[n_test:])
    raw = cs["As_raw"][industry]
    out = {"date": str(pd_date(date)), "n": n, "n_train": int(train.size), "n_test": int(test.size)}
    for label, index in (("full", np.arange(n)), ("train", train)):
        weights = raw[np.ix_(index, index)]
        adjacency = sp.row_normalize(weights)
        y = cs["Y"][index]
        x = cs["X"][index]
        _, resid, _ = ols(y, x)
        block = mc_counts(resid, x, adjacency, np.random.default_rng(2026))
        block["kappa"] = binary_kappa(weights)
        block["psi"] = float(np.linalg.svd(adjacency, compute_uv=False)[0] / np.linalg.svd(adjacency, compute_uv=False).sum())
        out[label] = block
    path = Path("results/revision/august_industry_mc.json")
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


def pd_date(value):
    import pandas as pd

    return pd.Timestamp(value).date()


if __name__ == "__main__":
    main()
