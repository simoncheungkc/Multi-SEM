"""Beta(1, 2) refit of the August 2026 S&P showcase.

Same 80/20 split, seed, and isotropic proposal as run_sp100.main. Only the
prior on L changes. Does not overwrite the archived showcase posterior.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import run_sp100 as sp
from mnreg.model import GibbsConfig, gibbs_multi_sem
from run_revision_tasks import _hist_mode

OUT = Path(__file__).resolve().parent / "results" / "revision" / "august_beta_l1.json"


def showcase():
    wiki = sp.load_constituents()
    tickers = wiki["Symbol"].dropna().unique().tolist()
    px = sp.download_prices(tickers)
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    fund = sp.fetch_fundamentals(tickers, wiki)
    panel = sp.monthly_panel(px)
    ret = panel["ret"].dropna(how="all")
    dates = ret.index[(ret.index >= "2021-01-01") & (ret.index <= "2026-08-31")]
    cs_show = None
    showcase_date = None
    for date in dates[::-1]:
        cs = sp.align_cross_section(panel, fund, date)
        if cs is not None and len(cs["tickers"]) >= 70:
            showcase_date = date
            cs_show = cs
            break
    if cs_show is None:
        raise RuntimeError("Could not build the August showcase")
    scr = sp.screen_networks(
        cs_show["Y"], cs_show["X"], cs_show["As"], names=cs_show["names"], alpha=0.05, fdr=True
    )
    sel_idx = np.where(scr["selected"])[0].tolist()
    if not sel_idx:
        sel_idx = np.where(scr["pvalues"] < 0.10)[0].tolist()
    if not sel_idx:
        sel_idx = [int(np.argmin(scr["pvalues"]))]
    keep, _, _ = sp.drop_collinear_networks(
        [cs_show["As_raw"][i] for i in sel_idx], [cs_show["names"][i] for i in sel_idx]
    )
    sel_idx = [sel_idx[i] for i in keep]
    rng = np.random.default_rng(2026)
    idx = rng.permutation(len(cs_show["Y"]))
    n_test = max(int(0.2 * len(cs_show["Y"])), 15)
    train_idx = np.sort(idx[n_test:])
    names = [cs_show["names"][i] for i in sel_idx]
    matrices = [
        sp.row_normalize(cs_show["As_raw"][i][np.ix_(train_idx, train_idx)]) for i in sel_idx
    ]
    return showcase_date, cs_show, train_idx, names, matrices


def main():
    t0 = time.time()
    showcase_date, cs_show, train_idx, names, matrices = showcase()
    print(
        f"showcase {showcase_date.date()} n={len(cs_show['tickers'])} "
        f"n_train={len(train_idx)} layers={names}",
        flush=True,
    )
    config = GibbsConfig(
        n_iter=8000,
        burn=4000,
        n_chains=4,
        seed=2026,
        lambda_prior="beta_l1",
        adapt_covariance=False,
    )
    result = gibbs_multi_sem(
        cs_show["Y"][train_idx],
        cs_show["X"][train_idx],
        matrices,
        names=names,
        config=config,
    )
    lam = result.samples["lambda"].reshape(-1, result.samples["lambda"].shape[-1])
    total = np.abs(lam).sum(axis=1)
    payload = {
        "date": str(pd_date(showcase_date)),
        "n": int(len(cs_show["tickers"])),
        "n_train": int(len(train_idx)),
        "layers": names,
        "lambda_prior": "beta_l1",
        "adapt_covariance": False,
        "n_iter": 8000,
        "burn": 4000,
        "n_chains": 4,
        "seed": 2026,
        "mean_L": float(total.mean()),
        "median_L": float(np.median(total)),
        "mode_L": _hist_mode(total),
        "q025_L": float(np.quantile(total, 0.025)),
        "q975_L": float(np.quantile(total, 0.975)),
        "p_L_gt_095": float(np.mean(total > 0.95)),
        "lambda_mean": lam.mean(axis=0).astype(float).tolist(),
        "acceptance": result.acceptance.astype(float).tolist(),
        "paper_prior_mean_L": 0.430,
        "elapsed_sec": time.time() - t0,
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


def pd_date(value):
    import pandas as pd

    return pd.Timestamp(value).date()


if __name__ == "__main__":
    main()
