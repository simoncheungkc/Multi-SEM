"""Strict-BH 63-month RMSE and the radial-prior identity check."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_sp100 as sp
from mnreg.model import screen_networks
from mnreg.networks import row_normalize
from run_postprocess_revision import verify_radial_prior
from run_v28_analyses import _split_indices, _sp_showcase

ROOT = Path(__file__).resolve().parent


def strict_bh_rmse() -> dict:
    saved = pd.read_csv(ROOT / "results" / "unified" / "sp100_running_oos.csv")
    saved["date"] = pd.to_datetime(saved["date"])
    by_date = {pd.Timestamp(row.date).normalize(): row for row in saved.itertuples(index=False)}
    panel, fund, dates, _showcase, _date = _sp_showcase()
    staged = []
    for date in dates:
        if pd.Timestamp(date) < pd.Timestamp("2021-06-30"):
            continue
        match = by_date.get(pd.Timestamp(date).normalize())
        if match is None:
            continue
        cs = sp.align_cross_section(panel, fund, date)
        if cs is None:
            continue
        staged.append((date, cs, match))
    august = staged[-1][1]
    _train, test = _split_indices(len(august["Y"]))
    holdout = set(np.array(august["tickers"])[test])
    chosen = []
    for date, cs, match in staged:
        tickers = np.array(cs["tickers"])
        is_test = np.array([ticker in holdout for ticker in tickers])
        train_i = np.where(~is_test)[0]
        if is_test.sum() < 8 or train_i.size < 30:
            continue
        trained = [row_normalize(A[np.ix_(train_i, train_i)]) for A in cs["As_raw"]]
        screen = screen_networks(
            cs["Y"][train_i], cs["X"][train_i], trained, names=cs["names"], alpha=0.05, fdr=True
        )
        active = bool(np.any(screen["selected"]))
        ols_rmse = float(match.OLS_RMSE)
        multi_rmse = float(match.MultiSEM_RMSE)
        ols_mae = float(match.OLS_MAE)
        multi_mae = float(match.MultiSEM_MAE)
        chosen.append({
            "rmse": multi_rmse if active else ols_rmse,
            "mae": multi_mae if active else ols_mae,
            "bh_active": active,
        })
    frame = pd.DataFrame(chosen)
    payload = {
        "months": int(len(frame)),
        "bh_months": int(frame["bh_active"].sum()),
        "safeguard_months": int((~frame["bh_active"]).sum()),
        "strict_bh_or_ols_rmse": float(frame["rmse"].mean()),
        "strict_bh_or_ols_mae": float(frame["mae"].mean()),
    }
    print(payload, flush=True)
    out = ROOT / "results" / "v28" / "strict_bh.json"
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    verify_radial_prior()
    strict_bh_rmse()
