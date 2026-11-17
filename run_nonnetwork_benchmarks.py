"""Industry fixed effects and one-factor imputation on the saved S&P 100 splits.

Does not re-estimate Multi-SEM. Rebuilding each cross-section from the cached
prices is required only so the holdout returns and industry labels line up with
results/sp100_running_oos.csv. A month is kept only when the reconstructed OLS
holdout RMSE matches the saved file.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mnreg.model import add_intercept, ols
from run_sp100 import (
    align_cross_section,
    download_prices,
    fetch_fundamentals,
    load_constituents,
    monthly_panel,
)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def rmse(y, yhat):
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.sqrt(np.mean(e ** 2)))


def mae(y, yhat):
    e = np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)
    return float(np.mean(np.abs(e)))


def corr(y, yhat):
    a = np.asarray(y, dtype=float)
    b = np.asarray(yhat, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def industry_fe_predict(Y, X, train, test, industries):
    """Y = industry intercepts + slope coefficients on the non-intercept columns of X.

    A holdout industry with no training member receives the mean training intercept.
    """
    ind = np.asarray(industries, dtype=object)
    tr_levels = pd.unique(ind[train])
    D_tr = np.column_stack([(ind[train] == g).astype(float) for g in tr_levels])
    X_tr = np.column_stack([D_tr, X[train, 1:]])
    beta, *_ = np.linalg.lstsq(X_tr, Y[train], rcond=None)
    n_g = D_tr.shape[1]
    intercepts = beta[:n_g]
    slopes = beta[n_g:]
    mean_intercept = float(np.mean(intercepts))
    level_of = {g: intercepts[i] for i, g in enumerate(tr_levels)}
    pred = np.empty(len(test))
    for j, i in enumerate(test):
        pred[j] = level_of.get(ind[i], mean_intercept) + float(X[i, 1:] @ slopes)
    return pred


def peer_mean_predict(Y, train, test, industries):
    """Holdout prediction equal to the same-industry training mean return.

    Industries with no training member use the grand training mean.
    """
    ind = np.asarray(industries, dtype=object)
    grand = float(np.mean(Y[train]))
    pred = np.empty(len(test))
    for j, i in enumerate(test):
        peers = train[ind[train] == ind[i]]
        pred[j] = float(np.mean(Y[peers])) if len(peers) else grand
    return pred


def factor_predict(panel_ret, tickers, date, Y, train, test, lookback=36, shrink=0.5):
    """Beta-adjusted imputation: trailing beta times the contemporaneous training mean.

    Betas are OLS slopes of each name on the equal-weighted cross-section mean,
    using only months strictly before `date`. Missing histories shrink to 1.
    """
    hist = panel_ret.loc[panel_ret.index < date, tickers].dropna(how="all")
    hist = hist.tail(lookback)
    market = hist.mean(axis=1)
    m_t = float(np.mean(Y[train]))
    pred = np.empty(len(test))
    for j, i in enumerate(test):
        series = hist.iloc[:, i] if i < hist.shape[1] else None
        # hist columns follow `tickers`, which is the cross-section order.
        col = hist[tickers[i]] if tickers[i] in hist.columns else None
        if col is None:
            beta = 1.0
        else:
            pair = pd.concat([col, market], axis=1).dropna()
            if len(pair) < 12 or float(pair.iloc[:, 1].var()) <= 1e-12:
                beta = 1.0
            else:
                x = np.column_stack([np.ones(len(pair)), pair.iloc[:, 1].values])
                coef, *_ = np.linalg.lstsq(x, pair.iloc[:, 0].values, rcond=None)
                beta = float(coef[1])
        # Light shrinkage toward the market beta of 1 (Vasicek-style, fixed weight).
        beta = (1.0 - shrink) * beta + shrink * 1.0
        pred[j] = beta * m_t
    return pred


def selection_counts(oos: pd.DataFrame) -> dict:
    corr = sec = ind = corr_only = boost_only = both = 0
    for raw in oos["selected"]:
        names = set(json.loads(raw))
        has_c = "Correlation (5-NN)" in names
        has_b = "Boosted correlation resolvent" in names
        if has_c or has_b:
            corr += 1
        if has_c and has_b:
            both += 1
        elif has_c:
            corr_only += 1
        elif has_b:
            boost_only += 1
        if "Sector affiliation" in names:
            sec += 1
        if "Industry (related companies)" in names:
            ind += 1
    n = len(oos)
    return {
        "n_months": n,
        "correlation_or_resolvent": corr,
        "correlation_only": corr_only,
        "resolvent_only": boost_only,
        "correlation_and_resolvent": both,
        "sector": sec,
        "industry": ind,
        "correlation_share": corr / n,
        "sector_share": sec / n,
        "industry_share": ind / n,
    }


def main():
    oos = pd.read_csv(RESULTS / "sp100_running_oos.csv")
    counts = selection_counts(oos)
    print("Selection counts:", counts, flush=True)

    wiki = load_constituents()
    tickers = wiki["Symbol"].dropna().unique().tolist()
    px = download_prices(tickers)
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    fund = fetch_fundamentals(tickers, wiki)
    panel = monthly_panel(px)
    ret = panel["ret"].dropna(how="all")
    dates = list(ret.index[(ret.index >= "2021-01-01") & (ret.index <= "2026-08-31")])

    cs_show = None
    showcase_date = None
    for d in dates[::-1]:
        cs = align_cross_section(panel, fund, d)
        if cs is not None and len(cs["tickers"]) >= 70:
            cs_show = cs
            showcase_date = d
            break
    if cs_show is None:
        raise RuntimeError("No showcase cross-section")

    n = len(cs_show["Y"])
    rng = np.random.default_rng(2026)
    perm = rng.permutation(n)
    n_test = max(int(0.2 * n), 15)
    test_idx = np.sort(perm[:n_test])
    train_idx = np.sort(perm[n_test:])
    holdout = set(np.array(cs_show["tickers"])[test_idx])
    industries = cs_show["fund"]["Industry"].tolist()
    Y, X = cs_show["Y"], cs_show["X"]
    b_ols, *_ = ols(Y[train_idx], add_intercept(X[train_idx]))
    # X from align_cross_section already has an intercept.
    pred_ols = X[test_idx] @ ols(Y[train_idx], X[train_idx])[0]
    pred_fe = industry_fe_predict(Y, X, train_idx, test_idx, industries)
    pred_peer = peer_mean_predict(Y, train_idx, test_idx, industries)
    pred_fac = factor_predict(panel["ret"], cs_show["tickers"], showcase_date, Y, train_idx, test_idx)
    pred_fac_raw = factor_predict(
        panel["ret"], cs_show["tickers"], showcase_date, Y, train_idx, test_idx, shrink=0.0
    )
    showcase = {
        "date": str(pd.Timestamp(showcase_date).date()),
        "n": n,
        "n_test": int(len(test_idx)),
        "OLS_RMSE": rmse(Y[test_idx], pred_ols),
        "IndustryFE_RMSE": rmse(Y[test_idx], pred_fe),
        "IndustryFE_MAE": mae(Y[test_idx], pred_fe),
        "IndustryFE_corr": corr(Y[test_idx], pred_fe),
        "IndustryPeer_RMSE": rmse(Y[test_idx], pred_peer),
        "Factor_RMSE": rmse(Y[test_idx], pred_fac),
        "Factor_MAE": mae(Y[test_idx], pred_fac),
        "Factor_corr": corr(Y[test_idx], pred_fac),
        "FactorRaw_RMSE": rmse(Y[test_idx], pred_fac_raw),
    }
    saved = json.loads((RESULTS / "sp100_showcase.json").read_text(encoding="utf-8"))
    showcase["saved_OLS_RMSE"] = saved["OLS"]["RMSE"]
    showcase["saved_MultiSEM_RMSE"] = saved["MultiSEM"]["RMSE"]
    showcase["ols_matches_saved"] = abs(showcase["OLS_RMSE"] - showcase["saved_OLS_RMSE"]) < 1e-8
    print("Showcase benchmarks:", showcase, flush=True)

    rows = []
    matched = 0
    by_date = {str(pd.Timestamp(d).date()): d for d in dates}
    for _, rec in oos.iterrows():
        d = by_date.get(rec["date"])
        if d is None:
            rows.append({"date": rec["date"], "matched": False})
            continue
        cs = align_cross_section(panel, fund, d)
        if cs is None:
            rows.append({"date": rec["date"], "matched": False})
            continue
        tk = np.array(cs["tickers"])
        is_test = np.array([t in holdout for t in tk])
        is_train = ~is_test
        train_i = np.where(is_train)[0]
        test_i = np.where(is_test)[0]
        Yc, Xc = cs["Y"], cs["X"]
        pred_ols = Xc[test_i] @ ols(Yc[train_i], Xc[train_i])[0]
        ols_rmse = rmse(Yc[test_i], pred_ols)
        ok = abs(ols_rmse - float(rec["OLS_RMSE"])) < 1e-8
        matched += int(ok)
        ind = cs["fund"]["Industry"].tolist()
        pred_fe = industry_fe_predict(Yc, Xc, train_i, test_i, ind)
        pred_peer = peer_mean_predict(Yc, train_i, test_i, ind)
        pred_fac = factor_predict(panel["ret"], cs["tickers"], d, Yc, train_i, test_i)
        pred_fac_raw = factor_predict(panel["ret"], cs["tickers"], d, Yc, train_i, test_i, shrink=0.0)
        rows.append(
            {
                "date": rec["date"],
                "matched": ok,
                "OLS_RMSE_recomputed": ols_rmse,
                "OLS_RMSE_saved": float(rec["OLS_RMSE"]),
                "IndustryFE_RMSE": rmse(Yc[test_i], pred_fe),
                "IndustryFE_MAE": mae(Yc[test_i], pred_fe),
                "IndustryPeer_RMSE": rmse(Yc[test_i], pred_peer),
                "Factor_RMSE": rmse(Yc[test_i], pred_fac),
                "Factor_MAE": mae(Yc[test_i], pred_fac),
                "FactorRaw_RMSE": rmse(Yc[test_i], pred_fac_raw),
                "MultiSEM_RMSE_saved": float(rec["MultiSEM_RMSE"]),
                "IndustrySEM_RMSE_saved": float(rec["IndustrySEM_RMSE"]),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "nonnetwork_benchmarks.csv", index=False)
    use = df[df["matched"] == True]  # noqa: E712
    summary = {
        "selection_counts": counts,
        "showcase": showcase,
        "n_months_matched": int(matched),
        "n_months_saved": int(len(oos)),
        "mean_OLS_RMSE": float(use["OLS_RMSE_saved"].mean()) if len(use) else None,
        "mean_IndustryFE_RMSE": float(use["IndustryFE_RMSE"].mean()) if len(use) else None,
        "mean_IndustryFE_MAE": float(use["IndustryFE_MAE"].mean()) if len(use) else None,
        "mean_IndustryPeer_RMSE": float(use["IndustryPeer_RMSE"].mean()) if len(use) else None,
        "mean_Factor_RMSE": float(use["Factor_RMSE"].mean()) if len(use) else None,
        "mean_Factor_MAE": float(use["Factor_MAE"].mean()) if len(use) else None,
        "mean_FactorRaw_RMSE": float(use["FactorRaw_RMSE"].mean()) if len(use) else None,
        "mean_IndustrySEM_RMSE": float(use["IndustrySEM_RMSE_saved"].mean()) if len(use) else None,
        "mean_MultiSEM_RMSE": float(use["MultiSEM_RMSE_saved"].mean()) if len(use) else None,
        "showcase_month_in_running": None,
    }
    show_row = df.loc[df["date"] == showcase["date"]]
    if len(show_row):
        summary["showcase_month_in_running"] = show_row.iloc[0].to_dict()
    (RESULTS / "nonnetwork_benchmarks_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
