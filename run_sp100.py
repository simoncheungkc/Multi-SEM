"""S&P 100 empirical analysis: multiplex networks, Gibbs posterior, running OOS.

The headquarters distance layer is estimated by run_sp100_geo.py, which
writes results/sp100_geo_oos.csv and does not replace the four-layer files
produced here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mnreg.model import (
    GibbsConfig,
    add_intercept,
    drop_collinear_networks,
    fit_ml_multi_sem,
    gelman_rubin,
    gibbs_multi_sem,
    metrics_rmse_mae,
    ols,
    posterior_summary,
    predict_kriging,
    predict_mean_only,
    screen_networks,
    standardize_columns,
)
from mnreg.networks import (
    affiliation_network,
    boosted_resolvent,
    correlation_network,
    degree_summary,
    row_normalize,
)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
FIG = RESULTS / "figures"
DATA.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

# Read the frozen CSV written from the parquet cache. Live Yahoo downloads
# revise adjusted prices, so the default path does not call yfinance.
USE_STATIC_CACHE = True
ALLOW_DOWNLOAD = False
SNAPSHOT = DATA / "sp100_clean_snapshot.csv"
PRICE_CACHE = DATA / "sp100_prices.parquet"

# Fallback S&P 100 tickers (Yahoo Finance symbols). Wikipedia is preferred.
SP100_FALLBACK = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GOOGL", "GS", "HD", "HON", "IBM",
    "INTC", "INTU", "ISRG", "JNJ", "JPM", "KO", "LIN", "LLY", "LMT", "LOW",
    "MA", "MCD", "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS",
    "MSFT", "NEE", "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM",
    "PYPL", "QCOM", "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO",
    "TMUS", "TSLA", "TXN", "UNH", "UNP", "UPS", "USB", "V", "VZ", "WFC",
    "WMT", "XOM",
]


def _normalize_ticker(sym: str) -> str:
    return str(sym).strip().replace(".", "-")


def load_constituents() -> pd.DataFrame:
    cache = DATA / "sp100_constituents.csv"
    if cache.exists():
        return pd.read_csv(cache)
    tables = []
    try:
        tables = pd.read_html("https://en.wikipedia.org/wiki/S%26P_100")
    except Exception as exc:
        print(f"  Wikipedia read failed ({exc}); using fallback ticker list", flush=True)
    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("symbol" in c or "ticker" in c for c in cols):
            df = t.copy()
            break
    if df is None:
        df = pd.DataFrame({"Symbol": SP100_FALLBACK, "Sector": np.nan, "Name": SP100_FALLBACK})
    rename = {}
    for c in df.columns:
        cl = str(c).lower()
        if "symbol" in cl or cl == "ticker":
            rename[c] = "Symbol"
        elif "sector" in cl:
            rename[c] = "Sector"
        elif "sub" in cl and "industr" in cl:
            rename[c] = "Industry"
        elif cl in ("name", "company", "security"):
            rename[c] = "Name"
    df = df.rename(columns=rename)
    if "Symbol" not in df.columns:
        df = pd.DataFrame({"Symbol": SP100_FALLBACK})
    df["Symbol"] = df["Symbol"].map(_normalize_ticker)
    df = df.drop_duplicates("Symbol")
    df = df[~df["Symbol"].isin(["GOOG"])]  # keep GOOGL only if both present
    df.to_csv(cache, index=False)
    return df


def _read_snapshot_prices() -> pd.DataFrame:
    text = SNAPSHOT.read_text(encoding="utf-8")
    price_lines = []
    in_prices = False
    for line in text.splitlines():
        if line.startswith("# block=adjusted_close"):
            in_prices = True
            continue
        if line.startswith("# block=") and in_prices:
            break
        if in_prices and line.strip():
            price_lines.append(line)
    if not price_lines:
        raise RuntimeError(f"{SNAPSHOT.name} has no adjusted_close block")
    from io import StringIO

    return pd.read_csv(StringIO("\n".join(price_lines)), index_col=0, parse_dates=True)


def write_static_snapshot(px: pd.DataFrame) -> Path:
    """Freeze adjusted closes, daily returns, and monthly covariates."""
    px = px.sort_index()
    daily = px.pct_change().dropna(how="all")
    panel = monthly_panel(px)
    rows = []
    for date in panel["ret"].index:
        frame = pd.DataFrame(
            {
                "lag_ret": panel["ret"].shift(1).loc[date],
                "vol": panel["vol"].shift(1).loc[date] if date in panel["vol"].shift(1).index else None,
                "mom12": panel["mom"].shift(1).loc[date] if date in panel["mom"].shift(1).index else None,
                "log_price": panel["log_price"].shift(1).loc[date],
            }
        )
        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if frame.empty:
            continue
        standardized = frame.copy()
        standardized[:] = (frame - frame.mean()) / frame.std(ddof=0).replace(0, np.nan)
        frame = frame.join(standardized.add_suffix("_z"))
        frame.insert(0, "month", pd.Timestamp(date).strftime("%Y-%m-%d"))
        frame.insert(0, "symbol", frame.index)
        rows.append(frame.reset_index(drop=True))
    covariates = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    parts = [
        "# sp100_clean_snapshot",
        f"# equities={px.shape[1]} price_rows={px.shape[0]} "
        f"daily_return_rows={daily.shape[0]} "
        f"start={px.index.min().date()} end={px.index.max().date()}",
        "# block=adjusted_close",
        px.to_csv(),
        "# block=daily_return",
        daily.to_csv(),
        "# block=monthly_covariates",
        covariates.to_csv(index=False),
    ]
    SNAPSHOT.write_text("\n".join(part if part.endswith("\n") else part + "\n" for part in parts), encoding="utf-8")
    return SNAPSHOT


def download_prices(tickers, start="2018-01-01", end="2026-09-16") -> pd.DataFrame:
    if not ALLOW_DOWNLOAD and not SNAPSHOT.exists():
        raise FileNotFoundError(
            f"{SNAPSHOT} is missing. Pass --download to fetch live Yahoo Finance prices."
        )
    if not ALLOW_DOWNLOAD and SNAPSHOT.exists():
        px = _read_snapshot_prices()
        have = [t for t in tickers if t in px.columns]
        if len(have) >= 80:
            return px[have]
        raise RuntimeError(
            f"{SNAPSHOT.name} does not cover the requested tickers "
            f"({len(have)} of {len(tickers)})"
        )
    if PRICE_CACHE.exists() and not ALLOW_DOWNLOAD:
        px = pd.read_parquet(PRICE_CACHE)
        have = [t for t in tickers if t in px.columns]
        if len(have) >= 80:
            px = px[have]
            if USE_STATIC_CACHE and not SNAPSHOT.exists():
                write_static_snapshot(px)
            return px
    if not ALLOW_DOWNLOAD:
        raise FileNotFoundError(
            f"{SNAPSHOT.name} is missing. Pass --download to fetch live Yahoo Finance prices."
        )
    import yfinance as yf

    print(f"  Downloading {len(tickers)} tickers from Yahoo Finance...", flush=True)
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            px = raw["Close"].copy()
        else:
            px = raw.xs("Close", axis=1, level=0, drop_level=True)
    else:
        px = raw.copy()
    px = px.sort_index()
    px = px.dropna(axis=1, how="all")
    px.to_parquet(cache)
    return px


def fetch_fundamentals(tickers: list[str], wiki: pd.DataFrame) -> pd.DataFrame:
    cache = DATA / "sp100_fundamentals.csv"
    if cache.exists():
        return pd.read_csv(cache)
    rows = []
    for i, t in enumerate(tickers):
        sector = industry = None
        if "Sector" in wiki.columns:
            hit = wiki.loc[wiki["Symbol"] == t, "Sector"]
            if len(hit):
                sector = hit.iloc[0]
        if "Industry" in wiki.columns:
            hit = wiki.loc[wiki["Symbol"] == t, "Industry"]
            if len(hit):
                industry = hit.iloc[0]
        try:
            info = yf.Ticker(t).info or {}
        except Exception:
            info = {}
        rows.append(
            {
                "Symbol": t,
                "Sector": info.get("sector") or sector,
                "Industry": info.get("industry") or industry,
                "marketCap": info.get("marketCap"),
                "beta": info.get("beta"),
                "trailingPE": info.get("trailingPE"),
                "profitMargins": info.get("profitMargins"),
            }
        )
        if (i + 1) % 20 == 0:
            print(f"    fundamentals {i+1}/{len(tickers)}", flush=True)
        time.sleep(0.05)
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def monthly_panel(px: pd.DataFrame) -> dict:
    daily = px.pct_change().dropna(how="all")
    monthly_px = px.resample("ME").last()
    monthly_ret = monthly_px.pct_change()
    monthly_vol = daily.pow(2).resample("ME").sum().pow(0.5)
    monthly_dvol = daily.resample("ME").count()
    # Dollar-volume proxy: skip; use log price as size and traded-day count
    log_price = np.log(monthly_px.replace(0, np.nan))
    mom_12 = monthly_ret.rolling(12).sum() - monthly_ret  # 12-1 momentum
    return {
        "ret": monthly_ret,
        "vol": monthly_vol,
        "log_price": log_price,
        "mom": mom_12,
        "ndays": monthly_dvol,
        "daily": daily,
    }


def align_cross_section(panel, fundamentals, date, lookback_days=252):
    """Build Y, X, and candidate networks for a month-end date."""
    ret = panel["ret"]
    if date not in ret.index:
        return None
    loc = ret.index.get_loc(date)
    if loc < 13:
        return None
    y = ret.loc[date]
    lag = ret.index[loc - 1]
    x_lag = ret.loc[lag]
    x_vol = panel["vol"].loc[lag] if lag in panel["vol"].index else None
    x_mom = panel["mom"].loc[lag] if lag in panel["mom"].index else None
    x_size = panel["log_price"].loc[lag] if lag in panel["log_price"].index else None
    if x_vol is None or x_mom is None or x_size is None:
        return None
    df = pd.DataFrame(
        {
            "Y": y,
            "lag_ret": x_lag,
            "vol": x_vol,
            "mom12": x_mom,
            "log_price": x_size,
        }
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    tickers = df.index.tolist()
    if len(tickers) < 40:
        return None

    # Daily returns in prior ~12 months for the correlation network (no look-ahead)
    daily = panel["daily"]
    end_daily = daily.index[daily.index <= lag]
    if len(end_daily) < 60:
        return None
    last = end_daily[-1]
    start = last - pd.Timedelta(days=lookback_days + 20)
    R = daily.loc[start:last, tickers].dropna(how="all")
    R = R.dropna(axis=1, thresh=max(40, int(0.6 * len(R))))
    keep = [t for t in tickers if t in R.columns]
    df = df.loc[keep]
    R = R[keep]
    tickers = keep
    if len(tickers) < 40:
        return None

    A_corr_raw = correlation_network(R.values, method="knn", k=5)
    fund = fundamentals.set_index("Symbol").reindex(tickers)
    A_sector_raw = affiliation_network(fund["Sector"].tolist())
    A_ind_raw = affiliation_network(fund["Industry"].tolist())
    # Boost the correlation network (multi-hop co-movement). Clique-like
    # affiliation matrices are invariant to the resolvent after row-normalization.
    A_corr_boost_raw = boosted_resolvent(row_normalize(A_corr_raw), alpha=0.5)

    names = [
        "Correlation (5-NN)",
        "Boosted correlation resolvent",
        "Sector affiliation",
        "Industry (related companies)",
    ]
    As_raw = [A_corr_raw, A_corr_boost_raw, A_sector_raw, A_ind_raw]
    # Symmetrized 0-1 graphs are row-standardized here so non-isolated rows
    # sum to 1 and ρ(A_s) ≤ 1 (Equation 2' / intercept remark).
    As = [row_normalize(A) for A in As_raw]
    meta = []
    for nm, A in zip(names, As_raw):
        info = degree_summary(A)
        info["name"] = nm
        meta.append(info)
    X = add_intercept(df[["lag_ret", "vol", "mom12", "log_price"]].values)
    X, _, _ = standardize_columns(X)
    Y = df["Y"].values.astype(float)
    return {
        "date": date,
        "tickers": tickers,
        "Y": Y,
        "X": X,
        "As": As,
        "As_raw": As_raw,
        "names": names,
        "meta": meta,
        "fund": fund,
        "df": df,
    }


def plot_pred(y, panels, title, path):
    k = len(panels)
    fig, axes = plt.subplots(1, k, figsize=(3.8 * k, 3.7), sharex=True, sharey=True)
    if k == 1:
        axes = [axes]
    lo = min(y.min(), min(p.min() for p, _, _ in panels))
    hi = max(y.max(), max(p.max() for p, _, _ in panels))
    for ax, (pred, name, met) in zip(axes, panels):
        ax.scatter(y, pred, s=22, c="#4C6A92", alpha=0.85, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="#B85C38", lw=1)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Actual monthly return")
        ax.set_ylabel("Predicted monthly return")
        ax.text(
            0.04,
            0.96,
            f"RMSE={met['RMSE']:.4f}\nMAE={met['MAE']:.4f}\ncorr={met['corr']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv=None):
    global ALLOW_DOWNLOAD
    import argparse

    parser = argparse.ArgumentParser(description="S&P 100 Multi-SEM panel")
    parser.add_argument(
        "--download",
        action="store_true",
        help="Fetch live Yahoo Finance prices. Without this flag a missing snapshot raises FileNotFoundError.",
    )
    args = parser.parse_args(argv)
    ALLOW_DOWNLOAD = bool(args.download)
    t_all = time.time()
    print("=== S&P 100 constituents and prices ===", flush=True)
    wiki = load_constituents()
    tickers = wiki["Symbol"].dropna().unique().tolist()
    print(f"  {len(tickers)} tickers", flush=True)
    px = download_prices(tickers)
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    print(f"  price matrix {px.shape}", flush=True)
    fund = fetch_fundamentals(tickers, wiki)
    panel = monthly_panel(px)

    # Choose a recent stable month for posterior showcase, then running OOS
    ret = panel["ret"].dropna(how="all")
    dates = ret.index[(ret.index >= "2021-01-01") & (ret.index <= "2026-08-31")]
    print(f"  candidate month-ends: {len(dates)}", flush=True)

    # Showcase month: last month with a complete cross-section in 2025 if possible
    showcase_date = None
    cs_show = None
    for d in dates[::-1]:
        cs = align_cross_section(panel, fund, d)
        if cs is not None and len(cs["tickers"]) >= 70:
            showcase_date = d
            cs_show = cs
            break
    if cs_show is None:
        raise RuntimeError("Could not build a showcase cross-section")
    print(
        f"  showcase date {pd.Timestamp(showcase_date).date()} n={len(cs_show['tickers'])}",
        flush=True,
    )
    pd.DataFrame(cs_show["meta"]).to_csv(RESULTS / "sp100_network_meta.csv", index=False)

    print("=== Moran I screening (showcase month) ===", flush=True)
    scr = screen_networks(
        cs_show["Y"], cs_show["X"], cs_show["As"], names=cs_show["names"], alpha=0.05, fdr=True
    )
    tests = pd.DataFrame(scr["tests"])
    tests.to_csv(RESULTS / "sp100_moran_tests.csv", index=False)
    print(tests[["name", "I", "z", "pvalue"]].to_string(index=False), flush=True)
    print("  selected:", scr["selected_names"], flush=True)

    As_raw, As, names = cs_show["As_raw"], cs_show["As"], cs_show["names"]
    X, Y = cs_show["X"], cs_show["Y"]

    # If FDR selects none, keep networks with p < 0.10 so the multi-network model is identifiable
    sel_idx = np.where(scr["selected"])[0].tolist()
    if len(sel_idx) == 0:
        sel_idx = np.where(scr["pvalues"] < 0.10)[0].tolist()
        print("  FDR empty; relaxing to p<0.10:", [cs_show["names"][i] for i in sel_idx], flush=True)
    if len(sel_idx) == 0:
        sel_idx = [int(np.argmin(scr["pvalues"]))]
        print("  still empty; keeping smallest p-value network", flush=True)
    keep, _, _ = drop_collinear_networks(
        [As_raw[i] for i in sel_idx], [names[i] for i in sel_idx]
    )
    sel_idx = [sel_idx[i] for i in keep]
    print("  after collinearity filter:", [names[i] for i in sel_idx], flush=True)

    print("=== Gibbs posterior (showcase, 80/20 node split on training stocks) ===", flush=True)
    n = len(cs_show["Y"])
    rng = np.random.default_rng(2026)
    idx = rng.permutation(n)
    n_test = max(int(0.2 * n), 15)
    test_idx = np.sort(idx[:n_test])
    train_idx = np.sort(idx[n_test:])
    X, Y = cs_show["X"], cs_show["Y"]
    As_raw, As, names = cs_show["As_raw"], cs_show["As"], cs_show["names"]
    As_tr_sel = [row_normalize(As_raw[i][np.ix_(train_idx, train_idx)]) for i in sel_idx]
    names_sel = [names[i] for i in sel_idx]

    gcfg = GibbsConfig(n_iter=8000, burn=4000, n_chains=4, seed=2026, adapt_covariance=True)
    result = gibbs_multi_sem(
        Y[train_idx], X[train_idx], As_tr_sel, names=names_sel, config=gcfg
    )
    beta_names = ["intercept", "lag_ret", "vol", "mom12", "log_price"]
    summ = posterior_summary(result, beta_names=beta_names)
    post_tbl = pd.DataFrame(summ["table"])
    post_tbl.to_csv(RESULTS / "sp100_posterior.csv", index=False)
    print(post_tbl.to_string(index=False), flush=True)
    print("  MH accept rates:", result.acceptance, flush=True)

    lam = result.samples["lambda"]
    fig, axes = plt.subplots(lam.shape[2], 1, figsize=(8.5, 2.2 * max(lam.shape[2], 1)), sharex=True)
    if lam.shape[2] == 1:
        axes = [axes]
    for s, ax in enumerate(axes):
        for c in range(lam.shape[0]):
            ax.plot(
                lam[c, :, s],
                lw=0.6,
                alpha=0.85,
                color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][c % 4],
                label=f"Chain {c + 1}" if s == 0 else None,
            )
        if s == 0:
            ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
        ax.set_ylabel(names_sel[s], fontsize=8)
        ax.set_ylim(-1, 1)
    axes[-1].set_xlabel("Post-burn iteration")
    fig.suptitle(f"S&P 100 Gibbs traces — {pd.Timestamp(showcase_date).date()}")
    fig.tight_layout()
    fig.savefig(FIG / "sp100_gibbs_traces.png", dpi=140)
    plt.close(fig)

    # Showcase holdout prediction
    b_ols, _, _ = ols(Y[train_idx], X[train_idx])
    pred_ols = predict_mean_only(X[test_idx], b_ols)
    pred_multi = predict_kriging(
        Y[train_idx],
        X[train_idx],
        X[test_idx],
        train_idx,
        test_idx,
        summ["means"]["lambda"],
        summ["means"]["beta"],
        [As[i] for i in sel_idx],
    )
    # Single-network ML for each candidate
    single_preds = {}
    for i, nm in enumerate(names):
        Atr = row_normalize(As_raw[i][np.ix_(train_idx, train_idx)])
        fit = fit_ml_multi_sem(Y[train_idx], X[train_idx], [Atr])
        pred = predict_kriging(
            Y[train_idx],
            X[train_idx],
            X[test_idx],
            train_idx,
            test_idx,
            fit["lambda"],
            fit["beta"],
            [As[i]],
        )
        single_preds[nm] = (pred, metrics_rmse_mae(Y[test_idx], pred), float(fit["lambda"][0]))

    m_ols = metrics_rmse_mae(Y[test_idx], pred_ols)
    m_multi = metrics_rmse_mae(Y[test_idx], pred_multi)
    panels = [(pred_ols, "OLS (no network)", m_ols), (pred_multi, "Multi-SEM (selected networks)", m_multi)]
    plot_pred(
        Y[test_idx],
        panels,
        f"S&P 100 holdout, {pd.Timestamp(showcase_date).strftime('%b %Y')} (n_test={len(test_idx)})",
        FIG / "sp100_pred_vs_actual.png",
    )

    showcase_metrics = {
        "date": str(pd.Timestamp(showcase_date).date()),
        "n": n,
        "n_test": int(len(test_idx)),
        "selected": names_sel,
        "OLS": m_ols,
        "MultiSEM": m_multi,
        "single": {nm: {"metrics": met, "lambda": lamh} for nm, (_, met, lamh) in single_preds.items()},
        "pvalues": {nm: float(p) for nm, p in zip(names, scr["pvalues"])},
        "acceptance": result.acceptance.tolist(),
    }
    with open(RESULTS / "sp100_showcase.json", "w") as f:
        json.dump(showcase_metrics, f, indent=2)

    print("=== Running monthly out-of-sample predictions ===", flush=True)
    # Fixed 20% holdout of tickers that appear in most months; re-estimate each month
    # Each month is refit with the covariance-adapted sampler in Algorithm 2.
    oos_rows = []
    run_dates = [d for d in dates if d >= pd.Timestamp("2021-06-30")]
    # Determine a stable ticker universe from the showcase
    holdout_tickers = set(np.array(cs_show["tickers"])[test_idx])
    for d in run_dates:
        cs = align_cross_section(panel, fund, d)
        if cs is None:
            continue
        tk = np.array(cs["tickers"])
        is_test = np.array([t in holdout_tickers for t in tk])
        is_train = ~is_test
        if is_test.sum() < 8 or is_train.sum() < 30:
            continue
        train_i = np.where(is_train)[0]
        test_i = np.where(is_test)[0]
        Yc, Xc, Asr, Asf, nms = cs["Y"], cs["X"], cs["As_raw"], cs["As"], cs["names"]
        # Screen on training nodes
        As_tr = [row_normalize(A[np.ix_(train_i, train_i)]) for A in Asr]
        scr_t = screen_networks(Yc[train_i], Xc[train_i], As_tr, names=nms, alpha=0.05, fdr=True)
        sel_t = np.where(scr_t["selected"])[0].tolist()
        if not sel_t:
            sel_t = np.where(scr_t["pvalues"] < 0.10)[0].tolist()
        if not sel_t:
            sel_t = [int(np.argmin(scr_t["pvalues"]))]
        keep_t, _, _ = drop_collinear_networks(
            [Asr[i] for i in sel_t], [nms[i] for i in sel_t]
        )
        sel_t = [sel_t[i] for i in keep_t]

        b_ols, _, _ = ols(Yc[train_i], Xc[train_i])
        p_ols = predict_mean_only(Xc[test_i], b_ols)
        m_ols_t = metrics_rmse_mae(Yc[test_i], p_ols)

        # Single-network SEMs and the screened multiplex use Algorithm 2.
        month_cfg = GibbsConfig(
            n_iter=8000, burn=4000, n_chains=4, seed=2026, adapt_covariance=True
        )
        single_month = {}
        for i, key in enumerate(["Corr", "BoostCorr", "Sector", "Industry"]):
            fit_i = gibbs_multi_sem(
                Yc[train_i], Xc[train_i], [As_tr[i]], names=[nms[i]], config=month_cfg
            )
            lam_i = fit_i.samples["lambda"].reshape(-1, fit_i.samples["lambda"].shape[-1]).mean(axis=0)
            beta_i = fit_i.samples["beta"].reshape(-1, fit_i.samples["beta"].shape[-1]).mean(axis=0)
            p_i = predict_kriging(
                Yc[train_i], Xc[train_i], Xc[test_i], train_i, test_i,
                lam_i, beta_i, [Asf[i]],
            )
            single_month[key] = metrics_rmse_mae(Yc[test_i], p_i)

        fit_m = gibbs_multi_sem(
            Yc[train_i],
            Xc[train_i],
            [As_tr[i] for i in sel_t],
            names=[nms[i] for i in sel_t],
            config=month_cfg,
        )
        lam_m = fit_m.samples["lambda"].reshape(-1, fit_m.samples["lambda"].shape[-1]).mean(axis=0)
        beta_m = fit_m.samples["beta"].reshape(-1, fit_m.samples["beta"].shape[-1]).mean(axis=0)
        p_m = predict_kriging(
            Yc[train_i],
            Xc[train_i],
            Xc[test_i],
            train_i,
            test_i,
            lam_m,
            beta_m,
            [Asf[i] for i in sel_t],
        )
        m_m = metrics_rmse_mae(Yc[test_i], p_m)

        # One-month-ahead (mean structure only)
        p_ahead = predict_mean_only(Xc[test_i], beta_m)
        m_ahead = metrics_rmse_mae(Yc[test_i], p_ahead)

        oos_rows.append(
            {
                "date": str(pd.Timestamp(d).date()),
                "n_train": int(is_train.sum()),
                "n_test": int(is_test.sum()),
                "selected": json.dumps([nms[i] for i in sel_t]),
                "OLS_RMSE": m_ols_t["RMSE"],
                "OLS_MAE": m_ols_t["MAE"],
                "OLS_corr": m_ols_t["corr"],
                "CorrSEM_RMSE": single_month["Corr"]["RMSE"],
                "CorrSEM_MAE": single_month["Corr"]["MAE"],
                "SectorSEM_RMSE": single_month["Sector"]["RMSE"],
                "SectorSEM_MAE": single_month["Sector"]["MAE"],
                "IndustrySEM_RMSE": single_month["Industry"]["RMSE"],
                "IndustrySEM_MAE": single_month["Industry"]["MAE"],
                "MultiSEM_RMSE": m_m["RMSE"],
                "MultiSEM_MAE": m_m["MAE"],
                "MultiSEM_corr": m_m["corr"],
                "Ahead_RMSE": m_ahead["RMSE"],
                "Ahead_MAE": m_ahead["MAE"],
                "lambda_hat": json.dumps(lam_m.tolist()),
                "rhat_max": float(max(gelman_rubin(fit_m.samples["lambda"][:, :, s]) for s in range(lam_m.size))),
            }
        )
        if len(oos_rows) % 6 == 0:
            print(f"    {d.date()}  OLS {m_ols_t['RMSE']:.4f}  Multi {m_m['RMSE']:.4f}", flush=True)

    oos = pd.DataFrame(oos_rows)
    oos.to_csv(RESULTS / "sp100_running_oos.csv", index=False)
    if len(oos):
        oos["date"] = pd.to_datetime(oos["date"])
        fig, ax = plt.subplots(figsize=(9.2, 4.2))
        ax.plot(oos["date"], oos["OLS_RMSE"], label="OLS", lw=1.4)
        ax.plot(oos["date"], oos["CorrSEM_RMSE"], label="Correlation SEM", lw=1.2)
        ax.plot(oos["date"], oos["SectorSEM_RMSE"], label="Sector SEM", lw=1.2)
        ax.plot(oos["date"], oos["IndustrySEM_RMSE"], label="Industry SEM", lw=1.2)
        ax.plot(oos["date"], oos["MultiSEM_RMSE"], label="Multi-SEM (Moran-selected)", lw=1.6)
        ax.set_ylabel("Holdout RMSE (monthly return)")
        ax.set_xlabel("Month")
        ax.set_title("S&P 100 running out-of-sample RMSE (fixed 20% ticker holdout)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / "sp100_running_rmse.png", dpi=140)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8.2, 3.8))
        cols = ["OLS_RMSE", "CorrSEM_RMSE", "SectorSEM_RMSE", "IndustrySEM_RMSE", "MultiSEM_RMSE", "Ahead_RMSE"]
        labels = ["OLS", "Correlation\nSEM", "Sector\nSEM", "Industry\nSEM", "Multi-SEM\n(kriging)", "Multi-SEM\n(X beta only)"]
        means = oos[cols].mean()
        ax.bar(labels, means.values, color="#4C6A92")
        ax.set_ylabel("Mean monthly holdout RMSE")
        ax.set_title("S&P 100 average out-of-sample RMSE, Jun 2021 to Aug 2026")
        fig.tight_layout()
        fig.savefig(FIG / "sp100_oos_mean_rmse.png", dpi=140)
        plt.close(fig)

    payload = {
        "n_tickers_prices": int(px.shape[1]),
        "showcase": showcase_metrics,
        "running_mean_rmse": oos[
            ["OLS_RMSE", "CorrSEM_RMSE", "SectorSEM_RMSE", "IndustrySEM_RMSE", "MultiSEM_RMSE", "Ahead_RMSE"]
        ].mean().to_dict()
        if len(oos)
        else {},
        "running_mean_mae": oos[
            ["OLS_MAE", "CorrSEM_MAE", "SectorSEM_MAE", "IndustrySEM_MAE", "MultiSEM_MAE", "Ahead_MAE"]
        ].mean().to_dict()
        if len(oos)
        else {},
        "n_running_months": int(len(oos)),
        "elapsed_sec": time.time() - t_all,
    }
    with open(RESULTS / "sp100_meta.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"S&P 100 done in {payload['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
