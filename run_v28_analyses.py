"""Cheap v28 updates: prior refits, spectral-radius maximum, cluster connectivity, BH months."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components

import run_boston as boston
import run_sp100 as sp
from mnreg.model import (
    GibbsConfig,
    _lambda_log_prior,
    gibbs_multi_sem,
    metrics_rmse_mae,
    ols,
    predict_kriging,
    screen_networks,
)
from mnreg.networks import row_normalize
from run_boston_spatial_cv import _kmeans
from run_unified_revision import _conditional_replicates, _predictive_metrics, _thin_draws

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v28"
OUT.mkdir(parents=True, exist_ok=True)


def _l_stats(lam: np.ndarray) -> dict:
    total = np.sum(np.abs(lam), axis=1)
    return {
        "mean_L": float(total.mean()),
        "L_0.95": float(np.quantile(total, 0.95)),
        "p_L_gt_095": float(np.mean(total > 0.95)),
        "lambda_mean": lam.mean(axis=0).tolist(),
        "lambda_lo": np.quantile(lam, 0.025, axis=0).tolist(),
        "lambda_hi": np.quantile(lam, 0.975, axis=0).tolist(),
    }


def verify_beta_jacobian() -> None:
    lam = np.array([0.2, -0.3, 0.1])
    length = float(np.sum(np.abs(lam)))
    expected = (1 - 3) * np.log(length) + np.log(1.0 - length)
    got = _lambda_log_prior(lam, 1.0, "beta_l1")
    if abs(got - expected) > 1e-12:
        raise RuntimeError(f"beta log-prior {got} != {expected}")


def rho_summary() -> dict:
    rows = []
    for path in sorted((ROOT / "results" / "unified").glob("boundary_*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        if row.get("ok"):
            rows.append(float(row["rho"]))
    values = np.asarray(rows, dtype=float)
    payload = {
        "reps": int(values.size),
        "rho_mean": float(values.mean()),
        "rho_max": float(values.max()),
        "rho_lt_1": int(np.sum(values < 1.0)),
    }
    print("rho", payload, flush=True)
    return payload


def cluster_connectivity() -> dict:
    data = boston.load_boston()
    networks = boston.build_networks(data)
    labels = _kmeans(data["xy"], k=10, n_init=50, seed=2026)
    geo = np.asarray(networks["adjacency"]["geo"], dtype=float)
    undirected = ((geo + geo.T) > 0).astype(int)
    np.fill_diagonal(undirected, 0)
    report = []
    for cluster in range(10):
        nodes = np.flatnonzero(labels == cluster)
        block = undirected[np.ix_(nodes, nodes)]
        count, comp = connected_components(block, directed=False)
        sizes = np.bincount(comp).tolist()
        report.append({
            "cluster": cluster,
            "n": int(len(nodes)),
            "components": int(count),
            "largest": int(max(sizes)),
            "connected": bool(count == 1),
        })
    payload = {
        "connected_clusters": int(sum(row["connected"] for row in report)),
        "clusters": report,
    }
    print("connectivity", payload["connected_clusters"], "/10", flush=True)
    return payload


def _predictive(result, y, x, train, test, full, rng):
    beta, sigma, lam = _thin_draws(result, 2000)
    stack = _conditional_replicates(y, x, train, test, beta, sigma, lam, full, rng, 20)
    return _predictive_metrics(stack, y[test])


def _fit_record(y_train, x_train, mats, names, seed, prior):
    config = GibbsConfig(
        n_iter=8000, burn=4000, n_chains=4, seed=seed,
        adapt_covariance=True, lambda_prior=prior,
    )
    result = gibbs_multi_sem(y_train, x_train, mats, names=names, config=config)
    lam = result.samples["lambda"].reshape(-1, len(names))
    stats = _l_stats(lam)
    stats["acceptance"] = result.acceptance.astype(float).tolist()
    stats["names"] = names
    return result, stats


def boston_beta_holdout() -> dict:
    data = boston.load_boston()
    networks = boston.build_networks(data)
    record = json.loads((ROOT / "results" / "boston_adaptive" / "split_05.json").read_text(encoding="utf-8"))
    part = np.random.default_rng([int(record["seed"]), 1])
    train = np.sort(part.choice(boston.N, size=boston.N_TRAIN, replace=False))
    test = np.sort(np.setdiff1d(np.arange(boston.N), train))
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = np.where(raw[train].std(axis=0, ddof=0) < 1e-12, 1.0, raw[train].std(axis=0, ddof=0))
    scaled = (raw - center) / spread
    x = np.column_stack([np.ones(boston.N), scaled])
    y = data["y"]
    layers = list(record["active"])
    induced = [boston.induced_pair(networks["weights"][name], train)[1] for name in layers]
    full = [networks["adjacency"][name] for name in layers]
    chain_seeds = np.random.default_rng([int(record["seed"]), 3]).integers(1, 2**31 - 1, size=4)
    result, stats = _fit_record(y[train], x[train], induced, layers, int(chain_seeds[2]), "beta_l1")
    pred = _predictive(result, y, x, train, test, full, np.random.default_rng(2026))
    stats["holdout"] = pred
    print("boston beta", stats["mean_L"], pred["rmse"], pred["crps"], flush=True)
    return stats


def _sp_showcase():
    wiki = sp.load_constituents()
    tickers = wiki["Symbol"].dropna().unique().tolist()
    px = sp.download_prices(tickers)
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    fund = sp.fetch_fundamentals(tickers, wiki)
    panel = sp.monthly_panel(px)
    ret = panel["ret"].dropna(how="all")
    dates = ret.index[(ret.index >= "2021-01-01") & (ret.index <= "2026-08-31")]
    showcase = None
    showcase_date = None
    for date in dates[::-1]:
        cs = sp.align_cross_section(panel, fund, date)
        if cs is not None and len(cs["tickers"]) >= 70:
            showcase = cs
            showcase_date = date
            break
    return panel, fund, dates, showcase, showcase_date


def _split_indices(n):
    rng = np.random.default_rng(2026)
    idx = rng.permutation(n)
    n_test = max(int(0.2 * n), 15)
    test = np.sort(idx[:n_test])
    train = np.sort(idx[n_test:])
    return train, test


def sp_priors(showcase) -> dict:
    y = showcase["Y"]
    x = showcase["X"]
    train, test = _split_indices(len(y))
    scr = screen_networks(y, x, showcase["As"], names=showcase["names"], alpha=0.05, fdr=True)
    selected = np.where(scr["selected"])[0].tolist()
    bh = bool(len(selected))
    if not selected:
        selected = np.where(scr["pvalues"] < 0.10)[0].tolist() or [int(np.argmin(scr["pvalues"]))]
    mats = [row_normalize(showcase["As_raw"][i][np.ix_(train, train)]) for i in selected]
    full = [showcase["As"][i] for i in selected]
    names = [showcase["names"][i] for i in selected]
    rows = {"bh_on_full_sample": bh, "names": names}
    for prior in ("gaussian", "beta_l1"):
        result, stats = _fit_record(y[train], x[train], mats, names, 2026, prior)
        stats["holdout"] = _predictive(result, y, x, train, test, full, np.random.default_rng(2026))
        rows[prior] = stats
        print("sp", prior, stats["lambda_mean"], stats["mean_L"], stats["holdout"]["rmse"], flush=True)
    # Training-only screen, then one adapted Gaussian fit.
    train_mats = [row_normalize(showcase["As_raw"][i][np.ix_(train, train)]) for i in range(len(showcase["names"]))]
    scr_t = screen_networks(y[train], x[train], train_mats, names=showcase["names"], alpha=0.05, fdr=True)
    sel_t = np.where(scr_t["selected"])[0].tolist()
    regime = "benjamini_hochberg"
    if not sel_t:
        sel_t = np.where(scr_t["pvalues"] < 0.10)[0].tolist()
        regime = "fallback_p_lt_0.10"
    if not sel_t:
        sel_t = [int(np.argmin(scr_t["pvalues"]))]
        regime = "fallback_minimum_p"
    fit_mats = [train_mats[i] for i in sel_t]
    full_t = [showcase["As"][i] for i in sel_t]
    names_t = [showcase["names"][i] for i in sel_t]
    result, stats = _fit_record(y[train], x[train], fit_mats, names_t, 2027, "gaussian")
    pred = predict_kriging(
        y[train], x[train], x[test], train, test, stats["lambda_mean"],
        result.samples["beta"].reshape(-1, x.shape[1]).mean(axis=0), full_t,
    )
    scored = metrics_rmse_mae(y[test], pred)
    rows["training_screen"] = {
        "regime": regime,
        "names": names_t,
        "pvalues": {name: float(p) for name, p in zip(showcase["names"], scr_t["pvalues"])},
        "lambda_mean": stats["lambda_mean"],
        "holdout_rmse": float(scored["RMSE"]),
        "holdout_mae": float(scored["MAE"]),
    }
    b_ols, _, _ = ols(y[train], x[train])
    rows["training_screen"]["ols_rmse"] = float(metrics_rmse_mae(y[test], x[test] @ b_ols)["RMSE"])
    print("training screen", rows["training_screen"]["regime"], names_t, rows["training_screen"]["holdout_rmse"], flush=True)
    return rows


def bh_months(panel, fund, dates) -> dict:
    saved = pd.read_csv(ROOT / "results" / "unified" / "sp100_running_oos.csv")
    saved["date"] = pd.to_datetime(saved["date"])
    by_date = {row.date: row for row in saved.itertuples(index=False)}
    rows = []
    for date in dates:
        if pd.Timestamp(date) < pd.Timestamp("2021-06-30"):
            continue
        cs = sp.align_cross_section(panel, fund, date)
        if cs is None:
            continue
        key = pd.Timestamp(date)
        # The saved panel uses month-end dates; match on date.
        match = None
        for saved_date, item in by_date.items():
            if pd.Timestamp(saved_date).normalize() == pd.Timestamp(date).normalize():
                match = item
                break
        if match is None:
            continue
        y, x = cs["Y"], cs["X"]
        n = len(y)
        # The rolling design holds out the showcase tickers, not a fresh permutation.
        # Classification uses the training block implied by n_train in the archive.
        n_test = int(match.n_test)
        # Reconstruct the showcase holdout tickers from the August split below is not
        # available here; screen the training rows of a deterministic 80/20 is wrong.
        # Screen on the same ticker holdout used by run_sp100: the first showcase split.
        rows.append((date, cs, match, n_test))
    return _classify(rows)


def _classify(staged) -> dict:
    if not staged:
        return {"months": 0}
    # Showcase holdout tickers from the August cross-section and seed 2026.
    august = staged[-1][1]
    train, test = _split_indices(len(august["Y"]))
    holdout = set(np.array(august["tickers"])[test])
    classified = []
    for date, cs, match, _n_test in staged:
        tickers = np.array(cs["tickers"])
        is_test = np.array([t in holdout for t in tickers])
        is_train = ~is_test
        if is_test.sum() < 8 or is_train.sum() < 30:
            continue
        train_i = np.where(is_train)[0]
        as_tr = [row_normalize(A[np.ix_(train_i, train_i)]) for A in cs["As_raw"]]
        scr = screen_networks(cs["Y"][train_i], cs["X"][train_i], as_tr, names=cs["names"], alpha=0.05, fdr=True)
        bh = bool(np.any(scr["selected"]))
        classified.append({
            "date": str(pd.Timestamp(date).date()),
            "bh_active": bh,
            "ols_rmse": float(match.OLS_RMSE),
            "multi_rmse": float(match.MultiSEM_RMSE),
        })
    frame = pd.DataFrame(classified)
    def block(flag):
        part = frame[frame["bh_active"] == flag]
        return {
            "months": int(len(part)),
            "ols_rmse": float(part["ols_rmse"].mean()) if len(part) else None,
            "multi_rmse": float(part["multi_rmse"].mean()) if len(part) else None,
        }
    payload = {
        "months": int(len(frame)),
        "bh": block(True),
        "fallback": block(False),
    }
    print("bh months", payload, flush=True)
    return payload


def boston_prior_table(beta_holdout) -> dict:
    gauss = np.load(ROOT / "results" / "revision" / "split5_gaussian.npz")["draws_lambda"]
    beta = np.load(ROOT / "results" / "revision" / "split5_beta.npz")["draws_lambda"]
    s13 = json.loads((ROOT / "results" / "unified" / "table_s13.json").read_text(encoding="utf-8"))
    return {
        "gaussian": {**_l_stats(gauss), "holdout": {"rmse": s13["induced"]["rmse"], "crps": s13["induced"]["crps"]}},
        "beta_l1": {**_l_stats(beta), "holdout": beta_holdout["holdout"], "refit_mean_L": beta_holdout["mean_L"]},
    }


def refresh_figure_s10() -> None:
    gauss = np.load(ROOT / "results" / "revision" / "split5_gaussian.npz")["draws_lambda"]
    beta = np.load(ROOT / "results" / "revision" / "split5_beta.npz")["draws_lambda"]
    l_post = np.sum(np.abs(gauss), axis=1)
    l_beta = np.sum(np.abs(beta), axis=1)
    rng = np.random.default_rng(2026)
    accepted = []
    while sum(len(chunk) for chunk in accepted) < 200_000:
        draw = rng.normal(size=(500_000, 3))
        length = np.sum(np.abs(draw), axis=1)
        accepted.append(length[length < 1.0])
    l_gauss = np.concatenate(accepted)[:200_000]
    grid = np.linspace(0.001, 0.999, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].hist(l_gauss, bins=60, density=True, alpha=0.55, color="#2A9D8F", label=fr"Truncated Gaussian, mean {l_gauss.mean():.3f}")
    axes[0].plot(grid, 3.0 * grid ** 2, color="black", lw=1.4, label=r"Uniform shell, $3L^{2}$")
    axes[0].hist(l_post, bins=50, density=True, alpha=0.4, color="steelblue", label="Split 5 posterior")
    axes[0].axvline(l_gauss.mean(), color="#1D6A62", lw=1.0)
    axes[0].set_xlabel(r"$L=\|\lambda\|_1$")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(grid, 2.0 * (1.0 - grid), color="0.35", lw=1.5, label=r"Beta$(1,2)$ on $L$")
    axes[1].hist(l_beta, bins=50, density=True, alpha=0.75, color="#B85C38", label="Beta(1, 2) posterior")
    axes[1].set_xlabel(r"$L=\|\lambda\|_1$")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = ROOT / "results" / "figures" / "figure_s10_prior_posterior.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    print("figure s10", float(l_gauss.mean()), flush=True)


def main():
    verify_beta_jacobian()
    payload = {
        "jacobian": "(1-k) ln L + ln(1-L)",
        "rho": rho_summary(),
        "connectivity": cluster_connectivity(),
    }
    (OUT / "partial.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    panel, fund, dates, showcase, showcase_date = _sp_showcase()
    payload["showcase_date"] = str(pd.Timestamp(showcase_date).date())
    payload["sp_priors"] = sp_priors(showcase)
    payload["bh_months"] = bh_months(panel, fund, dates)
    payload["boston_beta"] = boston_beta_holdout()
    payload["boston_priors"] = boston_prior_table(payload["boston_beta"])
    refresh_figure_s10()
    (OUT / "analyses.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote", OUT / "analyses.json", flush=True)


if __name__ == "__main__":
    main()
