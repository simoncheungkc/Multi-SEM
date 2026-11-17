"""Five-layer S&P 100 analysis with a headquarters distance network.

The archived four-layer results in results/sp100_running_oos.csv are not
overwritten. Screening for the new specification uses the training tickers
only. Structural pruning uses off-diagonal correlations and does not use
the outcome.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_rel

from mnreg.model import (
    GibbsConfig,
    benjamini_hochberg,
    drop_collinear_networks,
    fit_ml_multi_sem,
    gibbs_multi_sem,
    metrics_rmse_mae,
    moran_i_test,
    ols,
    posterior_summary,
    predict_kriging,
    predict_mean_only,
    screen_networks,
)
from mnreg.networks import (
    build_headquarters_geographic_network,
    geographic_topology,
    offdiagonal_correlation,
    row_normalize,
)
from run_sp100 import (
    ROOT,
    align_cross_section,
    download_prices,
    fetch_fundamentals,
    load_constituents,
    monthly_panel,
)

RESULTS = ROOT / "results"
DATA = ROOT / "data"
HQ = DATA / "sp100_headquarters.csv"
GEO_NAME = "Headquarters geography"


def loss_tests(loss: np.ndarray) -> dict:
    loss = np.asarray(loss, dtype=float)
    wins = int(np.sum(loss > 0))
    months = int(len(loss))
    ties = int(np.sum(loss == 0))
    sign = binomtest(wins, months - ties, 0.5, alternative="two-sided")
    mean = float(loss.mean())
    centered = loss - mean
    gamma0 = float(np.dot(centered, centered) / months)
    variance = gamma0
    lag = 3
    for ell in range(1, lag + 1):
        gamma = float(np.dot(centered[ell:], centered[:-ell]) / months)
        variance += 2.0 * (1.0 - ell / (lag + 1)) * gamma
    stat = mean / np.sqrt(variance / months)
    from scipy.special import ndtr

    return {
        "months": months,
        "wins": wins,
        "ties": ties,
        "sign_pvalue": float(sign.pvalue),
        "mean_rmse_differential": mean,
        "newey_west_lag": lag,
        "newey_west_stat": float(stat),
        "newey_west_pvalue": float(2.0 * (1.0 - ndtr(abs(stat)))),
    }


def paired_t(baseline: np.ndarray, alternative: np.ndarray) -> dict:
    result = ttest_rel(baseline, alternative)
    return {"statistic": float(result.statistic), "pvalue": float(result.pvalue)}


def binary_kappa(raw: np.ndarray) -> float:
    degree = (np.asarray(raw) > 0).sum(axis=1).astype(float)
    # The builder already has a zero diagonal. Guard the other raw matrices too.
    if np.trace(np.asarray(raw) > 0):
        degree = degree - np.diag(np.asarray(raw) > 0).astype(float)
    mean_degree = float(degree.mean())
    if mean_degree <= 0:
        return float("nan")
    return float(degree.max() / mean_degree)


def singular_psi(adjacency: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(adjacency, dtype=float), compute_uv=False)
    total = float(singular.sum())
    if total <= 0:
        return float("nan")
    return float(singular[0] / total)


def gaussian_mc_pvalues(residuals, design, adjacency, observed_z, triggers, rng, draws: int = 499):
    """Analytical Gaussian tail, unless degree or singular-value concentration triggers Monte Carlo."""
    pvalues = []
    references = []
    for z_value, adjacency_s, trigger in zip(observed_z, adjacency, triggers):
        if not trigger:
            from scipy.special import ndtr

            pvalues.append(float(2.0 * (1.0 - ndtr(abs(z_value)))))
            references.append("Gaussian")
            continue
        exceed = 0
        n = len(residuals)
        projector = np.eye(n) - design @ np.linalg.pinv(design.T @ design) @ design.T
        for _ in range(draws):
            simulated = projector @ rng.normal(size=n)
            z_draw = moran_i_test(simulated, adjacency_s, design)["z"]
            if abs(z_draw) >= abs(z_value) - 1e-12:
                exceed += 1
        pvalues.append((1.0 + exceed) / (draws + 1.0))
        references.append("Monte Carlo")
    return np.asarray(pvalues, dtype=float), references


def attach_geography(cs: dict) -> dict:
    weights, adjacency = build_headquarters_geographic_network(HQ, cs["tickers"], threshold_km=800.0)
    out = dict(cs)
    out["As_raw"] = [weights] + list(cs["As_raw"])
    out["As"] = [adjacency] + list(cs["As"])
    out["names"] = [GEO_NAME] + list(cs["names"])
    return out


def training_screen(cs, train_idx, rng):
    raw = [A[np.ix_(train_idx, train_idx)] for A in cs["As_raw"]]
    standardized = [row_normalize(A) for A in raw]
    names = list(cs["names"])
    correlations = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            correlations.append(
                {
                    "left": names[i],
                    "right": names[j],
                    "r_vec": offdiagonal_correlation(standardized[i], standardized[j]),
                }
            )
    keep, kept_names, _ = drop_collinear_networks(standardized, names, corr_max=0.99)
    screened_raw = [raw[i] for i in keep]
    screened = [standardized[i] for i in keep]
    screened_names = [names[i] for i in keep]
    Y = cs["Y"][train_idx]
    X = cs["X"][train_idx]
    tests = screen_networks(Y, X, screened, names=screened_names, alpha=0.05, fdr=False)
    kappa = [binary_kappa(A) for A in screened_raw]
    psi = [singular_psi(A) for A in screened]
    triggers = [(k > 3.0 or p > 0.20) for k, p in zip(kappa, psi)]
    observed_z = [row["z"] for row in tests["tests"]]
    pvalues, references = gaussian_mc_pvalues(
        tests["resid"], X, screened, observed_z, triggers, rng, draws=499
    )
    selected = benjamini_hochberg(pvalues, alpha=0.05)
    rows = []
    for i, row in enumerate(tests["tests"]):
        rows.append(
            {
                "name": screened_names[i],
                "I": row["I"],
                "z": row["z"],
                "kappa": kappa[i],
                "psi": psi[i],
                "reference": references[i],
                "pvalue": float(pvalues[i]),
                "selected": bool(selected[i]),
            }
        )
    return {
        "correlations": correlations,
        "tests": rows,
        "selected_index": [keep[i] for i, flag in enumerate(selected) if flag],
        "selected_names": [row["name"] for row in rows if row["selected"]],
    }


def holdout_prediction(cs, train_idx, test_idx, layer_index):
    raw = [cs["As_raw"][i] for i in layer_index]
    train_layers = [row_normalize(A[np.ix_(train_idx, train_idx)]) for A in raw]
    fit = fit_ml_multi_sem(cs["Y"][train_idx], cs["X"][train_idx], train_layers)
    prediction = predict_kriging(
        cs["Y"][train_idx],
        cs["X"][train_idx],
        cs["X"][test_idx],
        train_idx,
        test_idx,
        fit["lambda"],
        fit["beta"],
        [cs["As"][i] for i in layer_index],
    )
    return fit, metrics_rmse_mae(cs["Y"][test_idx], prediction)


def main():
    started = time.time()
    wiki = load_constituents()
    tickers = [t for t in wiki["Symbol"].dropna().unique().tolist() if t != "GOOG"]
    prices = download_prices(tickers)
    tickers = [t for t in tickers if t in prices.columns]
    prices = prices[tickers]
    fundamentals = fetch_fundamentals(tickers, wiki)
    panel = monthly_panel(prices)
    returns = panel["ret"].dropna(how="all")
    dates = returns.index[(returns.index >= "2021-06-30") & (returns.index <= "2026-08-31")]

    showcase = None
    showcase_date = None
    for date in dates[::-1]:
        cross_section = align_cross_section(panel, fundamentals, date)
        if cross_section is not None and len(cross_section["tickers"]) >= 70:
            showcase = attach_geography(cross_section)
            showcase_date = date
            break
    if showcase is None:
        raise RuntimeError("no showcase cross-section")

    # Topology and collinearity use the network matrices only.
    weights = showcase["As_raw"][0]
    adjacency = showcase["As"][0]
    topology = geographic_topology(weights, adjacency)
    full_correlations = []
    for i, name in enumerate(showcase["names"][1:], start=1):
        full_correlations.append(
            {
                "left": GEO_NAME,
                "right": name,
                "r_vec": offdiagonal_correlation(adjacency, showcase["As"][i]),
            }
        )
    diagnostics = {
        "showcase_date": str(pd.Timestamp(showcase_date).date()),
        "n": len(showcase["tickers"]),
        "threshold_km": 800.0,
        "topology_full_sample": topology,
        "full_sample_correlations": full_correlations,
        "max_abs_r": max(abs(row["r_vec"]) for row in full_correlations),
    }
    (RESULTS / "sp100_geo_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print("full-sample geo", topology, flush=True)
    print("max |r|", diagnostics["max_abs_r"], flush=True)

    n = len(showcase["Y"])
    rng = np.random.default_rng(2026)
    permutation = rng.permutation(n)
    n_test = max(int(0.2 * n), 15)
    test_idx = np.sort(permutation[:n_test])
    train_idx = np.sort(permutation[n_test:])
    holdout = set(np.array(showcase["tickers"])[test_idx])
    screen = training_screen(showcase, train_idx, np.random.default_rng(2026))
    print("showcase train screen", screen["tests"], flush=True)
    print("selected", screen["selected_names"], flush=True)

    # Showcase predictions. Gibbs is reserved for the selected specification.
    b_ols, _, _ = ols(showcase["Y"][train_idx], showcase["X"][train_idx])
    ols_metrics = metrics_rmse_mae(showcase["Y"][test_idx], predict_mean_only(showcase["X"][test_idx], b_ols))
    single = {}
    for i, name in enumerate(showcase["names"]):
        fit, metrics = holdout_prediction(showcase, train_idx, test_idx, [i])
        single[name] = {"metrics": metrics, "lambda": fit["lambda"].tolist(), "success": fit["success"]}
        print(f"  single {name} RMSE {metrics['RMSE']:.4f}", flush=True)
    if screen["selected_index"]:
        multi_fit, multi_metrics = holdout_prediction(showcase, train_idx, test_idx, screen["selected_index"])
    else:
        multi_fit, multi_metrics = None, ols_metrics
    showcase_payload = {
        "date": str(pd.Timestamp(showcase_date).date()),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "screen": screen,
        "OLS": ols_metrics,
        "single": single,
        "MultiSEM5": None if multi_fit is None else {
            "metrics": multi_metrics,
            "lambda": multi_fit["lambda"].tolist(),
            "names": screen["selected_names"],
            "success": multi_fit["success"],
        },
    }
    (RESULTS / "sp100_geo_showcase.json").write_text(json.dumps(showcase_payload, indent=2), encoding="utf-8")

    rows = []
    for date in dates:
        cross_section = align_cross_section(panel, fundamentals, date)
        if cross_section is None:
            continue
        cross_section = attach_geography(cross_section)
        labels = np.array(cross_section["tickers"])
        test_i = np.where(np.array([t in holdout for t in labels]))[0]
        train_i = np.where(np.array([t not in holdout for t in labels]))[0]
        if len(test_i) < 8 or len(train_i) < 30:
            continue
        month_screen = training_screen(cross_section, train_i, np.random.default_rng(2026))
        b_month, _, _ = ols(cross_section["Y"][train_i], cross_section["X"][train_i])
        ols_month = metrics_rmse_mae(
            cross_section["Y"][test_i], predict_mean_only(cross_section["X"][test_i], b_month)
        )
        singles = {}
        for i, name in enumerate(cross_section["names"]):
            _, metrics = holdout_prediction(cross_section, train_i, test_i, [i])
            singles[name] = metrics
        if month_screen["selected_index"]:
            fit5, metrics5 = holdout_prediction(cross_section, train_i, test_i, month_screen["selected_index"])
            lambda5 = fit5["lambda"].tolist()
        else:
            metrics5 = ols_month
            lambda5 = []
        # Relational four-layer screen, same training rows, for the comparison column.
        relational_names = cross_section["names"][1:]
        relational = {
            "names": relational_names,
            "As_raw": cross_section["As_raw"][1:],
            "As": cross_section["As"][1:],
            "Y": cross_section["Y"],
            "X": cross_section["X"],
        }
        relational_screen = training_screen(relational, train_i, np.random.default_rng(2026))
        if relational_screen["selected_index"]:
            _, metrics4 = holdout_prediction(cross_section, train_i, test_i, [i + 1 for i in relational_screen["selected_index"]])
        else:
            metrics4 = ols_month
        geo_row = next(row for row in month_screen["tests"] if row["name"] == GEO_NAME)
        rows.append(
            {
                "date": str(pd.Timestamp(date).date()),
                "n_train": int(len(train_i)),
                "n_test": int(len(test_i)),
                "selected5": json.dumps(month_screen["selected_names"]),
                "geo_selected": bool(geo_row["selected"]),
                "geo_I": geo_row["I"],
                "geo_z": geo_row["z"],
                "geo_kappa": geo_row["kappa"],
                "geo_psi": geo_row["psi"],
                "geo_reference": geo_row["reference"],
                "geo_p": geo_row["pvalue"],
                "OLS_RMSE": ols_month["RMSE"],
                "OLS_MAE": ols_month["MAE"],
                "GeoSEM_RMSE": singles[GEO_NAME]["RMSE"],
                "GeoSEM_MAE": singles[GEO_NAME]["MAE"],
                "SectorSEM_RMSE": singles["Sector affiliation"]["RMSE"],
                "IndustrySEM_RMSE": singles["Industry (related companies)"]["RMSE"],
                "CorrSEM_RMSE": singles["Correlation (5-NN)"]["RMSE"],
                "BoostSEM_RMSE": singles["Boosted correlation resolvent"]["RMSE"],
                "Multi4_RMSE": metrics4["RMSE"],
                "Multi4_MAE": metrics4["MAE"],
                "Multi5_RMSE": metrics5["RMSE"],
                "Multi5_MAE": metrics5["MAE"],
                "lambda5": json.dumps(lambda5),
            }
        )
        print(
            f"{date.date()} OLS {ols_month['RMSE']:.4f} Geo {singles[GEO_NAME]['RMSE']:.4f} "
            f"M5 {metrics5['RMSE']:.4f} geo_selected {geo_row['selected']}",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(RESULTS / "sp100_geo_oos.csv", index=False)

    oos = pd.DataFrame(rows)
    summary = {
        "n_months": int(len(oos)),
        "geo_months": int(oos["geo_selected"].sum()),
        "mean_rmse": oos[
            ["OLS_RMSE", "GeoSEM_RMSE", "SectorSEM_RMSE", "IndustrySEM_RMSE", "CorrSEM_RMSE", "BoostSEM_RMSE", "Multi4_RMSE", "Multi5_RMSE"]
        ].mean().to_dict(),
        "mean_mae": oos[["OLS_MAE", "GeoSEM_MAE", "Multi4_MAE", "Multi5_MAE"]].mean().to_dict(),
        "geo_vs_ols": loss_tests(oos["OLS_RMSE"] - oos["GeoSEM_RMSE"]),
        "multi5_vs_ols": loss_tests(oos["OLS_RMSE"] - oos["Multi5_RMSE"]),
        "multi4_vs_ols": loss_tests(oos["OLS_RMSE"] - oos["Multi4_RMSE"]),
        "paired_t_multi5": paired_t(oos["OLS_RMSE"].values, oos["Multi5_RMSE"].values),
        "paired_t_geo": paired_t(oos["OLS_RMSE"].values, oos["GeoSEM_RMSE"].values),
        "elapsed_sec": time.time() - started,
    }
    # The placeholder in loss_tests is unused; paired_t above is the t-test.
    (RESULTS / "sp100_geo_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)

    print("=== showcase Gibbs ===", flush=True)
    if screen["selected_index"]:
        layers = [row_normalize(showcase["As_raw"][i][np.ix_(train_idx, train_idx)]) for i in screen["selected_index"]]
        result = gibbs_multi_sem(
            showcase["Y"][train_idx],
            showcase["X"][train_idx],
            layers,
            names=screen["selected_names"],
            config=GibbsConfig(n_iter=8000, burn=4000, n_chains=4, seed=2026),
        )
        table = posterior_summary(result, beta_names=["intercept", "lag_ret", "vol", "mom12", "log_price"])
        pd.DataFrame(table["table"]).to_csv(RESULTS / "sp100_geo_posterior.csv", index=False)
        print(pd.DataFrame(table["table"]).to_string(index=False), flush=True)
    geo_layers = [row_normalize(showcase["As_raw"][0][np.ix_(train_idx, train_idx)])]
    geo_result = gibbs_multi_sem(
        showcase["Y"][train_idx],
        showcase["X"][train_idx],
        geo_layers,
        names=[GEO_NAME],
        config=GibbsConfig(n_iter=8000, burn=4000, n_chains=4, seed=2027),
    )
    geo_table = posterior_summary(geo_result, beta_names=["intercept", "lag_ret", "vol", "mom12", "log_price"])
    pd.DataFrame(geo_table["table"]).to_csv(RESULTS / "sp100_geo_only_posterior.csv", index=False)
    print(pd.DataFrame(geo_table["table"]).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
