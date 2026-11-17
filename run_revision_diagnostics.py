"""Draw-based posterior diagnostics, parametric bootstrap size, and recovery Monte Carlo.

Writes results/revision_diagnostics.json. Does not overwrite the published posterior CSVs.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from mnreg.model import (
    GibbsConfig,
    assemble_V,
    effective_sample_size,
    gelman_rubin,
    gibbs_multi_sem,
    moran_i_test,
    ols,
    posterior_summary,
    screen_networks,
)
from mnreg.networks import powerlaw_configuration, row_normalize
from run_simulation import TRUE_BETA, TRUE_LAMBDA, make_X, make_networks, simulate_y

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "revision_diagnostics.json"


def boundary_stats(lam: np.ndarray) -> dict:
    flat = lam.reshape(-1, lam.shape[-1])
    L = np.abs(flat).sum(axis=1)
    return {
        "P_positive": [float(x) for x in (flat > 0).mean(axis=0)],
        "L_mean": float(L.mean()),
        "L_p95": float(np.quantile(L, 0.95)),
        "P_L_gt_095": float(np.mean(L > 0.95)),
        "L_max": float(L.max()),
        "n_draws": int(flat.shape[0]),
    }


def fast_forward_moran_rng(rng: np.random.Generator) -> None:
    """Match the parent-stream consumption of run_moran_experiments in run_simulation.main."""
    for n in (100, 200, 500):
        _ = int(rng.integers(1, 2**31 - 1))
        for _setting in ("H0", "H1"):
            for _r in range(500):
                rng.normal(size=n)


def replay_table4() -> dict:
    rng = np.random.default_rng(2026)
    fast_forward_moran_rng(rng)
    n = 200
    child = np.random.default_rng(int(rng.integers(1, 2**31 - 1)))
    As, _, names, _ = make_networks(n, child)
    X = make_X(n, rng)
    Y = simulate_y(X, TRUE_BETA, TRUE_LAMBDA, As, 1.0, rng)
    gcfg = GibbsConfig(n_iter=6000, burn=3000, n_chains=4, seed=2026, target_accept=0.30)
    t0 = time.time()
    result = gibbs_multi_sem(Y, X, As, names=names, config=gcfg)
    elapsed = time.time() - t0
    beta_names = ["intercept"] + [f"x{j}" for j in range(1, 6)]
    table = pd.DataFrame(posterior_summary(result, beta_names=beta_names)["table"])
    published = pd.read_csv(RESULTS / "gibbs_recovery_posterior.csv")
    stats = boundary_stats(result.samples["lambda"])
    stats["names"] = list(names)
    stats["elapsed_sec"] = elapsed
    stats["means"] = [float(x) for x in table["mean"]]
    stats["published_means"] = [float(x) for x in published["mean"]]
    stats["mean_abs_diff"] = float(np.max(np.abs(np.array(stats["means"]) - np.array(stats["published_means"][: len(stats["means"])]))))
    np.savez(
        RESULTS / "gibbs_recovery_samples.npz",
        lambda_samples=result.samples["lambda"],
        beta_samples=result.samples["beta"],
        sigma2=result.samples["sigma2"],
    )
    return stats


def showcase_draws() -> dict:
    from run_sp100 import (
        align_cross_section,
        download_prices,
        drop_collinear_networks,
        fetch_fundamentals,
        load_constituents,
        monthly_panel,
        row_normalize,
    )

    wiki = load_constituents()
    tickers = wiki["Symbol"].dropna().unique().tolist()
    px = download_prices(tickers)
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    fund = fetch_fundamentals(tickers, wiki)
    panel = monthly_panel(px)
    ret = panel["ret"].dropna(how="all")
    dates = ret.index[(ret.index >= "2021-01-01") & (ret.index <= "2026-08-31")]
    cs_show = None
    showcase_date = None
    for d in dates[::-1]:
        cs = align_cross_section(panel, fund, d)
        if cs is not None and len(cs["tickers"]) >= 70:
            showcase_date = d
            cs_show = cs
            break
    scr = screen_networks(cs_show["Y"], cs_show["X"], cs_show["As"], names=cs_show["names"], alpha=0.05, fdr=True)
    sel_idx = np.where(scr["selected"])[0].tolist()
    if len(sel_idx) == 0:
        sel_idx = np.where(scr["pvalues"] < 0.10)[0].tolist()
    keep, _, _ = drop_collinear_networks([cs_show["As_raw"][i] for i in sel_idx], [cs_show["names"][i] for i in sel_idx])
    sel_idx = [sel_idx[i] for i in keep]
    n = len(cs_show["Y"])
    rng = np.random.default_rng(2026)
    idx = rng.permutation(n)
    n_test = max(int(0.2 * n), 15)
    train_idx = np.sort(idx[n_test:])
    As_tr = [row_normalize(cs_show["As_raw"][i][np.ix_(train_idx, train_idx)]) for i in sel_idx]
    names = [cs_show["names"][i] for i in sel_idx]
    gcfg = GibbsConfig(n_iter=8000, burn=4000, n_chains=4, seed=2026)
    t0 = time.time()
    result = gibbs_multi_sem(cs_show["Y"][train_idx], cs_show["X"][train_idx], As_tr, names=names, config=gcfg)
    elapsed = time.time() - t0
    stats = boundary_stats(result.samples["lambda"])
    stats["names"] = names
    stats["date"] = str(pd.Timestamp(showcase_date).date())
    stats["n_train"] = int(len(train_idx))
    stats["elapsed_sec"] = elapsed
    beta_names = ["intercept", "lag_ret", "vol", "mom12", "log_price"]
    table = pd.DataFrame(posterior_summary(result, beta_names=beta_names)["table"])
    published = pd.read_csv(RESULTS / "sp100_posterior.csv")
    stats["means"] = {str(r.parameter): float(r.mean) for r in table.itertuples()}
    stats["published_means"] = {str(r.parameter): float(r.mean) for r in published.itertuples()}
    np.savez(RESULTS / "sp100_showcase_lambda.npz", lambda_samples=result.samples["lambda"], names=np.array(names))
    return stats


def _one_null_screen(seed_pack):
    n, rep, B, net_seed = seed_pack
    net_rng = np.random.default_rng(net_seed)
    A = row_normalize(powerlaw_configuration(n, 2.2, net_rng))
    Z = net_rng.normal(size=(n, 5))
    X = np.column_stack([np.ones(n), Z])
    X[:, 1:] = (X[:, 1:] - X[:, 1:].mean(0)) / X[:, 1:].std(0, ddof=0)
    rng = np.random.default_rng(10_000_000 + 1000 * n + rep)
    Y = X @ TRUE_BETA + rng.normal(size=n)
    beta, resid, _ = ols(Y, X)
    z_obs = moran_i_test(resid, A, X)["z"]
    gauss_p = moran_i_test(resid, A, X)["pvalue"]
    boot_rng = np.random.default_rng(20_000_000 + 1000 * n + rep)
    n_ge = 0
    for _b in range(B):
        Yb = X @ beta + boot_rng.normal(scale=np.sqrt(float(resid @ resid) / max(n - X.shape[1], 1)), size=n)
        _, resid_b, _ = ols(Yb, X)
        zb = moran_i_test(resid_b, A, X)["z"]
        if abs(zb) >= abs(z_obs):
            n_ge += 1
    boot_p = (1.0 + n_ge) / (B + 1.0)
    return n, int(gauss_p < 0.05), int(boot_p < 0.05)


def bootstrap_size(n_list=(100, 200, 500), n_reps=500, B=199) -> dict:
    """Fixed power-law graph and design per n; parametric residual bootstrap under the Gaussian null."""
    packs = []
    net_seeds = {100: 2026100, 200: 2026200, 500: 2026500}
    for n in n_list:
        for rep in range(n_reps):
            packs.append((n, rep, B, net_seeds[n]))
    # networks are regenerated inside each worker from the same seed, so the graph is fixed per n
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i, out in enumerate(pool.map(_one_null_screen, packs, chunksize=5)):
            rows.append(out)
            if (i + 1) % 200 == 0:
                print(f"  bootstrap {i+1}/{len(packs)}", flush=True)
    df = pd.DataFrame(rows, columns=["n", "gauss", "boot"])
    summary = {}
    for n, g in df.groupby("n"):
        summary[str(int(n))] = {
            "reps": int(len(g)),
            "B": B,
            "gaussian_size": float(g["gauss"].mean()),
            "bootstrap_size": float(g["boot"].mean()),
            "gaussian_rejects": int(g["gauss"].sum()),
            "bootstrap_rejects": int(g["boot"].sum()),
        }
    summary["elapsed_sec"] = time.time() - t0
    summary["gamma"] = 2.2
    summary["design"] = "one power-law graph and one standardized design per n; null Y = X beta + N(0,I); parametric residual bootstrap of the OLS fit"
    return summary


def _one_recovery(task):
    seed, n_iter, burn, n_chains = task
    rng = np.random.default_rng(seed)
    n = 200
    As, _, names, _ = make_networks(n, np.random.default_rng(seed + 17))
    X = make_X(n, rng)
    Y = simulate_y(X, TRUE_BETA, TRUE_LAMBDA, As, 1.0, rng)
    cfg = GibbsConfig(
        n_iter=n_iter, burn=burn, n_chains=n_chains, seed=seed, target_accept=0.30, adapt_covariance=True
    )
    result = gibbs_multi_sem(Y, X, As, names=names, config=cfg)
    lam_ch = result.samples["lambda"]
    beta_ch = result.samples["beta"]
    lam = lam_ch.reshape(-1, lam_ch.shape[-1])
    beta = beta_ch.reshape(-1, beta_ch.shape[-1])
    def pack(draws, truth):
        lo = np.quantile(draws, 0.025, axis=0)
        hi = np.quantile(draws, 0.975, axis=0)
        mean = draws.mean(axis=0)
        return {
            "mean": mean.tolist(),
            "cover": ((lo <= truth) & (truth <= hi)).astype(int).tolist(),
            "bias_draw": (mean - truth).tolist(),
        }
    rhats = [gelman_rubin(lam_ch[:, :, s]) for s in range(lam_ch.shape[-1])]
    rhats += [gelman_rubin(beta_ch[:, :, j]) for j in range(beta_ch.shape[-1])]
    network_ess = [
        sum(effective_sample_size(lam_ch[chain, :, index]) for chain in range(lam_ch.shape[0]))
        for index in range(lam_ch.shape[-1])
    ]
    return {
        "lambda": pack(lam, TRUE_LAMBDA),
        "beta": pack(beta, TRUE_BETA),
        "names": names,
        "rhat_max": float(np.nanmax(rhats)),
        "min_network_ess": float(np.min(network_ess)),
    }


def monte_carlo(R=200, n_iter=4000, burn=2000, n_chains=4) -> dict:
    tasks = [(50_000 + r, n_iter, burn, n_chains) for r in range(R)]
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=min(16, os.cpu_count() or 4)) as pool:
        for i, out in enumerate(pool.map(_one_recovery, tasks, chunksize=1)):
            rows.append(out)
            if (i + 1) % 10 == 0:
                print(f"  recovery {i+1}/{R} elapsed {time.time()-t0:.0f}s", flush=True)
    lam = np.array([r["lambda"]["bias_draw"] for r in rows])
    bet = np.array([r["beta"]["bias_draw"] for r in rows])
    lam_c = np.array([r["lambda"]["cover"] for r in rows])
    bet_c = np.array([r["beta"]["cover"] for r in rows])
    # RMSE from posterior-mean error across replications
    def summarize(err, cover, truth, labels):
        out = []
        count = float(len(err))
        for j, lab in enumerate(labels):
            coverage = float(cover[:, j].mean())
            squared = err[:, j] ** 2
            rmse = float(np.sqrt(np.mean(squared)))
            rmse_mcse = float(np.std(squared, ddof=1) / (2.0 * rmse * np.sqrt(count))) if rmse > 0 else 0.0
            out.append({
                "parameter": lab,
                "truth": float(truth[j]),
                "bias": float(err[:, j].mean()),
                "rmse": rmse,
                "rmse_mcse": rmse_mcse,
                "coverage": coverage,
                "coverage_mcse": float(np.sqrt(coverage * (1.0 - coverage) / count)),
            })
        return out
    labels_l = rows[0]["names"]
    labels_b = ["intercept", "x1", "x2", "x3", "x4", "x5"]
    return {
        "R": R,
        "n": 200,
        "n_iter": n_iter,
        "burn": burn,
        "n_chains": n_chains,
        "elapsed_sec": time.time() - t0,
        "lambda": summarize(lam, lam_c, TRUE_LAMBDA, labels_l),
        "beta": summarize(bet, bet_c, TRUE_BETA, labels_b),
        "rhat_max": float(np.nanmax([row["rhat_max"] for row in rows])),
        "adapt_covariance": True,
    }


def main():
    payload = {}
    print("=== Table 4 replay ===", flush=True)
    payload["table4"] = replay_table4()
    print(json.dumps({k: payload["table4"][k] for k in ("L_mean", "L_p95", "P_L_gt_095", "P_positive", "mean_abs_diff", "elapsed_sec")}, indent=2), flush=True)
    _dump(payload)
    print("=== Showcase draws ===", flush=True)
    payload["table7"] = showcase_draws()
    print(json.dumps({k: payload["table7"][k] for k in ("L_mean", "L_p95", "P_L_gt_095", "P_positive", "names", "elapsed_sec")}, indent=2), flush=True)
    _dump(payload)
    print("=== Bootstrap size ===", flush=True)
    payload["bootstrap"] = bootstrap_size()
    print(json.dumps(payload["bootstrap"], indent=2), flush=True)
    _dump(payload)
    print("=== Monte Carlo recovery ===", flush=True)
    payload["monte_carlo"] = monte_carlo()
    print(json.dumps(payload["monte_carlo"], indent=2), flush=True)
    _dump(payload)
    print("done", flush=True)


def _dump(payload):
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
