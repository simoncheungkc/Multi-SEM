"""Simulation suite: Poisson + power-law multiplex networks, Moran I, Gibbs, OOS."""

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
    assemble_V,
    feasible_lambda,
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
from mnreg.networks import degree_summary, erdos_renyi, powerlaw_configuration, row_normalize


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
RESULTS.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)


CHAIN_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # blue, orange, green, red
TRUE_BETA = np.array([0.10, -0.20, 0.40, -0.10, 0.25, 0.15])
TRUE_LAMBDA = np.array([0.40, 0.35, 0.00, 0.00, 0.00])  # Poisson + PL(γ=2.2)
NETWORK_SPEC = [
    ("Poisson d=8", "poisson", {"mean_degree": 8.0}),
    ("Power-law gamma=2.2", "powerlaw", {"gamma": 2.2}),
    ("Power-law gamma=2.5", "powerlaw", {"gamma": 2.5}),
    ("Power-law gamma=3.0", "powerlaw", {"gamma": 3.0}),
    ("Poisson d=4 (null)", "poisson", {"mean_degree": 4.0}),
]


def simulate_y(X, beta, lambdas, As, sigma, rng):
    n = X.shape[0]
    V = assemble_V(lambdas, As)
    eps = rng.normal(0.0, sigma, size=n)
    try:
        u = np.linalg.solve(V, eps)
    except np.linalg.LinAlgError:
        u = np.linalg.pinv(V) @ eps
    return X @ beta + u


def make_networks(n: int, rng: np.random.Generator):
    As_raw, names, meta = [], [], []
    for name, kind, kw in NETWORK_SPEC:
        if kind == "poisson":
            A = erdos_renyi(n, kw["mean_degree"], rng)
        else:
            A = powerlaw_configuration(n, kw["gamma"], rng)
        names.append(name)
        As_raw.append(A)
        info = degree_summary(A)
        info["name"] = name
        info["kind"] = kind
        info.update(kw)
        meta.append(info)
    As = [row_normalize(A) for A in As_raw]
    return As, As_raw, names, meta


def make_X(n: int, rng: np.random.Generator):
    Z = rng.normal(size=(n, 5))
    X = add_intercept(Z)
    X, _, _ = standardize_columns(X)
    return X


def run_moran_experiments(n_list, n_reps, rng):
    rows = []
    pvalue_store = {}
    for n in n_list:
        print(f"  Moran I screens at n={n}, reps={n_reps}", flush=True)
        net_rng = np.random.default_rng(rng.integers(1, 2**31 - 1))
        As, _, names, meta = make_networks(n, net_rng)
        X = make_X(n, net_rng)
        pvalue_store[n] = {"H0": {nm: [] for nm in names}, "H1": {nm: [] for nm in names}}
        for setting in ("H0", "H1"):
            for r in range(n_reps):
                if setting == "H0":
                    Y = X @ TRUE_BETA + rng.normal(size=n)
                else:
                    Y = simulate_y(X, TRUE_BETA, TRUE_LAMBDA, As, 1.0, rng)
                scr = screen_networks(Y, X, As, names=names, alpha=0.05, fdr=False)
                scr_fdr = screen_networks(Y, X, As, names=names, alpha=0.05, fdr=True)
                for i, nm in enumerate(names):
                    p = float(scr["pvalues"][i])
                    pvalue_store[n][setting][nm].append(p)
                    rows.append(
                        {
                            "n": n,
                            "rep": r,
                            "setting": setting,
                            "network": nm,
                            "true_lambda": float(TRUE_LAMBDA[i]),
                            "pvalue": p,
                            "z": float(scr["tests"][i]["z"]),
                            "I": float(scr["tests"][i]["I"]),
                            "reject_05": int(p < 0.05),
                            "reject_fdr": int(scr_fdr["selected"][i]),
                        }
                    )
                if (r + 1) % 100 == 0:
                    print(f"    {setting} n={n} rep {r+1}/{n_reps}", flush=True)
        # keep network meta
        pd.DataFrame(meta).to_csv(RESULTS / f"network_meta_n{n}.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "moran_replications.csv", index=False)
    return df, pvalue_store


def summarize_size_power(df: pd.DataFrame) -> pd.DataFrame:
    g = (
        df.groupby(["n", "setting", "network", "true_lambda"], as_index=False)
        .agg(
            reject_rate=("reject_05", "mean"),
            fdr_reject_rate=("reject_fdr", "mean"),
            mean_pvalue=("pvalue", "mean"),
            mean_z=("z", "mean"),
            n_reps=("pvalue", "size"),
        )
    )
    g["rate_type"] = np.where(g["setting"] == "H0", "size (Type I)", "power")
    g.to_csv(RESULTS / "moran_size_power.csv", index=False)
    return g


def plot_size_power(summary: pd.DataFrame):
    for n in sorted(summary["n"].unique()):
        sub = summary[summary["n"] == n]
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharey=False)
        for ax, setting, title, hline in (
            (axes[0], "H0", f"Type I error at n={n} (nominal 5%)", 0.05),
            (axes[1], "H1", f"Power at n={n} (α = 0.05)", None),
        ):
            s = sub[sub["setting"] == setting]
            ax.bar(np.arange(len(s)), s["reject_rate"].values, color="#4C6A92", width=0.7)
            ax.set_xticks(np.arange(len(s)))
            ax.set_xticklabels(s["network"].values, rotation=25, ha="right")
            ax.set_ylabel("Rejection rate")
            ax.set_title(title)
            ax.set_ylim(0, 1.05)
            if hline is not None:
                ax.axhline(hline, color="#B85C38", ls="--", lw=1, label="nominal 0.05")
                ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(FIG / f"moran_size_power_n{n}.png", dpi=140)
        plt.close(fig)


def plot_pvalue_histograms(store, n=200):
    names = list(store[n]["H1"].keys())
    fig, axes = plt.subplots(2, len(names), figsize=(3.1 * len(names), 6.2), sharex=True, sharey=True)
    bins = np.linspace(0, 1, 21)
    for row, setting in enumerate(("H0", "H1")):
        for col, nm in enumerate(names):
            ax = axes[row, col]
            ax.hist(store[n][setting][nm], bins=bins, color="#4C6A92", edgecolor="white", density=True)
            ax.set_xlim(0, 1)
            ax.set_xlabel("Two-sided p-value")
            if col == 0:
                ax.set_ylabel("Density")
            ttl = "Null (no network)" if setting == "H0" else "DGP with two active networks"
            ax.set_title(f"{nm}\n{ttl}", fontsize=9)
            ax.axvline(0.05, color="#B85C38", ls="--", lw=0.9)
    fig.suptitle(f"Moran I p-value distributions, n={n} (should lie in [0, 1])", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / f"moran_pvalues_n{n}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_gibbs_recovery(n, rng, config: GibbsConfig):
    print(f"  Gibbs posterior recovery at n={n}", flush=True)
    As, _, names, _ = make_networks(n, np.random.default_rng(rng.integers(1, 2**31 - 1)))
    X = make_X(n, rng)
    Y = simulate_y(X, TRUE_BETA, TRUE_LAMBDA, As, 1.0, rng)
    # Screen then fit the selected + always report fit on true active set
    scr = screen_networks(Y, X, As, names=names, alpha=0.05, fdr=True)
    print("    Moran selected:", scr["selected_names"], flush=True)

    t0 = time.time()
    result = gibbs_multi_sem(Y, X, As, names=names, config=config)
    print(f"    Gibbs finished in {time.time()-t0:.1f}s, accept={result.acceptance}", flush=True)
    beta_names = ["intercept"] + [f"x{j}" for j in range(1, 6)]
    summ = posterior_summary(result, beta_names=beta_names)
    table = pd.DataFrame(summ["table"])
    truth = list(TRUE_LAMBDA) + list(TRUE_BETA) + [1.0]
    table["truth"] = truth[: len(table)]
    table["abs_error"] = (table["mean"] - table["truth"]).abs()
    table.to_csv(RESULTS / "gibbs_recovery_posterior.csv", index=False)

    # Trace plots for lambda and sigma2
    lam = result.samples["lambda"]
    sig = result.samples["sigma2"]
    k = lam.shape[2]
    fig, axes = plt.subplots(k + 1, 1, figsize=(8.5, 1.7 * (k + 1)), sharex=True)
    for s in range(k):
        for c in range(lam.shape[0]):
            axes[s].plot(
                lam[c, :, s],
                lw=0.6,
                alpha=0.85,
                color=CHAIN_COLORS[c % len(CHAIN_COLORS)],
                label=f"Chain {c+1}" if s == 0 else None,
            )
        axes[s].axhline(TRUE_LAMBDA[s], color="black", ls="--", lw=1)
        axes[s].set_ylabel(names[s], fontsize=8)
        axes[s].set_ylim(-1, 1)
    if k:
        axes[0].legend(frameon=False, fontsize=7, ncol=4, loc="upper right")
    for c in range(sig.shape[0]):
        axes[-1].plot(sig[c], lw=0.6, alpha=0.85, color=CHAIN_COLORS[c % len(CHAIN_COLORS)])
    axes[-1].axhline(1.0, color="black", ls="--", lw=1)
    axes[-1].set_ylabel(r"$\sigma^2$")
    axes[-1].set_xlabel("Post-burn iteration")
    fig.suptitle(f"Gibbs traces, n={n} (dashed = true value; four chains)")
    fig.tight_layout()
    fig.savefig(FIG / "gibbs_traces_recovery.png", dpi=140)
    plt.close(fig)

    # Posterior densities for lambda
    fig, axes = plt.subplots(1, k, figsize=(3.0 * k, 3.3), sharey=True)
    for s, ax in enumerate(axes):
        draws = lam[:, :, s].ravel()
        ax.hist(draws, bins=30, density=True, color="#4C6A92", edgecolor="white")
        ax.axvline(TRUE_LAMBDA[s], color="#B85C38", ls="--", label="truth")
        ax.axvline(np.mean(draws), color="black", lw=1, label="posterior mean")
        ax.set_title(names[s], fontsize=9)
        ax.set_xlabel(r"$\lambda$")
        ax.set_xlim(-1, 1)
        if s == 0:
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "gibbs_lambda_posteriors.png", dpi=140)
    plt.close(fig)

    np.savez(
        RESULTS / "gibbs_recovery_samples.npz",
        lambda_samples=lam,
        beta_samples=result.samples["beta"],
        sigma2=sig,
        Y=Y,
        X=X,
    )
    return {
        "summary": summ,
        "selected": scr["selected_names"],
        "pvalues": {nm: float(p) for nm, p in zip(names, scr["pvalues"])},
        "Y": Y,
        "X": X,
        "As": As,
        "names": names,
        "result": result,
    }


def _sub_row_normalize(A, idx):
    return row_normalize(A[np.ix_(idx, idx)])


def run_oos(n, n_reps, rng, gibbs_cfg: GibbsConfig):
    print(f"  Out-of-sample predictions at n={n}, reps={n_reps}", flush=True)
    As, As_raw, names, _ = make_networks(n, np.random.default_rng(rng.integers(1, 2**31 - 1)))
    rows = []
    showcase = None
    n_test = max(int(0.2 * n), 20)
    for r in range(n_reps):
        X = make_X(n, rng)
        Y = simulate_y(X, TRUE_BETA, TRUE_LAMBDA, As, 1.0, rng)
        idx = rng.permutation(n)
        test_idx = np.sort(idx[:n_test])
        train_idx = np.sort(idx[n_test:])
        Xtr, Xte = X[train_idx], X[test_idx]
        Ytr, Yte = Y[train_idx], Y[test_idx]
        As_tr = [_sub_row_normalize(A, train_idx) for A in As_raw]

        # OLS
        b_ols, _, _ = ols(Ytr, Xtr)
        pred_ols = predict_mean_only(Xte, b_ols)
        m_ols = metrics_rmse_mae(Yte, pred_ols)

        # Single-network ML
        single = {}
        for i, nm in enumerate(names):
            fit = gibbs_multi_sem(Ytr, Xtr, [As_tr[i]], names=[nm], config=gibbs_cfg)
            lam_s = fit.samples["lambda"].reshape(-1).mean()
            beta_s = fit.samples["beta"].reshape(-1, fit.samples["beta"].shape[-1]).mean(axis=0)
            pred = predict_kriging(
                Ytr, Xtr, Xte, train_idx, test_idx, np.array([lam_s]), beta_s, [As[i]]
            )
            single[nm] = metrics_rmse_mae(Yte, pred)
            single[nm]["lambda"] = float(lam_s)

        # Multi-SEM on true active networks (0,1)
        active = [0, 1]
        As_tr_act = [As_tr[i] for i in active]
        As_full_act = [As[i] for i in active]
        result = gibbs_multi_sem(
            Ytr,
            Xtr,
            As_tr_act,
            names=[names[i] for i in active],
            config=gibbs_cfg,
        )
        summ = posterior_summary(result)
        lam_hat = summ["means"]["lambda"]
        beta_hat = summ["means"]["beta"]
        pred_multi = predict_kriging(
            Ytr, Xtr, Xte, train_idx, test_idx, lam_hat, beta_hat, As_full_act
        )
        m_multi = metrics_rmse_mae(Yte, pred_multi)

        # Multi-SEM on Moran-selected networks
        scr = screen_networks(Ytr, Xtr, As_tr, names=names, alpha=0.05, fdr=True)
        sel = np.where(scr["selected"])[0].tolist()
        if len(sel) == 0:
            pred_sel = pred_ols
            m_sel = m_ols
            lam_sel = []
        else:
            fit_sel = gibbs_multi_sem(
                Ytr,
                Xtr,
                [As_tr[i] for i in sel],
                names=[names[i] for i in sel],
                config=gibbs_cfg,
            )
            lam_sel_hat = fit_sel.samples["lambda"].reshape(-1, fit_sel.samples["lambda"].shape[-1]).mean(axis=0)
            beta_sel_hat = fit_sel.samples["beta"].reshape(-1, fit_sel.samples["beta"].shape[-1]).mean(axis=0)
            pred_sel = predict_kriging(
                Ytr,
                Xtr,
                Xte,
                train_idx,
                test_idx,
                lam_sel_hat,
                beta_sel_hat,
                [As[i] for i in sel],
            )
            m_sel = metrics_rmse_mae(Yte, pred_sel)
            lam_sel = lam_sel_hat.tolist()

        rec = {
            "rep": r,
            "OLS_RMSE": m_ols["RMSE"],
            "OLS_MAE": m_ols["MAE"],
            "OLS_corr": m_ols["corr"],
            "MultiSEM_RMSE": m_multi["RMSE"],
            "MultiSEM_MAE": m_multi["MAE"],
            "MultiSEM_corr": m_multi["corr"],
            "SelectedSEM_RMSE": m_sel["RMSE"],
            "SelectedSEM_MAE": m_sel["MAE"],
            "selected": json.dumps([names[i] for i in sel]),
            "lambda_hat": json.dumps(lam_hat.tolist()),
            "lambda_sel": json.dumps(lam_sel),
        }
        for nm, met in single.items():
            rec[f"Single[{nm}]_RMSE"] = met["RMSE"]
            rec[f"Single[{nm}]_MAE"] = met["MAE"]
        rows.append(rec)
        if r == 0:
            showcase = {
                "Yte": Yte,
                "pred_ols": pred_ols,
                "pred_multi": pred_multi,
                "pred_sel": pred_sel,
                "m_ols": m_ols,
                "m_multi": m_multi,
                "m_sel": m_sel,
            }
        print(
            f"    OOS rep {r+1}/{n_reps}  OLS RMSE={m_ols['RMSE']:.3f}  "
            f"MultiSEM RMSE={m_multi['RMSE']:.3f}  selected={ [names[i] for i in sel] }",
            flush=True,
        )

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "simulation_oos.csv", index=False)

    # Predicted vs actual (first replication)
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7), sharex=True, sharey=True)
    panels = [
        (showcase["pred_ols"], "OLS (no network)", showcase["m_ols"]),
        (showcase["pred_multi"], "Multi-SEM (true active networks)", showcase["m_multi"]),
        (showcase["pred_sel"], "Multi-SEM (Moran-selected)", showcase["m_sel"]),
    ]
    lo = min(showcase["Yte"].min(), min(p.min() for p, _, _ in panels))
    hi = max(showcase["Yte"].max(), max(p.max() for p, _, _ in panels))
    for ax, (pred, title, met) in zip(axes, panels):
        ax.scatter(showcase["Yte"], pred, s=18, c="#4C6A92", alpha=0.8, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], color="#B85C38", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Actual Y (test)")
        ax.set_ylabel("Predicted Y")
        ax.text(
            0.04,
            0.96,
            f"RMSE={met['RMSE']:.3f}\nMAE={met['MAE']:.3f}\ncorr={met['corr']:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )
    fig.suptitle(f"Simulation out-of-sample, n={n}, 20% node holdout")
    fig.tight_layout()
    fig.savefig(FIG / "simulation_pred_vs_actual.png", dpi=140)
    plt.close(fig)

    # Summary bars — labels match Table 5 in the report
    display = [
        ("OLS_RMSE", "OLS (no network)"),
        ("Single[Poisson d=8]_RMSE", "Single SEM, Poisson d = 8"),
        ("Single[Power-law gamma=2.2]_RMSE", "Single SEM, PL γ = 2.2"),
        (None, "Single SEM, inactive networks"),
        ("SelectedSEM_RMSE", "Multi-SEM, Moran-selected"),
        ("MultiSEM_RMSE", "Multi-SEM, true active pair"),
    ]
    inactive_cols = [
        "Single[Power-law gamma=2.5]_RMSE",
        "Single[Power-law gamma=3.0]_RMSE",
        "Single[Poisson d=4 (null)]_RMSE",
    ]
    values, labels = [], []
    for col, lab in display:
        labels.append(lab)
        if col is None:
            values.append(float(df[inactive_cols].mean().mean()))
        else:
            values.append(float(df[col].mean()))
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.barh(labels[::-1], values[::-1], color="#4C6A92")
    ax.set_xlabel("Mean test RMSE across replications")
    ax.set_title(f"Out-of-sample RMSE by model, n={n}, {n_reps} replications")
    fig.tight_layout()
    fig.savefig(FIG / "simulation_oos_rmse.png", dpi=140)
    plt.close(fig)
    return df, showcase


# Scenario 2: opposite-sign spatial coefficients inside the open l1 ball.
# Signed sum is 0.20; the l1 norm is 0.80. Networks are redrawn each replication.
OPPOSITE_LAMBDA = np.array([0.50, -0.30, 0.00, 0.00, 0.00])


def opposite_sign_replication(
    rep: int,
    n: int = 200,
    n_iter: int = 6000,
    burn: int = 3000,
    n_chains: int = 4,
) -> dict:
    """One Monte Carlo replication of the opposite-sign multiplex DGP."""
    rng = np.random.default_rng([202611, int(rep)])
    As, _, names, _ = make_networks(n, rng)
    X = make_X(n, rng)
    Y = simulate_y(X, TRUE_BETA, OPPOSITE_LAMBDA, As, 1.0, rng)
    cfg = GibbsConfig(
        n_iter=n_iter,
        burn=burn,
        n_chains=n_chains,
        seed=int(rng.integers(1, 2**31 - 1)),
        adapt_covariance=True,
        target_accept=0.30,
    )
    result = gibbs_multi_sem(Y, X, As, names=names, config=cfg)
    lam = result.samples["lambda"]
    flat = lam.reshape(-1, lam.shape[-1])
    mean = flat.mean(axis=0)
    q025 = np.quantile(flat, 0.025, axis=0)
    q975 = np.quantile(flat, 0.975, axis=0)
    truth = OPPOSITE_LAMBDA
    cover = (truth >= q025) & (truth <= q975)
    return {
        "rep": int(rep),
        "mean": mean.astype(float).tolist(),
        "q025": q025.astype(float).tolist(),
        "q975": q975.astype(float).tolist(),
        "cover": cover.astype(int).tolist(),
        "bias": (mean - truth).astype(float).tolist(),
        "rhat": [float(gelman_rubin(lam[:, :, s])) for s in range(lam.shape[-1])],
        "accept": result.acceptance.astype(float).tolist(),
        "truth": truth.astype(float).tolist(),
        "names": list(names),
        "l1_truth": float(np.sum(np.abs(truth))),
        "signed_sum_truth": float(np.sum(truth)),
    }


def summarize_opposite_sign(rows: list[dict]) -> dict:
    means = np.array([row["mean"] for row in rows], dtype=float)
    truth = OPPOSITE_LAMBDA
    bias = means.mean(axis=0) - truth
    rmse = np.sqrt(np.mean((means - truth) ** 2, axis=0))
    cover = np.array([row["cover"] for row in rows], dtype=float).mean(axis=0)
    rhats = np.array([row["rhat"] for row in rows], dtype=float)
    names = rows[0]["names"]
    columns = []
    for s, name in enumerate(names):
        columns.append({
            "parameter": name,
            "truth": float(truth[s]),
            "mean_posterior_mean": float(means[:, s].mean()),
            "bias": float(bias[s]),
            "rmse": float(rmse[s]),
            "coverage_95": float(cover[s]),
            "max_rhat": float(np.nanmax(rhats[:, s])),
        })
    return {
        "n_reps": len(rows),
        "n": 200,
        "covariates": "intercept plus five i.i.d. standard normal regressors, column-standardized",
        "lambda_truth": truth.tolist(),
        "l1_norm": float(np.sum(np.abs(truth))),
        "signed_sum": float(np.sum(truth)),
        "columns": columns,
        "gibbs": {"n_iter": 6000, "burn": 3000, "n_chains": 4, "adapt_covariance": True},
    }


def run_opposite_sign_experiment(rerun: bool = False, n_reps: int = 200) -> dict:
    """Table 4b. The default reads the finished 200-replication archive."""
    archive = ROOT / "results" / "revision" / "opposite_sign_reps.jsonl"
    rows = []
    if archive.exists() and not rerun:
        for line in archive.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(opposite_sign_replication, rep) for rep in range(n_reps)]
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["rep"])
    if len(rows) != n_reps:
        raise RuntimeError(f"expected {n_reps} opposite-sign replications, found {len(rows)}")
    summary = summarize_opposite_sign(rows)
    out = ROOT / "results" / "revision" / "table4b.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    expected = {
        "Poisson d=8": (-0.106, 0.142, 0.910),
        "Power-law gamma=2.2": (0.066, 0.100, 0.915),
    }
    for column in summary["columns"]:
        if column["parameter"] not in expected:
            continue
        got = (round(column["bias"], 3), round(column["rmse"], 3), round(column["coverage_95"], 3))
        if got != expected[column["parameter"]]:
            raise RuntimeError(f"{column['parameter']} {got} != {expected[column['parameter']]}")
    print(f"Table 4b matches the archived opposite-sign experiment -> {out}", flush=True)
    return summary


def table_s4_prior_grid():
    """Prior settings for Supplementary Table S4. The fit is run_table_s4_prior.py."""
    from run_table_s4_prior import SETTINGS

    return SETTINGS


def main():
    rng = np.random.default_rng(2026)
    n_list = [100, 200, 500]
    n_reps_test = 500
    n_oos = 30
    n_gibbs = 200

    t_all = time.time()
    print("=== 1. Moran I size and power ===", flush=True)
    df_m, store = run_moran_experiments(n_list, n_reps_test, rng)
    summary = summarize_size_power(df_m)
    plot_size_power(summary)
    plot_pvalue_histograms(store, n=200)
    print(summary.to_string(index=False), flush=True)

    print("=== 2. Gibbs posterior recovery ===", flush=True)
    gcfg = GibbsConfig(
        n_iter=6000, burn=3000, n_chains=4, seed=2026, target_accept=0.30, adapt_covariance=True
    )
    recovery = run_gibbs_recovery(n_gibbs, rng, gcfg)
    print(pd.DataFrame(recovery["summary"]["table"]).to_string(index=False), flush=True)

    print("=== 3. Out-of-sample predictions ===", flush=True)
    oos_cfg = GibbsConfig(
        n_iter=2500, burn=1200, n_chains=2, seed=7, target_accept=0.30, adapt_covariance=True
    )
    oos_df, _ = run_oos(200, n_oos, rng, oos_cfg)
    print(oos_df[[c for c in oos_df.columns if "RMSE" in c]].mean().to_string(), flush=True)

    payload = {
        "true_beta": TRUE_BETA.tolist(),
        "true_lambda": TRUE_LAMBDA.tolist(),
        "network_spec": NETWORK_SPEC,
        "moran_selected_on_recovery": recovery["selected"],
        "recovery_pvalues": recovery["pvalues"],
        "elapsed_sec": time.time() - t_all,
    }
    with open(RESULTS / "simulation_meta.json", "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Done in {payload['elapsed_sec']:.1f}s", flush=True)


if __name__ == "__main__":
    import sys

    if "--opposite-sign" in sys.argv:
        run_opposite_sign_experiment(rerun="--rerun" in sys.argv)
    else:
        main()
