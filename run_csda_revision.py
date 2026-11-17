"""CSDA revision benchmarks.

1. Recover the n=100 power-law graph used for Table 3 and calibrate
   residual Moran p-values with conditional Gaussian Monte Carlo.
2. Compare induced-subgraph estimation with the coherent training
   marginal likelihood (Schur complement of the full precision) at n=200.
3. Check whether a diagonal-purged clique resolvent equals the base clique.

Results are written under results/. The S&P 100 study is not re-estimated.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mnreg.model import (
    GibbsConfig,
    add_intercept,
    assemble_V,
    feasible_lambda,
    logdet_signed,
    ols,
    predict_kriging,
    _draw_invgamma,
    _log_posterior_lambda,
)
from mnreg.networks import (
    affiliation_network,
    boosted_resolvent,
    degree_summary,
    row_normalize,
)
from run_simulation import TRUE_BETA, TRUE_LAMBDA, make_X, make_networks, simulate_y

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

PL_NAME = "Power-law gamma=2.2"
N_REPS_MORAN = 500
B_BOOT = 499
N_MARGINAL = 30
N_HOLD = 40  # 20% of n=200


def clique_resolvent_check() -> dict:
    """Branch A if purging the resolvent diagonal and re-standardizing returns C."""
    labels = np.array([0] * 6 + [1] * 11 + [2] * 4 + [3] * 1 + [4] * 8)
    raw = affiliation_network(labels)
    C = row_normalize(raw)
    G = boosted_resolvent(C, alpha=0.5)
    G_purged = row_normalize(G)
    diff = np.max(np.abs(G_purged - C))
    # Off-diagonal correlation of the un-purged resolvent with the base clique.
    mask = ~np.eye(C.shape[0], dtype=bool)
    a = C[mask] - C[mask].mean()
    b = G[mask] - G[mask].mean()
    r = float(a @ b / np.sqrt((a @ a) * (b @ b)))
    out = {
        "max_abs_diff_after_purge_and_row_standardize": float(diff),
        "mean_resolvent_diagonal_before_purge": float(np.mean(np.diag(G))),
        "offdiag_corr_unpurged_resolvent_vs_clique": r,
        "branch": "A" if diff < 1e-8 else "B",
        "note": (
            "row_normalize zeros the diagonal before dividing by the row sum. "
            "The model matrices in run_sp100 pass through row_normalize, so "
            "diagonals that the resolvent creates are removed before estimation."
        ),
    }
    (RESULTS / "clique_resolvent_check.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Clique resolvent check:", out, flush=True)
    return out


def recover_table3_n100():
    """Replay run_moran_experiments' first draw (n=100, master seed 2026)."""
    rng = np.random.default_rng(2026)
    net_seed = int(rng.integers(1, 2**31 - 1))
    net_rng = np.random.default_rng(net_seed)
    As, As_raw, names, meta = make_networks(100, net_rng)
    X = make_X(100, net_rng)
    saved = pd.read_csv(RESULTS / "network_meta_n100.csv")
    pl = next(m for m in meta if m["name"] == PL_NAME)
    row = saved.loc[saved["name"] == PL_NAME].iloc[0]
    keys = ["n_edges", "mean_degree", "median_degree", "max_degree", "isolates", "min_positive_degree"]
    mismatches = {k: (pl[k], row[k]) for k in keys if abs(float(pl[k]) - float(row[k])) > 1e-6}
    if mismatches:
        raise RuntimeError(f"Recovered graph does not match network_meta_n100.csv: {mismatches}")
    print(
        f"Recovered Table 3 graph (seed {net_seed}): "
        f"edges={pl['n_edges']}, max degree={pl['max_degree']}, mean={pl['mean_degree']}",
        flush=True,
    )
    return rng, As, As_raw, names, X, meta, net_seed


def _moran_constants(A, X):
    n, q = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    trA = float(np.trace(A))
    trAP = float(np.trace(xtx_inv @ (X.T @ (A @ X))))
    trAM = trA - trAP
    df = n - q
    EI = trAM / df
    AX = A @ X
    AM = A - AX @ (xtx_inv @ X.T)
    MAM = AM - X @ (xtx_inv @ (X.T @ AM))
    trMAMA = float(np.sum(AM * MAM.T))
    VI = 2.0 * trMAMA / (df * (df + 2)) - 2.0 * (trAM ** 2) / (df ** 2 * (df + 2))
    VI = max(VI, 1e-16)
    return xtx_inv, EI, np.sqrt(VI)


def _residuals(Z, X, xtx_inv):
    """Z is (n, B). Return OLS residuals with the same shape."""
    beta = xtx_inv @ (X.T @ Z)
    return Z - X @ beta


def _moran_z_from_resid(E, A, EI, se):
    """E is (n, B)."""
    AE = A @ E
    num = np.sum(E * AE, axis=0)
    den = np.sum(E * E, axis=0)
    return (num / den - EI) / se


def spectral_and_degree(A_raw, A_row):
    deg = A_raw.sum(axis=1)
    kappa = float(deg.max() / deg.mean()) if deg.mean() > 0 else np.nan
    # Singular values are nonnegative, so the ratio is defined for a zero-diagonal A.
    theta = np.linalg.svd(A_row, compute_uv=False)
    psi = float(theta[0] / theta.sum()) if theta.sum() > 0 else np.nan
    return kappa, psi, theta


def replay_and_bootstrap():
    rng, As, As_raw, names, X, meta, net_seed = recover_table3_n100()
    pl_idx = names.index(PL_NAME)
    A = As[pl_idx]
    A_raw = As_raw[pl_idx]
    kappa, psi, _ = spectral_and_degree(A_raw, A)
    print(f"  kappa={kappa:.3f}, psi={psi:.3f} (trigger if kappa>3 or psi>0.20)", flush=True)

    xtx_inv, EI, se = _moran_constants(A, X)
    boot_rng = np.random.default_rng(20261021)

    rows = []
    for setting in ("H0", "H1"):
        z_obs = np.empty(N_REPS_MORAN)
        p_analytic = np.empty(N_REPS_MORAN)
        p_boot = np.empty(N_REPS_MORAN)
        t0 = time.time()
        for r in range(N_REPS_MORAN):
            if setting == "H0":
                Y = X @ TRUE_BETA + rng.normal(size=X.shape[0])
            else:
                Y = simulate_y(X, TRUE_BETA, TRUE_LAMBDA, As, 1.0, rng)
            beta, resid, _ = ols(Y, X)
            z = float(_moran_z_from_resid(resid.reshape(-1, 1), A, EI, se)[0])
            z_obs[r] = z
            p_analytic[r] = float(2.0 * (1.0 - ndtr_abs(abs(z))))
            Zb = boot_rng.normal(size=(X.shape[0], B_BOOT))
            Eb = _residuals(Zb, X, xtx_inv)
            zb = _moran_z_from_resid(Eb, A, EI, se)
            p_boot[r] = (1.0 + np.sum(np.abs(zb) >= abs(z))) / (B_BOOT + 1)
            if (r + 1) % 100 == 0:
                print(f"    {setting} {r+1}/{N_REPS_MORAN} ({time.time()-t0:.1f}s)", flush=True)
        rows.append(
            {
                "setting": setting,
                "network": PL_NAME,
                "n": 100,
                "n_reps": N_REPS_MORAN,
                "B": B_BOOT,
                "gaussian_reject_rate": float(np.mean(p_analytic < 0.05)),
                "bootstrap_reject_rate": float(np.mean(p_boot < 0.05)),
                "mean_gaussian_p": float(np.mean(p_analytic)),
                "mean_bootstrap_p": float(np.mean(p_boot)),
            }
        )
        print(
            f"  {setting}: Gaussian {rows[-1]['gaussian_reject_rate']:.4f}, "
            f"bootstrap {rows[-1]['bootstrap_reject_rate']:.4f}",
            flush=True,
        )

    summary = {
        "net_seed": net_seed,
        "graph_matches_network_meta_n100": True,
        "kappa": kappa,
        "psi_singular_value_share": psi,
        "rule": "bootstrap if kappa>3 or psi>0.20",
        "rule_triggers": bool(kappa > 3.0 or psi > 0.20),
        "degree": degree_summary(A_raw),
        "rates": rows,
    }
    pd.DataFrame(rows).to_csv(RESULTS / "bootstrap_matched_table3.csv", index=False)
    (RESULTS / "bootstrap_matched_table3.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def ndtr_abs(z):
    from math import erf, sqrt

    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def schur_training_factor(lambdas, As_full, train_idx, test_idx):
    """Return S such that the training marginal precision is S / sigma^2.

    S = M_tt - M_t* M_**^{-1} M_*t with M = V'V on the full graph.
    """
    if not feasible_lambda(lambdas):
        return None
    V = assemble_V(lambdas, As_full)
    M = V.T @ V
    Mss = M[np.ix_(test_idx, test_idx)]
    Mst = M[np.ix_(test_idx, train_idx)]
    Mts = M[np.ix_(train_idx, test_idx)]
    Mtt = M[np.ix_(train_idx, train_idx)]
    try:
        solved = np.linalg.solve(Mss, Mst)
    except np.linalg.LinAlgError:
        return None
    S = Mtt - Mts @ solved
    S = 0.5 * (S + S.T)
    return S


def log_posterior_marginal(lambdas, u_t, As_full, train_idx, test_idx, sigma2, xi):
    S = schur_training_factor(lambdas, As_full, train_idx, test_idx)
    if S is None:
        return -np.inf, None
    sign, ld = np.linalg.slogdet(S)
    if sign <= 0 or not np.isfinite(ld):
        return -np.inf, None
    quad = float(u_t @ (S @ u_t))
    if quad < 0 and quad > -1e-8:
        quad = 0.0
    if quad < 0 or not np.isfinite(quad):
        return -np.inf, None
    # 0.5 log det(S) matches log|det V| on a model that is exactly the training filter.
    lp = 0.5 * float(ld) - 0.5 * quad / sigma2 - 0.5 * xi * float(np.dot(lambdas, lambdas))
    return lp, S


def _gibbs_induced(Y, X, As, cfg, rng, start):
    """One chain of the existing induced-subgraph sampler, timed inside the caller."""
    n, q = X.shape
    k = len(As)
    Sigma0_inv = np.eye(q) / cfg.Sigma0_diag
    mu0 = np.full(q, cfg.mu0_scale)
    beta_ols, _, s2_ols = ols(Y, X)
    lam = np.asarray(start, dtype=float).copy()
    beta = beta_ols.copy()
    sigma2 = max(float(s2_ols), 1e-8)
    tau = 0.08
    n_adapt = cfg.burn
    n_keep = cfg.n_iter - cfg.burn
    lam_s = np.zeros((n_keep, k))
    beta_s = np.zeros((n_keep, q))
    keep = 0
    logp, V = _log_posterior_lambda(lam, Y - X @ beta, As, sigma2, cfg.xi)
    if not np.isfinite(logp):
        lam = np.zeros(k)
        logp, V = _log_posterior_lambda(lam, Y - X @ beta, As, sigma2, cfg.xi)
    for t in range(cfg.n_iter):
        resid = Y - X @ beta
        prop = lam + tau * rng.normal(size=k)
        logp_prop, V_prop = _log_posterior_lambda(prop, resid, As, sigma2, cfg.xi)
        if np.isfinite(logp_prop) and np.log(rng.uniform()) < (logp_prop - logp):
            lam, V, logp = prop, V_prop, logp_prop
        if t < n_adapt and (t + 1) % 50 == 0:
            # tau adaptation is intentionally light; both methods share the schedule logic
            tau = float(np.clip(tau, 1e-4, 0.5))
        VX, VY = V @ X, V @ Y
        prec = Sigma0_inv + (VX.T @ VX) / sigma2
        prec = 0.5 * (prec + prec.T)
        cov = np.linalg.inv(prec)
        mean = cov @ (Sigma0_inv @ mu0 + (VX.T @ VY) / sigma2)
        beta = rng.multivariate_normal(mean, 0.5 * (cov + cov.T) + 1e-10 * np.eye(q))
        resid = Y - X @ beta
        vr = V @ resid
        sse = float(vr @ vr)
        sigma2 = float(_draw_invgamma(rng, cfg.alpha0 + n / 2.0, cfg.gamma0 + 0.5 * sse))
        sigma2 = max(sigma2, 1e-12)
        logp, V = _log_posterior_lambda(lam, resid, As, sigma2, cfg.xi)
        if t >= cfg.burn:
            lam_s[keep] = lam
            beta_s[keep] = beta
            keep += 1
    return lam_s.mean(axis=0), beta_s.mean(axis=0)


def _gibbs_marginal(Y_t, X_t, As_full, train_idx, test_idx, cfg, rng, start):
    n_t, q = X_t.shape
    k = len(As_full)
    Sigma0_inv = np.eye(q) / cfg.Sigma0_diag
    mu0 = np.full(q, cfg.mu0_scale)
    beta_ols, _, s2_ols = ols(Y_t, X_t)
    lam = np.asarray(start, dtype=float).copy()
    beta = beta_ols.copy()
    sigma2 = max(float(s2_ols), 1e-8)
    tau = 0.08
    n_keep = cfg.n_iter - cfg.burn
    lam_s = np.zeros((n_keep, k))
    beta_s = np.zeros((n_keep, q))
    keep = 0
    u = Y_t - X_t @ beta
    logp, S = log_posterior_marginal(lam, u, As_full, train_idx, test_idx, sigma2, cfg.xi)
    if not np.isfinite(logp):
        lam = np.zeros(k)
        logp, S = log_posterior_marginal(lam, u, As_full, train_idx, test_idx, sigma2, cfg.xi)
    for t in range(cfg.n_iter):
        u = Y_t - X_t @ beta
        prop = lam + tau * rng.normal(size=k)
        logp_prop, S_prop = log_posterior_marginal(prop, u, As_full, train_idx, test_idx, sigma2, cfg.xi)
        if np.isfinite(logp_prop) and np.log(rng.uniform()) < (logp_prop - logp):
            lam, S, logp = prop, S_prop, logp_prop
        if t < cfg.burn and (t + 1) % 50 == 0:
            tau = float(np.clip(tau, 1e-4, 0.5))
        SX = S @ X_t
        SY = S @ Y_t
        prec = Sigma0_inv + (X_t.T @ SX) / sigma2
        prec = 0.5 * (prec + prec.T)
        cov = np.linalg.inv(prec)
        mean = cov @ (Sigma0_inv @ mu0 + (X_t.T @ SY) / sigma2)
        beta = rng.multivariate_normal(mean, 0.5 * (cov + cov.T) + 1e-10 * np.eye(q))
        u = Y_t - X_t @ beta
        quad = float(u @ (S @ u))
        sigma2 = float(_draw_invgamma(rng, cfg.alpha0 + n_t / 2.0, cfg.gamma0 + 0.5 * quad))
        sigma2 = max(sigma2, 1e-12)
        logp, S = log_posterior_marginal(lam, u, As_full, train_idx, test_idx, sigma2, cfg.xi)
        if t >= cfg.burn:
            lam_s[keep] = lam
            beta_s[keep] = beta
            keep += 1
    return lam_s.mean(axis=0), beta_s.mean(axis=0)


def concentrated_ml(kind, Y_t, X_t, As_induced, As_full, train_idx, test_idx, xi=1.0):
    from scipy.optimize import minimize

    k = len(As_induced) if kind == "induced" else len(As_full)

    def obj(z):
        z = np.asarray(z, dtype=float)
        if not feasible_lambda(z):
            return 1e12
        if kind == "induced":
            val = _log_posterior_lambda(z, Y_t - X_t @ np.zeros(X_t.shape[1]), As_induced, 1.0, 0.0)[0]
            # Use the concentrated induced likelihood, matching fit_ml_multi_sem.
            V = assemble_V(z, As_induced)
            ld = logdet_signed(V)
            if not np.isfinite(ld):
                return 1e12
            Yt, Xt = V @ Y_t, V @ X_t
            beta, *_ = np.linalg.lstsq(Xt, Yt, rcond=None)
            r = Yt - Xt @ beta
            sse = float(r @ r)
            n = Y_t.shape[0]
            return float(-ld + 0.5 * n * np.log(sse / n) + 0.5 * xi * float(z @ z))
        V = assemble_V(z, As_full)
        M = V.T @ V
        Mss = M[np.ix_(test_idx, test_idx)]
        Mst = M[np.ix_(test_idx, train_idx)]
        Mts = M[np.ix_(train_idx, test_idx)]
        Mtt = M[np.ix_(train_idx, train_idx)]
        try:
            S = Mtt - Mts @ np.linalg.solve(Mss, Mst)
        except np.linalg.LinAlgError:
            return 1e12
        S = 0.5 * (S + S.T)
        sign, ld = np.linalg.slogdet(S)
        if sign <= 0 or not np.isfinite(ld):
            return 1e12
        # GLS concentration: profile beta, then sigma2.
        try:
            # Solve S^{1/2} via cholesky if possible.
            chol = np.linalg.cholesky(S + 1e-10 * np.eye(S.shape[0]))
        except np.linalg.LinAlgError:
            return 1e12
        Yw = chol.T @ Y_t
        Xw = chol.T @ X_t
        beta, *_ = np.linalg.lstsq(Xw, Yw, rcond=None)
        r = Yw - Xw @ beta
        sse = float(r @ r)
        n = Y_t.shape[0]
        return float(-0.5 * ld + 0.5 * n * np.log(sse / n) + 0.5 * xi * float(z @ z))

    bounds = [(-0.95, 0.95)] * k
    cons = {"type": "ineq", "fun": lambda z: 0.95 - np.sum(np.abs(z))}
    res = minimize(obj, np.zeros(k), method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 80, "ftol": 1e-10})
    lam = np.asarray(res.x, dtype=float)
    if not feasible_lambda(lam) or not res.success:
        lam = np.zeros(k)
    if kind == "induced":
        V = assemble_V(lam, As_induced)
        beta, *_ = np.linalg.lstsq(V @ X_t, V @ Y_t, rcond=None)
    else:
        S = schur_training_factor(lam, As_full, train_idx, test_idx)
        chol = np.linalg.cholesky(S + 1e-10 * np.eye(S.shape[0]))
        beta, *_ = np.linalg.lstsq(chol.T @ X_t, chol.T @ Y_t, rcond=None)
    return lam, beta, bool(res.success)


def holdout_rmse(Y, X, train_idx, test_idx, lam, beta, As_full):
    pred = predict_kriging(
        Y[train_idx], X[train_idx], X[test_idx], train_idx, test_idx, lam, beta, As_full
    )
    err = Y[test_idx] - pred
    return float(np.sqrt(np.mean(err ** 2)))


def marginal_benchmark():
    """30 trials at n=200. Two active networks. Same full-graph kriging predictor."""
    cfg = GibbsConfig(n_iter=800, burn=400, n_chains=1, seed=7, xi=1.0)
    rows = []
    master = np.random.default_rng(2026)
    # Active DGP uses the first two networks of NETWORK_SPEC, lambda (0.40, 0.35).
    true_lam = np.array([0.40, 0.35])
    for rep in range(N_MARGINAL):
        t_rep = time.time()
        net_rng = np.random.default_rng(int(master.integers(1, 2**31 - 1)))
        As, As_raw, names, _ = make_networks(200, net_rng)
        # Keep the two networks that are active in the paper DGP.
        As2 = As[:2]
        names2 = names[:2]
        X = make_X(200, net_rng)
        Y = simulate_y(X, TRUE_BETA, np.array([0.40, 0.35, 0, 0, 0]), As, 1.0, net_rng)
        perm = master.permutation(200)
        test_idx = np.sort(perm[:N_HOLD])
        train_idx = np.sort(perm[N_HOLD:])
        Y_t, X_t = Y[train_idx], X[train_idx]
        As_tr = [row_normalize(A[np.ix_(train_idx, train_idx)]) for A in As_raw[:2]]

        # Likelihood-evaluation timing at the true parameter (40 repeats).
        u = Y_t - X_t @ TRUE_BETA
        t0 = time.perf_counter()
        for _ in range(40):
            _log_posterior_lambda(true_lam, u, As_tr, 1.0, 1.0)
        t_ind_like = (time.perf_counter() - t0) / 40.0
        t0 = time.perf_counter()
        for _ in range(40):
            log_posterior_marginal(true_lam, u, As2, train_idx, test_idx, 1.0, 1.0)
        t_mar_like = (time.perf_counter() - t0) / 40.0

        t0 = time.perf_counter()
        lam_i_ml, beta_i_ml, ok_i = concentrated_ml(
            "induced", Y_t, X_t, As_tr, As2, train_idx, test_idx
        )
        t_ind_ml = time.perf_counter() - t0
        t0 = time.perf_counter()
        lam_m_ml, beta_m_ml, ok_m = concentrated_ml(
            "marginal", Y_t, X_t, As_tr, As2, train_idx, test_idx
        )
        t_mar_ml = time.perf_counter() - t0

        rng_i = np.random.default_rng(10_000 + rep)
        rng_m = np.random.default_rng(20_000 + rep)
        t0 = time.perf_counter()
        lam_i, beta_i = _gibbs_induced(Y_t, X_t, As_tr, cfg, rng_i, lam_i_ml)
        t_ind_mc = time.perf_counter() - t0
        t0 = time.perf_counter()
        lam_m, beta_m = _gibbs_marginal(Y_t, X_t, As2, train_idx, test_idx, cfg, rng_m, lam_m_ml)
        t_mar_mc = time.perf_counter() - t0

        rmse_i_ml = holdout_rmse(Y, X, train_idx, test_idx, lam_i_ml, beta_i_ml, As2)
        rmse_m_ml = holdout_rmse(Y, X, train_idx, test_idx, lam_m_ml, beta_m_ml, As2)
        rmse_i = holdout_rmse(Y, X, train_idx, test_idx, lam_i, beta_i, As2)
        rmse_m = holdout_rmse(Y, X, train_idx, test_idx, lam_m, beta_m, As2)
        row = {
            "rep": rep,
            "lam_ind_ml_1": float(lam_i_ml[0]),
            "lam_ind_ml_2": float(lam_i_ml[1]),
            "lam_mar_ml_1": float(lam_m_ml[0]),
            "lam_mar_ml_2": float(lam_m_ml[1]),
            "lam_ind_mc_1": float(lam_i[0]),
            "lam_ind_mc_2": float(lam_i[1]),
            "lam_mar_mc_1": float(lam_m[0]),
            "lam_mar_mc_2": float(lam_m[1]),
            "beta_rel_ml": float(np.linalg.norm(beta_i_ml - beta_m_ml) / max(np.linalg.norm(beta_m_ml), 1e-8)),
            "beta_rel_mc": float(np.linalg.norm(beta_i - beta_m) / max(np.linalg.norm(beta_m), 1e-8)),
            "rmse_ind_ml": rmse_i_ml,
            "rmse_mar_ml": rmse_m_ml,
            "rmse_ind_mc": rmse_i,
            "rmse_mar_mc": rmse_m,
            "seconds_induced_mcmc": t_ind_mc,
            "seconds_marginal_mcmc": t_mar_mc,
            "seconds_induced_ml": t_ind_ml,
            "seconds_marginal_ml": t_mar_ml,
            "seconds_induced_likelihood": t_ind_like,
            "seconds_marginal_likelihood": t_mar_like,
            "ml_success_induced": ok_i,
            "ml_success_marginal": ok_m,
        }
        rows.append(row)
        print(
            f"  rep {rep+1}/{N_MARGINAL}: "
            f"ML lam ind {np.round(lam_i_ml, 3)} mar {np.round(lam_m_ml, 3)}; "
            f"RMSE {rmse_i_ml:.4f} vs {rmse_m_ml:.4f}; "
            f"MCMC {t_ind_mc:.1f}s vs {t_mar_mc:.1f}s "
            f"(rep {time.time()-t_rep:.1f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(RESULTS / "marginal_vs_induced.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "marginal_vs_induced.csv", index=False)

    def rel(a, b):
        return float(np.mean(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))

    summary = {
        "n": 200,
        "n_holdout": N_HOLD,
        "n_reps": N_MARGINAL,
        "true_lambda": [0.40, 0.35],
        "mcmc": {"n_iter": cfg.n_iter, "burn": cfg.burn, "n_chains": 1},
        "mean_lambda_induced_ml": [float(df.lam_ind_ml_1.mean()), float(df.lam_ind_ml_2.mean())],
        "mean_lambda_marginal_ml": [float(df.lam_mar_ml_1.mean()), float(df.lam_mar_ml_2.mean())],
        "mean_lambda_induced_mcmc": [float(df.lam_ind_mc_1.mean()), float(df.lam_ind_mc_2.mean())],
        "mean_lambda_marginal_mcmc": [float(df.lam_mar_mc_1.mean()), float(df.lam_mar_mc_2.mean())],
        "mean_abs_rel_lambda_ml": float(
            np.mean(
                [
                    rel(df.lam_ind_ml_1, df.lam_mar_ml_1),
                    rel(df.lam_ind_ml_2, df.lam_mar_ml_2),
                ]
            )
        ),
        "mean_abs_rel_lambda_mcmc": float(
            np.mean(
                [
                    rel(df.lam_ind_mc_1, df.lam_mar_mc_1),
                    rel(df.lam_ind_mc_2, df.lam_mar_mc_2),
                ]
            )
        ),
        "mean_abs_rel_beta_ml": float(df.beta_rel_ml.mean()),
        "mean_rmse_induced_ml": float(df.rmse_ind_ml.mean()),
        "mean_rmse_marginal_ml": float(df.rmse_mar_ml.mean()),
        "mean_rmse_induced_mcmc": float(df.rmse_ind_mc.mean()),
        "mean_rmse_marginal_mcmc": float(df.rmse_mar_mc.mean()),
        "rmse_rel_diff_ml": float(
            abs(df.rmse_ind_ml.mean() - df.rmse_mar_ml.mean()) / df.rmse_mar_ml.mean()
        ),
        "rmse_rel_diff_mcmc": float(
            abs(df.rmse_ind_mc.mean() - df.rmse_mar_mc.mean()) / df.rmse_mar_mc.mean()
        ),
        "median_mcmc_speedup_marginal_over_induced": float(
            np.median(df.seconds_marginal_mcmc / df.seconds_induced_mcmc)
        ),
        "median_likelihood_speedup": float(
            np.median(df.seconds_marginal_likelihood / df.seconds_induced_likelihood)
        ),
        "mean_seconds_induced_mcmc": float(df.seconds_induced_mcmc.mean()),
        "mean_seconds_marginal_mcmc": float(df.seconds_marginal_mcmc.mean()),
    }
    (RESULTS / "marginal_vs_induced_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def _one_design(n, n_reps, n_hold, seed, do_mcmc, n_iter, burn, label):
    """Concentrated-likelihood comparison, with optional longer Gibbs runs."""
    cfg = GibbsConfig(n_iter=n_iter, burn=burn, n_chains=1, seed=seed, xi=1.0)
    master = np.random.default_rng(seed)
    true_full = np.array([0.40, 0.35, 0.0, 0.0, 0.0])
    rows = []
    for rep in range(n_reps):
        t_rep = time.time()
        net_rng = np.random.default_rng(int(master.integers(1, 2**31 - 1)))
        As, As_raw, names, _ = make_networks(n, net_rng)
        As2 = As[:2]
        X = make_X(n, net_rng)
        Y = simulate_y(X, TRUE_BETA, true_full, As, 1.0, net_rng)
        perm = master.permutation(n)
        test_idx = np.sort(perm[:n_hold])
        train_idx = np.sort(perm[n_hold:])
        Y_t, X_t = Y[train_idx], X[train_idx]
        As_tr = [row_normalize(A[np.ix_(train_idx, train_idx)]) for A in As_raw[:2]]

        u = Y_t - X_t @ TRUE_BETA
        t0 = time.perf_counter()
        _log_posterior_lambda(np.array([0.40, 0.35]), u, As_tr, 1.0, 1.0)
        t_ind_like = time.perf_counter() - t0
        t0 = time.perf_counter()
        log_posterior_marginal(np.array([0.40, 0.35]), u, As2, train_idx, test_idx, 1.0, 1.0)
        t_mar_like = time.perf_counter() - t0

        t0 = time.perf_counter()
        lam_i_ml, beta_i_ml, ok_i = concentrated_ml("induced", Y_t, X_t, As_tr, As2, train_idx, test_idx)
        t_ind_ml = time.perf_counter() - t0
        t0 = time.perf_counter()
        lam_m_ml, beta_m_ml, ok_m = concentrated_ml("marginal", Y_t, X_t, As_tr, As2, train_idx, test_idx)
        t_mar_ml = time.perf_counter() - t0

        row = {
            "design": label,
            "n": n,
            "n_hold": n_hold,
            "rep": rep,
            "lam_ind_1": float(lam_i_ml[0]),
            "lam_ind_2": float(lam_i_ml[1]),
            "lam_mar_1": float(lam_m_ml[0]),
            "lam_mar_2": float(lam_m_ml[1]),
            "beta_rel": float(np.linalg.norm(beta_i_ml - beta_m_ml) / max(np.linalg.norm(beta_m_ml), 1e-8)),
            "slope_rel": float(
                np.linalg.norm(beta_i_ml[1:] - beta_m_ml[1:]) / max(np.linalg.norm(beta_m_ml[1:]), 1e-8)
            ),
            "rmse_ind": holdout_rmse(Y, X, train_idx, test_idx, lam_i_ml, beta_i_ml, As2),
            "rmse_mar": holdout_rmse(Y, X, train_idx, test_idx, lam_m_ml, beta_m_ml, As2),
            "seconds_induced_ml": t_ind_ml,
            "seconds_marginal_ml": t_mar_ml,
            "seconds_induced_likelihood": t_ind_like,
            "seconds_marginal_likelihood": t_mar_like,
            "ml_success_induced": ok_i,
            "ml_success_marginal": ok_m,
        }
        if do_mcmc:
            t0 = time.perf_counter()
            lam_i, beta_i = _gibbs_induced(
                Y_t, X_t, As_tr, cfg, np.random.default_rng(10_000 + rep), lam_i_ml
            )
            t_ind_mc = time.perf_counter() - t0
            t0 = time.perf_counter()
            lam_m, beta_m = _gibbs_marginal(
                Y_t, X_t, As2, train_idx, test_idx, cfg, np.random.default_rng(20_000 + rep), lam_m_ml
            )
            t_mar_mc = time.perf_counter() - t0
            row.update(
                {
                    "lam_ind_mc_1": float(lam_i[0]),
                    "lam_ind_mc_2": float(lam_i[1]),
                    "lam_mar_mc_1": float(lam_m[0]),
                    "lam_mar_mc_2": float(lam_m[1]),
                    "rmse_ind_mc": holdout_rmse(Y, X, train_idx, test_idx, lam_i, beta_i, As2),
                    "rmse_mar_mc": holdout_rmse(Y, X, train_idx, test_idx, lam_m, beta_m, As2),
                    "seconds_induced_mcmc": t_ind_mc,
                    "seconds_marginal_mcmc": t_mar_mc,
                }
            )
        rows.append(row)
        print(
            f"  {label} rep {rep + 1}/{n_reps}: "
            f"ML ind {np.round(lam_i_ml, 3)} mar {np.round(lam_m_ml, 3)} "
            f"RMSE {row['rmse_ind']:.4f} vs {row['rmse_mar']:.4f} "
            f"({time.time() - t_rep:.1f}s)",
            flush=True,
        )
        pd.DataFrame(rows).to_csv(RESULTS / f"marginal_vs_induced_{label}.csv", index=False)
    return rows


def _summarize_design(rows):
    df = pd.DataFrame(rows)

    def rel(a, b):
        return np.abs(a - b) / np.maximum(np.abs(b), 1e-8)

    out = {
        "n": int(df["n"].iloc[0]),
        "n_hold": int(df["n_hold"].iloc[0]),
        "n_reps": int(len(df)),
        "mean_lambda_induced": [float(df.lam_ind_1.mean()), float(df.lam_ind_2.mean())],
        "mean_lambda_marginal": [float(df.lam_mar_1.mean()), float(df.lam_mar_2.mean())],
        "bias_induced": [float(df.lam_ind_1.mean() - 0.40), float(df.lam_ind_2.mean() - 0.35)],
        "bias_marginal": [float(df.lam_mar_1.mean() - 0.40), float(df.lam_mar_2.mean() - 0.35)],
        "median_rel_lambda": [
            float(np.median(rel(df.lam_ind_1, df.lam_mar_1))),
            float(np.median(rel(df.lam_ind_2, df.lam_mar_2))),
        ],
        "mean_rmse_induced": float(df.rmse_ind.mean()),
        "mean_rmse_marginal": float(df.rmse_mar.mean()),
        "rmse_rel_diff_of_means": float(abs(df.rmse_ind.mean() - df.rmse_mar.mean()) / df.rmse_mar.mean()),
        "median_rmse_rel": float(np.median(np.abs(df.rmse_ind - df.rmse_mar) / df.rmse_mar)),
        "median_beta_rel": float(df.beta_rel.median()),
        "median_slope_rel": float(df.slope_rel.median()),
        "median_likelihood_speedup": float(np.median(df.seconds_marginal_likelihood / df.seconds_induced_likelihood)),
        "ml_success_induced": int(df.ml_success_induced.sum()),
        "ml_success_marginal": int(df.ml_success_marginal.sum()),
    }
    if "seconds_induced_mcmc" in df.columns:
        out["mean_lambda_induced_mcmc"] = [float(df.lam_ind_mc_1.mean()), float(df.lam_ind_mc_2.mean())]
        out["mean_lambda_marginal_mcmc"] = [float(df.lam_mar_mc_1.mean()), float(df.lam_mar_mc_2.mean())]
        out["median_rel_lambda_mcmc"] = [
            float(np.median(rel(df.lam_ind_mc_1, df.lam_mar_mc_1))),
            float(np.median(rel(df.lam_ind_mc_2, df.lam_mar_mc_2))),
        ]
        out["mean_rmse_induced_mcmc"] = float(df.rmse_ind_mc.mean())
        out["mean_rmse_marginal_mcmc"] = float(df.rmse_mar_mc.mean())
        out["median_mcmc_speedup"] = float(np.median(df.seconds_marginal_mcmc / df.seconds_induced_mcmc))
        out["mean_seconds_induced_mcmc"] = float(df.seconds_induced_mcmc.mean())
        out["mean_seconds_marginal_mcmc"] = float(df.seconds_marginal_mcmc.mean())
    return out


def large_benchmark():
    """50 trials at n=200 with a longer Gibbs run, and 20 ML trials at n=500."""
    t0 = time.time()
    print("=== Large comparison: n=200, 50 reps, MCMC 2000/1000 ===", flush=True)
    rows_200 = _one_design(200, 50, 40, seed=2026, do_mcmc=True, n_iter=2000, burn=1000, label="n200")
    print("=== Large comparison: n=500, 20 reps, concentrated likelihood ===", flush=True)
    rows_500 = _one_design(500, 20, 100, seed=2027, do_mcmc=False, n_iter=1, burn=0, label="n500")
    pd.concat([pd.DataFrame(rows_200), pd.DataFrame(rows_500)], ignore_index=True).to_csv(
        RESULTS / "marginal_vs_induced_large.csv", index=False
    )
    summary = {
        "n200": _summarize_design(rows_200),
        "n500": _summarize_design(rows_500),
        "minutes": (time.time() - t0) / 60.0,
    }
    (RESULTS / "marginal_vs_induced_large_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Finished in {summary['minutes']:.1f} min", flush=True)


def main():
    t0 = time.time()
    print("=== Induced subgraph vs coherent marginal likelihood ===", flush=True)
    marginal_benchmark()
    print(f"Finished in {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "large":
        large_benchmark()
    else:
        main()
