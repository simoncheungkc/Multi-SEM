"""Residual Moran I screening, Multi-SEM Gibbs sampler, and kriging predictions.

Unified specification (Multi-SEM):
    Y = X beta + u
    u = sum_s lambda_s A_s u + eps,  eps ~ N(0, sigma2 I)
    V = I - sum_s lambda_s A_s
Stationarity / invertibility (sufficient): sum_s |lambda_s| < 1 for row-stochastic A_s,
which implies rho(sum lambda_s A_s) <= sum |lambda_s| rho(A_s) <= sum |lambda_s| < 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.linalg import slogdet
from scipy.optimize import minimize
from scipy.special import ndtr
from scipy.stats import chi2


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def add_intercept(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    ones = np.ones((X.shape[0], 1))
    if np.allclose(X[:, 0], 1.0):
        return X
    return np.hstack([ones, X])


def standardize_columns(X: np.ndarray, mean=None, std=None, skip_intercept: bool = True):
    X = np.asarray(X, dtype=float).copy()
    start = 1 if skip_intercept and np.allclose(X[:, 0], 1.0) else 0
    if mean is None:
        mean = X[:, start:].mean(axis=0)
    if std is None:
        std = X[:, start:].std(axis=0, ddof=0)
        std = np.where(std < 1e-12, 1.0, std)
    X[:, start:] = (X[:, start:] - mean) / std
    return X, np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


def ols(Y: np.ndarray, X: np.ndarray):
    Y = np.asarray(Y, dtype=float).reshape(-1)
    X = np.asarray(X, dtype=float)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    n, q = X.shape
    sse = float(resid @ resid)
    sigma2 = sse / max(n - q, 1)
    return beta, resid, sigma2


def assemble_V(lambdas: np.ndarray, As: list[np.ndarray]) -> np.ndarray:
    n = As[0].shape[0]
    V = np.eye(n)
    for lam, A in zip(lambdas, As):
        if lam != 0.0:
            V -= lam * A
    return V


def logdet_signed(V: np.ndarray) -> float:
    """Exact log|V| via dense LU (numpy.linalg.slogdet).

    Appropriate for n <= 500 as in this study. For n > 2000, replace this
    helper with a Chebyshev / Barry-Pace or Hutchinson trace approximation
    without changing the Gibbs update order.
    """
    sign, ld = slogdet(V)
    if sign <= 0 or not np.isfinite(ld):
        return -np.inf
    return float(ld)


def feasible_lambda(lambdas: np.ndarray, tol: float = 1e-8) -> bool:
    """Sufficient invertibility: sum |lambda_s| < 1 (and each |lambda_s| < 1)."""
    lambdas = np.asarray(lambdas, dtype=float)
    return (
        bool(np.sum(np.abs(lambdas)) < 1.0 - tol)
        and bool(np.all(np.abs(lambdas) < 1.0 - tol))
        and bool(np.all(np.isfinite(lambdas)))
    )


def spectral_radius(matrix: np.ndarray) -> float:
    """Largest absolute eigenvalue. The infinity norm is returned when it is already below 1."""
    matrix = np.asarray(matrix, dtype=float)
    bound = float(np.max(np.sum(np.abs(matrix), axis=1))) if matrix.size else 0.0
    if bound < 1.0:
        return bound
    values = np.linalg.eigvals(matrix)
    return float(np.max(np.abs(values)))


def spectral_admissible(lambdas, As, tol: float = 1e-8) -> bool:
    """Exact admissibility: rho(sum lambda_s A_s) < 1.

    Row-substochastic layers satisfy rho(B) <= ||lambda||_1, so the ell_1 ball
    is accepted without an eigendecomposition.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    if not np.all(np.isfinite(lambdas)):
        return False
    if len(As) != len(lambdas):
        raise ValueError("lambda and network counts differ")
    if float(np.sum(np.abs(lambdas))) < 1.0 - tol:
        return True
    combined = np.zeros_like(np.asarray(As[0], dtype=float))
    for weight, adjacency in zip(lambdas, As):
        combined += float(weight) * np.asarray(adjacency, dtype=float)
    return spectral_radius(combined) < 1.0 - tol


def proposal_admissible(lambdas, As=None, constraint: str = "l1", tol: float = 1e-8) -> bool:
    if constraint == "l1":
        return feasible_lambda(lambdas, tol)
    if constraint == "spectral":
        if As is None:
            raise ValueError("spectral admissibility requires the network list")
        return spectral_admissible(lambdas, As, tol)
    raise ValueError(f"unknown constraint: {constraint}")


def offdiag_correlation(A: np.ndarray, B: np.ndarray) -> float:
    """Pearson correlation of two network matrices on off-diagonal entries only."""
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if A.shape != B.shape or A.shape[0] != A.shape[1]:
        raise ValueError("off-diagonal correlation requires square matrices of equal shape")
    mask = ~np.eye(A.shape[0], dtype=bool)
    a = A[mask]
    b = B[mask]
    a = a - a.mean()
    b = b - b.mean()
    da = float(np.sqrt(a @ a))
    db = float(np.sqrt(b @ b))
    if da < 1e-12 and db < 1e-12:
        return 1.0
    if da < 1e-12 or db < 1e-12:
        return 0.0
    return float((a @ b) / (da * db))


def drop_collinear_networks(As, names, corr_max: float = 0.99):
    """Drop structural duplicates using off-diagonal Pearson correlation.

    The correlation does not use the response, so this screen is ex ante.
    Call it before residual Moran tests when the family size must be fixed.
    """
    keep_idx = []
    mats = [np.asarray(A, dtype=float) for A in As]
    for i, A in enumerate(mats):
        is_dup = False
        for j in keep_idx:
            c = offdiag_correlation(A, mats[j])
            if np.isfinite(c) and abs(c) >= corr_max:
                is_dup = True
                break
        if not is_dup:
            keep_idx.append(i)
    return keep_idx, [names[i] for i in keep_idx], [As[i] for i in keep_idx]


# ---------------------------------------------------------------------------
# Residual Moran I
# ---------------------------------------------------------------------------

def moran_i_test(resid: np.ndarray, A: np.ndarray, X: np.ndarray) -> dict:
    """Two-sided residual Moran I on OLS residuals, following the Multi-SEM note.

    I = (e' A e) / (e' e)
    E(I) = tr(A M) / (n - q)
    Var(I) = 2 tr(M A M A) / [(n-q)(n-q+2)] - 2 [tr(M A)]^2 / [(n-q)^2 (n-q+2)]
    z = (I - E) / sqrt(Var),  p = 2 (1 - Phi(|z|))
    """
    e = np.asarray(resid, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    X = np.asarray(X, dtype=float)
    n, q = X.shape
    denom = float(e @ e)
    if denom <= 0:
        return {"I": np.nan, "EI": np.nan, "VI": np.nan, "z": np.nan, "pvalue": 1.0}
    I = float(e @ (A @ e) / denom)

    # Residual maker M = I - P_X
    try:
        xtx = X.T @ X
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        xtx_inv = np.linalg.pinv(xtx)
    # tr(A M) = tr(A) - tr(A P) = tr(A) - tr(X^+ A X) wait: P = X (X'X)^{-1} X'
    # tr(A P) = tr((X'X)^{-1} X' A X)
    trA = float(np.trace(A))
    trAP = float(np.trace(xtx_inv @ (X.T @ (A @ X))))
    trAM = trA - trAP
    df = n - q
    EI = trAM / df

    # tr(M A M A) = tr(A M A M). Compute MA = A - A P, then tr(MA MA) is wrong
    # because M is symmetric: M A M A, tr = ||M^{1/2} A M^{1/2}||_F^2 analog.
    # MA = A @ M = A - (A X) (X'X)^{-1} X'
    AX = A @ X
    AM = A - AX @ (xtx_inv @ X.T)
    # M AM = AM - P AM
    MAM = AM - X @ (xtx_inv @ (X.T @ AM))
    trMAMA = float(np.sum(AM * MAM.T))  # tr(AM @ MAM) = tr(A M A M)

    VI = 2.0 * trMAMA / (df * (df + 2)) - 2.0 * (trAM ** 2) / (df ** 2 * (df + 2))
    VI = max(VI, 1e-16)
    z = (I - EI) / np.sqrt(VI)
    pvalue = float(2.0 * (1.0 - ndtr(abs(z))))
    pvalue = min(max(pvalue, 0.0), 1.0)
    return {"I": I, "EI": EI, "VI": VI, "z": float(z), "pvalue": pvalue}


def joint_multi_weight_lm(e, A_list):
    """Omnibus and robust partial LM-error tests of H0: lambda = 0.

    Score g_s = (e' A_s e) / sigma_hat^2 with sigma_hat^2 = e'e / n, and
    Fisher information I_rs = tr((A_r + A_r') A_s). The joint statistic is
    g' I^{-1} g on chi^2_{k'}. Each partial statistic is the squared
    information-orthogonalized score divided by the corresponding diagonal
    of I^{-1}, referred to chi^2_1 (Anselin, 1988; Anselin et al., 1996).
    """
    e = np.asarray(e, dtype=float).reshape(-1)
    A_list = [np.asarray(A, dtype=float) for A in A_list]
    n = len(e)
    k = len(A_list)
    sigma2_hat = float(np.dot(e, e)) / n
    g = np.array([float(e @ (A @ e)) / sigma2_hat for A in A_list])
    sym_A = [A + A.T for A in A_list]
    I_mat = np.zeros((k, k))
    for r in range(k):
        for s in range(r, k):
            val = float(np.sum(sym_A[r] * A_list[s].T))
            I_mat[r, s] = I_mat[s, r] = val
    I_inv = np.linalg.pinv(I_mat)
    v = I_inv @ g
    lm_joint = float(g @ v)
    p_joint = float(chi2.sf(lm_joint, df=k))
    lm_partial = (v ** 2) / np.maximum(np.diag(I_inv), 1e-12)
    p_partial = np.array([float(chi2.sf(stat, df=1)) for stat in lm_partial])
    return {
        "lm_joint": lm_joint,
        "df": k,
        "p_joint": p_joint,
        "lm_partial": lm_partial,
        "p_partial": p_partial,
        "score": g,
        "information": I_mat,
    }


def benjamini_hochberg(pvalues: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    thresh = alpha * (np.arange(1, m + 1) / m)
    passed = ranked <= thresh
    reject = np.zeros(m, dtype=bool)
    if passed.any():
        kmax = np.max(np.where(passed)[0])
        reject[order[: kmax + 1]] = True
    return reject


def screen_networks(
    Y: np.ndarray,
    X: np.ndarray,
    As: list[np.ndarray],
    names: list[str] | None = None,
    alpha: float = 0.05,
    fdr: bool = True,
) -> dict:
    Y = np.asarray(Y, dtype=float).reshape(-1)
    X = add_intercept(X)
    _, resid, _ = ols(Y, X)
    names = names or [f"A{s+1}" for s in range(len(As))]
    tests = []
    for name, A in zip(names, As):
        out = moran_i_test(resid, A, X)
        out["name"] = name
        tests.append(out)
    pvals = np.array([t["pvalue"] for t in tests])
    if fdr:
        selected = benjamini_hochberg(pvals, alpha=alpha)
    else:
        selected = pvals < alpha
    return {
        "tests": tests,
        "pvalues": pvals,
        "selected": selected,
        "selected_names": [n for n, s in zip(names, selected) if s],
        "resid": resid,
    }


# ---------------------------------------------------------------------------
# Maximum-likelihood (concentrated) fit — used for init and fast baselines
# ---------------------------------------------------------------------------

def concentrated_nll(lambdas, Y, X, As, xi=1.0):
    lambdas = np.asarray(lambdas, dtype=float)
    if not feasible_lambda(lambdas):
        return 1e12
    V = assemble_V(lambdas, As)
    ld = logdet_signed(V)
    if not np.isfinite(ld):
        return 1e12
    Yt = V @ Y
    Xt = V @ X
    try:
        beta, *_ = np.linalg.lstsq(Xt, Yt, rcond=None)
    except np.linalg.LinAlgError:
        return 1e12
    r = Yt - Xt @ beta
    sse = float(r @ r)
    n = Y.shape[0]
    if sse <= 0:
        return 1e12
    # Gaussian nll up to constants, plus weakly informative N(0, 1/xi) prior
    nll = -ld + 0.5 * n * np.log(sse / n) + 0.5 * xi * float(lambdas @ lambdas)
    return float(nll)


def fit_ml_multi_sem(Y, X, As, xi=1.0):
    Y = np.asarray(Y, dtype=float).reshape(-1)
    X = add_intercept(X)
    k = len(As)
    x0 = np.zeros(k)

    def obj(z):
        return concentrated_nll(z, Y, X, As, xi=xi)

    bounds = [(-0.95, 0.95)] * k
    cons = {"type": "ineq", "fun": lambda z: 0.95 - np.sum(np.abs(z))}
    res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    lam = np.asarray(res.x, dtype=float)
    if not feasible_lambda(lam) or not res.success:
        lam = np.zeros(k)
    V = assemble_V(lam, As)
    Yt, Xt = V @ Y, V @ X
    beta, *_ = np.linalg.lstsq(Xt, Yt, rcond=None)
    r = Yt - Xt @ beta
    n = Y.shape[0]
    sigma2 = float(r @ r) / n
    return {"lambda": lam, "beta": beta, "sigma2": sigma2, "success": bool(res.success)}


# ---------------------------------------------------------------------------
# Gibbs sampler
# ---------------------------------------------------------------------------

@dataclass
class GibbsConfig:
    n_iter: int = 8000
    burn: int = 4000
    n_chains: int = 4
    mu0_scale: float = 0.0
    Sigma0_diag: float = 100.0
    alpha0: float = 0.001
    gamma0: float = 0.001
    xi: float = 1.0
    target_accept: float = 0.30
    seed: int = 2026
    thin: int = 1
    adapt_until: int | None = None
    adapt_covariance: bool = False
    # "gaussian": exp(-xi ||lambda||^2 / 2) on the open l1 ball.
    # "beta_l1": L=||lambda||_1 ~ Beta(1, 2), |lambda|/L ~ Dirichlet(1..1),
    # independent signs. The Lebesgue density on the ball is (1-L) / L^{k-1}.
    lambda_prior: str = "gaussian"
    # "l1": reject ||lambda||_1 >= 1. "spectral": reject rho(sum lambda_s A_s) >= 1.
    constraint: str = "l1"


@dataclass
class GibbsResult:
    samples: dict = field(default_factory=dict)
    acceptance: np.ndarray = field(default_factory=lambda: np.array([]))
    config: GibbsConfig | None = None
    names: list[str] = field(default_factory=list)
    proposal_correlation: np.ndarray | None = None
    tau: np.ndarray = field(default_factory=lambda: np.array([]))


def _draw_invgamma(rng, shape, rate):
    return 1.0 / rng.gamma(shape, 1.0 / rate)


def _lambda_log_prior(lambdas, xi, lambda_prior):
    if lambda_prior == "gaussian":
        return -0.5 * xi * float(lambdas @ lambdas)
    if lambda_prior != "beta_l1":
        raise ValueError(f"unknown lambda prior: {lambda_prior}")
    # Proper pushforward of L ~ Beta(1, 2) and a uniform signed composition.
    # p(lambda) ∝ (1 - L) / L^{k-1} on the open l1 ball, which is zero at L = 1.
    total = float(np.sum(np.abs(lambdas)))
    k = int(np.asarray(lambdas).size)
    if total <= 1e-8:
        return -np.inf
    return float(np.log1p(-total) - (k - 1) * np.log(total))


def _log_posterior_lambda(lambdas, resid, As, sigma2, xi, lambda_prior="gaussian", constraint="l1"):
    n = As[0].shape[0]
    if not proposal_admissible(lambdas, As, constraint):
        return -np.inf, np.eye(n), -np.inf
    V = assemble_V(lambdas, As)
    ld = logdet_signed(V)
    if not np.isfinite(ld):
        return -np.inf, V, -np.inf
    vr = V @ resid
    sse = float(vr @ vr)
    lp = ld - 0.5 * sse / sigma2 + _lambda_log_prior(lambdas, xi, lambda_prior)
    return lp, V, ld


def _gibbs_chain(Y, X, As, cfg: GibbsConfig, rng: np.random.Generator, start_lambda=None):
    n, q = X.shape
    k = len(As)
    Sigma0_inv = np.eye(q) / cfg.Sigma0_diag
    mu0 = np.full(q, cfg.mu0_scale)

    beta_ols, _, s2_ols = ols(Y, X)
    if start_lambda is None:
        lam = rng.normal(0.0, 0.15, size=k)
        if not proposal_admissible(lam, As, cfg.constraint):
            lam *= 0.2
    else:
        lam = np.asarray(start_lambda, dtype=float).copy()
    beta = beta_ols + rng.normal(0.0, 0.05, size=q)
    sigma2 = max(float(s2_ols), 1e-8)
    tau = 0.08 if k else 0.08
    n_adapt = cfg.adapt_until if cfg.adapt_until is not None else cfg.burn
    adapt_cov = bool(cfg.adapt_covariance) and k > 0
    cov_count = 0
    cov_mean = np.zeros(k)
    cov_m2 = np.zeros((k, k))
    frozen_factor = None
    proposal_corr = np.eye(k)
    scale = (2.38 ** 2) / k if k else 1.0
    min_history = max(2 * k, 50)
    window_accept = 0
    window_count = 0

    def _covariance_update(draw: np.ndarray) -> None:
        nonlocal cov_count, cov_mean, cov_m2
        cov_count += 1
        delta = draw - cov_mean
        cov_mean = cov_mean + delta / cov_count
        cov_m2 += np.outer(delta, draw - cov_mean)

    def _proposal_factor(empirical: np.ndarray) -> np.ndarray:
        cov = scale * empirical + 1e-5 * np.eye(k)
        cov = 0.5 * (cov + cov.T)
        try:
            return np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            return np.linalg.cholesky(cov + 1e-4 * np.eye(k))

    def _empirical_covariance() -> np.ndarray | None:
        if cov_count <= min_history:
            return None
        return cov_m2 / (cov_count - 1)

    n_keep = (cfg.n_iter - cfg.burn) // cfg.thin
    lam_s = np.zeros((n_keep, k))
    beta_s = np.zeros((n_keep, q))
    sig_s = np.zeros(n_keep)
    n_accept = 0
    n_prop = 0
    keep_i = 0
    logp, V, ld = _log_posterior_lambda(
        lam, Y - X @ beta, As, sigma2, cfg.xi, cfg.lambda_prior, cfg.constraint
    )
    if not np.isfinite(logp):
        # The Beta(1, 2) density is unbounded at L = 0 but the point mass at the
        # origin is not a valid start: the log density there is -inf.
        lam = np.full(k, 0.4 / max(k, 1)) if cfg.lambda_prior == "beta_l1" else np.zeros(k)
        logp, V, ld = _log_posterior_lambda(
            lam, Y - X @ beta, As, sigma2, cfg.xi, cfg.lambda_prior, cfg.constraint
        )
    if adapt_cov:
        _covariance_update(lam)

    for t in range(cfg.n_iter):
        resid = Y - X @ beta
        # --- lambda block RW-MH ---
        # Oriented proposals use the burn-in history only. The first draws stay
        # isotropic so the covariance is not estimated from a chain that has
        # not yet moved. After burn-in the factor is frozen.
        if adapt_cov and t >= n_adapt and frozen_factor is not None:
            prop = lam + tau * (frozen_factor @ rng.normal(size=k))
        elif adapt_cov:
            empirical = _empirical_covariance()
            if empirical is None:
                prop = lam + tau * rng.normal(size=k)
            else:
                prop = lam + tau * (_proposal_factor(empirical) @ rng.normal(size=k))
        else:
            prop = lam + tau * rng.normal(size=k)
        n_prop += 1
        logp_prop, V_prop, ld_prop = _log_posterior_lambda(
            prop, resid, As, sigma2, cfg.xi, cfg.lambda_prior, cfg.constraint
        )
        accepted = False
        if np.isfinite(logp_prop) and np.log(rng.uniform()) < (logp_prop - logp):
            lam = prop
            V = V_prop
            logp = logp_prop
            ld = ld_prop
            n_accept += 1
            accepted = True
        if adapt_cov and t < n_adapt:
            _covariance_update(lam)
            window_count += 1
            window_accept += int(accepted)
            if window_count == 50:
                acc = window_accept / window_count
                if acc < cfg.target_accept - 0.08:
                    tau *= 0.85
                elif acc > cfg.target_accept + 0.08:
                    tau *= 1.15
                tau = float(np.clip(tau, 1e-4, 1.0))
                window_accept = 0
                window_count = 0
            if t + 1 == n_adapt:
                empirical = _empirical_covariance()
                if empirical is None:
                    frozen_factor = np.eye(k)
                    proposal_corr = np.eye(k)
                else:
                    frozen_factor = _proposal_factor(empirical)
                    sd = np.sqrt(np.clip(np.diag(empirical), 1e-15, None))
                    proposal_corr = empirical / np.outer(sd, sd)
                    proposal_corr = np.clip(0.5 * (proposal_corr + proposal_corr.T), -1.0, 1.0)
        elif t < n_adapt and n_prop > 0 and (t + 1) % 50 == 0:
            acc = n_accept / n_prop
            if acc < cfg.target_accept - 0.08:
                tau *= 0.85
            elif acc > cfg.target_accept + 0.08:
                tau *= 1.15
            tau = float(np.clip(tau, 1e-4, 0.6))

        # --- beta | rest ---
        VX = V @ X
        VY = V @ Y
        prec = Sigma0_inv + (VX.T @ VX) / sigma2
        prec = 0.5 * (prec + prec.T)
        try:
            cov = np.linalg.inv(prec)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(prec)
        mean = cov @ (Sigma0_inv @ mu0 + (VX.T @ VY) / sigma2)
        cov = 0.5 * (cov + cov.T)
        jitter = 1e-10 * np.eye(q)
        try:
            beta = rng.multivariate_normal(mean, cov + jitter, method="cholesky")
        except np.linalg.LinAlgError:
            beta = rng.multivariate_normal(mean, cov + 1e-6 * np.eye(q))

        # --- sigma2 | rest ---
        resid = Y - X @ beta
        vr = V @ resid
        sse = float(vr @ vr)
        shape = cfg.alpha0 + n / 2.0
        rate = cfg.gamma0 + 0.5 * sse
        sigma2 = float(_draw_invgamma(rng, shape, rate))
        sigma2 = max(sigma2, 1e-12)
        # V and log|V| depend only on lambda, which this block does not change.
        logp = ld - 0.5 * sse / sigma2 + _lambda_log_prior(lam, cfg.xi, cfg.lambda_prior)

        if t >= cfg.burn and ((t - cfg.burn) % cfg.thin == 0):
            if keep_i < n_keep:
                lam_s[keep_i] = lam
                beta_s[keep_i] = beta
                sig_s[keep_i] = sigma2
                keep_i += 1

    return {
        "lambda": lam_s[:keep_i],
        "beta": beta_s[:keep_i],
        "sigma2": sig_s[:keep_i],
        "accept_rate": n_accept / max(n_prop, 1),
        "tau": tau,
        "proposal_correlation": proposal_corr,
    }


def gibbs_multi_sem(Y, X, As, names=None, config: GibbsConfig | None = None, starts=None) -> GibbsResult:
    cfg = config or GibbsConfig()
    Y = np.asarray(Y, dtype=float).reshape(-1)
    X = add_intercept(X)
    names = names or [f"lambda_{s+1}" for s in range(len(As))]
    rng0 = np.random.default_rng(cfg.seed)
    seeds = rng0.integers(1, 2**31 - 1, size=cfg.n_chains)
    if starts is None:
        ml = fit_ml_multi_sem(Y, X, As)
        if cfg.lambda_prior == "beta_l1":
            origin = np.full(len(As), 0.4 / max(len(As), 1))
        else:
            origin = np.zeros(len(As))
        starts = [origin, ml["lambda"]]
        while len(starts) < cfg.n_chains:
            z = rng0.normal(0, 0.2, size=len(As))
            if proposal_admissible(z, As, cfg.constraint):
                starts.append(z)
            else:
                starts.append(0.3 * z)
        starts = starts[: cfg.n_chains]

    chains = []
    for c in range(cfg.n_chains):
        rng = np.random.default_rng(int(seeds[c]))
        chains.append(_gibbs_chain(Y, X, As, cfg, rng, start_lambda=starts[c]))

    result = GibbsResult(
        samples={
            "lambda": np.stack([ch["lambda"] for ch in chains], axis=0),
            "beta": np.stack([ch["beta"] for ch in chains], axis=0),
            "sigma2": np.stack([ch["sigma2"] for ch in chains], axis=0),
        },
        acceptance=np.array([ch["accept_rate"] for ch in chains]),
        config=cfg,
        names=list(names),
        proposal_correlation=np.mean(
            [ch["proposal_correlation"] for ch in chains], axis=0
        ),
        tau=np.array([ch["tau"] for ch in chains], dtype=float),
    )
    return result


def gelman_rubin(chains: np.ndarray) -> float:
    """Split-R-hat for array shape (n_chains, n_samples)."""
    x = np.asarray(chains, dtype=float)
    if x.ndim != 2:
        raise ValueError("chains must be 2-d")
    m, n = x.shape
    if n < 4:
        return np.nan
    # split each chain
    half = n // 2
    x = x[:, : 2 * half].reshape(m * 2, half)
    m2, n2 = x.shape
    chain_means = x.mean(axis=1)
    W = np.mean(x.var(axis=1, ddof=1))
    if W <= 0:
        return 1.0
    B = n2 * np.var(chain_means, ddof=1)
    var_hat = ((n2 - 1) / n2) * W + B / n2
    return float(np.sqrt(var_hat / W))


def effective_sample_size(x: np.ndarray, max_lag: int | None = None) -> float:
    x = np.asarray(x, dtype=float).reshape(-1)
    n = len(x)
    if n < 8:
        return float(n)
    x = x - x.mean()
    var = float(np.dot(x, x) / n)
    if var <= 0:
        return float(n)
    max_lag = max_lag or min(n // 3, 400)
    acf_sum = 0.0
    for lag in range(1, max_lag + 1):
        c = float(np.dot(x[:-lag], x[lag:]) / n / var)
        if lag > 5 and c < 0.05:
            break
        acf_sum += c
    ess = n / (1.0 + 2.0 * acf_sum)
    return float(np.clip(ess, 1.0, n))


def posterior_summary(result: GibbsResult, beta_names=None) -> dict:
    lam = result.samples["lambda"]  # C x T x k
    beta = result.samples["beta"]
    sig = result.samples["sigma2"]
    C, T, k = lam.shape
    q = beta.shape[2]
    beta_names = beta_names or [f"beta_{j}" for j in range(q)]
    rows = []
    flat_lam = lam.reshape(C * T, k)
    flat_beta = beta.reshape(C * T, q)
    flat_sig = sig.reshape(C * T)

    def _row(name, draws_ct, draws_flat):
        return {
            "parameter": name,
            "mean": float(draws_flat.mean()),
            "median": float(np.median(draws_flat)),
            "sd": float(draws_flat.std(ddof=1)),
            "q025": float(np.quantile(draws_flat, 0.025)),
            "q975": float(np.quantile(draws_flat, 0.975)),
            "Rhat": float(gelman_rubin(draws_ct)),
            "ESS": float(np.sum([effective_sample_size(draws_ct[c]) for c in range(draws_ct.shape[0])])),
        }

    for s in range(k):
        rows.append(_row(result.names[s], lam[:, :, s], flat_lam[:, s]))
    for j in range(q):
        rows.append(_row(beta_names[j], beta[:, :, j], flat_beta[:, j]))
    rows.append(_row("sigma2", sig, flat_sig))
    means = {
        "lambda": flat_lam.mean(axis=0),
        "beta": flat_beta.mean(axis=0),
        "sigma2": float(flat_sig.mean()),
        "lambda_median": np.median(flat_lam, axis=0),
        "beta_median": np.median(flat_beta, axis=0),
    }
    return {"table": rows, "means": means, "acceptance": result.acceptance.tolist()}


# ---------------------------------------------------------------------------
# Kriging-style out-of-sample prediction
# ---------------------------------------------------------------------------

def predict_kriging(Y_train, X_train, X_test, train_idx, test_idx, lambdas, beta, As_full):
    """Point prediction using the partitioned error covariance Omega = (V'V)^{-1}.

    Ŷ_* = X_* β + Ω_{*t} Ω_{tt}^{-1} (Y_t - X_t β)
    """
    lambdas = np.asarray(lambdas, dtype=float)
    beta = np.asarray(beta, dtype=float).reshape(-1)
    V = assemble_V(lambdas, As_full)
    # Omega proportional to (V' V)^{-1}
    VtV = V.T @ V
    try:
        Omega = np.linalg.inv(VtV)
    except np.linalg.LinAlgError:
        Omega = np.linalg.pinv(VtV)
    Omega_tt = Omega[np.ix_(train_idx, train_idx)]
    Omega_st = Omega[np.ix_(test_idx, train_idx)]
    resid_t = np.asarray(Y_train, dtype=float).reshape(-1) - np.asarray(X_train, dtype=float) @ beta
    try:
        w = np.linalg.solve(Omega_tt + 1e-8 * np.eye(len(train_idx)), resid_t)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(Omega_tt) @ resid_t
    mean = np.asarray(X_test, dtype=float) @ beta + Omega_st @ w
    return mean


def predict_mean_only(X_test, beta):
    return np.asarray(X_test, dtype=float) @ np.asarray(beta, dtype=float).reshape(-1)


def metrics_rmse_mae(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    err = y_pred - y_true
    return {
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAE": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 2 else np.nan,
        "n": int(len(y_true)),
    }
