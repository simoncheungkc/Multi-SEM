"""Covariance-adapted reruns and the four new reviewer experiments.

Writes results/unified/. Does not overwrite the archived isotropic CSVs.
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mnreg.model import (
    GibbsConfig,
    assemble_V,
    effective_sample_size,
    gelman_rubin,
    gibbs_multi_sem,
    ols,
    spectral_radius,
)
from mnreg.networks import (
    boosted_resolvent,
    erdos_renyi,
    offdiagonal_correlation,
    powerlaw_configuration,
    row_normalize,
)
from run_csda_revision import log_posterior_marginal, schur_training_factor
from run_simulation import TRUE_BETA, make_X, simulate_y

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "unified"
FIG = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

WORKERS = min(16, os.cpu_count() or 4)
BOUNDARY_TRUE = np.array([0.75, -0.55])
COLLINEAR_TRUE = np.array([0.20, 0.55, 0.0, 0.0, 0.0])


def _dump(name, payload):
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def _chain_config(seed, constraint="l1", adapt=True, n_iter=6000, burn=3000, n_chains=4):
    return GibbsConfig(
        n_iter=n_iter,
        burn=burn,
        n_chains=n_chains,
        seed=int(seed),
        target_accept=0.30,
        adapt_covariance=bool(adapt),
        constraint=constraint,
    )


def _posterior_record(result, truth):
    lam = result.samples["lambda"]
    flat = lam.reshape(-1, lam.shape[-1])
    mean = flat.mean(axis=0)
    lower = np.quantile(flat, 0.025, axis=0)
    upper = np.quantile(flat, 0.975, axis=0)
    total = np.abs(flat).sum(axis=1)
    rhats = [float(gelman_rubin(lam[:, :, s])) for s in range(lam.shape[-1])]
    ess = [
        float(sum(effective_sample_size(lam[c, :, s]) for c in range(lam.shape[0])))
        for s in range(lam.shape[-1])
    ]
    centered = flat - flat.mean(axis=0)
    if flat.shape[0] > 2 and flat.shape[1] >= 2:
        posterior_corr = float(np.corrcoef(flat[:, 0], flat[:, 1])[0, 1])
    else:
        posterior_corr = float("nan")
    proposal = result.proposal_correlation
    proposal_corr = float(proposal[0, 1]) if proposal is not None and proposal.shape[0] >= 2 else float("nan")
    return {
        "mean": mean.astype(float).tolist(),
        "bias": (mean - np.asarray(truth, dtype=float)).astype(float).tolist(),
        "cover": ((lower <= truth) & (truth <= upper)).astype(int).tolist(),
        "mean_L": float(total.mean()),
        "p_L_gt_095": float(np.mean(total > 0.95)),
        "rhat": rhats,
        "ess": ess,
        "posterior_corr": posterior_corr,
        "proposal_corr": proposal_corr,
        "acceptance": result.acceptance.astype(float).tolist(),
    }


def _holdout_rmse(Y, X, As_raw, train_idx, test_idx, config, truth_unused=None):
    As_train = [row_normalize(A[np.ix_(train_idx, train_idx)]) for A in As_raw]
    result = gibbs_multi_sem(Y[train_idx], X[train_idx], As_train, config=config)
    record = _posterior_record(result, np.zeros(len(As_train)))
    lam = np.asarray(record["mean"], dtype=float)
    beta = result.samples["beta"].reshape(-1, result.samples["beta"].shape[-1]).mean(axis=0)
    from mnreg.model import predict_kriging, metrics_rmse_mae

    As_full = [row_normalize(A) for A in As_raw]
    pred = predict_kriging(
        Y[train_idx], X[train_idx], X[test_idx], train_idx, test_idx, lam, beta, As_full
    )
    metrics = metrics_rmse_mae(Y[test_idx], pred)
    return float(metrics["RMSE"]), record


def _split(n, rng):
    n_test = max(int(0.2 * n), 20)
    order = rng.permutation(n)
    test_idx = np.sort(order[:n_test])
    train_idx = np.sort(order[n_test:])
    return train_idx, test_idx


def boundary_replication(rep: int) -> dict:
    rng = np.random.default_rng(30_000 + rep)
    n = 200
    rho = 1.0
    As_raw = None
    for _ in range(25):
        raw = [
            erdos_renyi(n, 8.0, rng),
            powerlaw_configuration(n, 2.2, rng),
        ]
        As = [row_normalize(A) for A in raw]
        combined = BOUNDARY_TRUE[0] * As[0] + BOUNDARY_TRUE[1] * As[1]
        rho = spectral_radius(combined)
        if rho < 0.98:
            As_raw = raw
            break
    if As_raw is None:
        return {"rep": rep, "ok": False, "rho": rho}
    As = [row_normalize(A) for A in As_raw]
    X = make_X(n, rng)
    Y = simulate_y(X, TRUE_BETA, BOUNDARY_TRUE, As, 1.0, rng)
    train_idx, test_idx = _split(n, rng)
    rows = {}
    draws = {}
    for name, constraint in (("l1", "l1"), ("spectral", "spectral")):
        config = _chain_config(40_000 + rep, constraint=constraint, adapt=True)
        result = gibbs_multi_sem(Y, X, As, config=config)
        record = _posterior_record(result, BOUNDARY_TRUE)
        holdout, _train_record = _holdout_rmse(
            Y, X, As_raw, train_idx, test_idx, _chain_config(50_000 + rep, constraint=constraint, adapt=True)
        )
        record["holdout_rmse"] = holdout
        rows[name] = record
        if rep == 0:
            draws[name] = result.samples["lambda"].reshape(-1, 2)[::8].astype(float).tolist()
    payload = {"rep": rep, "ok": True, "rho": float(rho), "estimators": rows, "draws": draws}
    (OUT / f"boundary_{rep:04d}.json").write_text(json.dumps(payload), encoding="utf-8")
    print(f"boundary {rep} rho={rho:.3f}", flush=True)
    return payload


def _calibrated_resolvent(adjacency):
    best = None
    for alpha in np.linspace(0.05, 0.92, 24):
        resolvent = boosted_resolvent(adjacency, float(alpha))
        np.fill_diagonal(resolvent, 0.0)
        resolvent = row_normalize(resolvent)
        corr = float(offdiagonal_correlation(adjacency, resolvent))
        if best is None or abs(corr - 0.97) < abs(best[0] - 0.97):
            best = (corr, float(alpha), resolvent)
        if 0.96 <= corr <= 0.98:
            return corr, float(alpha), resolvent
    return best


def collinear_replication(rep: int) -> dict:
    rng = np.random.default_rng(60_000 + rep)
    n = 200
    chosen = None
    for _ in range(20):
        base = row_normalize(erdos_renyi(n, 8.0, rng))
        corr, alpha, resolvent = _calibrated_resolvent(base)
        if 0.96 <= corr <= 0.98:
            decoys = [
                row_normalize(powerlaw_configuration(n, 2.5, rng)),
                row_normalize(powerlaw_configuration(n, 3.0, rng)),
                row_normalize(erdos_renyi(n, 4.0, rng)),
            ]
            chosen = (base, resolvent, decoys, corr, alpha)
            break
    if chosen is None:
        return {"rep": rep, "ok": False}
    base, resolvent, decoys, corr, alpha = chosen
    As = [base, resolvent, *decoys]
    X = make_X(n, rng)
    Y = simulate_y(X, TRUE_BETA, COLLINEAR_TRUE, As, 1.0, rng)
    train_idx, test_idx = _split(n, rng)
    As_raw = As
    samplers = {}
    for name, adapt in (("isotropic", False), ("adapted", True)):
        config = _chain_config(70_000 + rep, constraint="l1", adapt=adapt)
        result = gibbs_multi_sem(Y, X, As, config=config)
        record = _posterior_record(result, COLLINEAR_TRUE)
        holdout, _ignored = _holdout_rmse(
            Y, X, As_raw, train_idx, test_idx, _chain_config(75_000 + rep, constraint="l1", adapt=adapt)
        )
        record["holdout_rmse"] = holdout
        samplers[name] = record
    # The holdout helper measured bias against zero. Refit is already done; replace bias
    # by reading the stored mean against the true lambda. Means are for the induced fit.
    payload = {
        "rep": rep,
        "ok": True,
        "r_vec": float(corr),
        "alpha": float(alpha),
        "samplers": samplers,
        "truth": COLLINEAR_TRUE.tolist(),
    }
    (OUT / f"collinear_{rep:04d}.json").write_text(json.dumps(payload), encoding="utf-8")
    if rep % 10 == 0:
        print(f"collinear {rep} r={corr:.3f}", flush=True)
    return payload


def _aggregate_boundary(rows):
    ok = [row for row in rows if row.get("ok")]
    summary = {"reps": len(ok), "failed": len(rows) - len(ok), "rho_mean": float(np.mean([r["rho"] for r in ok]))}
    estimators = {}
    for key in ("l1", "spectral"):
        means = np.array([r["estimators"][key]["mean"] for r in ok])
        bias = means.mean(axis=0) - BOUNDARY_TRUE
        rmse = np.sqrt(np.mean((means - BOUNDARY_TRUE) ** 2, axis=0))
        cover = np.array([r["estimators"][key]["cover"] for r in ok]).mean(axis=0)
        estimators[key] = {
            "posterior_mean": means.mean(axis=0).astype(float).tolist(),
            "bias": bias.astype(float).tolist(),
            "rmse": rmse.astype(float).tolist(),
            "coverage": cover.astype(float).tolist(),
            "mean_L": float(np.mean([r["estimators"][key]["mean_L"] for r in ok])),
            "p_L_gt_095": float(np.mean([r["estimators"][key]["p_L_gt_095"] for r in ok])),
            "holdout_rmse": float(np.mean([r["estimators"][key]["holdout_rmse"] for r in ok])),
            "rhat_max": float(np.max([max(r["estimators"][key]["rhat"]) for r in ok])),
        }
    summary["estimators"] = estimators
    return summary


def _figure_s11(sample):
    draws = sample.get("draws") or {}
    if "spectral" not in draws:
        return
    spec = np.asarray(draws["spectral"], dtype=float)
    constrained = np.asarray(draws["l1"], dtype=float)
    fig, ax = plt.subplots(figsize=(5.4, 5.2))
    ax.scatter(spec[:, 0], spec[:, 1], s=8, alpha=0.35, c="#4C6A92", label="Exact spectral admissible")
    ax.scatter(constrained[:, 0], constrained[:, 1], s=8, alpha=0.35, c="#B85C38", label="ell_1 constrained")
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], dtype=float)
    ax.plot(diamond[:, 0], diamond[:, 1], color="black", lw=1.2, label=r"$\Vert\lambda\Vert_1=1$")
    ax.scatter([0.75], [-0.55], marker="x", c="black", s=60, label="truth")
    ax.set_xlabel(r"$\lambda_1$")
    ax.set_ylabel(r"$\lambda_2$")
    ax.set_title("Replication 1 posterior, outside the ell_1 ball")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "figure_s11_boundary.png", dpi=140)
    plt.close(fig)


def run_boundary():
    done = []
    pending = []
    for rep in range(200):
        path = OUT / f"boundary_{rep:04d}.json"
        if path.exists():
            done.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(rep)
    print(f"boundary pending {len(pending)} of 200", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for payload in pool.map(boundary_replication, pending, chunksize=1):
                done.append(payload)
    done.sort(key=lambda row: row["rep"])
    summary = _aggregate_boundary(done)
    summary["elapsed_note"] = "full-sample filter posterior plus a separate 80/20 induced holdout fit"
    _dump("table4c.json", summary)
    first = next((row for row in done if row["rep"] == 0 and row.get("draws")), None)
    if first:
        _figure_s11(first)
    return summary


def run_collinear():
    done = []
    pending = []
    for rep in range(200):
        path = OUT / f"collinear_{rep:04d}.json"
        if path.exists():
            done.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            pending.append(rep)
    print(f"collinear pending {len(pending)} of 200", flush=True)
    if pending:
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            for payload in pool.map(collinear_replication, pending, chunksize=1):
                done.append(payload)
    ok = [row for row in done if row.get("ok")]
    # Bias in the stored record is against zero because _holdout_rmse used a zero target.
    # Reconstruct RMSE from the posterior means and the known truth.
    summary = {"reps": len(ok), "failed": len(done) - len(ok)}
    samplers = {}
    for key in ("isotropic", "adapted"):
        means = np.array([row["samplers"][key]["mean"] for row in ok])
        truth = COLLINEAR_TRUE
        rmse = np.sqrt(np.mean((means - truth) ** 2, axis=0))
        ess = [min(row["samplers"][key]["ess"]) for row in ok]
        rhat = [max(row["samplers"][key]["rhat"]) for row in ok]
        proposal = [row["samplers"][key]["proposal_corr"] for row in ok]
        holdout = [row["samplers"][key]["holdout_rmse"] for row in ok]
        samplers[key] = {
            "rmse": rmse.astype(float).tolist(),
            "min_ess_mean": float(np.mean(ess)),
            "rhat_max": float(np.max(rhat)),
            "proposal_corr_mean": float(np.nanmean(proposal)),
            "lambda1_rmse": float(rmse[0]),
            "lambda2_rmse": float(rmse[1]),
            "holdout_rmse": float(np.mean(holdout)),
            "r_vec_mean": float(np.mean([row["r_vec"] for row in ok])),
        }
    summary["samplers"] = samplers
    _dump("table_s12.json", summary)
    return summary


def crps_ensemble(samples, observed):
    ordered = np.sort(np.asarray(samples, dtype=float), axis=0)
    observed = np.asarray(observed, dtype=float).reshape(1, -1)
    count = ordered.shape[0]
    index = np.arange(1, count + 1, dtype=float)[:, None]
    absolute = np.mean(np.abs(ordered - observed), axis=0)
    pairwise = 2.0 * np.sum((2.0 * index - count - 1.0) * ordered, axis=0)
    return absolute - pairwise / (2.0 * count * count)


def _predictive_metrics(draws, observed):
    mean = draws.mean(axis=0)
    err = mean - observed
    q05, q95 = np.quantile(draws, [0.05, 0.95], axis=0)
    q025, q975 = np.quantile(draws, [0.025, 0.975], axis=0)
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "cover90": float(np.mean((observed >= q05) & (observed <= q95))),
        "cover95": float(np.mean((observed >= q025) & (observed <= q975))),
        "width95": float(np.mean(q975 - q025)),
        "crps": float(np.mean(crps_ensemble(draws, observed))),
    }


def _conditional_replicates(Y, X, train_idx, test_idx, beta_draws, sigma_draws, lambda_draws, As_full, rng, per_draw):
    replicates = []
    for beta, sigma2, lam in zip(beta_draws, sigma_draws, lambda_draws):
        factor = assemble_V(lam, As_full)
        gram = factor.T @ factor
        hold = gram[np.ix_(test_idx, test_idx)]
        cross = gram[np.ix_(test_idx, train_idx)]
        resid = Y[train_idx] - X[train_idx] @ beta
        try:
            correction = np.linalg.solve(hold, cross @ resid)
            chol = np.linalg.cholesky(hold + 1e-10 * np.eye(len(test_idx)))
        except np.linalg.LinAlgError:
            continue
        mean = X[test_idx] @ beta - correction
        scale = float(np.sqrt(max(sigma2, 1e-12)))
        noise = scale * np.linalg.solve(chol, rng.normal(size=(len(test_idx), per_draw)))
        replicates.append(mean[:, None] + noise)
    if not replicates:
        raise RuntimeError("no predictive replicates")
    return np.concatenate(replicates, axis=1).T


def _thin_draws(result, count):
    lam = result.samples["lambda"].reshape(-1, result.samples["lambda"].shape[-1])
    beta = result.samples["beta"].reshape(-1, result.samples["beta"].shape[-1])
    sigma = result.samples["sigma2"].reshape(-1)
    step = max(len(sigma) // count, 1)
    return beta[::step][:count], sigma[::step][:count], lam[::step][:count]


def gibbs_schur(Y, X, As_full, train_idx, test_idx, config: GibbsConfig):
    """Metropolis-within-Gibbs on the Schur training likelihood, ell_1 constrained."""
    Yt = np.asarray(Y, dtype=float)[train_idx]
    Xt = np.asarray(X, dtype=float)[train_idx]
    n, q = Xt.shape
    k = len(As_full)
    rng0 = np.random.default_rng(config.seed)
    seeds = rng0.integers(1, 2**31 - 1, size=config.n_chains)
    chains = []
    for seed in seeds:
        rng = np.random.default_rng(int(seed))
        lam = np.zeros(k)
        beta, _, s2 = ols(Yt, Xt)
        sigma2 = max(float(s2), 1e-8)
        tau = 0.08
        logp, _ = log_posterior_marginal(lam, Yt - Xt @ beta, As_full, train_idx, test_idx, sigma2, config.xi)
        keep_l, keep_b, keep_s = [], [], []
        accept = 0
        cov_count = 0
        cov_mean = np.zeros(k)
        cov_m2 = np.zeros((k, k))
        frozen = None
        for t in range(config.n_iter):
            resid = Yt - Xt @ beta
            if config.adapt_covariance and t >= config.burn and frozen is not None:
                prop = lam + tau * (frozen @ rng.normal(size=k))
            elif config.adapt_covariance and cov_count > max(2 * k, 50):
                empirical = cov_m2 / (cov_count - 1)
                scale = (2.38 ** 2) / k
                cov = scale * empirical + 1e-5 * np.eye(k)
                try:
                    factor = np.linalg.cholesky(0.5 * (cov + cov.T))
                except np.linalg.LinAlgError:
                    factor = np.eye(k)
                prop = lam + tau * (factor @ rng.normal(size=k))
            else:
                prop = lam + tau * rng.normal(size=k)
            logp_prop, _ = log_posterior_marginal(prop, resid, As_full, train_idx, test_idx, sigma2, config.xi)
            accepted = np.isfinite(logp_prop) and np.log(rng.uniform()) < (logp_prop - logp)
            if accepted:
                lam = prop
                logp = logp_prop
                accept += 1
            if config.adapt_covariance and t < config.burn:
                cov_count += 1
                delta = lam - cov_mean
                cov_mean = cov_mean + delta / cov_count
                cov_m2 += np.outer(delta, lam - cov_mean)
                if (t + 1) % 50 == 0:
                    rate = accept / max(t + 1, 1)
                    if rate < 0.22:
                        tau *= 0.85
                    elif rate > 0.38:
                        tau *= 1.15
                    tau = float(np.clip(tau, 1e-4, 1.0))
                if t + 1 == config.burn and cov_count > k + 1:
                    empirical = cov_m2 / (cov_count - 1)
                    scale = (2.38 ** 2) / k
                    cov = scale * empirical + 1e-5 * np.eye(k)
                    try:
                        frozen = np.linalg.cholesky(0.5 * (cov + cov.T))
                    except np.linalg.LinAlgError:
                        frozen = np.eye(k)
            factor_s = schur_training_factor(lam, As_full, train_idx, test_idx)
            if factor_s is None:
                factor_s = np.eye(n)
            xts = Xt.T @ factor_s
            precision = np.eye(q) / config.Sigma0_diag + (xts @ Xt) / sigma2
            precision = 0.5 * (precision + precision.T)
            cov = np.linalg.pinv(precision)
            mean = cov @ ((xts @ Yt) / sigma2)
            try:
                beta = rng.multivariate_normal(mean, cov + 1e-10 * np.eye(q))
            except np.linalg.LinAlgError:
                beta = mean
            resid = Yt - Xt @ beta
            sse = float(resid @ (factor_s @ resid))
            sigma2 = 1.0 / rng.gamma(config.alpha0 + n / 2.0, 1.0 / (config.gamma0 + 0.5 * max(sse, 0.0)))
            sigma2 = max(float(sigma2), 1e-12)
            logp, _ = log_posterior_marginal(lam, resid, As_full, train_idx, test_idx, sigma2, config.xi)
            if t >= config.burn:
                keep_l.append(lam.copy())
                keep_b.append(beta.copy())
                keep_s.append(sigma2)
        chains.append((np.array(keep_l), np.array(keep_b), np.array(keep_s), accept / config.n_iter))
    class Pack:
        pass
    pack = Pack()
    pack.samples = {
        "lambda": np.stack([c[0] for c in chains], axis=0),
        "beta": np.stack([c[1] for c in chains], axis=0),
        "sigma2": np.stack([c[2] for c in chains], axis=0),
    }
    pack.acceptance = np.array([c[3] for c in chains])
    return pack


def _bayes_ols(Y, X, count, rng):
    """Gibbs draws under beta ~ N(0, 100 I) and sigma^2 ~ IG(0.001, 0.001), with no network."""
    n, q = X.shape
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    sigma2 = max(float(np.sum((Y - X @ beta) ** 2) / n), 1e-8)
    keep_b, keep_s = [], []
    for t in range(2000 + 2 * count):
        precision = np.eye(q) / 100.0 + (X.T @ X) / sigma2
        precision = 0.5 * (precision + precision.T)
        cov = np.linalg.inv(precision)
        mean = cov @ ((X.T @ Y) / sigma2)
        beta = rng.multivariate_normal(mean, cov + 1e-10 * np.eye(q))
        sse = float(np.sum((Y - X @ beta) ** 2))
        sigma2 = 1.0 / rng.gamma(0.001 + n / 2.0, 1.0 / (0.001 + 0.5 * max(sse, 0.0)))
        if t >= 2000 and len(keep_b) < count:
            keep_b.append(beta.copy())
            keep_s.append(sigma2)
    return np.asarray(keep_b), np.asarray(keep_s)


def run_schur():
    import run_boston as boston

    data = boston.load_boston()
    networks = boston.build_networks(data)
    record = json.loads((ROOT / "results" / "boston_adaptive" / "split_05.json").read_text(encoding="utf-8"))
    seed = int(record["seed"])
    part = np.random.default_rng([seed, 1])
    train = np.sort(part.choice(boston.N, size=boston.N_TRAIN, replace=False))
    test = np.sort(np.setdiff1d(np.arange(boston.N), train))
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = np.where(raw[train].std(axis=0, ddof=0) < 1e-12, 1.0, raw[train].std(axis=0, ddof=0))
    scaled = (raw - center) / spread
    X = np.column_stack([np.ones(boston.N), scaled])
    Y = data["y"]
    active = list(record["active"])
    configs = {
        "ols": [],
        "geo": ["geo"],
        "induced": active,
        "schur": active,
    }
    rng = np.random.default_rng(2026)
    rows = {}
    per_draw = 20
    target = 2000
    for name, layers in configs.items():
        print(f"predictive {name}", flush=True)
        full = [networks["adjacency"][layer] for layer in layers]
        if name == "ols":
            beta_draws, sigma_draws = _bayes_ols(Y[train], X[train], target, rng)
            draws = []
            for b, s2 in zip(beta_draws, sigma_draws):
                mean = X[test] @ b
                noise = np.sqrt(s2) * rng.normal(size=(len(test), per_draw))
                draws.append(mean[:, None] + noise)
            stack = np.concatenate(draws, axis=1).T
        elif name == "schur":
            result = gibbs_schur(
                Y, X, full, train, test, _chain_config(2026, n_iter=8000, burn=4000, n_chains=4)
            )
            beta, sigma, lam = _thin_draws(result, target)
            stack = _conditional_replicates(Y, X, train, test, beta, sigma, lam, full, rng, per_draw)
        else:
            induced = [boston.induced_pair(networks["weights"][layer], train)[1] for layer in layers]
            result = gibbs_multi_sem(
                Y[train], X[train], induced, config=_chain_config(2026, n_iter=8000, burn=4000, n_chains=4)
            )
            beta, sigma, lam = _thin_draws(result, target)
            stack = _conditional_replicates(Y, X, train, test, beta, sigma, lam, full, rng, per_draw)
        metrics = _predictive_metrics(stack, Y[test])
        metrics["draws"] = int(stack.shape[0])
        metrics["layers"] = layers
        rows[name] = metrics
        print(name, metrics, flush=True)
    _dump("table_s13.json", rows)
    return rows


def loto_fold(town: int) -> dict:
    import run_boston as boston

    state = loto_fold.state
    data = state["data"]
    networks = state["networks"]
    labels = data["town_id"]
    test = np.flatnonzero(labels == town)
    train = np.flatnonzero(labels != town)
    if len(test) == 0 or len(train) < 30:
        return {"town": int(town), "ok": False}
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = np.where(raw[train].std(axis=0, ddof=0) < 1e-12, 1.0, raw[train].std(axis=0, ddof=0))
    scaled = (raw - center) / spread
    x_train = np.column_stack([np.ones(len(train)), scaled[train]])
    x_test = np.column_stack([np.ones(len(test)), scaled[test]])
    y_train = data["y"][train]
    y_test = data["y"][test]
    induced = []
    for name in networks["kept"]:
        weights, adjacency = boston.induced_pair(networks["weights"][name], train)
        induced.append((name, weights, adjacency))
    rng = np.random.default_rng(80_000 + int(town))
    tests = boston.screen_split(y_train, x_train, induced, rng)
    active = [row["name"] for row in tests if row["selected"]]
    if not active:
        active = [networks["kept"][0]]
    beta, _, _ = ols(y_train, x_train)
    ols_hat = x_test @ beta
    config = _chain_config(90_000 + int(town), n_iter=8000, burn=4000, n_chains=4)

    def predict(layers):
        mats = [dict((name, adj) for name, _, adj in induced)[name] for name in layers]
        result = gibbs_multi_sem(y_train, x_train, mats, names=layers, config=config)
        lam = result.samples["lambda"].reshape(-1, len(layers)).mean(axis=0)
        bhat = result.samples["beta"].reshape(-1, x_train.shape[1]).mean(axis=0)
        full = [networks["adjacency"][name] for name in layers]
        return boston.precision_kriging(y_train, x_train, x_test, train, test, lam, bhat, full)

    from mnreg.model import metrics_rmse_mae

    geo = predict(["geo"])
    multi = predict(active)
    payload = {
        "town": int(town),
        "ok": True,
        "n_test": int(len(test)),
        "active": active,
        "ols": metrics_rmse_mae(y_test, ols_hat),
        "geo": metrics_rmse_mae(y_test, geo),
        "multi": metrics_rmse_mae(y_test, multi),
    }
    print(f"loto town {town} n={len(test)} multi {payload['multi']['RMSE']:.3f}", flush=True)
    return payload


def _loto_init(state):
    loto_fold.state = state


def run_loto():
    import run_boston as boston

    data = boston.load_boston()
    networks = boston.build_networks(data)
    towns = sorted(int(v) for v in np.unique(data["town_id"]))
    state = {"data": data, "networks": networks}
    rows = []
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_loto_init, initargs=(state,)) as pool:
        for payload in pool.map(loto_fold, towns, chunksize=1):
            rows.append(payload)
    ok = [row for row in rows if row.get("ok")]

    def pooled(key):
        sse = 0.0
        sae = 0.0
        yy = 0.0
        yp = 0.0
        n = 0
        # Rebuild from per-town RMSE is not the pooled RMSE. Weight by n_test using RMSE^2.
        weight = np.array([row["n_test"] for row in ok], dtype=float)
        rmse = np.sqrt(np.sum(weight * np.array([row[key]["RMSE"] ** 2 for row in ok])) / weight.sum())
        mae = float(np.sum(weight * np.array([row[key]["MAE"] for row in ok])) / weight.sum())
        corr = float(np.mean([row[key]["corr"] for row in ok]))
        return {"rmse": float(rmse), "mae": mae, "corr": corr, "folds": len(ok)}

    summary = {"ols": pooled("ols"), "geo": pooled("geo"), "multi": pooled("multi")}
    _dump("table_s14.json", summary)
    return summary


def _check_crps():
    rng = np.random.default_rng(0)
    draws = rng.normal(size=(20000, 1))
    value = float(crps_ensemble(draws, np.array([0.0]))[0])
    # CRPS of N(0, 1) at 0 equals sqrt(2/pi) - 1/sqrt(pi).
    if abs(value - 0.233694977) > 0.02:
        raise RuntimeError(f"CRPS self-check failed: {value}")


def main():
    import sys

    _check_crps()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "boundary"):
        run_boundary()
    if which in ("all", "collinear"):
        run_collinear()
    if which in ("all", "schur"):
        run_schur()
    if which in ("all", "loto"):
        run_loto()


if __name__ == "__main__":
    main()
