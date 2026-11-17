"""MCSE, Figure S10 prior overlay, Figure S11 spectral boundary, and plug-in Table S13.

Uses the finished unified archives. Does not refit the Monte Carlo experiments.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

import run_boston as boston
from mnreg.model import assemble_V, spectral_radius
from mnreg.networks import erdos_renyi, powerlaw_configuration, row_normalize

ROOT = Path(__file__).resolve().parent
UNI = ROOT / "results" / "unified"
FIG = ROOT / "results" / "figures"
REV = ROOT / "results" / "revision"


def _mcse_coverage(coverage: float, reps: int) -> float:
    return float(np.sqrt(coverage * (1.0 - coverage) / reps))


def _rmse_mcse(errors: np.ndarray) -> tuple[float, float]:
    squared = np.asarray(errors, dtype=float) ** 2
    rmse = float(np.sqrt(np.mean(squared)))
    if rmse == 0:
        return 0.0, 0.0
    mcse = float(np.std(squared, ddof=1) / (2.0 * rmse * np.sqrt(len(squared))))
    return rmse, mcse


def annotate_existing() -> None:
    boundary_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(UNI.glob("boundary_*.json"))]
    boundary_rows = [row for row in boundary_rows if row.get("ok")]
    summary = json.loads((UNI / "table4c.json").read_text(encoding="utf-8"))
    for key in ("l1", "spectral"):
        item = summary["estimators"][key]
        for index in range(2):
            errors = np.array([row["estimators"][key]["mean"][index] for row in boundary_rows]) - np.array([0.75, -0.55])[index]
            _rmse, mcse = _rmse_mcse(errors)
            item.setdefault("rmse_mcse", [None, None])
            item["rmse_mcse"][index] = mcse
            item.setdefault("coverage_mcse", [None, None])
            item["coverage_mcse"][index] = _mcse_coverage(item["coverage"][index], len(boundary_rows))
    (UNI / "table4c.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    collinear_rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(UNI.glob("collinear_*.json"))]
    collinear_rows = [row for row in collinear_rows if row.get("ok")]
    truth = np.array([0.20, 0.55])
    collinear = json.loads((UNI / "table_s12.json").read_text(encoding="utf-8"))
    for key in ("isotropic", "adapted"):
        item = collinear["samplers"][key]
        for index, label in ((0, "lambda1"), (1, "lambda2")):
            errors = np.array([row["samplers"][key]["mean"][index] for row in collinear_rows]) - truth[index]
            _rmse, mcse = _rmse_mcse(errors)
            item[f"{label}_rmse_mcse"] = mcse
    (UNI / "table_s12.json").write_text(json.dumps(collinear, indent=2), encoding="utf-8")

    table4b = json.loads((REV / "table4b.json").read_text(encoding="utf-8"))
    reps = int(table4b["n_reps"])
    for column in table4b["columns"]:
        column["coverage_mcse"] = _mcse_coverage(column["coverage_95"], reps)
    (REV / "table4b.json").write_text(json.dumps(table4b, indent=2), encoding="utf-8")
    print("annotated MCSE", flush=True)


def figure_s10() -> None:
    import run_revision_tasks as revision

    gaussian = json.loads((REV / "split5_gaussian.json").read_text(encoding="utf-8"))
    beta = json.loads((REV / "split5_beta.json").read_text(encoding="utf-8"))
    g_draws = np.load(REV / "split5_gaussian.npz")["draws_lambda"]
    b_draws = np.load(REV / "split5_beta.npz")["draws_lambda"]
    l_post = np.sum(np.abs(g_draws), axis=1)
    l_beta = np.sum(np.abs(b_draws), axis=1)
    rng = np.random.default_rng(2026)
    k = 3
    n_prior = 200_000
    raw = rng.exponential(scale=1.0, size=(n_prior, k))
    signs = rng.choice([-1.0, 1.0], size=(n_prior, k))
    norm1 = np.sum(raw, axis=1)
    radius = rng.uniform(0.0, 1.0, size=n_prior) ** (1.0 / k)
    l_uniform = np.sum(np.abs((raw / norm1[:, None]) * radius[:, None] * signs), axis=1)
    accepted = []
    while sum(len(chunk) for chunk in accepted) < n_prior:
        draw = rng.normal(size=(500_000, k))
        length = np.sum(np.abs(draw), axis=1)
        accepted.append(length[length < 1.0])
    l_gauss = np.concatenate(accepted)[:n_prior]
    grid = np.linspace(0.001, 0.999, 400)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].hist(l_uniform, bins=50, density=True, alpha=0.35, color="0.55", label=r"Uniform ball, $p(L)\propto L^{2}$")
    axes[0].hist(l_gauss, bins=50, density=True, alpha=0.45, color="#2A9D8F", label=r"Truncated Gaussian, $\xi=1$")
    axes[0].hist(l_post, bins=50, density=True, alpha=0.55, color="steelblue", label="Split 5 posterior")
    axes[0].set_xlabel(r"$L=\|\lambda\|_1$")
    axes[0].set_ylabel("Density")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(grid, 2.0 * (1.0 - grid), color="0.35", lw=1.5, label=r"Beta$(1,2)$ on $L$")
    axes[1].hist(l_beta, bins=50, density=True, alpha=0.75, color="#B85C38", label="Beta(1, 2) posterior")
    axes[1].set_xlabel(r"$L=\|\lambda\|_1$")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "figure_s10_prior_posterior.png"
    fig.savefig(path, dpi=160)
    fig.savefig(REV / "figure_s10_prior_vs_posterior_L.png", dpi=160)
    plt.close(fig)
    summary = {
        "figure": str(path),
        "truncated_gaussian_mean_L": float(np.mean(l_gauss)),
        "uniform_ball_mean_L": float(np.mean(l_uniform)),
        "paper_posterior_mean_L": float(np.mean(l_post)),
        "beta_posterior_mean_L": float(np.mean(l_beta)),
        "stored_gaussian_mean_L": gaussian["mean_L"],
        "stored_beta_mean_L": beta["mean_L"],
        "jacobian_note": "Beta(1,2) on L with Dirichlet(1..1) weights includes the Jacobian L^{1-k} in Cartesian lambda coordinates.",
    }
    (FIG / "figure_s10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("figure s10", summary["truncated_gaussian_mean_L"], flush=True)
    del revision


def _rep0_graphs():
    rng = np.random.default_rng(30_000)
    truth = np.array([0.75, -0.55])
    for _ in range(25):
        raw = [erdos_renyi(200, 8.0, rng), powerlaw_configuration(200, 2.2, rng)]
        graphs = [row_normalize(A) for A in raw]
        rho = spectral_radius(truth[0] * graphs[0] + truth[1] * graphs[1])
        if rho < 0.98:
            return graphs, rho
    raise RuntimeError("replication 0 graph was not admissible")


def figure_s11() -> None:
    graphs, rho = _rep0_graphs()
    draws = json.loads((UNI / "boundary_0000.json").read_text(encoding="utf-8"))["draws"]
    grid = np.linspace(-1.15, 1.15, 31)
    radius = np.empty((len(grid), len(grid)))
    for i, lam1 in enumerate(grid):
        for j, lam2 in enumerate(grid):
            radius[j, i] = spectral_radius(lam1 * graphs[0] + lam2 * graphs[1])
    fig, ax = plt.subplots(figsize=(5.6, 5.3))
    spectral = np.asarray(draws["spectral"], dtype=float)
    constrained = np.asarray(draws["l1"], dtype=float)
    ax.scatter(spectral[:, 0], spectral[:, 1], s=8, alpha=0.35, c="#4C6A92", label="Spectral-stability, ρ(B(λ)) < 1")
    ax.scatter(constrained[:, 0], constrained[:, 1], s=8, alpha=0.35, c="#B85C38", label="ℓ₁ constrained")
    diamond = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]], dtype=float)
    ax.plot(diamond[:, 0], diamond[:, 1], color="black", lw=1.2, label="‖λ‖₁ = 1")
    ax.contour(grid, grid, radius, levels=[1.0], colors="#1B4332", linewidths=1.4)
    ax.scatter([0.75], [-0.55], marker="x", c="black", s=60, label="truth")
    ax.set_xlabel("λ₁")
    ax.set_ylabel("λ₂")
    ax.set_title(f"Replication 1, spectral radius at truth {rho:.3f}")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path = FIG / "figure_s11_boundary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("figure s11", rho, flush=True)


def _plugin_metrics(y, mean, variance) -> dict:
    std = np.sqrt(np.maximum(variance, 1e-12))
    z90, z95 = norm.ppf(0.95), norm.ppf(0.975)
    z = (y - mean) / std
    crps = std * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
    return {
        "rmse": float(np.sqrt(np.mean((mean - y) ** 2))),
        "mae": float(np.mean(np.abs(mean - y))),
        "cover90": float(np.mean(np.abs(y - mean) <= z90 * std)),
        "cover95": float(np.mean(np.abs(y - mean) <= z95 * std)),
        "width95": float(np.mean(2.0 * z95 * std)),
        "crps": float(np.mean(crps)),
    }


def _conditional(y, x, train, test, beta, sigma2, lam, graphs):
    factor = assemble_V(np.asarray(lam, dtype=float), graphs)
    gram = factor.T @ factor
    hold = gram[np.ix_(test, test)]
    cross = gram[np.ix_(test, train)]
    resid = y[train] - x[train] @ np.asarray(beta, dtype=float)
    correction = np.linalg.solve(hold, cross @ resid)
    mean = x[test] @ np.asarray(beta, dtype=float) - correction
    variance = sigma2 * np.diag(np.linalg.inv(hold))
    return _plugin_metrics(y[test], mean, variance)


def table_s13_plugin() -> None:
    data = boston.load_boston()
    networks = boston.build_networks(data)
    raw = data["raw"]
    y = data["y"]
    rows = {"geo": [], "town": [], "multi": [], "ols": []}
    for path in sorted((ROOT / "results" / "boston_adaptive").glob("split_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        part = np.random.default_rng([int(record["seed"]), 1])
        train = np.sort(part.choice(boston.N, size=boston.N_TRAIN, replace=False))
        test = np.sort(np.setdiff1d(np.arange(boston.N), train))
        center = raw[train].mean(axis=0)
        spread = np.where(raw[train].std(axis=0, ddof=0) < 1e-12, 1.0, raw[train].std(axis=0, ddof=0))
        scaled = (raw - center) / spread
        x = np.column_stack([np.ones(boston.N), scaled])
        ols_beta, _, sigma_ols = boston.ols(y[train], x[train])
        resid = y[test] - x[test] @ ols_beta
        rows["ols"].append(_plugin_metrics(y[test], x[test] @ ols_beta, np.full(len(test), sigma_ols)))
        for key, layers in (
            ("geo", ["geo"]),
            ("town", ["town"]),
            ("multi", record["active"]),
        ):
            block = record["geo" if key == "geo" else "town_sem" if key == "town" else "multi"]
            graphs = [networks["adjacency"][name] for name in layers]
            rows[key].append(_conditional(y, x, train, test, block["beta"], block["sigma2"], block["lambda"], graphs))
        print(f"plugin split {record['split']}", flush=True)

    def average(items):
        keys = items[0].keys()
        return {key: float(np.mean([item[key] for item in items])) for key in keys}

    payload = {key: average(value) for key, value in rows.items()}
    payload["split5_full_posterior"] = json.loads((UNI / "table_s13.json").read_text(encoding="utf-8"))
    payload["note"] = (
        "The 50-split scores condition on the archived posterior means. "
        "The split files do not retain the MCMC draws. Split 5 full posterior predictive scores are nested."
    )
    (UNI / "table_s13_50.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("wrote table_s13_50", flush=True)


def verify_radial_prior() -> None:
    """Beta(1, 2) on L, times a uniform direction, has Cartesian log-density
    (1-k) ln L + ln(1-L). The factor L^{1-k} cancels the shell volume L^{k-1}.
    """
    from scipy.stats import beta as beta_density

    from mnreg.model import _lambda_log_prior

    for dimension in (2, 3, 5):
        lam = np.linspace(0.05, 0.05 * dimension, dimension)
        lam = lam / np.sum(np.abs(lam)) * 0.4
        length = float(np.sum(np.abs(lam)))
        expected = (1 - dimension) * np.log(length) + np.log(1.0 - length)
        got = _lambda_log_prior(lam, 1.0, "beta_l1")
        if abs(got - expected) > 1e-12:
            raise RuntimeError(f"radial log-prior mismatch at k={dimension}: {got} != {expected}")
        shell = length ** (dimension - 1)
        jacobian = length ** (1 - dimension)
        if abs(shell * jacobian - 1.0) > 1e-12:
            raise RuntimeError(f"Jacobian did not cancel at k={dimension}")
    grid = np.linspace(0.0, 1.0, 5)
    marginal = 2.0 * (1.0 - grid)
    if abs(float(np.trapezoid(marginal, grid)) - 1.0) > 1e-12:
        raise RuntimeError("Beta(1, 2) marginal is not normalized")
    if not np.allclose(marginal, beta_density(1, 2).pdf(grid)):
        raise RuntimeError("2(1-L) is not the Beta(1, 2) density")
    print("radial prior: Jacobian L^{k-1} cancels for every k, marginal is Beta(1, 2)", flush=True)


def main() -> None:
    verify_radial_prior()
    annotate_existing()
    figure_s10()
    figure_s11()
    table_s13_plugin()


if __name__ == "__main__":
    main()
