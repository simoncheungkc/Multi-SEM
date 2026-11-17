"""Boston housing Multi-SEM: screening, Gibbs estimation, and precision kriging.

The design follows the requested protocol. Screening, residuals, and estimation
use the 405 training tracts only. Kriging uses the posterior means and the
full 506-by-506 networks. The Gibbs settings are the package defaults except
for the requested length: 4 chains, 6,000 iterations, 3,000 burn-in.
"""

from __future__ import annotations

import json
import os
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from pathlib import Path

import numpy as np
from scipy.special import ndtr
from scipy.stats import wilcoxon

from mnreg.model import (
    GibbsConfig,
    benjamini_hochberg,
    effective_sample_size,
    gelman_rubin,
    gibbs_multi_sem,
    metrics_rmse_mae,
    moran_i_test,
    offdiag_correlation,
    ols,
    posterior_summary,
)
from mnreg.networks import boosted_resolvent, row_normalize

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "boston_corrected.txt"
OUT_DIR = ROOT / "results" / "boston"
SOURCE_URL = "https://lib.stat.cmu.edu/datasets/boston_corrected.txt"

N = 506
N_TRAIN = 405
N_TEST = 101
K_NEIGHBORS = 6
ALPHA_RESOLVENT = 0.5
FDR_Q = 0.05
KAPPA_MAX = 3.0
PSI_MAX = 0.20
MC_DRAWS = 499
N_SPLITS = 50
MASTER_SEED = 2026
LAYER_ORDER = ("geo", "resolvent", "town", "radial")
LAYER_LABELS = {
    "geo": "Geographic 6-NN",
    "resolvent": "Geographic resolvent",
    "town": "Town jurisdiction",
    "radial": "Radial highway",
}
COVARIATE_NAMES = (
    "intercept",
    "rm",
    "age",
    "crim",
    "zn",
    "indus",
    "log_lstat",
    "tax",
    "ptratio",
    "nox",
    "log_dis",
)

_STATE: dict = {}


def utm_zone19n(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """WGS84 longitude/latitude to UTM zone 19N easting and northing, in metres."""
    a = 6378137.0
    f = 1.0 / 298.257223563
    k0 = 0.9996
    e2 = f * (2.0 - f)
    ep2 = e2 / (1.0 - e2)
    lon0 = np.radians((19 - 1) * 6 - 180 + 3)
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    tan_lat = np.tan(lat)
    n = a / np.sqrt(1.0 - e2 * sin_lat**2)
    t = tan_lat**2
    c = ep2 * cos_lat**2
    aa = cos_lat * (lon - lon0)
    m = a * (
        (1 - e2 / 4 - 3 * e2**2 / 64 - 5 * e2**3 / 256) * lat
        - (3 * e2 / 8 + 3 * e2**2 / 32 + 45 * e2**3 / 1024) * np.sin(2 * lat)
        + (15 * e2**2 / 256 + 45 * e2**3 / 1024) * np.sin(4 * lat)
        - (35 * e2**3 / 3072) * np.sin(6 * lat)
    )
    easting = k0 * n * (
        aa
        + (1 - t + c) * aa**3 / 6
        + (5 - 18 * t + t**2 + 72 * c - 58 * ep2) * aa**5 / 120
    ) + 500000.0
    northing = k0 * (
        m
        + n * tan_lat * (
            aa**2 / 2
            + (5 - t + 9 * c + 4 * c**2) * aa**4 / 24
            + (61 - 58 * t + t**2 + 600 * c - 330 * ep2) * aa**6 / 720
        )
    )
    return np.column_stack([easting, northing])


def load_boston(path: Path = DATA_PATH) -> dict:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 21 or not parts[0].isdigit():
            continue
        tail = parts[-19:]
        town = " ".join(parts[1:-19])
        nums = [float(v) for v in tail]
        rows.append((int(parts[0]), town, *nums))
    if len(rows) != N:
        raise RuntimeError(f"Expected {N} tracts, parsed {len(rows)} from {path}")
    town = np.array([r[1] for r in rows])
    town_id = np.array([int(r[2]) for r in rows], dtype=int)
    lon = np.array([r[4] for r in rows])
    lat = np.array([r[5] for r in rows])
    cmedv = np.array([r[7] for r in rows])
    crim = np.array([r[8] for r in rows])
    zn = np.array([r[9] for r in rows])
    indus = np.array([r[10] for r in rows])
    nox = np.array([r[12] for r in rows])
    rm = np.array([r[13] for r in rows])
    age = np.array([r[14] for r in rows])
    dis = np.array([r[15] for r in rows])
    rad = np.array([int(r[16]) for r in rows], dtype=int)
    tax = np.array([r[17] for r in rows])
    ptratio = np.array([r[18] for r in rows])
    lstat = np.array([r[20] for r in rows])
    if np.any(cmedv <= 0) or np.any(lstat <= 0) or np.any(dis <= 0):
        raise RuntimeError("log transforms require positive cmedv, lstat, and dis")
    y = np.log(cmedv)
    raw = np.column_stack([rm, age, crim, zn, indus, np.log(lstat), tax, ptratio, nox, np.log(dis)])
    return {
        "town": town,
        "town_id": town_id,
        "lon": lon,
        "lat": lat,
        "rad": rad,
        "y": y,
        "cmedv": cmedv,
        "raw": raw,
        "xy": utm_zone19n(lon, lat),
    }


def knn_weights(xy: np.ndarray, k: int = K_NEIGHBORS) -> np.ndarray:
    diff = xy[:, None, :] - xy[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    nn = np.argsort(dist, axis=1)[:, :k]
    n = xy.shape[0]
    weights = np.zeros((n, n), dtype=float)
    rows = np.repeat(np.arange(n), k)
    cols = nn.ravel()
    weights[rows, cols] = 1.0 / dist[rows, cols]
    weights = np.maximum(weights, weights.T)
    np.fill_diagonal(weights, 0.0)
    return weights


def group_weights(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    weights = np.zeros((n, n), dtype=float)
    for label in np.unique(labels):
        idx = np.flatnonzero(labels == label)
        if len(idx) < 2:
            continue
        weights[np.ix_(idx, idx)] = 1.0
    np.fill_diagonal(weights, 0.0)
    return weights


def clique_resolvent_gap(weights: np.ndarray, alpha: float = ALPHA_RESOLVENT) -> dict:
    """Largest town block: purged resolvent versus the zero-diagonal clique."""
    degrees = (weights > 0).sum(axis=1)
    anchor = int(np.argmax(degrees))
    members = np.flatnonzero(weights[anchor] > 0)
    members = np.sort(np.append(members, anchor))
    block = weights[np.ix_(members, members)]
    clique = row_normalize(block)
    resolvent = boosted_resolvent(clique, alpha=alpha)
    np.fill_diagonal(resolvent, 0.0)
    purged = row_normalize(resolvent)
    gap = np.max(np.abs(purged - clique))
    return {"town_size": int(len(members)), "max_abs_gap": float(gap)}


def build_networks(data: dict) -> dict:
    w_geo = knn_weights(data["xy"])
    a_geo = row_normalize(w_geo)
    resolvent = boosted_resolvent(a_geo, alpha=ALPHA_RESOLVENT)
    w_res = resolvent.copy()
    np.fill_diagonal(w_res, 0.0)
    a_res = row_normalize(w_res)
    w_town = group_weights(data["town_id"])
    a_town = row_normalize(w_town)
    w_rad = group_weights(data["rad"])
    a_rad = row_normalize(w_rad)
    weights = {"geo": w_geo, "resolvent": w_res, "town": w_town, "radial": w_rad}
    adjacency = {"geo": a_geo, "resolvent": a_res, "town": a_town, "radial": a_rad}
    names = list(LAYER_ORDER)
    matrices = [adjacency[name] for name in names]
    corr = {
        f"{names[i]}__{names[j]}": offdiag_correlation(matrices[i], matrices[j])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    }
    keep = []
    for i, name in enumerate(names):
        redundant = False
        for j in keep:
            value = abs(corr[f"{names[j]}__{name}"] if j < i else corr[f"{name}__{names[j]}"])
            # keys were stored with the earlier index first
            key = f"{names[min(i, j)]}__{names[max(i, j)]}"
            if abs(corr[key]) >= 0.99:
                redundant = True
                break
        if not redundant:
            keep.append(i)
    kept = [names[i] for i in keep]
    return {
        "weights": weights,
        "adjacency": adjacency,
        "correlation": corr,
        "kept": kept,
        "dropped": [name for name in names if name not in kept],
        "clique_check": clique_resolvent_gap(w_town),
    }


def layer_description(weights: np.ndarray, adjacency: np.ndarray, labels: np.ndarray | None = None) -> dict:
    binary = (np.asarray(weights) > 0).astype(float)
    np.fill_diagonal(binary, 0.0)
    degree = binary.sum(axis=1)
    row_sum = np.asarray(weights, dtype=float).sum(axis=1)
    singular = np.linalg.svd(adjacency, compute_uv=False)
    out = {
        "edges": int(np.sum(np.triu(binary, 1) > 0)),
        "isolates": int(np.sum(degree == 0)),
        "s0": float(adjacency.sum()),
        "mean_binary_degree": float(degree.mean()),
        "max_binary_degree": int(degree.max()),
        "kappa_row_sum": float(row_sum.max() / row_sum.mean()) if row_sum.mean() > 0 else None,
        "psi": float(singular[0] / singular.sum()) if singular.sum() > 0 else None,
    }
    if labels is not None:
        sizes = [int(np.sum(labels == lab)) for lab in np.unique(labels)]
        out["groups"] = int(len(sizes))
        out["singleton_groups"] = int(sum(size == 1 for size in sizes))
        out["largest_group"] = int(max(sizes))
    return out


def induced_pair(weights: np.ndarray, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    block = np.asarray(weights, dtype=float)[np.ix_(index, index)].copy()
    np.fill_diagonal(block, 0.0)
    return block, row_normalize(block)


def moran_proposition1(resid: np.ndarray, adjacency: np.ndarray, design: np.ndarray) -> dict:
    """Scaled residual Moran statistic and the exact normal moments in Proposition 1."""
    resid = np.asarray(resid, dtype=float).reshape(-1)
    adjacency = np.asarray(adjacency, dtype=float)
    design = np.asarray(design, dtype=float)
    n, q = design.shape
    dof = n - q
    sse = float(resid @ resid)
    active = float(adjacency.sum())
    if sse <= 0 or active <= 0 or dof <= 2:
        return {"I": np.nan, "EI": np.nan, "VI": np.nan, "z": np.nan, "pvalue": 1.0, "c": np.nan, "dof": int(dof)}
    scale = n / active
    statistic = scale * float(resid @ (adjacency @ resid)) / sse
    symmetric = 0.5 * (adjacency + adjacency.T)
    gram = design.T @ design
    try:
        gram_inv = np.linalg.inv(gram)
    except np.linalg.LinAlgError:
        gram_inv = np.linalg.pinv(gram)
    projected = symmetric @ design
    maker = symmetric - projected @ (gram_inv @ design.T)
    sandwiched = maker - design @ (gram_inv @ (design.T @ maker))
    trace_ma = float(np.trace(symmetric) - np.trace(gram_inv @ (design.T @ projected)))
    trace_square = float(np.sum(sandwiched * sandwiched.T))
    variance = scale**2 * (2.0 / (dof * (dof + 2.0))) * (trace_square - (trace_ma**2) / dof)
    variance = max(variance, 1e-18)
    z = (statistic - scale * trace_ma / dof) / np.sqrt(variance)
    # erfc avoids an exact zero when |z| is large enough that Phi saturates at 1.
    from math import erfc
    pvalue = float(erfc(abs(z) / np.sqrt(2.0)))
    return {
        "I": float(statistic),
        "EI": float(scale * trace_ma / dof),
        "VI": float(variance),
        "z": float(z),
        "pvalue": min(max(pvalue, 0.0), 1.0),
        "c": float(scale),
        "dof": int(dof),
    }


def monte_carlo_pvalue(observed_z, design, adjacency, scale, mean, sd, rng, draws=MC_DRAWS) -> float:
    n, _ = design.shape
    gram_inv = np.linalg.inv(design.T @ design)
    count = 0
    batch = 100
    done = 0
    while done < draws:
        width = min(batch, draws - done)
        gaussian = rng.normal(size=(n, width))
        residual = gaussian - design @ (gram_inv @ (design.T @ gaussian))
        quadratic = np.sum(residual * (adjacency @ residual), axis=0)
        denom = np.sum(residual * residual, axis=0)
        replicate = scale * quadratic / denom
        z = (replicate - mean) / sd
        count += int(np.sum(np.abs(z) >= abs(observed_z)))
        done += width
    return float((1 + count) / (draws + 1))


def screen_split(y_train, x_train, named_matrices, rng) -> list[dict]:
    beta, resid, _ = ols(y_train, x_train)
    del beta
    rows = []
    for name, weights, adjacency in named_matrices:
        row_sum = weights.sum(axis=1)
        kappa = float(row_sum.max() / row_sum.mean()) if row_sum.mean() > 0 else np.inf
        singular = np.linalg.svd(adjacency, compute_uv=False)
        psi = float(singular[0] / singular.sum()) if singular.sum() > 0 else np.nan
        test = moran_proposition1(resid, adjacency, x_train)
        reference = "undefined"
        pvalue = 1.0
        if np.isfinite(test["z"]):
            if kappa > KAPPA_MAX or psi > PSI_MAX:
                reference = "Monte Carlo"
                pvalue = monte_carlo_pvalue(
                    test["z"], x_train, adjacency, test["c"], test["EI"], np.sqrt(test["VI"]), rng
                )
            else:
                reference = "Gaussian"
                pvalue = test["pvalue"]
        rows.append({
            "name": name,
            "I": test["I"],
            "EI": test["EI"],
            "z": test["z"],
            "kappa": None if not np.isfinite(kappa) else kappa,
            "psi": psi,
            "c": test["c"],
            "dof": test["dof"],
            "reference": reference,
            "pvalue": pvalue,
            "code_z": moran_i_test(resid, adjacency, x_train)["z"],
        })
    selected = benjamini_hochberg(np.array([row["pvalue"] for row in rows]), alpha=FDR_Q)
    for row, flag in zip(rows, selected):
        row["selected"] = bool(flag)
    return rows


def precision_kriging(y_train, x_train, x_test, train_idx, test_idx, lam, beta, networks) -> np.ndarray:
    from mnreg.model import assemble_V

    spatial = assemble_V(np.asarray(lam, dtype=float), networks)
    precision = spatial.T @ spatial
    hold = precision[np.ix_(test_idx, test_idx)]
    cross = precision[np.ix_(test_idx, train_idx)]
    residual = np.asarray(y_train, dtype=float) - np.asarray(x_train, dtype=float) @ np.asarray(beta, dtype=float)
    correction = np.linalg.solve(hold, cross @ residual)
    return np.asarray(x_test, dtype=float) @ np.asarray(beta, dtype=float) - correction


def fit_spatial(
    y_train,
    x_train,
    networks,
    names,
    seed,
    adaptive: bool = False,
    covariate_names=None,
    lambda_prior: str = "gaussian",
    n_iter: int | None = None,
    burn: int | None = None,
    n_chains: int | None = None,
) -> dict:
    if len(networks) == 0:
        beta, resid, sigma2 = ols(y_train, x_train)
        return {
            "lambda": [],
            "beta": beta,
            "sigma2": sigma2,
            "names": [],
            "rhat_lambda": [],
            "rhat_beta": [],
            "ess_lambda": [],
            "ess_beta": [],
            "acceptance": [],
            "mean_L": 0.0,
            "p_L_gt_095": 0.0,
            "method": "OLS",
        }
    if adaptive:
        config = GibbsConfig(
            n_iter=8000 if n_iter is None else n_iter,
            burn=4000 if burn is None else burn,
            n_chains=4 if n_chains is None else n_chains,
            seed=int(seed),
            adapt_covariance=True,
            lambda_prior=lambda_prior,
        )
    else:
        config = GibbsConfig(
            n_iter=6000 if n_iter is None else n_iter,
            burn=3000 if burn is None else burn,
            n_chains=4 if n_chains is None else n_chains,
            seed=int(seed),
            lambda_prior=lambda_prior,
        )
    result = gibbs_multi_sem(y_train, x_train, networks, names=list(names), config=config)
    lam = result.samples["lambda"]
    beta = result.samples["beta"]
    sigma = result.samples["sigma2"]
    flat_l = lam.reshape(-1, lam.shape[-1])
    flat_b = beta.reshape(-1, beta.shape[-1])
    total_l = np.sum(np.abs(flat_l), axis=1)

    def pack(draws):
        return {
            "rhat": float(gelman_rubin(draws)),
            "ess": float(sum(effective_sample_size(draws[c]) for c in range(draws.shape[0]))),
        }

    lambda_diag = [pack(lam[:, :, s]) for s in range(lam.shape[-1])]
    beta_diag = [pack(beta[:, :, j]) for j in range(beta.shape[-1])]
    beta_names = list(covariate_names) if covariate_names is not None else list(COVARIATE_NAMES)
    if len(beta_names) != flat_b.shape[1]:
        beta_names = [f"beta_{j}" for j in range(flat_b.shape[1])]
    table = posterior_summary(result, beta_names)["table"]
    l_row = {
        "parameter": "L",
        "mean": float(total_l.mean()),
        "median": float(np.median(total_l)),
        "sd": float(total_l.std(ddof=1)),
        "q025": float(np.quantile(total_l, 0.025)),
        "q975": float(np.quantile(total_l, 0.975)),
        "q95": float(np.quantile(total_l, 0.95)),
        "Rhat": float(gelman_rubin(np.sum(np.abs(lam), axis=2))),
        "ESS": float(sum(
            effective_sample_size(np.sum(np.abs(lam[c]), axis=1)) for c in range(lam.shape[0])
        )),
    }
    table.append(l_row)
    return {
        "lambda": flat_l.mean(axis=0),
        "beta": flat_b.mean(axis=0),
        "sigma2": float(sigma.mean()),
        "names": list(names),
        "rhat_lambda": [item["rhat"] for item in lambda_diag],
        "rhat_beta": [item["rhat"] for item in beta_diag],
        "ess_lambda": [item["ess"] for item in lambda_diag],
        "ess_beta": [item["ess"] for item in beta_diag],
        "acceptance": result.acceptance.astype(float),
        "mean_L": float(total_l.mean()),
        "p_L_gt_095": float(np.mean(total_l > 0.95)),
        "max_L": float(total_l.max()),
        "p_L_ge_1": float(np.mean(total_l >= 1.0 - 1e-8)),
        "posterior": table,
        "proposal_correlation": None
        if result.proposal_correlation is None
        else result.proposal_correlation.astype(float),
        "tau": result.tau.astype(float),
        "adaptive_covariance": bool(adaptive),
        "method": "Gibbs",
        "lambda_q025": np.quantile(flat_l, 0.025, axis=0),
        "lambda_q975": np.quantile(flat_l, 0.975, axis=0),
    }


def town_fixed_effects(y_train, x_train, y_index_train, x_test, town_train, town_test):
    levels, counts = np.unique(town_train, return_counts=True)
    reference = int(levels[np.argmax(counts)])
    others = [int(level) for level in levels if int(level) != reference]
    column = {level: j for j, level in enumerate(others)}
    train_dummies = np.zeros((len(town_train), len(others)))
    for i, level in enumerate(town_train):
        slot = column.get(int(level))
        if slot is not None:
            train_dummies[i, slot] = 1.0
    test_dummies = np.zeros((len(town_test), len(others)))
    unseen = 0
    for i, level in enumerate(town_test):
        slot = column.get(int(level))
        if slot is not None:
            test_dummies[i, slot] = 1.0
        elif int(level) != reference:
            unseen += 1
    design = np.hstack([x_train, train_dummies])
    beta, *_ = np.linalg.lstsq(design, y_train, rcond=None)
    prediction = np.hstack([x_test, test_dummies]) @ beta
    return prediction, {
        "reference_town": reference,
        "training_towns": int(len(levels)),
        "unseen_holdout_tracts": int(unseen),
    }


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _init_worker(state: dict) -> None:
    _STATE.clear()
    _STATE.update(state)


def run_split(task: dict) -> dict:
    started = time.perf_counter()
    data = _STATE["data"]
    networks = _STATE["networks"]
    kept = _STATE["kept"]
    seed = int(task["seed"])
    split_id = int(task["split"])
    part = np.random.default_rng([seed, 1])
    monte = np.random.default_rng([seed, 2])
    gibbs_rng = np.random.default_rng([seed, 3])
    train = np.sort(part.choice(N, size=N_TRAIN, replace=False))
    test = np.sort(np.setdiff1d(np.arange(N), train))
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = raw[train].std(axis=0, ddof=0)
    spread = np.where(spread < 1e-12, 1.0, spread)
    scaled = (raw - center) / spread
    x_train = np.column_stack([np.ones(N_TRAIN), scaled[train]])
    x_test = np.column_stack([np.ones(N_TEST), scaled[test]])
    y_train = data["y"][train]
    y_test = data["y"][test]

    induced = []
    for name in kept:
        weights, adjacency = induced_pair(networks["weights"][name], train)
        induced.append((name, weights, adjacency))
    tests = screen_split(y_train, x_train, induced, monte)
    active = [row["name"] for row in tests if row["selected"]]

    def spatial_prediction(layer_names, gibbs_seed):
        mats = [dict((name, adj) for name, _, adj in induced)[name] for name in layer_names]
        fit = fit_spatial(
            y_train, x_train, mats, layer_names, gibbs_seed, adaptive=bool(_STATE.get("adaptive"))
        )
        full = [networks["adjacency"][name] for name in layer_names]
        if layer_names:
            yhat = precision_kriging(y_train, x_train, x_test, train, test, fit["lambda"], fit["beta"], full)
        else:
            yhat = x_test @ fit["beta"]
        mean_only = x_test @ fit["beta"]
        return fit, yhat, mean_only

    seeds = gibbs_rng.integers(1, 2**31 - 1, size=4)
    ols_beta, _, _ = ols(y_train, x_train)
    ols_hat = x_test @ ols_beta
    fe_hat, fe_info = town_fixed_effects(
        y_train, x_train, train, x_test, data["town_id"][train], data["town_id"][test]
    )
    geo_fit, geo_hat, _ = spatial_prediction(["geo"], int(seeds[0]))
    town_fit, town_hat, _ = spatial_prediction(["town"], int(seeds[1]))
    multi_fit, multi_hat, multi_mean = spatial_prediction(active, int(seeds[2]))

    def pack_metrics(yhat):
        return metrics_rmse_mae(y_test, yhat)

    record = {
        "split": split_id,
        "seed": seed,
        "tests": tests,
        "active": active,
        "ols": pack_metrics(ols_hat),
        "fixed_effects": {**pack_metrics(fe_hat), **fe_info},
        "geo": {**pack_metrics(geo_hat), **{k: v for k, v in geo_fit.items() if k != "method"}, "method": geo_fit["method"]},
        "town_sem": {**pack_metrics(town_hat), **{k: v for k, v in town_fit.items() if k != "method"}, "method": town_fit["method"]},
        "multi_mean": pack_metrics(multi_mean),
        "multi": {**pack_metrics(multi_hat), **{k: v for k, v in multi_fit.items() if k != "method"}, "method": multi_fit["method"]},
        "seconds": time.perf_counter() - started,
    }
    return _jsonable(record)


def aggregate(records: list[dict]) -> dict:
    keys = ("ols", "fixed_effects", "geo", "town_sem", "multi_mean", "multi")
    labels = {
        "ols": "OLS (no network)",
        "fixed_effects": "Municipal fixed effects",
        "geo": "Single SEM (geographic 6-NN)",
        "town_sem": "Single SEM (town boundaries)",
        "multi_mean": "Multi-SEM (no kriging)",
        "multi": "Multi-SEM (screened, precision kriging)",
    }
    ols_rmse = np.array([row["ols"]["RMSE"] for row in records])
    table = []
    for key in keys:
        rmse = np.array([row[key]["RMSE"] for row in records])
        mae = np.array([row[key]["MAE"] for row in records])
        corr = np.array([row[key]["corr"] for row in records])
        wins = int(np.sum(rmse < ols_rmse - 1e-15))
        if key == "ols":
            wins = None
        table.append({
            "model": labels[key],
            "mean_rmse": float(rmse.mean()),
            "mean_mae": float(mae.mean()),
            "mean_corr": float(np.nanmean(corr)),
            "wins_vs_ols": wins,
            "win_rate": None if wins is None else wins / len(records),
        })
    diff = ols_rmse**2 - np.array([row["multi"]["RMSE"] for row in records]) ** 2
    if len(diff) < 2 or np.allclose(diff, 0):
        signed = {"statistic": None, "pvalue": None, "n": int(len(diff))}
    else:
        signed_test = wilcoxon(diff, alternative="two-sided", zero_method="wilcox")
        signed = {
            "statistic": float(signed_test.statistic),
            "pvalue": float(signed_test.pvalue),
            "n": int(len(diff)),
        }
    counts = {name: 0 for name in _STATE["kept"]}
    empty = 0
    monte = 0
    gaussian = 0
    for row in records:
        if not row["active"]:
            empty += 1
        for name in row["active"]:
            counts[name] = counts.get(name, 0) + 1
        for test in row["tests"]:
            if test["reference"] == "Monte Carlo":
                monte += 1
            elif test["reference"] == "Gaussian":
                gaussian += 1
    spatial_keys = ("geo", "town_sem", "multi")
    convergence = {}
    for key in spatial_keys:
        rhats = []
        ess_l = []
        ess_b = []
        mean_l = []
        tail = []
        accept = []
        for row in records:
            if row[key].get("method") != "Gibbs":
                continue
            rhats.append(max(row[key]["rhat_lambda"] + row[key]["rhat_beta"]))
            ess_l.extend(row[key]["ess_lambda"])
            ess_b.extend(row[key]["ess_beta"])
            mean_l.append(row[key]["mean_L"])
            tail.append(row[key]["p_L_gt_095"])
            accept.extend(row[key]["acceptance"])
        convergence[key] = {
            "fits": len(rhats),
            "rhat_within_1_025": int(sum(value <= 1.025 for value in rhats)),
            "max_rhat": None if not rhats else float(max(rhats)),
            "min_ess_lambda": None if not ess_l else float(min(ess_l)),
            "min_ess_beta": None if not ess_b else float(min(ess_b)),
            "mean_E_L": None if not mean_l else float(np.mean(mean_l)),
            "mean_P_L_gt_095": None if not tail else float(np.mean(tail)),
            "acceptance_min": None if not accept else float(min(accept)),
            "acceptance_max": None if not accept else float(max(accept)),
        }
    z_gap = []
    for row in records:
        for test in row["tests"]:
            if test["z"] is not None and test["code_z"] is not None:
                z_gap.append(abs(test["z"] - test["code_z"]))
    return {
        "n_splits": len(records),
        "table": table,
        "wilcoxon": signed,
        "mean_squared_error_diff": float(diff.mean()),
        "selection_counts": counts,
        "empty_active_sets": empty,
        "gaussian_tests": gaussian,
        "monte_carlo_tests": monte,
        "max_abs_z_gap_vs_package": None if not z_gap else float(max(z_gap)),
        "convergence": convergence,
        "mean_seconds": float(np.mean([row["seconds"] for row in records])),
        "unseen_holdout_tracts_mean": float(np.mean([row["fixed_effects"]["unseen_holdout_tracts"] for row in records])),
    }


def prepare() -> dict:
    if not DATA_PATH.exists():
        import urllib.request

        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCE_URL, DATA_PATH)
    data = load_boston()
    built = build_networks(data)
    description = {}
    label_map = {"town": data["town_id"], "radial": data["rad"]}
    for name in LAYER_ORDER:
        description[name] = layer_description(
            built["weights"][name], built["adjacency"][name], label_map.get(name)
        )
    within = {}
    for col, title in enumerate(("rm", "age", "crim", "zn", "indus", "log_lstat", "tax", "ptratio", "nox", "log_dis")):
        values = data["raw"][:, col]
        spreads = []
        for town in np.unique(data["town_id"]):
            block = values[data["town_id"] == town]
            spreads.append(float(block.max() - block.min()) if len(block) else 0.0)
        within[title] = {"towns_with_variation": int(sum(v > 1e-10 for v in spreads))}
    meta = {
        "n": int(len(data["y"])),
        "towns": int(len(np.unique(data["town_id"]))),
        "cmedv_min": float(data["cmedv"].min()),
        "cmedv_max": float(data["cmedv"].max()),
        "utm_x_range": [float(data["xy"][:, 0].min()), float(data["xy"][:, 0].max())],
        "utm_y_range": [float(data["xy"][:, 1].min()), float(data["xy"][:, 1].max())],
        "kept": built["kept"],
        "dropped": built["dropped"],
        "correlation": built["correlation"],
        "clique_check": built["clique_check"],
        "layers": description,
        "within_town_variation": within,
        "rad_values": {str(int(v)): int(np.sum(data["rad"] == v)) for v in np.unique(data["rad"])},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "networks.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return {"data": data, "networks": built, "kept": built["kept"], "meta": meta}


def main(n_run: int = N_SPLITS, adaptive: bool = False, only: int | None = None) -> None:
    global OUT_DIR
    if adaptive:
        OUT_DIR = ROOT / "results" / "boston_adaptive"
    state = prepare()
    state["adaptive"] = adaptive
    _init_worker(state)
    print(json.dumps(state["meta"], indent=2))
    base = np.random.SeedSequence(MASTER_SEED)
    seeds = [
        int(np.random.default_rng(child).integers(1, 2**31 - 1))
        for child in base.spawn(N_SPLITS)
    ]
    tasks = [{"split": i, "seed": seeds[i]} for i in range(n_run)]
    if only is not None:
        tasks = [task for task in tasks if task["split"] == only]
    records = []
    pending = []
    for task in tasks:
        path = OUT_DIR / f"split_{task['split']:02d}.json"
        if path.exists():
            records.append(json.loads(path.read_text(encoding="utf-8")))
            print(f"split {task['split']} loaded", flush=True)
        else:
            pending.append(task)
    if pending:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        workers = min(12, os.cpu_count() or 1, len(pending))
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(state,)) as pool:
            futures = {pool.submit(run_split, task): task["split"] for task in pending}
            for future in as_completed(futures):
                split_id = futures[future]
                record = future.result()
                records.append(record)
                (OUT_DIR / f"split_{split_id:02d}.json").write_text(
                    json.dumps(record, indent=2), encoding="utf-8"
                )
                ess = record["multi"].get("ess_lambda") or [float("nan")]
                rhat = record["multi"].get("rhat_lambda") or [float("nan")]
                print(
                    f"split {split_id} rmse ols {record['ols']['RMSE']:.4f} "
                    f"multi {record['multi']['RMSE']:.4f} active {record['active']} "
                    f"ess {min(ess):.0f} rhat {max(rhat):.3f} "
                    f"sec {record['seconds']:.1f}",
                    flush=True,
                )
    records.sort(key=lambda row: row["split"])
    summary = aggregate(records)
    summary["master_seed"] = MASTER_SEED
    summary["gibbs"] = {
        "n_iter": 8000 if adaptive else 6000,
        "burn": 4000 if adaptive else 3000,
        "n_chains": 4,
        "adapt_covariance": adaptive,
        "scaling": "(2.38^2/k) * empirical covariance + 1e-5 I" if adaptive else "isotropic",
        "initial_isotropic_draws": 50 if adaptive else 0,
    }
    summary["standardization"] = "training-split mean and population standard deviation"
    summary["response"] = "log(cmedv)"
    summary["rm_transform"] = "level"
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# Fixed layer sets for the multiplex ablation. A and C are the stored
# single-layer SEMs; F is the stored screened Multi-SEM. B, D, and E are fit
# on the same 50 splits.
ABLATION_MODELS = (
    ("A_geo", ("geo",)),
    ("B_resolvent", ("resolvent",)),
    ("C_town", ("town",)),
    ("D_geo_town", ("geo", "town")),
    ("E_geo_resolvent", ("geo", "resolvent")),
    ("F_screened", None),
)


def build_ablation_jobs(records: list[dict], gibbs: dict | None = None) -> list[dict]:
    """Ablation evaluation loop: one Gibbs fit for each new layer set and split."""
    gibbs = dict(gibbs or {})
    jobs = []
    for record in records:
        for name, layers in ABLATION_MODELS:
            if layers is None or name in {"A_geo", "C_town"}:
                continue
            jobs.append({
                "study": "ablation",
                "model": name,
                "layers": list(layers),
                "split": int(record["split"]),
                "seed": int(record["seed"]),
                "screen": False,
                "gibbs": gibbs,
            })
    return jobs


def _revision_module():
    import run_revision_tasks as revision

    return revision


def write_ablation_summary(path: Path | None = None) -> list[dict]:
    """Aggregate the 50-split ablation. New Gibbs fits are reused from results/revision/fits."""
    revision = _revision_module()
    records = revision._load_adaptive_records()
    rows = revision.aggregate_ablation(records, revision.ROOT / "results" / "revision")
    path = path or (Path(__file__).resolve().parent / "results" / "boston_adaptive" / "ablation_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    expected = {
        "A. Geographic SEM": 0.1403,
        "B. Geographic resolvent only": 0.1377,
        "C. Municipal cliques only": 0.1499,
        "D. Geographic + municipal": 0.1384,
        "E. Geographic + resolvent": 0.1383,
        "F. Screened Multi-SEM": 0.1375,
    }
    for row in rows:
        got = round(row["mean_rmse"], 4)
        target = expected[row["model"]]
        if got != target:
            raise RuntimeError(f"{row['model']} mean RMSE {got} != {target}")
    print(f"ablation summary matches Table S9 -> {path}", flush=True)
    return rows


def write_sensitivity_summary(path: Path | None = None) -> dict:
    revision = _revision_module()
    records = revision._load_adaptive_records()
    tables = revision.aggregate_sensitivity(records, revision.ROOT / "results" / "revision")
    path = path or (Path(__file__).resolve().parent / "results" / "boston_adaptive" / "sensitivity_summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tables, indent=2), encoding="utf-8")
    checks = {
        "chas": (0.1378, 0.1904, 0.1410),
        "censor": (0.1289, 0.1806, 0.1297),
    }
    for key, (multi, ols, geo) in checks.items():
        by_name = {row["model"]: round(row["mean_rmse"], 4) for row in tables[key]["rows"]}
        got = (by_name["Screened Multi-SEM"], by_name["OLS"], by_name["Geographic SEM"])
        if got != (multi, ols, geo):
            raise RuntimeError(f"{key} RMSE {got} != {(multi, ols, geo)}")
    print(f"sensitivity summary matches Table S10 -> {path}", flush=True)
    return tables


def write_nadeau_bengio(path: Path | None = None) -> dict:
    revision = _revision_module()
    records = revision._load_adaptive_records()
    out = revision.task1(records)
    path = path or (Path(__file__).resolve().parent / "results" / "boston_adaptive" / "nadeau_bengio.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Nadeau-Bengio statistics -> {path}", flush=True)
    return out


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    if "--ablation" in argv:
        write_ablation_summary()
    elif "--include-chas" in argv or "--drop-censored" in argv:
        write_sensitivity_summary()
    elif "--nadeau-bengio" in argv:
        write_nadeau_bengio()
    else:
        adaptive = "--adaptive" in argv
        only = None
        if "--only" in argv:
            only = int(argv[argv.index("--only") + 1])
        numeric = [arg for arg in argv if arg.isdigit()]
        requested = int(numeric[0]) if numeric else N_SPLITS
        main(requested, adaptive=adaptive, only=only)
