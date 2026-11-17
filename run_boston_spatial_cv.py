"""Universal kriging, leave-one-town-out, and contiguous spatial-block CV for Boston.

Point predictions are pooled over all 506 tracts. Random-split RMSE for the
network models is read from the existing 50-split archive. Universal kriging
is fit on those same splits.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

import run_boston as boston
from mnreg.model import metrics_rmse_mae

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "boston"
CACHE = OUT / "spatial_cv"
MODELS = ("ols", "fixed_effects", "universal_kriging", "town", "geo", "multi")


def _kmeans(xy: np.ndarray, k: int = 10, n_init: int = 50, seed: int = 2026) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = (xy - xy.mean(axis=0)) / np.maximum(xy.std(axis=0), 1e-8)
    best_inertia = None
    best = None
    for _ in range(n_init):
        centers = z[rng.choice(len(z), size=k, replace=False)].copy()
        labels = np.zeros(len(z), dtype=int)
        for _step in range(50):
            distance = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distance.argmin(axis=1)
            updated = centers.copy()
            for cluster in range(k):
                members = z[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
            if np.allclose(updated, centers):
                centers = updated
                break
            centers = updated
        inertia = float(((z - centers[labels]) ** 2).sum())
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best = labels.copy()
    return best


def _matern32(distance: np.ndarray, phi: float) -> np.ndarray:
    scaled = phi * distance
    return (1.0 + scaled) * np.exp(-scaled)


def _uk_predict(y_train, x_train, xy_train, x_test, xy_test) -> np.ndarray:
    """REML Matérn-3/2 universal kriging. Coordinates are in kilometres."""
    train_km = xy_train / 1000.0
    test_km = xy_test / 1000.0
    distance = cdist(train_km, train_km)
    cross = cdist(test_km, train_km)
    np.fill_diagonal(distance, 0.0)

    def objective(theta):
        phi = np.exp(theta[0])
        nugget = 1.0 / (1.0 + np.exp(-theta[1]))
        kernel = _matern32(distance, phi)
        weight = (1.0 - nugget) * kernel + nugget * np.eye(len(y_train))
        weight = weight + 1e-8 * np.eye(len(y_train))
        try:
            chol = np.linalg.cholesky(weight)
        except np.linalg.LinAlgError:
            return 1e12
        whitened_x = np.linalg.solve(chol, x_train)
        whitened_y = np.linalg.solve(chol, y_train)
        gram = whitened_x.T @ whitened_x
        try:
            beta = np.linalg.solve(gram, whitened_x.T @ whitened_y)
        except np.linalg.LinAlgError:
            return 1e12
        residual = whitened_y - whitened_x @ beta
        rank = x_train.shape[1]
        sigma2 = float(residual @ residual) / (len(y_train) - rank)
        logdet_w = 2.0 * np.sum(np.log(np.diag(chol)))
        sign, logdet_g = np.linalg.slogdet(gram)
        if sign <= 0 or not np.isfinite(sigma2) or sigma2 <= 0:
            return 1e12
        return 0.5 * ((len(y_train) - rank) * np.log(sigma2) + logdet_w + logdet_g)

    start = np.array([np.log(0.3), 0.0])
    fit = minimize(objective, start, method="L-BFGS-B")
    phi = float(np.exp(fit.x[0]))
    nugget = float(1.0 / (1.0 + np.exp(-fit.x[1])))
    kernel = _matern32(distance, phi)
    weight = (1.0 - nugget) * kernel + nugget * np.eye(len(y_train)) + 1e-8 * np.eye(len(y_train))
    chol = np.linalg.cholesky(weight)
    whitened_x = np.linalg.solve(chol, x_train)
    whitened_y = np.linalg.solve(chol, y_train)
    beta = np.linalg.solve(whitened_x.T @ whitened_x, whitened_x.T @ whitened_y)
    train_resid = y_train - x_train @ beta
    solved = np.linalg.solve(chol.T, np.linalg.solve(chol, train_resid))
    cross_kernel = (1.0 - nugget) * _matern32(cross, phi)
    return x_test @ beta + cross_kernel @ solved


def _design(data, train, test):
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = np.where(raw[train].std(axis=0, ddof=0) < 1e-12, 1.0, raw[train].std(axis=0, ddof=0))
    scaled = (raw - center) / spread
    x_train = np.column_stack([np.ones(len(train)), scaled[train]])
    x_test = np.column_stack([np.ones(len(test)), scaled[test]])
    return x_train, x_test, data["y"][train], data["y"][test]


def _predict_fold(data, networks, train, test, seed: int) -> dict:
    x_train, x_test, y_train, y_test = _design(data, train, test)
    ols_beta, _, _ = boston.ols(y_train, x_train)
    fe_hat, _info = boston.town_fixed_effects(
        y_train, x_train, train, x_test, data["town_id"][train], data["town_id"][test]
    )
    uk_hat = _uk_predict(y_train, x_train, data["xy"][train], x_test, data["xy"][test])
    induced = []
    for name in networks["kept"]:
        weights, adjacency = boston.induced_pair(networks["weights"][name], train)
        induced.append((name, weights, adjacency))
    screened = boston.screen_split(y_train, x_train, induced, np.random.default_rng(seed))
    active = [row["name"] for row in screened if row["selected"]] or [networks["kept"][0]]
    by_name = {name: adj for name, _w, adj in induced}

    def krige(layers, gibbs_seed):
        mats = [by_name[name] for name in layers]
        fit = boston.fit_spatial(y_train, x_train, mats, layers, gibbs_seed, adaptive=True)
        full = [networks["adjacency"][name] for name in layers]
        return boston.precision_kriging(
            y_train, x_train, x_test, train, test, fit["lambda"], fit["beta"], full
        )

    return {
        "y": y_test,
        "ols": x_test @ ols_beta,
        "fixed_effects": fe_hat,
        "universal_kriging": uk_hat,
        "town": krige(["town"], seed + 1),
        "geo": krige(["geo"], seed + 2),
        "multi": krige(active, seed + 3),
        "active": active,
    }


def _scores(y, pred) -> dict:
    metrics = metrics_rmse_mae(y, pred)
    return {"rmse": float(metrics["RMSE"]), "mae": float(metrics["MAE"]), "n": int(len(y))}


def _load_fold(kind: str, key: str):
    path = CACHE / f"{kind}_{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_fold(kind: str, key: str, payload: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{kind}_{key}.json").write_text(json.dumps(payload), encoding="utf-8")


def _accumulate(store, test, predicted):
    for name in MODELS:
        store[name][test] = np.asarray(predicted[name], dtype=float)


def random_split_kriging(data) -> dict:
    rows = []
    archive = ROOT / "results" / "boston_adaptive"
    for path in sorted(archive.glob("split_*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        cached = _load_fold("random", f"{record['split']:02d}")
        if cached is None:
            part = np.random.default_rng([int(record["seed"]), 1])
            train = np.sort(part.choice(boston.N, size=boston.N_TRAIN, replace=False))
            test = np.sort(np.setdiff1d(np.arange(boston.N), train))
            x_train, x_test, y_train, _y_test = _design(data, train, test)
            hat = _uk_predict(y_train, x_train, data["xy"][train], x_test, data["xy"][test])
            metrics = _scores(data["y"][test], hat)
            cached = {"split": record["split"], **metrics}
            _save_fold("random", f"{record['split']:02d}", cached)
            print(f"random uk split {record['split']} rmse {metrics['rmse']:.4f}", flush=True)
        rows.append(cached)
    return {
        "rmse": float(np.mean([row["rmse"] for row in rows])),
        "mae": float(np.mean([row["mae"] for row in rows])),
        "splits": len(rows),
    }


def _worker_init(state):
    global _WORKER
    _WORKER = state


def _worker_fold(task):
    kind, key, test = task
    test = np.asarray(test, dtype=int)
    train = np.setdiff1d(np.arange(boston.N), test)
    predicted = _predict_fold(_WORKER["data"], _WORKER["networks"], train, test, seed=2026 + int(key))
    payload = {
        "key": key,
        "n_test": int(len(test)),
        "test": test.astype(int).tolist(),
        "active": predicted["active"],
    }
    for name in MODELS:
        payload[name] = np.asarray(predicted[name], dtype=float).tolist()
    y = _WORKER["data"]["y"][test]
    payload["multi_rmse"] = _scores(y, predicted["multi"])["rmse"]
    return kind, payload


def run_scheme(data, networks, folds: list[tuple[str, np.ndarray]], kind: str) -> dict:
    store = {name: np.full(boston.N, np.nan) for name in MODELS}
    pending = []
    for key, test in folds:
        cached = _load_fold(kind, key)
        if cached is None:
            pending.append((kind, key, test.astype(int).tolist()))
        else:
            _accumulate(store, np.asarray(cached["test"], dtype=int), cached)
    if pending:
        workers = min(8, os.cpu_count() or 4, len(pending))
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=({"data": data, "networks": networks},),
        ) as pool:
            for kind_name, payload in pool.map(_worker_fold, pending, chunksize=1):
                _save_fold(kind_name, payload["key"], payload)
                _accumulate(store, np.asarray(payload["test"], dtype=int), payload)
                print(f"{kind_name} {payload['key']} n={payload['n_test']} multi {payload['multi_rmse']:.3f}", flush=True)
    y = data["y"]
    if any(np.isnan(store[name]).any() for name in MODELS):
        missing = {name: int(np.isnan(store[name]).sum()) for name in MODELS}
        raise RuntimeError(f"incomplete {kind} predictions: {missing}")
    return {name: _scores(y, store[name]) for name in MODELS}


def published_random_split() -> dict:
    summary = json.loads((ROOT / "results" / "boston_adaptive" / "summary.json").read_text(encoding="utf-8"))
    by_name = {row["model"]: row for row in summary["table"]}
    return {
        "ols": by_name["OLS (no network)"]["mean_rmse"],
        "fixed_effects": by_name["Municipal fixed effects"]["mean_rmse"],
        "town": by_name["Single SEM (town boundaries)"]["mean_rmse"],
        "geo": by_name["Single SEM (geographic 6-NN)"]["mean_rmse"],
        "multi": by_name["Multi-SEM (screened, precision kriging)"]["mean_rmse"],
    }


def main() -> None:
    data = boston.load_boston()
    networks = boston.build_networks(data)
    print(f"towns {len(np.unique(data['town_id']))} kept {networks['kept']}", flush=True)
    uk_random = random_split_kriging(data)
    towns = sorted(int(v) for v in np.unique(data["town_id"]))
    loto_folds = [
        (str(town), np.flatnonzero(data["town_id"] == town))
        for town in towns
    ]
    labels = _kmeans(data["xy"], k=10, n_init=50, seed=2026)
    block_folds = [
        (str(cluster), np.flatnonzero(labels == cluster))
        for cluster in range(10)
    ]
    print("block sizes", [int(len(test)) for _key, test in block_folds], flush=True)
    loto = run_scheme(data, networks, loto_folds, "loto")
    block = run_scheme(data, networks, block_folds, "block")
    random_rmse = published_random_split()
    random_rmse["universal_kriging"] = uk_random["rmse"]
    summary = {
        "n": int(boston.N),
        "towns": len(towns),
        "random_interpolation_rmse": random_rmse,
        "universal_kriging_random_mae": uk_random["mae"],
        "loto": loto,
        "spatial_block": block,
        "block_sizes": [int(len(test)) for _key, test in block_folds],
        "sampler": {"n_iter": 8000, "burn": 4000, "n_chains": 4, "adapt_covariance": True, "seed": 2026},
        "pooling": "one prediction per tract, concatenated over folds",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "spatial_cv_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
