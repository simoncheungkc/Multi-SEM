"""Revision computations requested by the referee reports.

Task 1. Nadeau-Bengio corrected resampled t-test on the stored 50 splits.
Task 2. Boston multiplex ablation (new layer sets on those same splits).
Task 3. Prior versus posterior geometry of L = ||lambda||_1, plus a Beta(1, 2) refit.
Task 4. Charles River dummy and ceiling-censoring sensitivity.
Task 5. Opposite-sign Monte Carlo (implemented in run_simulation.py).
Task 6. Moran I on the spatially filtered training innovations.
"""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

import run_boston as boston
import run_simulation as simulation
from mnreg.model import assemble_V, ols

ROOT = Path(__file__).resolve().parent
ADAPTIVE = ROOT / "results" / "boston_adaptive"
REV = ROOT / "results" / "revision"
FIG_PATH = ROOT / "results" / "boston" / "figure_s10_prior_vs_posterior_L.png"

_STATE: dict = {}


def nadeau_bengio(mse_base, mse_target, n_t, n_star, name):
    d = np.asarray(mse_base, dtype=float) - np.asarray(mse_target, dtype=float)
    r = int(d.size)
    d_bar = float(np.mean(d))
    s2 = float(np.var(d, ddof=1))
    factor = (1.0 / r) + (float(n_star) / float(n_t))
    se = float(np.sqrt(factor * s2))
    t_stat = d_bar / se
    from scipy import stats

    p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=r - 1)))
    crit = float(stats.t.ppf(0.975, df=r - 1))
    return {
        "name": name,
        "R": r,
        "n_t": int(n_t),
        "n_star": int(n_star),
        "correction_factor": factor,
        "mean_mse_reduction": d_bar,
        "variance_d": s2,
        "se_corrected": se,
        "t": t_stat,
        "df": r - 1,
        "p": p_val,
        "ci95": [d_bar - crit * se, d_bar + crit * se],
        "wins": int(np.sum(d > 0)),
        "ties": int(np.sum(np.isclose(d, 0.0))),
        "losses": int(np.sum(d < 0)),
    }


def _load_adaptive_records():
    records = []
    for path in sorted(ADAPTIVE.glob("split_*.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    records.sort(key=lambda row: row["split"])
    if len(records) != 50:
        raise RuntimeError(f"expected 50 adaptive splits, found {len(records)}")
    return records


def _mse(record, key):
    return float(record[key]["RMSE"]) ** 2


def task1(records):
    mse_ols = np.array([_mse(row, "ols") for row in records])
    mse_geo = np.array([_mse(row, "geo") for row in records])
    mse_multi = np.array([_mse(row, "multi") for row in records])
    out = {
        "source": "results/boston_adaptive/split_*.json",
        "loss": "holdout MSE = RMSE^2",
        "comparisons": [
            nadeau_bengio(mse_ols, mse_multi, 405, 101, "Multi-SEM vs OLS"),
            nadeau_bengio(mse_geo, mse_multi, 405, 101, "Multi-SEM vs Geographic SEM"),
        ],
    }
    max_draw = max(float(row["multi"].get("max_L", 0.0)) for row in records)
    split5 = next(row for row in records if int(row["split"]) == 5)
    out["max_L_split5"] = float(split5["multi"]["max_L"])
    out["max_L_across_splits"] = float(max_draw)
    out["mean_rmse"] = {
        "ols": float(np.mean([row["ols"]["RMSE"] for row in records])),
        "geo": float(np.mean([row["geo"]["RMSE"] for row in records])),
        "multi": float(np.mean([row["multi"]["RMSE"] for row in records])),
        "multi_mean": float(np.mean([row["multi_mean"]["RMSE"] for row in records])),
        "town": float(np.mean([row["town_sem"]["RMSE"] for row in records])),
    }
    return out


def _indices(seed, n, n_train):
    part = np.random.default_rng([int(seed), 1])
    train = np.sort(part.choice(n, size=n_train, replace=False))
    test = np.sort(np.setdiff1d(np.arange(n), train))
    return train, test


def _design(data, train, test, extra=None):
    raw = np.asarray(data["raw"], dtype=float)
    if extra is not None:
        raw = np.column_stack([raw, np.asarray(extra, dtype=float)])
    center = raw[train].mean(axis=0)
    spread = raw[train].std(axis=0, ddof=0)
    spread = np.where(spread < 1e-12, 1.0, spread)
    scaled = (raw - center) / spread
    x_train = np.column_stack([np.ones(len(train)), scaled[train]])
    x_test = np.column_stack([np.ones(len(test)), scaled[test]])
    return x_train, x_test, data["y"][train], data["y"][test]


def _slim(fit, metrics, layers, seconds):
    return {
        "RMSE": float(metrics["RMSE"]),
        "MAE": float(metrics["MAE"]),
        "corr": None if not np.isfinite(metrics["corr"]) else float(metrics["corr"]),
        "lambda": np.asarray(fit.get("lambda", []), dtype=float).tolist(),
        "beta": np.asarray(fit.get("beta", []), dtype=float).tolist(),
        "names": list(layers),
        "mean_L": float(fit.get("mean_L", 0.0)),
        "max_L": float(fit.get("max_L", 0.0)),
        "rhat_lambda": [float(v) for v in fit.get("rhat_lambda", [])],
        "ess_lambda": [float(v) for v in fit.get("ess_lambda", [])],
        "acceptance": np.asarray(fit.get("acceptance", []), dtype=float).tolist(),
        "method": fit.get("method", "Gibbs"),
        "seconds": float(seconds),
    }


def _init_worker(state):
    _STATE.clear()
    _STATE.update(state)


def _score_layers(bundle, train, test, layers, seed, screen, covariate_names):
    data = bundle["data"]
    networks = bundle["networks"]
    extra = bundle.get("extra")
    x_train, x_test, y_train, y_test = _design(data, train, test, extra)
    induced = {}
    for name in boston.LAYER_ORDER:
        induced[name] = boston.induced_pair(networks["weights"][name], train)[1]
    chosen = list(layers)
    tests = None
    if screen:
        named = []
        for name in networks["kept"]:
            weights, adjacency = boston.induced_pair(networks["weights"][name], train)
            named.append((name, weights, adjacency))
        rng = np.random.default_rng([int(seed), 2, 9])
        tests = boston.screen_split(y_train, x_train, named, rng)
        chosen = [row["name"] for row in tests if row["selected"]]
    started = time.perf_counter()
    gibbs = bundle.get("gibbs") or {}
    if chosen:
        mats = [induced[name] for name in chosen]
        fit = boston.fit_spatial(
            y_train,
            x_train,
            mats,
            chosen,
            int(bundle["gibbs_seed"]),
            covariate_names=covariate_names,
            **gibbs,
        )
        full = [networks["adjacency"][name] for name in chosen]
        yhat = boston.precision_kriging(
            y_train, x_train, x_test, train, test, fit["lambda"], fit["beta"], full
        )
    else:
        beta, _, _ = ols(y_train, x_train)
        fit = {
            "lambda": [],
            "beta": beta,
            "names": [],
            "mean_L": 0.0,
            "max_L": 0.0,
            "rhat_lambda": [],
            "ess_lambda": [],
            "acceptance": [],
            "method": "OLS",
        }
        yhat = x_test @ beta
    metrics = boston.metrics_rmse_mae(y_test, yhat)
    elapsed = time.perf_counter() - started
    packed = _slim(fit, metrics, chosen, elapsed)
    if tests is not None:
        packed["active"] = chosen
        packed["screen_pvalues"] = {row["name"]: row["pvalue"] for row in tests}
    if bundle.get("save_draws") and chosen:
        packed["_draws"] = np.asarray(fit["lambda"], dtype=float)
    return packed


def fit_job(job):
    try:
        started = time.perf_counter()
        study = job["study"]
        if study == "split5":
            payload = _fit_split5(job)
        else:
            payload = _fit_predictive(job)
        payload["seconds_wall"] = time.perf_counter() - started
        path = Path(job["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        draws = payload.pop("_draws", None)
        path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
        if draws is not None:
            np.savez_compressed(path.with_suffix(".npz"), draws_lambda=draws)
        return {
            "ok": True,
            "study": study,
            "split": job.get("split"),
            "model": job.get("model"),
            "rmse": payload.get("RMSE"),
            "seconds": payload["seconds_wall"],
        }
    except Exception:
        return {
            "ok": False,
            "study": job.get("study"),
            "split": job.get("split"),
            "model": job.get("model"),
            "trace": traceback.format_exc(),
        }


def _fit_predictive(job):
    if job["study"] == "censor":
        bundle_src = _STATE["censor"]
        n = bundle_src["n"]
        n_train = bundle_src["n_train"]
    elif job["study"] == "chas":
        bundle_src = _STATE["chas"]
        n = boston.N
        n_train = boston.N_TRAIN
    else:
        bundle_src = _STATE["full"]
        n = boston.N
        n_train = boston.N_TRAIN
    train, test = _indices(job["seed"], n, n_train)
    gibbs_seed = int(
        np.random.default_rng([int(job["seed"]), 17, _model_code(job["model"])]).integers(1, 2**31 - 1)
    )
    bundle = {
        "data": bundle_src["data"],
        "networks": bundle_src["networks"],
        "extra": bundle_src.get("extra"),
        "gibbs": job.get("gibbs") or {},
        "gibbs_seed": gibbs_seed,
    }
    names = bundle_src.get("covariate_names")
    return _score_layers(
        bundle,
        train,
        test,
        job.get("layers") or [],
        job["seed"],
        bool(job.get("screen")),
        names,
    )


def _model_code(name):
    codes = {
        "B_resolvent": 2,
        "D_geo_town": 4,
        "E_geo_resolvent": 5,
        "geo": 11,
        "multi": 12,
        "split5_gaussian": 21,
        "split5_beta": 22,
    }
    return int(codes.get(name, 19))


def _fit_split5(job):
    from mnreg.model import GibbsConfig, gibbs_multi_sem

    record = job["record"]
    data = _STATE["full"]["data"]
    networks = _STATE["full"]["networks"]
    train, _test = _indices(record["seed"], boston.N, boston.N_TRAIN)
    x_train, _, y_train, _ = _design(data, train, _test)
    layers = list(record["active"])
    induced = [boston.induced_pair(networks["weights"][name], train)[1] for name in layers]
    gibbs_rng = np.random.default_rng([int(record["seed"]), 3])
    chain_seeds = gibbs_rng.integers(1, 2**31 - 1, size=4)
    gibbs = job.get("gibbs") or {}
    config = GibbsConfig(
        n_iter=gibbs.get("n_iter", 8000),
        burn=gibbs.get("burn", 4000),
        n_chains=gibbs.get("n_chains", 4),
        seed=int(chain_seeds[2]),
        adapt_covariance=True,
        lambda_prior=job["lambda_prior"],
    )
    result = gibbs_multi_sem(
        y_train, x_train, induced, names=layers, config=config
    )
    lam = result.samples["lambda"]
    flat = lam.reshape(-1, lam.shape[-1])
    total = np.sum(np.abs(flat), axis=1)
    return {
        "split": 5,
        "lambda_prior": job["lambda_prior"],
        "names": layers,
        "lambda_mean": flat.mean(axis=0).astype(float).tolist(),
        "mean_L": float(total.mean()),
        "median_L": float(np.median(total)),
        "mode_L": _hist_mode(total),
        "q025_L": float(np.quantile(total, 0.025)),
        "q975_L": float(np.quantile(total, 0.975)),
        "max_L": float(total.max()),
        "p_L_gt_095": float(np.mean(total > 0.95)),
        "acceptance": result.acceptance.astype(float).tolist(),
        "stored_lambda_mean": record["multi"]["lambda"],
        "stored_mean_L": record["multi"]["mean_L"],
        "_draws": flat.astype(np.float64),
    }


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.astype(float).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"not serializable: {type(value)}")


def _hist_mode(values, bins=80):
    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    slot = int(np.argmax(counts))
    return float(0.5 * (edges[slot] + edges[slot + 1]))


def task6(records):
    data = _STATE["full"]["data"]
    networks = _STATE["full"]["networks"]
    rows = []
    for record in records:
        train, test = _indices(record["seed"], boston.N, boston.N_TRAIN)
        x_train, _, y_train, _ = _design(data, train, test)
        beta_ols, resid, _ = ols(y_train, x_train)
        del beta_ols
        layers = list(record["active"])
        induced = {
            name: boston.induced_pair(networks["weights"][name], train)[1]
            for name in boston.LAYER_ORDER
        }
        mats = [induced[name] for name in layers]
        lam = np.asarray(record["multi"]["lambda"], dtype=float)
        beta = np.asarray(record["multi"]["beta"], dtype=float)
        innov = assemble_V(lam, mats) @ (y_train - x_train @ beta)
        ones = np.ones((len(train), 1))
        entry = {"split": int(record["split"]), "active": layers, "ols": {}, "filtered": {}}
        for name in boston.LAYER_ORDER:
            ols_stat = boston.moran_proposition1(resid, induced[name], x_train)
            fil_stat = boston.moran_proposition1(innov, induced[name], ones)
            entry["ols"][name] = {"I": ols_stat["I"], "p": ols_stat["pvalue"]}
            entry["filtered"][name] = {"I": fil_stat["I"], "p": fil_stat["pvalue"]}
        rows.append(entry)
    summary = {}
    for name in boston.LAYER_ORDER:
        ols_I = np.array([row["ols"][name]["I"] for row in rows])
        fil_I = np.array([row["filtered"][name]["I"] for row in rows])
        fil_p = np.array([row["filtered"][name]["p"] for row in rows])
        summary[name] = {
            "ols_mean_I": float(np.mean(ols_I)),
            "ols_split5_I": float(next(row["ols"][name]["I"] for row in rows if row["split"] == 5)),
            "filtered_mean_I": float(np.mean(fil_I)),
            "filtered_min_I": float(np.min(fil_I)),
            "filtered_max_I": float(np.max(fil_I)),
            "filtered_split5_I": float(next(row["filtered"][name]["I"] for row in rows if row["split"] == 5)),
            "filtered_split5_p": float(next(row["filtered"][name]["p"] for row in rows if row["split"] == 5)),
            "filtered_median_p": float(np.median(fil_p)),
            "filtered_frac_p_gt_030": float(np.mean(fil_p > 0.30)),
        }
    return {"per_split": rows, "summary": summary}


def _load_chas():
    values = []
    for line in boston.DATA_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 21 or not parts[0].isdigit():
            continue
        tail = parts[-19:]
        values.append(float(tail[9]))
    chas = np.asarray(values, dtype=float)
    if chas.shape != (boston.N,):
        raise RuntimeError(f"CHAS length {chas.size}")
    return chas


def _censored_bundle(full_data):
    keep = np.flatnonzero(~np.isclose(full_data["cmedv"], 50.0))
    if boston.N - len(keep) != 16:
        raise RuntimeError(f"expected 16 censored tracts, dropped {boston.N - len(keep)}")
    sub = {
        "town": full_data["town"][keep],
        "town_id": full_data["town_id"][keep],
        "lon": full_data["lon"][keep],
        "lat": full_data["lat"][keep],
        "rad": full_data["rad"][keep],
        "y": full_data["y"][keep],
        "cmedv": full_data["cmedv"][keep],
        "raw": full_data["raw"][keep],
        "xy": full_data["xy"][keep],
    }
    built = boston.build_networks(sub)
    return {
        "data": sub,
        "networks": built,
        "n": int(len(keep)),
        "n_train": 392,
        "n_test": 98,
        "index_in_full": keep,
        "covariate_names": list(boston.COVARIATE_NAMES),
    }


def _prepare(gibbs):
    state = boston.prepare()
    chas = _load_chas()
    full = {
        "data": state["data"],
        "networks": state["networks"],
        "extra": None,
        "covariate_names": list(boston.COVARIATE_NAMES),
    }
    chas_bundle = {
        "data": state["data"],
        "networks": state["networks"],
        "extra": chas,
        "covariate_names": list(boston.COVARIATE_NAMES) + ["chas"],
    }
    censor = _censored_bundle(state["data"])
    _init_worker({"full": full, "chas": chas_bundle, "censor": censor, "gibbs": gibbs})
    return {"full": full, "chas": chas_bundle, "censor": censor}


def _jobs(records, gibbs, out_dir: Path, include_predictive=True):
    jobs = []
    if include_predictive:
        for job in boston.build_ablation_jobs(records, gibbs):
            job["path"] = str(out_dir / "fits" / f"ablation__{job['model']}__split{job['split']:02d}.json")
            jobs.append(job)
        for record in records:
            seed = int(record["seed"])
            split = int(record["split"])
            for model, layers, screen in (
                ("geo", ["geo"], False),
                ("multi", [], True),
            ):
                for study, _bundle_name in (("chas", "chas"), ("censor", "censor")):
                    jobs.append({
                        "study": study,
                        "model": model,
                        "layers": layers,
                        "split": split,
                        "seed": seed,
                        "screen": screen,
                        "gibbs": gibbs,
                        "path": str(out_dir / "fits" / f"{study}__{model}__split{split:02d}.json"),
                    })
    split5 = next(row for row in records if int(row["split"]) == 5)
    for prior, model in (("gaussian", "split5_gaussian"), ("beta_l1", "split5_beta")):
        jobs.append({
            "study": "split5",
            "model": model,
            "split": 5,
            "lambda_prior": prior,
            "record": {
                "seed": int(split5["seed"]),
                "active": list(split5["active"]),
                "multi": {
                    "lambda": split5["multi"]["lambda"],
                    "mean_L": split5["multi"]["mean_L"],
                },
            },
            "gibbs": gibbs,
            "path": str(out_dir / f"{model}.json"),
        })
    pending = [job for job in jobs if not Path(job["path"]).exists()]
    return jobs, pending


def _run_pool(pending, workers):
    if not pending:
        return []
    from concurrent.futures import ProcessPoolExecutor, as_completed

    failures = []
    done = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(_STATE,),
    ) as pool:
        futures = {pool.submit(fit_job, job): job for job in pending}
        for future in as_completed(futures):
            done += 1
            info = future.result()
            print(f"[{done}/{len(pending)}] {info}", flush=True)
            if not info.get("ok"):
                failures.append(info)
    return failures


def _read_fit(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _table_from_rmse(name, rmse, mae, ols_rmse, geo_rmse, n_t, n_star):
    rmse = np.asarray(rmse, dtype=float)
    mae = np.asarray(mae, dtype=float)
    versus_geo = None
    if geo_rmse is not None and not np.allclose(rmse, geo_rmse):
        versus_geo = nadeau_bengio(np.asarray(geo_rmse) ** 2, rmse ** 2, n_t, n_star, name)
    elif geo_rmse is not None:
        versus_geo = {"name": name, "t": None, "p": None, "wins": 0, "note": "baseline"}
    return {
        "model": name,
        "mean_rmse": float(np.mean(rmse)),
        "mean_mae": float(np.mean(mae)),
        "wins_vs_ols": int(np.sum(rmse < np.asarray(ols_rmse) - 1e-15)),
        "nadeau_bengio_vs_geo": versus_geo,
    }


def aggregate_ablation(records, out_dir: Path):
    labels = {
        "A_geo": "A. Geographic SEM",
        "B_resolvent": "B. Geographic resolvent only",
        "C_town": "C. Municipal cliques only",
        "D_geo_town": "D. Geographic + municipal",
        "E_geo_resolvent": "E. Geographic + resolvent",
        "F_screened": "F. Screened Multi-SEM",
    }
    stored = {
        "A_geo": ("geo", "geo"),
        "C_town": ("town_sem", "town_sem"),
        "F_screened": ("multi", "multi"),
    }
    ols = np.array([row["ols"]["RMSE"] for row in records])
    geo = np.array([row["geo"]["RMSE"] for row in records])
    rows = []
    for key, label in labels.items():
        if key in stored:
            src, _ = stored[key]
            rmse = np.array([row[src]["RMSE"] for row in records])
            mae = np.array([row[src]["MAE"] for row in records])
        else:
            rmse, mae = [], []
            for record in records:
                path = out_dir / "fits" / f"ablation__{key}__split{int(record['split']):02d}.json"
                fit = _read_fit(path)
                rmse.append(fit["RMSE"])
                mae.append(fit["MAE"])
            rmse = np.asarray(rmse)
            mae = np.asarray(mae)
        rows.append(_table_from_rmse(label, rmse, mae, ols, geo, 405, 101))
    return rows


def aggregate_sensitivity(records, out_dir: Path):
    """Pair each sensitivity design with OLS, geographic SEM, and screened Multi-SEM."""
    tables = {}
    designs = {
        "chas": (405, 101, "Charles River dummy included"),
        "censor": (392, 98, "16 ceiling-censored tracts excluded"),
    }
    for study, (n_t, n_star, title) in designs.items():
        ols_rmse, geo_rmse, multi_rmse = [], [], []
        geo_mae, multi_mae, ols_mae = [], [], []
        for record in records:
            # OLS is deterministic given the split. Recompute is unnecessary for CHAS
            # because the fit file is the spatial model; OLS is stored by a side file
            # when present and otherwise computed from the geo file's companion.
            geo = _read_fit(out_dir / "fits" / f"{study}__geo__split{int(record['split']):02d}.json")
            multi = _read_fit(out_dir / "fits" / f"{study}__multi__split{int(record['split']):02d}.json")
            geo_rmse.append(geo["RMSE"])
            geo_mae.append(geo["MAE"])
            multi_rmse.append(multi["RMSE"])
            multi_mae.append(multi["MAE"])
            ols_path = out_dir / "fits" / f"{study}__ols__split{int(record['split']):02d}.json"
            if ols_path.exists():
                ols_fit = _read_fit(ols_path)
                ols_rmse.append(ols_fit["RMSE"])
                ols_mae.append(ols_fit["MAE"])
            else:
                raise RuntimeError(f"missing OLS baseline {ols_path.name}")
        ols_rmse = np.asarray(ols_rmse)
        geo_rmse = np.asarray(geo_rmse)
        multi_rmse = np.asarray(multi_rmse)
        tables[study] = {
            "title": title,
            "n_t": n_t,
            "n_star": n_star,
            "rows": [
                _table_from_rmse("OLS", ols_rmse, ols_mae, ols_rmse, None, n_t, n_star),
                _table_from_rmse("Geographic SEM", geo_rmse, geo_mae, ols_rmse, geo_rmse, n_t, n_star),
                _table_from_rmse("Screened Multi-SEM", multi_rmse, multi_mae, ols_rmse, geo_rmse, n_t, n_star),
            ],
        }
        # Geographic versus itself is not a test. Drop that cell.
        tables[study]["rows"][1]["nadeau_bengio_vs_geo"] = None
        multi_vs_ols = nadeau_bengio(ols_rmse ** 2, multi_rmse ** 2, n_t, n_star, "Multi-SEM vs OLS")
        tables[study]["multi_vs_ols"] = multi_vs_ols
        base_rmse = float(np.mean(ols_rmse))
        geo_mean = float(np.mean(geo_rmse))
        multi_mean = float(np.mean(multi_rmse))
        tables[study]["rmse_reduction_vs_ols"] = (base_rmse - multi_mean) / base_rmse
        tables[study]["rmse_reduction_vs_geo"] = (geo_mean - multi_mean) / geo_mean
    return tables


def _ols_jobs(records, gibbs, out_dir: Path):
    """OLS baselines for the two sensitivity designs. No Gibbs sampling."""
    jobs = []
    for record in records:
        for study in ("chas", "censor"):
            path = out_dir / "fits" / f"{study}__ols__split{int(record['split']):02d}.json"
            if path.exists():
                continue
            jobs.append({
                "study": study,
                "model": "ols",
                "split": int(record["split"]),
                "seed": int(record["seed"]),
                "path": str(path),
                "gibbs": gibbs,
            })
    return jobs


def fit_ols_job(job):
    try:
        if job["study"] == "censor":
            src = _STATE["censor"]
            n, n_train = src["n"], src["n_train"]
        else:
            src = _STATE["chas"]
            n, n_train = boston.N, boston.N_TRAIN
        train, test = _indices(job["seed"], n, n_train)
        x_train, x_test, y_train, y_test = _design(src["data"], train, test, src.get("extra"))
        beta, _, _ = ols(y_train, x_train)
        metrics = boston.metrics_rmse_mae(y_test, x_test @ beta)
        payload = {
            "RMSE": float(metrics["RMSE"]),
            "MAE": float(metrics["MAE"]),
            "corr": None if not np.isfinite(metrics["corr"]) else float(metrics["corr"]),
            "method": "OLS",
        }
        Path(job["path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(job["path"]).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return {"ok": True, "model": "ols", "study": job["study"], "split": job["split"]}
    except Exception:
        return {"ok": False, "trace": traceback.format_exc(), "job": job["study"]}


def make_figure(out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gaussian = json.loads((out_dir / "split5_gaussian.json").read_text(encoding="utf-8"))
    beta = json.loads((out_dir / "split5_beta.json").read_text(encoding="utf-8"))
    g_draws = np.load(out_dir / "split5_gaussian.npz")["draws_lambda"]
    b_draws = np.load(out_dir / "split5_beta.npz")["draws_lambda"]
    l_post = np.sum(np.abs(g_draws), axis=1)
    l_beta = np.sum(np.abs(b_draws), axis=1)

    rng = np.random.default_rng(2026)
    k = 3
    n_prior = 200_000
    raw = rng.exponential(scale=1.0, size=(n_prior, k))
    signs = rng.choice([-1.0, 1.0], size=(n_prior, k))
    norm1 = np.sum(raw, axis=1)
    radius = rng.uniform(0.0, 1.0, size=n_prior) ** (1.0 / k)
    lam_prior = (raw / norm1[:, None]) * radius[:, None] * signs
    l_prior = np.sum(np.abs(lam_prior), axis=1)

    grid = np.linspace(0.001, 0.999, 400)
    beta_density = 2.0 * (1.0 - grid)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].hist(l_prior, bins=50, density=True, alpha=0.45, color="0.55", label=r"Uniform-in-ball prior, $p(L)\propto L^{2}$")
    axes[0].hist(l_post, bins=50, density=True, alpha=0.75, color="steelblue", label="MCMC posterior under the paper prior")
    axes[0].axvline(np.mean(l_post), color="darkblue", ls="--", label=f"Posterior mean ({np.mean(l_post):.3f})")
    axes[0].set_xlabel(r"$L=\|\lambda\|_1$")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Split 5, paper prior")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(grid, beta_density, color="0.35", lw=1.5, label=r"Beta$(1,2)$ prior, $p(L)=2(1-L)$")
    axes[1].hist(l_beta, bins=50, density=True, alpha=0.75, color="#B85C38", label="MCMC posterior under Beta(1, 2)")
    axes[1].axvline(np.mean(l_beta), color="#7A2E16", ls="--", label=f"Posterior mean ({np.mean(l_beta):.3f})")
    axes[1].set_xlabel(r"$L=\|\lambda\|_1$")
    axes[1].set_title("Split 5, boundary-penalizing prior")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=300)
    fig.savefig(out_dir / "figure_s10_prior_vs_posterior_L.png", dpi=300)
    plt.close(fig)
    summary = {
        "gaussian_prior_posterior": {
            "mean_L": float(np.mean(l_post)),
            "mode_L": _hist_mode(l_post),
            "q025": float(np.quantile(l_post, 0.025)),
            "q975": float(np.quantile(l_post, 0.975)),
            "refit_mean_L": gaussian["mean_L"],
            "stored_mean_L": gaussian["stored_mean_L"],
            "lambda_mean": gaussian["lambda_mean"],
        },
        "beta_prior_posterior": {
            "mean_L": float(np.mean(l_beta)),
            "mode_L": _hist_mode(l_beta),
            "q025": float(np.quantile(l_beta, 0.025)),
            "q975": float(np.quantile(l_beta, 0.975)),
            "lambda_mean": beta["lambda_mean"],
        },
        "prior_uniform_ball_mean_L": float(np.mean(l_prior)),
        "figure": str(FIG_PATH),
    }
    return summary


def run_simulation_task(out_dir: Path, n_reps, n_iter, burn, n_chains, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_dir.mkdir(parents=True, exist_ok=True)
    done_path = out_dir / "opposite_sign_reps.jsonl"
    finished = set()
    if done_path.exists() and n_iter >= 6000:
        for line in done_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                finished.add(int(json.loads(line)["rep"]))
    pending = [rep for rep in range(n_reps) if rep not in finished]
    failures = []
    if pending:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(simulation.opposite_sign_replication, rep, 200, n_iter, burn, n_chains): rep
                for rep in pending
            }
            done = 0
            for future in as_completed(futures):
                done += 1
                rep = futures[future]
                try:
                    row = future.result()
                except Exception:
                    failures.append({"rep": rep, "trace": traceback.format_exc()})
                    print(f"[sim {done}/{len(pending)}] rep {rep} FAILED", flush=True)
                    continue
                with done_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row) + "\n")
                print(
                    f"[sim {done}/{len(pending)}] rep {rep} "
                    f"mean {np.round(row['mean'], 3).tolist()} rhat {np.round(row['rhat'], 3).tolist()}",
                    flush=True,
                )
    rows = []
    if done_path.exists():
        for line in done_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda row: row["rep"])
    summary = simulation.summarize_opposite_sign(rows) if rows else {}
    summary["failures"] = failures
    (out_dir / "table4b.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _print_nb(block):
    for row in block["comparisons"]:
        print(
            f"--- {row['name']} ---\n"
            f"Mean MSE reduction: {row['mean_mse_reduction']:.6f}\n"
            f"Wins: {row['wins']}/{row['R']}\n"
            f"Corrected t({row['df']}): {row['t']:.3f}, p-value: {row['p']:.4e}\n"
            f"95% Corrected CI: [{row['ci95'][0]:.6f}, {row['ci95'][1]:.6f}]\n",
            flush=True,
        )


def main(argv=None):
    import sys

    argv = list(sys.argv[1:] if argv is None else argv)
    smoke = "--smoke" in argv
    fast_only = "--fast" in argv
    out_dir = ROOT / ("results/revision_smoke" if smoke else "results/revision")
    out_dir.mkdir(parents=True, exist_ok=True)
    gibbs = (
        {"adaptive": True, "n_iter": 200, "burn": 100, "n_chains": 2}
        if smoke
        else {"adaptive": True, "n_iter": 8000, "burn": 4000, "n_chains": 4}
    )
    records = _load_adaptive_records()
    if smoke:
        records_run = [row for row in records if int(row["split"]) in {0, 5}]
    else:
        records_run = records

    prepared = _prepare(gibbs)
    nb = task1(records)
    (out_dir / "nadeau_bengio.json").write_text(json.dumps(nb, indent=2), encoding="utf-8")
    _print_nb(nb)
    print(
        f"max L split 5 = {nb['max_L_split5']:.6f}; "
        f"max L across splits = {nb['max_L_across_splits']:.6f}",
        flush=True,
    )

    moran = task6(records)
    moran_public = {"summary": moran["summary"]}
    (out_dir / "moran_innovations_summary.json").write_text(
        json.dumps(moran_public, indent=2), encoding="utf-8"
    )
    (out_dir / "moran_innovations.json").write_text(json.dumps(moran, indent=2), encoding="utf-8")
    print(json.dumps(moran_public, indent=2), flush=True)
    if fast_only:
        print("fast tasks written", flush=True)
        return

    # OLS baselines are cheap and keep the sensitivity table complete.
    ols_pending = _ols_jobs(records_run, gibbs, out_dir)
    if ols_pending:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=min(8, len(ols_pending)), initializer=_init_worker, initargs=(_STATE,)) as pool:
            list(pool.map(fit_ols_job, ols_pending))
        print(f"OLS baselines written: {len(ols_pending)}", flush=True)

    _jobs_all, pending = _jobs(records_run, gibbs, out_dir, include_predictive=True)
    print(f"MCMC jobs pending {len(pending)} of {len(_jobs_all)}", flush=True)
    # Dense factorizations saturate memory bandwidth well before all cores are busy.
    workers = 4 if smoke else 6
    sim_workers = 2 if smoke else 2
    sim_reps = 2 if smoke else 200
    sim_iter = 200 if smoke else 6000
    sim_burn = 100 if smoke else 3000
    sim_chains = 2 if smoke else 4

    # The opposite-sign Monte Carlo and the Boston Gibbs jobs run together.
    import threading

    sim_box = {}

    def _run_sim():
        try:
            sim_box["summary"] = run_simulation_task(
                out_dir, sim_reps, sim_iter, sim_burn, sim_chains, sim_workers
            )
        except Exception:
            sim_box["trace"] = traceback.format_exc()

    sim_thread = threading.Thread(target=_run_sim, name="opposite-sign")
    sim_thread.start()
    failures = _run_pool(pending, workers)
    sim_thread.join()
    if "trace" in sim_box:
        raise RuntimeError(sim_box["trace"])
    sim_summary = sim_box.get("summary") or {}
    if failures:
        (out_dir / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        raise RuntimeError(f"{len(failures)} MCMC jobs failed; see {out_dir / 'failures.json'}")

    ablation = aggregate_ablation(records_run, out_dir)
    (out_dir / "table_s9_ablation.json").write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    print("TABLE S9", flush=True)
    print(json.dumps(ablation, indent=2), flush=True)

    if not smoke:
        sensitivity = aggregate_sensitivity(records, out_dir)
        (out_dir / "table_s10_sensitivity.json").write_text(json.dumps(sensitivity, indent=2), encoding="utf-8")
        print("TABLE S10", flush=True)
        print(json.dumps(sensitivity, indent=2), flush=True)
        figure = make_figure(out_dir)
        (out_dir / "figure_s10_summary.json").write_text(json.dumps(figure, indent=2), encoding="utf-8")
        print("FIGURE S10", flush=True)
        print(json.dumps(figure, indent=2), flush=True)
    print("TABLE 4b", flush=True)
    print(json.dumps(sim_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
