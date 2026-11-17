"""Induced versus Schur predictive scores for August 2026 and Boston splits 0-9."""
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import run_boston as boston
from mnreg.model import gibbs_multi_sem, screen_networks
from mnreg.networks import row_normalize
from run_unified_revision import (
    _chain_config,
    _conditional_replicates,
    _predictive_metrics,
    _thin_draws,
    gibbs_schur,
)
from run_v28_analyses import _sp_showcase, _split_indices

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "v28"
CACHE = OUT / "schur"
CACHE.mkdir(parents=True, exist_ok=True)


def _score(result, y, x, train, test, full, seed):
    beta, sigma, lam = _thin_draws(result, 2000)
    stack = _conditional_replicates(
        y, x, train, test, beta, sigma, lam, full, np.random.default_rng(seed), 20
    )
    metrics = _predictive_metrics(stack, y[test])
    flat = result.samples["lambda"].reshape(-1, result.samples["lambda"].shape[-1])
    metrics["lambda_mean"] = flat.mean(axis=0).tolist()
    return metrics


def sp_schur() -> dict:
    _panel, _fund, _dates, showcase, date = _sp_showcase()
    y, x = showcase["Y"], showcase["X"]
    train, test = _split_indices(len(y))
    scr = screen_networks(y, x, showcase["As"], names=showcase["names"], alpha=0.05, fdr=True)
    selected = np.where(scr["selected"])[0].tolist()
    induced = [row_normalize(showcase["As_raw"][i][np.ix_(train, train)]) for i in selected]
    full = [showcase["As"][i] for i in selected]
    names = [showcase["names"][i] for i in selected]
    config = _chain_config(2026, n_iter=8000, burn=4000, n_chains=4)
    induced_fit = gibbs_multi_sem(y[train], x[train], induced, names=names, config=config)
    schur_fit = gibbs_schur(y, x, full, train, test, config)
    payload = {
        "date": str(date.date()) if hasattr(date, "date") else str(date),
        "names": names,
        "induced": _score(induced_fit, y, x, train, test, full, 11),
        "schur": _score(schur_fit, y, x, train, test, full, 12),
    }
    print("sp schur", payload["induced"]["rmse"], payload["schur"]["rmse"], flush=True)
    return payload


def _boston_one(split: int) -> dict:
    path = CACHE / f"boston_{split:02d}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    data = boston.load_boston()
    networks = boston.build_networks(data)
    record = json.loads((ROOT / "results" / "boston_adaptive" / f"split_{split:02d}.json").read_text(encoding="utf-8"))
    part = np.random.default_rng([int(record["seed"]), 1])
    train = np.sort(part.choice(boston.N, size=boston.N_TRAIN, replace=False))
    test = np.sort(np.setdiff1d(np.arange(boston.N), train))
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = np.where(raw[train].std(axis=0, ddof=0) < 1e-12, 1.0, raw[train].std(axis=0, ddof=0))
    scaled = (raw - center) / spread
    x = np.column_stack([np.ones(boston.N), scaled])
    y = data["y"]
    layers = list(record["active"])
    induced = [boston.induced_pair(networks["weights"][name], train)[1] for name in layers]
    full = [networks["adjacency"][name] for name in layers]
    config = _chain_config(3000 + split, n_iter=8000, burn=4000, n_chains=4)
    induced_fit = gibbs_multi_sem(y[train], x[train], induced, names=layers, config=config)
    schur_fit = gibbs_schur(y, x, full, train, test, config)
    payload = {
        "split": split,
        "layers": layers,
        "induced": _score(induced_fit, y, x, train, test, full, 100 + split),
        "schur": _score(schur_fit, y, x, train, test, full, 200 + split),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    print(f"boston split {split} induced {payload['induced']['rmse']:.3f} schur {payload['schur']['rmse']:.3f}", flush=True)
    return payload


def _average(rows, key):
    fields = ("rmse", "mae", "cover90", "cover95", "width95", "crps")
    return {field: float(np.mean([row[key][field] for row in rows])) for field in fields}


def main():
    sp = sp_schur()
    (OUT / "schur_sp.json").write_text(json.dumps(sp, indent=2), encoding="utf-8")
    with ProcessPoolExecutor(max_workers=2) as pool:
        boston_rows = list(pool.map(_boston_one, range(10), chunksize=1))
    boston_rows = sorted(boston_rows, key=lambda row: row["split"])
    summary = {
        "sp100": sp,
        "boston_splits": boston_rows,
        "boston_mean": {
            "induced": _average(boston_rows, "induced"),
            "schur": _average(boston_rows, "schur"),
        },
    }
    (OUT / "schur_panel.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote schur panel", summary["boston_mean"], flush=True)


if __name__ == "__main__":
    main()
