"""Joint and robust partial LM diagnostics on the saved training designs.

Uses the archived Boston split seeds and the offline S&P 100 snapshot.
Does not refit the Gibbs sampler.
"""

from __future__ import annotations

import json
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import numpy as np

from mnreg.model import joint_multi_weight_lm, moran_i_test, ols
from run_boston import N, N_TRAIN, ROOT, build_networks, induced_pair, load_boston
from run_sp100 import (
    align_cross_section,
    download_prices,
    fetch_fundamentals,
    load_constituents,
    monthly_panel,
    row_normalize,
)

OUT = ROOT / "results" / "v34" / "joint_lm.json"


def _boston():
    data = load_boston()
    networks = build_networks(data)
    kept = networks["kept"]
    records = []
    for path in sorted((ROOT / "results" / "boston_adaptive").glob("split_*.json")):
        stored = json.loads(path.read_text(encoding="utf-8"))
        seed = int(stored["seed"])
        part = np.random.default_rng([seed, 1])
        train = np.sort(part.choice(N, size=N_TRAIN, replace=False))
        raw = data["raw"]
        center = raw[train].mean(axis=0)
        spread = raw[train].std(axis=0, ddof=0)
        spread = np.where(spread < 1e-12, 1.0, spread)
        scaled = (raw - center) / spread
        x_train = np.column_stack([np.ones(N_TRAIN), scaled[train]])
        y_train = data["y"][train]
        names = []
        matrices = []
        for name in kept:
            _weights, adjacency = induced_pair(networks["weights"][name], train)
            names.append(name)
            matrices.append(adjacency)
        _beta, resid, _sigma = ols(y_train, x_train)
        lm = joint_multi_weight_lm(resid, matrices)
        moran = [moran_i_test(resid, A, x_train) for A in matrices]
        stored_z = {row["name"]: row["z"] for row in stored["tests"]}
        stored_p = {row["name"]: row["pvalue"] for row in stored["tests"]}
        records.append({
            "split": int(stored["split"]),
            "names": names,
            "lm_joint": lm["lm_joint"],
            "p_joint": lm["p_joint"],
            "lm_partial": lm["lm_partial"].tolist(),
            "p_partial": lm["p_partial"].tolist(),
            "moran_z": [row["z"] for row in moran],
            "archived_z": [stored_z[name] for name in names],
            "archived_p": [stored_p[name] for name in names],
            "condition": float(np.linalg.cond(lm["information"])),
        })
        print(
            f"boston split {stored['split']:02d} LM {lm['lm_joint']:.2f} "
            f"p {lm['p_joint']:.3e} partial {np.array2string(lm['lm_partial'], precision=2)}",
            flush=True,
        )
    names = records[0]["names"]
    joint = np.array([row["lm_joint"] for row in records])
    p_joint = np.array([row["p_joint"] for row in records])
    partial = np.array([row["lm_partial"] for row in records])
    p_partial = np.array([row["p_partial"] for row in records])
    archived_z = np.array([row["archived_z"] for row in records])
    archived_p = np.array([row["archived_p"] for row in records])
    summary = {
        "n_splits": len(records),
        "n_train": N_TRAIN,
        "names": names,
        "lm_joint_mean": float(joint.mean()),
        "lm_joint_min": float(joint.min()),
        "lm_joint_max": float(joint.max()),
        "p_joint_max": float(p_joint.max()),
        "n_p_joint_lt_1e_16": int(np.sum(p_joint < 1e-16)),
        "lm_partial_mean": partial.mean(axis=0).tolist(),
        "p_partial_median": np.median(p_partial, axis=0).tolist(),
        "n_partial_lt_0_05": np.sum(p_partial < 0.05, axis=0).astype(int).tolist(),
        "archived_z_mean": archived_z.mean(axis=0).tolist(),
        "n_archived_p_lt_0_05": np.sum(archived_p < 0.05, axis=0).astype(int).tolist(),
        "condition_median": float(np.median([row["condition"] for row in records])),
    }
    return {"summary": summary, "splits": records}


def _sp100():
    wiki = load_constituents()
    tickers = wiki["Symbol"].dropna().unique().tolist()
    px = download_prices(tickers)
    tickers = [t for t in tickers if t in px.columns]
    px = px[tickers]
    fund = fetch_fundamentals(tickers, wiki)
    panel = monthly_panel(px)
    ret = panel["ret"].dropna(how="all")
    dates = ret.index[(ret.index >= "2021-01-01") & (ret.index <= "2026-08-31")]
    showcase = None
    for date in dates[::-1]:
        cross = align_cross_section(panel, fund, date)
        if cross is not None and len(cross["tickers"]) >= 70:
            showcase = cross
            break
    if showcase is None:
        raise RuntimeError("August showcase cross-section was not rebuilt")
    n = len(showcase["Y"])
    rng = np.random.default_rng(2026)
    order = rng.permutation(n)
    n_test = max(int(0.2 * n), 15)
    train = np.sort(order[n_test:])
    names = showcase["names"]
    matrices = [
        row_normalize(A[np.ix_(train, train)]) for A in showcase["As_raw"]
    ]
    y = showcase["Y"][train]
    x = showcase["X"][train]
    _beta, resid, _sigma = ols(y, x)
    lm = joint_multi_weight_lm(resid, matrices)
    moran = [moran_i_test(resid, A, x) for A in matrices]
    print(
        f"sp100 n_train {len(train)} date {showcase['date']} "
        f"LM {lm['lm_joint']:.4f} p {lm['p_joint']:.4e}",
        flush=True,
    )
    for name, z, stat, p in zip(names, moran, lm["lm_partial"], lm["p_partial"]):
        print(f"  {name}: z {z['z']:.4f} LM* {stat:.4f} p {p:.4e}", flush=True)
    return {
        "date": str(showcase["date"]),
        "n_train": int(len(train)),
        "n_test": int(n_test),
        "names": names,
        "lm_joint": lm["lm_joint"],
        "p_joint": lm["p_joint"],
        "lm_partial": lm["lm_partial"].tolist(),
        "p_partial": lm["p_partial"].tolist(),
        "moran_z": [row["z"] for row in moran],
        "moran_p": [row["pvalue"] for row in moran],
        "condition": float(np.linalg.cond(lm["information"])),
    }


def main():
    boston = _boston()
    sp = _sp100()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        json.dumps({"boston": boston["summary"], "sp100": sp}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"boston": boston["summary"], "sp100": sp}, indent=2))


if __name__ == "__main__":
    main()
