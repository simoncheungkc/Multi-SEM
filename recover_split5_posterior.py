"""Replay the Split 5 Multi-SEM Gibbs fit and summarize the retained draws.

The seed path matches run_boston.run_split. Means are checked against
results/boston/split_05.json before the summary is written.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_boston as rb
from mnreg.model import effective_sample_size, gelman_rubin, gibbs_multi_sem, GibbsConfig

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "boston" / "split_05_posterior.json"


def summarize(draws: np.ndarray) -> dict:
    flat = draws.reshape(-1)
    return {
        "mean": float(flat.mean()),
        "median": float(np.median(flat)),
        "sd": float(flat.std(ddof=1)),
        "q025": float(np.quantile(flat, 0.025)),
        "q975": float(np.quantile(flat, 0.975)),
        "rhat": float(gelman_rubin(draws)),
        "ess": float(sum(effective_sample_size(draws[c]) for c in range(draws.shape[0]))),
    }


def main():
    record = json.loads((ROOT / "results" / "boston" / "split_05.json").read_text(encoding="utf-8"))
    data = rb.load_boston()
    built = rb.build_networks(data)
    seed = int(record["seed"])
    part = np.random.default_rng([seed, 1])
    train = np.sort(part.choice(rb.N, size=rb.N_TRAIN, replace=False))
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = raw[train].std(axis=0, ddof=0)
    spread = np.where(spread < 1e-12, 1.0, spread)
    scaled = (raw - center) / spread
    x_train = np.column_stack([np.ones(rb.N_TRAIN), scaled[train]])
    y_train = data["y"][train]
    names = list(record["multi"]["names"])
    mats = []
    for name in names:
        _, adjacency = rb.induced_pair(built["weights"][name], train)
        mats.append(adjacency)
    gibbs_rng = np.random.default_rng([seed, 3])
    seeds = gibbs_rng.integers(1, 2**31 - 1, size=4)
    config = GibbsConfig(n_iter=6000, burn=3000, n_chains=4, seed=int(seeds[2]))
    result = gibbs_multi_sem(y_train, x_train, mats, names=names, config=config)
    lam = result.samples["lambda"]
    beta = result.samples["beta"]
    sigma = result.samples["sigma2"]
    flat_l = lam.reshape(-1, lam.shape[-1])
    flat_b = beta.reshape(-1, beta.shape[-1])
    if not np.allclose(flat_l.mean(axis=0), record["multi"]["lambda"], atol=1e-10):
        raise SystemExit(f"lambda mean {flat_l.mean(axis=0)} != {record['multi']['lambda']}")
    if not np.allclose(flat_b.mean(axis=0), record["multi"]["beta"], atol=1e-10):
        raise SystemExit("beta mean does not match the archive")
    if abs(float(sigma.mean()) - record["multi"]["sigma2"]) > 1e-10:
        raise SystemExit("sigma2 mean does not match the archive")
    total = np.sum(np.abs(flat_l), axis=1)
    rows = []
    labels = [
        ("lambda_geo", "Geographic 6-NN"),
        ("lambda_resolvent", "Geographic resolvent"),
        ("lambda_town", "Town jurisdiction"),
    ]
    for index, (key, label) in enumerate(labels):
        item = summarize(lam[:, :, index])
        item.update({"key": key, "label": label})
        rows.append(item)
    beta_labels = [
        ("beta0", "Intercept"),
        ("rm", "Rooms"),
        ("age", "Age"),
        ("crim", "Crime"),
        ("zn", "Large-lot zoning"),
        ("indus", "Non-retail business acreage"),
        ("log_lstat", "log lower-status share"),
        ("tax", "Property-tax rate"),
        ("ptratio", "Pupil-teacher ratio"),
        ("nox", "Nitric oxides"),
        ("log_dis", "log distance to employment centres"),
    ]
    for index, (key, label) in enumerate(beta_labels):
        item = summarize(beta[:, :, index])
        item.update({"key": key, "label": label})
        rows.append(item)
    sigma_row = summarize(sigma)
    sigma_row.update({"key": "sigma2", "label": "Innovation variance"})
    rows.append(sigma_row)
    # L is a chain-wise scalar. Shape it as chains by draws.
    l_draws = np.abs(lam).sum(axis=2)
    l_row = summarize(l_draws)
    l_row.update({
        "key": "L",
        "label": "L = sum of absolute lambda",
        "q95": float(np.quantile(total, 0.95)),
        "p_gt_095": float(np.mean(total > 0.95)),
    })
    payload = {
        "split": 5,
        "n_train": int(rb.N_TRAIN),
        "names": names,
        "rows": rows,
        "L": l_row,
        "acceptance": result.acceptance.astype(float).tolist(),
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    for row in rows:
        print(
            f"{row['key']:16} mean {row['mean']:.4f} med {row['median']:.4f} "
            f"sd {row['sd']:.4f} ({row['q025']:.4f}, {row['q975']:.4f}) "
            f"Rhat {row['rhat']:.3f} ESS {row['ess']:.0f}"
        )
    print(
        f"L q95 {l_row['q95']:.4f} P(L>0.95) {l_row['p_gt_095']:.4f} "
        f"Rhat {l_row['rhat']:.3f} ESS {l_row['ess']:.0f}"
    )


if __name__ == "__main__":
    main()
