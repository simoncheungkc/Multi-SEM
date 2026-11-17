"""Quick sanity checks for Multi-SEM primitives."""

import numpy as np

from mnreg.model import (
    GibbsConfig,
    add_intercept,
    assemble_V,
    fit_ml_multi_sem,
    gibbs_multi_sem,
    moran_i_test,
    ols,
    posterior_summary,
    predict_kriging,
    screen_networks,
)
from mnreg.networks import erdos_renyi, powerlaw_configuration, row_normalize


def simulate_y(X, beta, lambdas, As, sigma, rng):
    V = assemble_V(lambdas, As)
    u = np.linalg.solve(V, rng.normal(0.0, sigma, size=X.shape[0]))
    return X @ beta + u


def main():
    rng = np.random.default_rng(1)
    n = 80
    A1 = row_normalize(erdos_renyi(n, 8, rng))
    A2 = row_normalize(powerlaw_configuration(n, 2.5, rng))
    A3 = row_normalize(erdos_renyi(n, 4, rng))
    X = add_intercept(rng.normal(size=(n, 3)))
    beta = np.array([0.2, -0.4, 0.3, 0.1])
    lam = np.array([0.45, 0.25, 0.0])
    Y = simulate_y(X, beta, lam, [A1, A2, A3], 1.0, rng)

    _, resid, _ = ols(Y, X)
    t1 = moran_i_test(resid, A1, X)
    t3 = moran_i_test(resid, A3, X)
    print("Moran p (active Poisson):", round(t1["pvalue"], 4), "z", round(t1["z"], 3))
    print("Moran p (null Poisson):  ", round(t3["pvalue"], 4), "z", round(t3["z"], 3))
    assert 0 <= t1["pvalue"] <= 1 and 0 <= t3["pvalue"] <= 1

    scr = screen_networks(Y, X, [A1, A2, A3], names=["P", "PL", "N"], alpha=0.05, fdr=False)
    print("selected", scr["selected_names"], "p", np.round(scr["pvalues"], 4))

    ml = fit_ml_multi_sem(Y, X, [A1, A2])
    print("ML lambda", np.round(ml["lambda"], 3), "sigma2", round(ml["sigma2"], 3))

    cfg = GibbsConfig(n_iter=400, burn=200, n_chains=2, seed=2)
    res = gibbs_multi_sem(Y, X, [A1, A2], names=["P", "PL"], config=cfg)
    s = posterior_summary(res)
    print("Gibbs lambda means", np.round(s["means"]["lambda"], 3), "accept", res.acceptance)

    idx = np.arange(n)
    test = idx[:16]
    train = idx[16:]
    pred = predict_kriging(
        Y[train], X[train], X[test], train, test, s["means"]["lambda"], s["means"]["beta"], [A1, A2]
    )
    print("pred length", len(pred), "rmse", float(np.sqrt(np.mean((pred - Y[test]) ** 2))))
    print("SMOKE OK")


if __name__ == "__main__":
    main()
