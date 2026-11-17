"""Network constructors for the Multi-SEM analysis."""

from __future__ import annotations

import numpy as np
import networkx as nx


def row_normalize(A: np.ndarray) -> np.ndarray:
    """Row-standardize so every non-isolated row sums to 1.

    Isolated nodes keep a zero row. For a nonnegative matrix the resulting W
    satisfies ||W||_∞ ≤ 1, hence ρ(W) ≤ 1, which is the spectral-radius bound
    used for invertibility and for the intercept identification remark
    V 1 = (1 − Σ λ_s) 1.
    """
    W = np.asarray(A, dtype=float).copy()
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    isolated = rs[:, 0] == 0.0
    rs[isolated] = 1.0
    W = W / rs
    return W


def _to_simple_undirected(g: nx.Graph) -> nx.Graph:
    h = nx.Graph(g)
    h.remove_edges_from(nx.selfloop_edges(h))
    return h


def erdos_renyi(n: int, mean_degree: float, rng: np.random.Generator) -> np.ndarray:
    """Undirected Erdos-Renyi graph with Poisson-like degree distribution."""
    p = float(mean_degree) / max(n - 1, 1)
    p = min(max(p, 0.0), 1.0)
    g = nx.gnp_random_graph(n, p, seed=int(rng.integers(1, 2**31 - 1)), directed=False)
    A = nx.to_numpy_array(g, dtype=float)
    return A


def powerlaw_configuration(
    n: int,
    gamma: float,
    rng: np.random.Generator,
    kmin: int = 2,
    kmax: int | None = None,
) -> np.ndarray:
    """Undirected configuration-model graph with P(k) proportional to k^{-gamma}."""
    if kmax is None:
        kmax = max(kmin + 1, min(n - 1, int(np.sqrt(n) * 4)))
    ks = np.arange(kmin, kmax + 1)
    weights = ks.astype(float) ** (-float(gamma))
    weights /= weights.sum()
    deg = rng.choice(ks, size=n, p=weights).astype(int)
    if int(deg.sum()) % 2 == 1:
        bumped = False
        for i in np.argsort(-deg):
            if deg[i] < kmax:
                deg[i] += 1
                bumped = True
                break
        if not bumped:
            for i in np.argsort(deg):
                if deg[i] > kmin:
                    deg[i] -= 1
                    break
    seed = int(rng.integers(1, 2**31 - 1))
    try:
        g = nx.configuration_model(deg.tolist(), seed=seed)
        g = _to_simple_undirected(g)
    except nx.NetworkXError:
        g = nx.havel_hakimi_graph(deg.tolist())
    A = nx.to_numpy_array(g, dtype=float, nodelist=list(range(n)))
    return A


def degree_summary(A: np.ndarray) -> dict:
    deg = (np.asarray(A) > 0).sum(axis=1)
    pos = deg[deg > 0]
    return {
        "n_nodes": int(A.shape[0]),
        "n_edges": int(np.nansum(np.triu(A, 1) > 0)),
        "mean_degree": float(deg.mean()),
        "median_degree": float(np.median(deg)),
        "max_degree": int(deg.max()) if len(deg) else 0,
        "isolates": int((deg == 0).sum()),
        "degree_cv": float(deg.std() / deg.mean()) if deg.mean() > 0 else np.nan,
        "min_positive_degree": int(pos.min()) if len(pos) else 0,
    }


def boosted_resolvent(A: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Matrix resolvent A(I - alpha A)^{-1} capturing multi-hop paths.

    For a row-normalized A, the spectral radius is at most 1, so alpha in (0, 1)
    guarantees that the Neumann series converges.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0, 1) for resolvent convergence")
    W = np.asarray(A, dtype=float)
    n = W.shape[0]
    eye = np.eye(n)
    try:
        boost = W @ np.linalg.solve(eye - alpha * W, eye)
    except np.linalg.LinAlgError:
        boost = W.copy()
    return boost


def haversine_km(latitude, longitude, radius_km: float = 6371.0) -> np.ndarray:
    """Great-circle distance matrix in kilometres.

    Latitude and longitude are in degrees. The diagonal is zero.
    """
    lat = np.radians(np.asarray(latitude, dtype=float).reshape(-1))
    lon = np.radians(np.asarray(longitude, dtype=float).reshape(-1))
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return radius_km * c


def build_headquarters_geographic_network(hq_csv_path, tickers, threshold_km: float = 800.0):
    """Row-standardized physical weights from headquarters great-circle distance.

    W_ij = 1/d_ij when i != j and d_ij is within threshold_km, and 0 otherwise.
    A_geo is the row-standardized version of W. A firm with no neighbour inside
    the threshold keeps a zero row.
    """
    import pandas as pd

    df = pd.read_csv(hq_csv_path).set_index("ticker").loc[list(tickers)]
    if df[["latitude", "longitude"]].isna().any().any():
        missing = df.index[df["latitude"].isna() | df["longitude"].isna()].tolist()
        raise ValueError(f"Headquarters coordinates missing for {missing}")
    distance_km = haversine_km(df["latitude"].values, df["longitude"].values)
    n = len(tickers)
    weights = np.zeros((n, n), dtype=np.float64)
    mask = (distance_km <= float(threshold_km)) & (distance_km > 0.0)
    weights[mask] = 1.0 / distance_km[mask]
    np.fill_diagonal(weights, 0.0)
    row_sums = weights.sum(axis=1)
    adjacency = np.zeros_like(weights)
    positive = row_sums > 0
    adjacency[positive] = weights[positive] / row_sums[positive, None]
    return weights, adjacency


def offdiagonal_correlation(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation of the off-diagonal entries of two square matrices."""
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    mask = ~np.eye(a.shape[0], dtype=bool)
    x = a[mask]
    y = b[mask]
    x = x - x.mean()
    y = y - y.mean()
    denom = np.sqrt(np.sum(x * x) * np.sum(y * y))
    if denom <= 0:
        return float("nan")
    return float(np.sum(x * y) / denom)


def geographic_topology(weights: np.ndarray, adjacency: np.ndarray) -> dict:
    """Degree heterogeneity of W and singular-value concentration of A."""
    binary = (np.asarray(weights) > 0).astype(float)
    np.fill_diagonal(binary, 0.0)
    degree = binary.sum(axis=1)
    mean_degree = float(degree.mean())
    kappa = float(degree.max() / mean_degree) if mean_degree > 0 else float("nan")
    singular = np.linalg.svd(np.asarray(adjacency, dtype=float), compute_uv=False)
    total = float(singular.sum())
    psi = float(singular[0] / total) if total > 0 else float("nan")
    return {
        "kappa": kappa,
        "psi": psi,
        "mean_degree": mean_degree,
        "max_degree": int(degree.max()) if len(degree) else 0,
        "isolates": int((degree == 0).sum()),
        "n_edges": int(np.sum(np.triu(binary, 1) > 0)),
        "leading_singular": float(singular[0]) if len(singular) else float("nan"),
    }


def affiliation_network(labels) -> np.ndarray:
    """Undirected co-membership network: edge if two nodes share a label."""
    labels = np.asarray(labels, dtype=object)
    n = len(labels)
    A = np.zeros((n, n), dtype=float)
    for i in range(n):
        if labels[i] is None or (isinstance(labels[i], float) and np.isnan(labels[i])):
            continue
        for j in range(i + 1, n):
            if labels[j] is None or (isinstance(labels[j], float) and np.isnan(labels[j])):
                continue
            if labels[i] == labels[j]:
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A


def correlation_network(
    returns: np.ndarray,
    method: str = "threshold",
    threshold: float | None = None,
    k: int = 5,
    min_obs: int = 20,
) -> np.ndarray:
    """Undirected correlation network from a T x n return matrix.

    method='threshold': keep |corr| > threshold (default set so mean degree ~ 2k).
    method='knn': keep the k strongest absolute correlations per node, then
    symmetrize with W = max(A, A'). The returned matrix is a symmetric 0-1
    adjacency; row-standardization (row sums 1, isolates 0) is applied
    afterwards so that ρ(W) ≤ 1.
    """
    R = np.asarray(returns, dtype=float)
    n = R.shape[1]
    with np.errstate(invalid="ignore"):
        C = np.corrcoef(R, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 0.0)
    absC = np.abs(C)
    if method == "knn":
        A = np.zeros((n, n), dtype=float)
        for i in range(n):
            idx = np.argpartition(absC[i], -k)[-k:]
            A[i, idx] = 1.0
        # Symmetrize first (undirected edges), then the caller row-standardizes.
        A = np.maximum(A, A.T)
        np.fill_diagonal(A, 0.0)
        return A
    if threshold is None:
        # Target about 2k edges per node on average (undirected => k neighbors).
        vals = absC[np.triu_indices(n, 1)]
        vals = vals[vals > 0]
        if len(vals) == 0:
            return np.zeros((n, n))
        q = 1.0 - min(0.5, (n * k / 2.0) / max(len(vals), 1))
        threshold = float(np.quantile(vals, q))
    A = (absC >= float(threshold)).astype(float)
    np.fill_diagonal(A, 0.0)
    return A
