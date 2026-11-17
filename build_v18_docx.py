"""Build v18 and supplement v16 from v17 and supplement v15.

Does not overwrite those source documents. Restores the archived Split 5
posterior in Table 8. Does not rerun the Gibbs sampler.
"""

from __future__ import annotations

import json

from lxml import etree

import build_v12_docx as b
from build_v10_docx import omath, set_cell_text
from mnreg.latex_omml import el

ROOT = b.ROOT
W, M = b.W, b.M
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v17.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v15.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v18.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v16.docx"
POSTERIOR = ROOT / "results" / "boston" / "split_05_posterior.json"

REJECTION = r"\left\lVert\lambda^{*}\right\rVert_{1}=\sum_{s=1}^{k}\left|\lambda_{s}^{*}\right|\geq 1"


def local(tag):
    return tag.split("}")[-1]


def flat_math(node):
    parts = []

    def walk(el):
        name = local(el.tag)
        if name == "t":
            parts.append(el.text or "")
            return
        if name == "d":
            beg = end = ""
            for ch in el:
                if local(ch.tag) == "dPr":
                    for item in ch:
                        if local(item.tag) == "begChr":
                            beg = item.get(f"{{{M}}}val") or ""
                        elif local(item.tag) == "endChr":
                            end = item.get(f"{{{M}}}val") or ""
            parts.append(beg)
            for ch in el:
                if local(ch.tag) != "dPr":
                    walk(ch)
            parts.append(end)
            return
        if name == "nary":
            symbol = "∑"
            for item in el.iter():
                if local(item.tag) == "chr":
                    symbol = item.get(f"{{{M}}}val") or symbol
                    break
            parts.append(symbol)
        for ch in el:
            if name == "nary" and local(ch.tag) == "naryPr":
                continue
            walk(ch)

    walk(node)
    return "".join(parts)


def fix_indicator_delimiters(root):
    """Use parentheses for I(sum |lambda_s| < 1), not bars around the inequality."""
    changed = []
    for delim in root.iter(b.q(M, "d")):
        text = flat_math(delim)
        if "<1" not in text or "|λ" not in text and "∣λ" not in text:
            continue
        dpr = delim.find(b.q(M, "dPr"))
        if dpr is None:
            dpr = el("dPr")
            delim.insert(0, dpr)
        beg = dpr.find(b.q(M, "begChr"))
        end = dpr.find(b.q(M, "endChr"))
        old = beg.get(f"{{{M}}}val") if beg is not None else ""
        if old == "(":
            continue
        if old not in ("", "|"):
            continue
        if beg is None:
            dpr.append(el("begChr", val="("))
        else:
            beg.set(f"{{{M}}}val", "(")
        if end is None:
            dpr.append(el("endChr", val=")"))
        else:
            end.set(f"{{{M}}}val", ")")
        changed.append((old, text[:80]))
    return changed


def expand_rejection_rule(root):
    replaced = 0
    formula = omath(REJECTION)
    for math in list(root.iter(b.q(M, "oMath"))):
        compact = flat_math(math).replace(" ", "")
        if compact in {"‖λ*‖1≥1", "‖λ∗‖1≥1"}:
            math.getparent().replace(math, etree.fromstring(etree.tostring(formula)))
            replaced += 1
    return replaced


def cell_text(tc):
    return "".join(t.text or "" for t in tc.iter(b.q(W, "t")))


def signed(value, digits):
    return f"{value:.{digits}f}".replace("-", "−")


def ess(value):
    return f"{int(round(value)):,}"


def restore_table8(body):
    posterior = json.loads(POSTERIOR.read_text(encoding="utf-8"))
    by_key = {row["key"]: row for row in posterior["rows"]}
    by_key["L"] = posterior["L"]
    caption = b.require_one(body, "Table 8. Posterior parameter estimates")
    table = caption.getnext()
    while table is not None and table.tag != b.q(W, "tbl"):
        table = table.getnext()
    if table is None:
        raise SystemExit("Table 8 has no following table")
    seen = set()
    for tr in table.findall(b.q(W, "tr"))[1:]:
        cells = tr.findall(b.q(W, "tc"))
        label = cell_text(cells[0]).lower()
        joined = " ".join(cell_text(tc).lower() for tc in cells)
        if "percentile" in joined or "p(l" in joined:
            key = "L"
        elif "6-nn" in label:
            key = "lambda_geo"
        elif "resolvent" in label:
            key = "lambda_resolvent"
        elif "municipal" in label:
            key = "lambda_town"
        elif "intercept" in label:
            key = "beta0"
        elif "rooms" in label:
            key = "rm"
        elif "lower status" in label:
            key = "log_lstat"
        elif "crime" in label:
            key = "crim"
        elif "tax" in label:
            key = "tax"
        elif "pupil" in label:
            key = "ptratio"
        elif "distance" in label:
            key = "log_dis"
        elif "nitric" in label or "oxide" in label:
            key = "nox"
        elif "zoning" in label:
            key = "zn"
        elif "acreage" in label or "industry" in label:
            key = "indus"
        elif "innovation" in label or "variance" in label:
            key = "sigma2"
        elif label.strip(" ()") == "age":
            key = "age"
        else:
            key = None
        if key is None or key in seen:
            raise SystemExit(f"unmatched Table 8 row: {label!r}")
        seen.add(key)
        row = by_key[key]
        digits = 4 if key == "sigma2" else 3
        set_cell_text(cells[1], signed(row["mean"], digits))
        set_cell_text(cells[2], signed(row["median"], digits))
        set_cell_text(cells[3], signed(row["sd"], digits))
        if key == "L":
            set_cell_text(cells[4], f"95th percentile: {row['q95']:.3f}")
            set_cell_text(cells[5], f"{row['rhat']:.3f}")
            set_cell_text(cells[6], f"P(L > 0.95) = {row['p_gt_095']:.3f}")
        else:
            set_cell_text(
                cells[4],
                f"({signed(row['q025'], digits)}, {signed(row['q975'], digits)})",
            )
            set_cell_text(cells[5], f"{row['rhat']:.3f}")
            set_cell_text(cells[6], ess(row["ess"]))
    if seen != set(by_key):
        raise SystemExit(f"Table 8 missed {set(by_key) - seen}")


def count_drawings(body, start, end):
    nodes, _ = b.span_nodes(body, start, end)
    return sum(1 for node in nodes for _ in node.iter(b.q(W, "drawing")))


def edit_main(body):
    if count_drawings(body, "Algorithm 1. Ex-ante", "Algorithm 2.") != 0:
        raise SystemExit("Algorithm 1 still contains a drawing")
    b.replace_containing(
        body,
        "We evaluate Multi-SEM using two substantive empirical benchmarks.",
        "We evaluate Multi-SEM using two substantive empirical benchmarks. "
        "In a spatial hedonic study of $n = 506$ Boston census tracts across 92 municipalities, "
        "Multi-SEM simultaneously captures geographic distance decay, regional multi-hop spillovers, "
        "and municipal boundaries. Across 50 repeated random holdout splits, Multi-SEM achieves an "
        "average holdout RMSE of 0.1374, a 28.5% reduction relative to OLS "
        "(0.1921; Wilcoxon $p = 1.78 \\times 10^{-15}$) and a 2.1% reduction relative to the "
        "single-layer geographic SEM (0.1403), and it outperforms OLS on all 50 splits. "
        "A 63-month longitudinal panel of S&P 100 equity returns confirms that the pipeline scales "
        "to dynamic, non-geographic networks, outperforming OLS in 44 of 63 months "
        "(sign test $p = 0.0022$; Newey–West $p = 0.072$).",
    )
    b.replace_containing(
        body,
        "Tobler's first law characterizes spatial dependence",
        "Tobler's first law characterizes spatial dependence through continuous distance decay, "
        "whereby nearby observations tend to be more strongly related than distant ones (Tobler, 1970). "
        "Yet administrative, legal, and functional boundaries often intersect with this geographic gradient. "
        "A residential property in Greater Boston has contiguous geographic neighbours, but also belongs "
        "to a municipal jurisdiction that determines property tax assessment, zoning bylaws, and public "
        "school provision (Harrison and Rubinfeld, 1978; Gilley and Pace, 1996). Similarly, a publicly "
        "traded firm belongs to an industry group while also being linked to corporate peers through "
        "return co-movement. Standard spatial regressions traditionally condition on a single weight "
        "matrix $W$ (Bivand et al., 2014; Bivand and Wong, 2018). A single spatial operator cannot "
        "simultaneously accommodate these distinct relational channels. Multi-SEM resolves this limitation "
        "by testing, screening, and combining candidate weight matrices within a unified model while "
        "guaranteeing an invertible spatial filter.",
    )
    b.replace_containing(
        body,
        "Spatial regression models traditionally condition on a single weight matrix",
        "Bivand and Wong (2018) compare implementations of global and local indicators of spatial "
        "association, including Moran's $I$, and show that the numerical results depend on the weights "
        "and on the software defaults used to construct them. Bivand, Gómez-Rubio, and Rue (2014) "
        "develop approximate Bayesian inference for spatial econometric models that each condition on "
        "one weight matrix, and they compare those models with a latent Gaussian Markov random field. "
        "Median housing prices in Boston census tracts are one of their applications.",
    )
    b.replace_span(
        body,
        "Hub-dominated layers can make the Gaussian tail",
        "At each rolling month",
        [
            "Before computing regression residuals or observing holdout responses, we calculate two "
            "ex-ante topological diagnostics: degree heterogeneity "
            "$\\kappa_s = \\max_i d_{s,i} / \\bar{d}_s$ from the unstandardized training adjacency "
            "$W_{s,TT}$, and singular-value concentration "
            "$\\psi_s = \\theta_{s,1} / \\sum_j \\theta_{s,j}$ from the row-standardized training matrix "
            "$A_{s,TT}$. The thresholds $\\kappa_s > 3.0$ and $\\psi_s > 0.20$ are pre-specified "
            "operational heuristics motivated by the finite-sample size distortions documented in "
            "Section 7.2. They identify candidate graphs whose projected residual spectrum is dominated "
            "by extreme eigenmodes and switch those layers to conditional Gaussian Monte Carlo "
            "calibration ($B = 499$ draws). Homogeneous layers satisfying $\\kappa_s \\leq 3.0$ and "
            "$\\psi_s \\leq 0.20$ use the analytical Gaussian tail. Because both diagnostics use only "
            "the training adjacency, the calibration choice is fixed before $Y$ is observed.",
        ],
    )
    b.replace_containing(
        body,
        "Table 8. Posterior parameter estimates for the Boston Housing",
        "Table 8. Posterior parameter estimates for a representative central Boston split "
        "(Split 5, $n_t = 405$ training tracts). The radial layer was not selected. "
        "The interval is the central 95 percent of the retained draws from the archived Gibbs seed.",
    )
    restore_table8(body)
    b.replace_containing(
        body,
        "(Note:",
        "On Split 5 the 95th percentile of $L$ is 0.957, and $P(L > 0.95 \\mid Y_t) = 0.058$. "
        "The geographic and resolvent coefficients on this split have effective sample sizes 69 and 66.",
    )
    b.replace_containing(
        body,
        "Pace and Gilley (1997) exploit the spatial configuration",
        "Pace and Gilley (1997) exploit the spatial configuration of these tracts to improve hedonic fit. "
        "Table 9 separates this continuous distance channel from the municipal-boundary channel. "
        "Relative to the OLS mean RMSE of 0.1921, the geographic six-nearest-neighbour spatial error "
        "model reduces mean RMSE to 0.1403, a 27.0% improvement. Screened Multi-SEM, which retains "
        "the geographic resolvent and the town layer on all 50 splits, reduces mean RMSE further to "
        "0.1374. That is a 2.1% reduction relative to the geographic SEM and a 28.5% reduction relative "
        "to OLS (Wilcoxon $p = 1.78 \\times 10^{-15}$). Multi-SEM outperforms OLS on all 50 splits. "
        "The town SEM alone has a mean RMSE of 0.1499, so the 28.5% figure is the gain of the full "
        "screened predictor over OLS, not the marginal contribution of the town layer. Physical "
        "contiguity and administrative jurisdiction still contribute complementary information: "
        "the multiplex predictor improves on the best single-layer geographic model.",
    )
    b.replace_containing(
        body,
        "MCMC diagnostics differ substantially",
        "MCMC convergence diagnostics reveal a structural distinction between single-layer and "
        "multi-layer specifications. The single-layer geographic and town samplers satisfy "
        "$\\hat{R} \\leq 1.025$ on all 50 splits, with maximum split-$\\hat{R}$ values of 1.006 and "
        "1.010, and every $\\lambda$ effective sample size exceeds 1,200. Split 5, one of the two "
        "central holdout partitions, shows the same multi-layer pattern (Table 8): the geographic "
        "and resolvent coefficients have split-$\\hat{R}$ of 1.122 and 1.133 and effective sample "
        "sizes of 69 and 66, and the town coefficient has split-$\\hat{R}$ 1.014 and effective sample "
        "size 335. Across the 50 screened fits, posterior mass concentrates near the boundary of the "
        "$\\ell_1$ ball. The mean of $E[L \\mid Y_t]$ is 0.886 and the mean of "
        "$P(L > 0.95 \\mid Y_t)$ is 0.185. The minimum $\\lambda$ effective sample size exceeds 400 "
        "on none of the 50 fits, and the largest split-$\\hat{R}$ is 1.257. The regression slopes "
        "remain thoroughly sampled: the $\\beta$ effective sample size exceeds 1,000 on every split "
        "and exceeds 8,900 on Split 5. Table 9 reports the holdout metrics from these original "
        "posterior means. The draws were not replaced by a longer chain.",
    )
    b.replace_containing(
        body,
        "Table 6 uses this matrix.",
        "Table 10 uses this industry matrix. The isolate count is unchanged, and the August 2026 Moran statistic is not recomputed.",
    )
    b.replace_containing(
        body,
        "August 2026 Screening and Posterior Inference",
        "August 2026 Archived Illustrative Snapshot",
    )
    b.insert_before_snippet(
        body,
        "Table 10 reports the archived August 2026 screening results.",
        [
            "To examine MCMC parameter mixing and precision-kriging mechanics on an overlapping network, "
            "this subsection uses the archived August 2026 cross-section. The exploratory screen for "
            "that illustrative snapshot used the full 100-firm cross-section, including the 20 outcomes "
            "later held out. The 63-month panel is the out-of-sample assessment: each month screens "
            "only the 80 training equities.",
        ],
    )


S15 = [
    "S1.5 Invariance of Diagonal-Purged Affiliation Resolvents",
    "Consider an affiliation network consisting of $M$ disjoint groups, where group $m$ contains "
    "$n_m \\geq 2$ nodes. Because self-loops are purged before row-standardization, each group is "
    "represented by the zero-diagonal clique matrix",
    "$C_m = (J_{n_m} - I_{n_m})/(n_m - 1)$.",
    "Two resolvent transformations have to be distinguished.",
    "Unpurged resolvent. Powers of the clique matrix satisfy "
    "$C_m^l \\in \\operatorname{span}\\{J_{n_m}, I_{n_m}\\}$ for every $l \\geq 1$. The unpurged series "
    "$G_{C_m} = \\sum_{l=1}^{\\infty} \\alpha^{l-1} C_m^l = C_m (I - \\alpha C_m)^{-1}$ remains in that "
    "span. Higher-order walks $i \\rightarrow j \\rightarrow i$ put positive mass on the diagonal. "
    "The off-diagonal Pearson correlation between the row-standardized unpurged resolvent and $C_m$ "
    "is about 0.999, so the ex-ante collinearity screen drops it.",
    "Purged resolvent, which is the operator used in Multi-SEM. Every candidate matrix is forced to "
    "a zero diagonal, $A_{s,ii} = 0$. Setting the diagonal of $G_{C_m}$ to zero and then "
    "row-standardizing reproduces the base clique,",
    "$\\widetilde{G}(C_m) = C_m$.",
    "The identity holds, up to numerical error, in both empirical applications. For the largest Boston "
    "municipality, a block of $n_m = 30$ tracts with damping $\\alpha = 0.5$, the largest absolute "
    "entrywise difference between $\\widetilde{G}(C)$ and $C$ is $1.4 \\times 10^{-17}$. Across the "
    "S&P 100 industry blocks the largest absolute entrywise difference is on the order of $10^{-16}$. "
    "In both applications the affiliation resolvent is a duplicate of the base clique and is excluded "
    "before residual screening.",
]


def edit_supplement(body):
    b.replace_span(body, "S1.5 Near-Collinearity", "Section S2.", S15)
    towns = b.require_one(body, "Cohasset, Dover, Duxbury")
    towns_text = b.ptext(towns) + "".join(t.text or "" for t in towns.iter(b.q(M, "t")))
    if "489" not in towns_text:
        raise SystemExit("S4.1 does not state S0,town = 489")
    step = b.require_one(body, "plot_boston_split5.py")
    step_text = b.ptext(step)
    if "Figure 3" not in step_text or "figure_s9.png" not in step_text:
        raise SystemExit("Step 3b does not name Figure 3 and figure_s9.png")
    caption = b.require_one(body, "Supplementary Figure S9. Boston tract centroids")
    if "Figure 3" not in b.ptext(caption):
        raise SystemExit("Figure S9 does not cross-reference Figure 3")


def bare_e(root):
    return sum(
        1
        for math in root.iter(b.q(M, "oMath"))
        for child in math
        if local(child.tag) == "e"
    )


def main():
    files, root = b.load_zip(MAIN_IN)
    print("main indicators", fix_indicator_delimiters(root))
    print("main rejection rules", expand_rejection_rule(root))
    edit_main(root.find(b.q(W, "body")))
    print("main bare e", bare_e(root))
    print("main words", b.word_count_root(root, "References"))
    b.store_zip(MAIN_OUT, files, root)

    files, root = b.load_zip(SUP_IN)
    print("supp indicators", fix_indicator_delimiters(root))
    print("supp rejection rules", expand_rejection_rule(root))
    edit_supplement(root.find(b.q(W, "body")))
    print("supp bare e", bare_e(root))
    print("supp words", b.word_count_root(root, "References"))
    b.store_zip(SUP_OUT, files, root)
    print(MAIN_OUT.name)
    print(SUP_OUT.name)


if __name__ == "__main__":
    main()
