"""Build manuscript v14 and supplement v12 from v13 and supplement v11.

Does not overwrite those source files. The August 2026 S&P screen remains the
archived full-sample calculation.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lxml import etree

import build_v12_docx as b
from build_v10_docx import append_row, insert_before, replace_with

ROOT = b.ROOT
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v13.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v11.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v14.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v12.docx"
FIG_PATH = ROOT / "results" / "boston" / "figure_s9.png"


def para(text, anchor):
    return b.make_p(b.blocks_from(text), anchor)


def insert_many(anchor, texts):
    made = [para(text, anchor) for text in texts]
    for node in made:
        anchor.addprevious(node)
    return made


def replace_para(body, snippet, text):
    target = b.require_one(body, snippet)
    replace_with(target, b.blocks_from(text))


def plain_table(headers, rows):
    from docx import Document
    from docx.shared import Pt

    document = Document()
    table = document.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    for j, header in enumerate(headers):
        table.rows[0].cells[j].text = header
        for run in table.rows[0].cells[j].paragraphs[0].runs:
            run.bold = True
            run.font.size = Pt(9)
            run.font.name = "Times New Roman"
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            table.rows[i + 1].cells[j].text = value
            for run in table.rows[i + 1].cells[j].paragraphs[0].runs:
                run.font.size = Pt(9)
                run.font.name = "Times New Roman"
    return table._tbl


def rewrite_sp_tables(nodes):
    """Renumber the existing S&P tables only. Boston tables are not in this list."""
    order = (
        ("Table 9.", "Table 12."),
        ("Table 8.", "Table 11."),
        ("Table 7.", "Table 10."),
        ("Table 6.", "Table 9."),
        ("in Table 7", "in Table 10"),
        ("in Table 6", "in Table 9"),
        ("Table 6 is", "Table 9 is"),
        ("Table 6,", "Table 9,"),
    )
    for node in nodes:
        if node.tag != b.q(b.W, "p"):
            continue
        for old, new in order:
            for t in node.iter(b.q(b.W, "t")):
                if t.text and old in t.text:
                    t.text = t.text.replace(old, new)


def retitle_sp(nodes):
    replacements = (
        ("8. Empirical Application: S&P 100 Equity Co-Movement and Return Imputation",
         "8. Empirical Applications"),
        ("8.1 Data Construction and Candidate Network Layers",
         "8.2 Dynamic Relational Imputation: S&P 100 Panel"),
        ("8.2 Network Screening and Bayesian Posterior Inference (Showcase Month: August 2026)",
         "August 2026 screening and posterior inference"),
        ("8.3 Out-of-Sample Evaluation: August 2026 Showcase",
         "August 2026 holdout"),
        ("8.4 Longitudinal Rolling-Window Evaluation (June 2021 – August 2026)",
         "Sixty-three month panel"),
    )
    for node in nodes:
        if node.tag != b.q(b.W, "p"):
            continue
        text = b.ptext(node).strip()
        for old, new in replacements:
            if text.startswith(old):
                for t in node.iter(b.q(b.W, "t")):
                    if t.text and old in t.text:
                        t.text = t.text.replace(old, new, 1)
                        break


def boston_nodes(anchor):
    texts_before_tables = [
        "The same pipeline is applied to a geographic hedonic cross-section and to a non-geographic equity panel. Section 8.1 is the Boston tract study. Section 8.2 is the S&P 100 panel.",
        "8.1 Spatial Hedonic Imputation in Greater Boston",
        "The spatial study uses the corrected Harrison–Rubinfeld file distributed as boston_corrected.txt (Harrison and Rubinfeld, 1978; Gilley and Pace, 1996; Pace and Gilley, 1997). It contains 506 census tracts in 92 municipalities. The outcome is $Y = \\log(\\mathrm{cmedv})$, with value in thousands of dollars. Sixteen tracts remain at the reporting ceiling of 50, so those outcomes are censored at $\\log 50$.",
        "The design has an intercept and ten regressors: rooms in levels, age, crime, large-lot zoning, non-retail business acreage, $\\log(\\mathrm{lstat})$, the property-tax rate, the pupil–teacher ratio, nitric oxides, and $\\log(\\mathrm{dis})$. Charles River adjacency and the racial-composition variable are not included. On each split the ten columns are standardized with the training mean and the training population standard deviation, and those constants are applied to the holdout. Longitude and latitude are projected to WGS84 UTM zone 19N before distances are computed. Easting runs from 310,852 m to 350,236 m, and northing from 4,654,767 m to 4,693,945 m.",
        "Four candidate layers are built without $Y$. Geographic proximity is a symmetrized inverse-distance graph on the six nearest neighbours. The resolvent is $A_{\\mathrm{geo}}(I - 0.5 A_{\\mathrm{geo}})^{-1}$ with the diagonal removed and the rows standardized again. The town layer is the block of tracts that share a municipal code. The radial layer is the block of tracts that share the highway-access index. Table 6 records the full-sample topology. Kappa is the maximum row sum of the unstandardized weights divided by the mean row sum. Psi is the leading singular value of the row-standardized matrix divided by the sum of the singular values.",
    ]
    mid = [
        "The town layer has 17 isolate tracts, one in each singleton municipality, so $S_{0,\\mathrm{town}} = 489$. The largest off-diagonal correlation among the four matrices is 0.975, between geography and its resolvent, which is below the 0.99 prune. All four layers stay in the candidate set. On the largest municipality, a block of 30 tracts, the diagonal-purged and row-standardized resolvent of the town clique matches the clique. The largest absolute entrywise difference is $1.4 \\times 10^{-17}$, which is why a town resolvent is not added.",
        "Each of 50 splits, drawn from seed 2026, keeps 405 tracts for training and 101 for the holdout. Residuals, kappa, psi, Moran statistics, and the Gibbs sampler use the induced training blocks only. Holdout outcomes are not used until the predictions are scored. There is no fallback that retains a layer with $p < 0.10$ after Benjamini–Hochberg control at 0.05. Table 7 summarizes the 50 training screens.",
    ]
    after7 = [
        "The geographic layer uses the $B = 499$ Monte Carlo reference on all 50 splits because its mean kappa is 4.12. That kappa is a ratio of inverse-distance row sums, not a binary degree ratio. The town layer, whose weights are zero-one, uses the Monte Carlo reference on 25 splits; its mean kappa is 3.02. The resolvent and the radial layer stay on the normal tail. Psi never reaches 0.20. Geography, the resolvent, and the town layer are retained in every split. The radial layer is retained in 28 splits.",
        "The sampler is Algorithm 2: four chains, 6,000 iterations, and 3,000 burn-in draws, with the prior in Table 1. Predictions use the posterior means and the full 506-by-506 networks. This is induced-subgraph estimation with full-graph imputation. The holdout correction factorizes the $101 \\times 101$ precision block rather than the $405 \\times 405$ training covariance. The ratio of those cubic terms is $(405/101)^3 \\approx 64.5$. Each Metropolis proposal still factorizes the training filter. Table 8 reports the holdout scores on the log-value scale.",
    ]
    after8 = [
        "Screened Multi-SEM with precision kriging has mean RMSE 0.1374, against 0.1921 for OLS, a 28.5% reduction in the ratio of the means. Its RMSE is smaller on all 50 splits, so the Wilcoxon signed-rank statistic is 0 and the two-sided $p$-value is $1.78 \\times 10^{-15}$.",
        "The geographic SEM alone reaches 0.1403 and already beats OLS on 49 splits. The multiplex predictor is lower still, so the town blocks are not redundant once Euclidean neighbours are included. Municipal fixed effects reach 0.1520, which is worse than the single town SEM at 0.1499. Zoning, industry acreage, the tax rate, and the pupil–teacher ratio do not vary inside a town, so the fixed-effects fit absorbs them into the town intercept. A holdout tract whose town is absent from the training sample is predicted at the reference-town intercept; that occurs for 4.92 tracts per split on average. The town SEM keeps $E[Y \\mid X] = X\\beta$ and puts the municipal clustering in the residual covariance.",
        "Multi-SEM without kriging has mean RMSE 0.2072 and beats OLS on 4 of the 50 splits. The 28.5% reduction is the precision correction from contemporaneous training residuals, not a different regression mean.",
        "The one-layer geographic and town samplers meet $\\hat{R} \\leq 1.025$ on all 50 splits. Their largest split-$\\hat{R}$ values are 1.006 and 1.010, and every lambda effective sample size on those fits exceeds 1,200. The screened fits, which retain three or four layers, do not mix to the same standard. The mean of $E[L \\mid Y_t]$ is 0.886 and the mean of $P(L > 0.95 \\mid Y_t)$ is 0.185. Lambda effective sample size exceeds 400 on 0 of 50 screened fits, while slope effective sample size exceeds 1,000 on all 50. The largest screened split-$\\hat{R}$ is 1.257. Table 8 uses those posterior means. They were not replaced by a longer run.",
    ]
    table6 = plain_table(
        ["Layer", "Edges", "Isolates", "S0", "Kappa", "Psi", "Role"],
        [
            ["Geographic 6-NN", "1,858", "0", "506", "3.75", "0.006", "Local distance decay"],
            ["Geographic resolvent", "127,765", "0", "506", "1.05", "0.009", "Multi-hop spillover"],
            ["Town jurisdiction", "2,434", "17", "489", "3.01", "0.007", "Municipal boundaries"],
            ["Radial highway", "23,102", "0", "506", "1.43", "0.056", "Commuting corridors"],
        ],
    )
    table7 = plain_table(
        ["Layer", "Mean I", "Mean Z", "Mean kappa", "Mean psi", "Monte Carlo", "Retained"],
        [
            ["Geographic 6-NN", "0.442", "15.03", "4.12", "0.007", "50 / 50", "50 / 50"],
            ["Geographic resolvent", "0.406", "18.19", "1.26", "0.010", "0 / 50", "50 / 50"],
            ["Town jurisdiction", "0.450", "13.23", "3.02", "0.007", "25 / 50", "50 / 50"],
            ["Radial highway", "0.013", "2.13", "1.43", "0.056", "0 / 50", "28 / 50"],
        ],
    )
    table8 = plain_table(
        ["Model", "Mean RMSE", "Mean MAE", "Correlation", "Wins versus OLS"],
        [
            ["OLS (no network)", "0.1921", "0.1404", "0.879", "Reference"],
            ["Municipal fixed effects", "0.1520", "0.1097", "0.927", "48 / 50"],
            ["Single SEM (town)", "0.1499", "0.1097", "0.928", "50 / 50"],
            ["Single SEM (geographic 6-NN)", "0.1403", "0.1007", "0.938", "49 / 50"],
            ["Multi-SEM, no kriging", "0.2072", "0.1490", "0.861", "4 / 50"],
            ["Multi-SEM, screened, precision kriging", "0.1374", "0.0983", "0.940", "50 / 50"],
        ],
    )
    nodes = [para(text, anchor) for text in texts_before_tables]
    nodes.append(para(
        "Table 6. Full-sample Boston candidate networks. Kappa uses unstandardized row sums. Psi uses the singular values of the row-standardized matrix.",
        anchor,
    ))
    nodes.append(table6)
    nodes.extend(para(text, anchor) for text in mid)
    nodes.append(para(
        "Table 7. Residual Moran screening on the 405 training tracts, averaged over 50 splits. Monte Carlo is the share of splits in which kappa or psi selected the $B = 499$ reference.",
        anchor,
    ))
    nodes.append(table7)
    nodes.extend(para(text, anchor) for text in after7)
    nodes.append(para(
        "Table 8. Mean holdout error on $\\log(\\mathrm{cmedv})$ over 50 random splits of 101 tracts. A win is a strictly smaller RMSE than OLS on that split.",
        anchor,
    ))
    nodes.append(table8)
    nodes.extend(para(text, anchor) for text in after8)
    return nodes


def make_figure():
    import run_boston as rb

    data = rb.load_boston()
    built = rb.build_networks(data)
    record = json.loads((ROOT / "results" / "boston" / "split_05.json").read_text(encoding="utf-8"))
    seed = int(record["seed"])
    part = np.random.default_rng([seed, 1])
    train = np.sort(part.choice(rb.N, size=rb.N_TRAIN, replace=False))
    test = np.sort(np.setdiff1d(np.arange(rb.N), train))
    raw = data["raw"]
    center = raw[train].mean(axis=0)
    spread = raw[train].std(axis=0, ddof=0)
    spread = np.where(spread < 1e-12, 1.0, spread)
    scaled = (raw - center) / spread
    x_train = np.column_stack([np.ones(rb.N_TRAIN), scaled[train]])
    x_test = np.column_stack([np.ones(rb.N_TEST), scaled[test]])
    y_train = data["y"][train]
    y_test = data["y"][test]
    names = record["multi"]["names"]
    yhat = rb.precision_kriging(
        y_train, x_train, x_test, train, test,
        np.asarray(record["multi"]["lambda"], dtype=float),
        np.asarray(record["multi"]["beta"], dtype=float),
        [built["adjacency"][name] for name in names],
    )
    rmse = float(np.sqrt(np.mean((yhat - y_test) ** 2)))
    stored = float(record["multi"]["RMSE"])
    if abs(rmse - stored) > 5e-4:
        raise SystemExit(f"split 5 RMSE recomputed {rmse} stored {stored}")
    xy = data["xy"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6))
    axes[0].scatter(xy[:, 0], xy[:, 1], c=data["town_id"], s=12, cmap="nipy_spectral", linewidths=0)
    axes[0].set_title("Tract centroids by municipality")
    axes[0].set_xlabel("UTM easting (m)")
    axes[0].set_ylabel("UTM northing (m)")
    axes[0].set_aspect("equal")
    residual = y_test - yhat
    scale = max(float(np.max(np.abs(residual))), 1e-6)
    sc = axes[1].scatter(
        xy[test, 0], xy[test, 1], c=residual, s=28, cmap="coolwarm",
        vmin=-scale, vmax=scale, linewidths=0,
    )
    axes[1].set_title("Holdout residual, split 5")
    axes[1].set_xlabel("UTM easting (m)")
    axes[1].set_aspect("equal")
    fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04, label="log value")
    for ax in axes:
        ax.tick_params(labelsize=8)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=160)
    plt.close(fig)


def edit_main():
    files, root = b.load_zip(MAIN_IN)
    body = root.find(b.q(b.W, "body"))
    replace_para(
        body,
        "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation",
        "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation Across Geographic and Network Domains",
    )
    replace_para(
        body,
        "Empirical entities frequently interact",
        "Many empirical systems are multiplex: interconnected entities participate simultaneously across physical space, administrative jurisdictions, and functional relational networks. This paper develops Multiplex Spatial Error Regression (Multi-SEM) as an integrated econometric pipeline combining ex-ante structural pruning, topology-calibrated residual screening, $\\ell_1$-constrained Bayesian estimation, and precision-form spatial kriging.",
    )
    replace_para(
        body,
        "Multi-SEM addresses four methodological challenges",
        "We resolve four methodological challenges: (i) an additive disturbance specification preserves linear conditional-mean interpretability ($E[Y \\mid X] = X\\beta$) under exogeneity without spatial-lag endogeneity; (ii) the parameter restriction $\\lVert \\lambda \\rVert_1 = \\sum_{s=1}^{k} |\\lambda_s| < 1$ provides a uniform sufficient condition ensuring spectral stability across row-substochastic layers; (iii) candidate layers are screened via residual Moran's $I$ using exact design-conditional null moments with isolate scaling, triggering conditional Monte Carlo calibration under hub-dominated topologies prior to false discovery rate control; and (iv) unobserved nodes are imputed via precision kriging ($Q = \\sigma^{-2} V_\\lambda' V_\\lambda$), reducing computational complexity from $O(n_t^3)$ to $O(n_*^3)$.",
    )
    replace_para(
        body,
        "In synthetic simulations, Multi-SEM enforces",
        "We evaluate Multi-SEM across two substantive empirical benchmarks. In a spatial hedonic study of $n = 506$ Boston census tracts across 92 municipalities, Multi-SEM simultaneously captures geographic distance decay, regional multi-hop spillovers, and municipal boundaries, achieving a 28.5% holdout RMSE reduction over OLS across 50 repeated random splits (Wilcoxon $p = 1.78 \\times 10^{-15}$). A 63-month longitudinal panel of S&P 100 equity returns confirms that the pipeline scales to dynamic, non-geographic networks, outperforming OLS in 44 of 63 months ($p = 0.0022$).",
    )
    replace_para(
        body,
        "Keywords:",
        "Keywords: Multiple spatial weight matrices; Spatial Error Model; Residual Moran's I; Precision kriging; Hedonic housing; S&P 100.",
    )
    replace_para(
        body,
        "Spatial econometrics models such dependence through spatial lag",
        "Tobler's first law treats dependence as continuous distance decay: near observations are more strongly related than distant ones (Tobler, 1970). Administrative and functional boundaries cut across that gradient. A housing tract in Greater Boston has a geographic neighbour and, separately, a municipal jurisdiction that sets the tax rate and the schools (Harrison and Rubinfeld, 1978; Gilley and Pace, 1996). An S&P 100 firm has an industry label and, separately, a set of names with which its returns co-move. A specification with one weight matrix has to discard one of those channels.",
    )
    # The snippet above may not match because paragraph 10's actual opening differs.
    # require_one will have raised if so. The call is ordered after we know the text.
    replace_para(
        body,
        "On synthetic graphs, Multi-SEM enforces spectral invertibility",
        "On synthetic graphs the oracle specification reduces holdout RMSE by 8.6% relative to OLS, and the screened specification by 6.1%. On the corrected Boston tracts, screened Multi-SEM with precision kriging reduces mean holdout RMSE by 28.5% across 50 random splits (Wilcoxon $p = 1.78 \\times 10^{-15}$). On 63 months of S&P 100 returns it beats OLS in 44 months (sign test $p = 0.0022$). Proofs, prior checks, and the replication code are in the Supplementary Materials.",
    )
    anchor_33 = b.require_one(body, "3.4 Affiliation Clique")
    insert_many(anchor_33, [
        "Singleton groups produce those zero rows without adding a self-loop. The Boston town layer has 17 singleton municipalities, so $n_{\\mathrm{isolate}} = 17$, $S_{0,\\mathrm{town}} = 506 - 17 = 489$, and the Moran scale is $c_{\\mathrm{town}} = 506/489 \\approx 1.035$. The S&P 100 industry layer has the same isolate count, with $S_{0,\\mathrm{ind}} = 83$ and $c_{\\mathrm{ind}} = 100/83 \\approx 1.205$. A zero row stops $V_\\lambda 1_n$ from shrinking on every node. The filtered intercept contracts only on the connected rows, and the information loss appears as variance inflation rather than a global rank collapse.",
    ])
    anchor_35 = b.require_one(body, "3.5 Likelihood Evaluation")
    insert_many(anchor_35, [
        "The identity is numerical as well as algebraic. On the largest Boston municipality, a block of 30 tracts, purging the diagonal of $C(I - 0.5 C)^{-1}$ and row-standardizing recovers $C$. The largest absolute entrywise difference is $1.4 \\times 10^{-17}$. Town resolvents are therefore excluded before screening. The purged S&P 100 industry cliques match their base blocks to about $10^{-16}$.",
    ])
    replace_para(
        body,
        "Hub-dominated layers make the Gaussian tail liberal.",
        "Hub-dominated layers make the Gaussian tail liberal. Before any residual is formed, compute $\\kappa_s = \\max_i d_{s,i} / \\bar{d}_s$ from the row sums of the unstandardized training block $W_{s,TT}$, and $\\psi_s = \\theta_{s,1} / \\sum_j \\theta_{s,j}$ from the singular values of the row-standardized block $A_{s,TT}$. For a zero-one weight matrix the row sum is the degree. If $\\kappa_s > 3$ or $\\psi_s > 0.20$, the $p$-value is a conditional Gaussian Monte Carlo probability with $B = 499$ draws. Otherwise it is the analytical tail $2(1 - \\Phi(|Z_s|))$.",
    )
    replace_para(
        body,
        "Step 1. For every pair",
        "Step 1. For every pair in $\\{A_1, \\ldots, A_K\\}$, compute the off-diagonal Pearson correlation $r_{\\mathrm{vec}}(A_r, A_s)$. If $|r_{\\mathrm{vec}}| \\geq 0.99$, drop the redundant layer. Call the retained set $\\{A_1, \\ldots, A_{K'}\\}$. This step does not use $Y$.",
    )
    replace_para(
        body,
        "Step 2. On the training rows",
        "Step 2. On the training rows only, compute $e_t = M_t Y_t$ with $M_t = I_{n_t} - X_t (X_t' X_t)^{-1} X_t'$.",
    )
    replace_para(
        body,
        "from the moments in Proposition 1",
        "Step 3. For each retained layer, compute $\\kappa_s$ on $W_{s,TT}$ and $\\psi_s$ on $A_{s,TT}$, and compute $Z_s$ from Proposition 1. If $\\kappa_s > 3$ or $\\psi_s > 0.20$, compute $p_s$ from $B = 499$ conditional Gaussian draws. Otherwise set $p_s = 2(1 - \\Phi(|Z_s|))$.",
    )
    replace_para(
        body,
        "Step 4. Apply Benjamini",
        "Step 4. Apply Benjamini–Hochberg control at $q^* = 0.05$ and retain $\\mathcal{S} = \\{s : p_s \\leq p_{(k^*)}\\}$. There is no later step that drops a selected layer because another selected layer has a smaller Moran $p$-value.",
    )
    anchor_63 = b.require_one(body, "6.3 Kriging versus")
    insert_many(anchor_63, [
        "For a Boston split, $n_t = 405$ and $n_* = 101$, so the holdout factorization is smaller than a dense factor of the $405 \\times 405$ training covariance by $(405/101)^3 \\approx 64.5$. For an S&P 100 month the same ratio is $(80/20)^3 = 64$. Both numbers describe the imputation solve. The Metropolis loop still factorizes the $n_t \\times n_t$ filter $V_\\lambda$ once per proposal.",
    ])
    sp_nodes = []
    seen = False
    for child in list(body):
        if child.tag == b.q(b.W, "p") and b.ptext(child).startswith("8. Empirical Application"):
            seen = True
        if seen and child.tag == b.q(b.W, "p") and b.ptext(child).startswith("9. Conclusion"):
            break
        if seen:
            sp_nodes.append(child)
    if len(sp_nodes) < 20:
        raise SystemExit(f"S&P section span is only {len(sp_nodes)} blocks")
    old_81 = b.require_one(body, "8.1 Data Construction")
    for node in boston_nodes(old_81):
        old_81.addprevious(node)
    retitle_sp(sp_nodes)
    rewrite_sp_tables(sp_nodes)
    replace_para(
        body,
        "In the simulations the oracle specification reduces holdout RMSE by 8.6%",
        "In the simulations the oracle specification reduces holdout RMSE by 8.6% relative to OLS, and the screened specification by 6.1%. Active-coefficient intervals at $n = 200$ undercover, at 87.5% and 89.5%. The induced-subgraph fit is a fast surrogate when cross-block ties are missing; the Schur precision $P_T$ is the coherent training likelihood when they are observed. On the Boston tracts, screened precision kriging reduces mean holdout RMSE by 28.5% across 50 splits, and the predictor that does not condition on peer residuals is worse than OLS. On 63 months of S&P 100 returns, Multi-SEM beats OLS in 44 months (sign test $p = 0.0022$). The gain there is also in the same-date kriging correction. The Newey–West test of the monthly loss differential has $p = 0.072$. Extensions to dynamic panels and to non-Gaussian outcomes are left open.",
    )
    add_references(body)
    b.store_zip(MAIN_OUT, files, root)


def add_reference(body, before_snippet, text):
    anchor = b.require_one(body, before_snippet)
    anchor.addprevious(para(text, anchor))


def add_references(body):
    add_reference(
        body,
        "Godoy-Lorite",
        "Gilley, O. W., and Pace, R. K. (1996). On the Harrison and Rubinfeld data. Journal of Environmental Economics and Management, 31(3), 403–405.",
    )
    add_reference(
        body,
        "Hastie, T.",
        "Harrison, D., and Rubinfeld, D. L. (1978). Hedonic housing prices and the demand for clean air. Journal of Environmental Economics and Management, 5(1), 81–102.",
    )
    add_reference(
        body,
        "Pan, M.",
        "Pace, R. K., and Gilley, O. W. (1997). Using the spatial configuration of the data to improve estimation. Journal of Real Estate Finance and Economics, 14(3), 333–340.",
    )
    add_reference(
        body,
        "Yu, D.",
        "Tobler, W. R. (1970). A computer movie simulating urban growth in the Detroit region. Economic Geography, 46, 234–240.",
    )


def edit_supplement():
    files, root = b.load_zip(SUP_IN)
    body = root.find(b.q(b.W, "body"))
    replace_para(
        body,
        "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation",
        "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation Across Geographic and Network Domains",
    )
    anchor = b.require_one(body, "Section S2.")
    insert_many(anchor, [
        "The purged resolvent was evaluated on both empirical clique families. For the largest Boston municipality, a 30-tract town clique with damping 0.5, the largest absolute entrywise difference between the diagonal-purged row-standardized resolvent and the original clique is $1.4 \\times 10^{-17}$. For the S&P 100 industry affiliation cliques the corresponding difference is on the order of $10^{-16}$. In both cases the affiliation resolvent is excluded before screening.",
    ])
    heading = b.require_one(body, "Section S4.")
    for t in heading.iter(b.q(b.W, "t")):
        if t.text and t.text.startswith("Section S4."):
            t.text = "Section S4.  Empirical Manifests and Data Specifications"
            break
    renames = (
        ("S4.5 Network-Selection Dynamics", "S4.6 Network-Selection Dynamics"),
        ("S4.4 Longitudinal", "S4.5 Longitudinal"),
        ("S4.3 August 2026 Showcase", "S4.4 August 2026 Showcase"),
        ("S4.2 Construction of Firm-Level Covariates", "S4.3 Construction of Firm-Level Covariates"),
        ("S4.1 Constituent Listing", "S4.2 Constituent Listing"),
    )
    for old, new in renames:
        node = b.require_one(body, old)
        for t in node.iter(b.q(b.W, "t")):
            if t.text and old in t.text:
                t.text = t.text.replace(old, new, 1)
                break
    s42 = b.require_one(body, "S4.2 Constituent Listing")
    insert_many(s42, [
        "S4.1 Boston Housing Data Specification",
        "The Boston file is StatLib boston_corrected.txt, saved as data/boston_corrected.txt. It has 506 tracts and 92 town codes, numbered 0 through 91. The outcome is the natural logarithm of corrected median value. Sixteen tracts still have corrected value 50. The ten regressors are rooms in levels, age, crime, the large-lot zoning share, non-retail business acreage, log lower-status share, the property-tax rate, the pupil–teacher ratio, nitric oxides, and log distance to employment centres. Tract longitude and latitude are projected to WGS84 UTM zone 19N. Easting on the sample runs from 310,852 m to 350,236 m, and northing from 4,654,767 m to 4,693,945 m.",
        "Four regressors do not vary inside any municipality: the large-lot zoning share, non-retail business acreage, the property-tax rate, and the pupil–teacher ratio. A municipal fixed-effects regression therefore absorbs those four columns into the town intercepts. Nitric oxides varies inside 12 towns, and log distance varies inside 51 towns.",
        "Seventeen towns contain a single tract and are isolates in the town network: Cohasset, Dover, Duxbury, Hamilton, Hanover, Hull, Lincoln, Manchester, Medfield, Middleton, Millis, Nahant, Norfolk, Norwell, Sherborn, Topsfield, and Wenham. The remaining towns have between 2 and 30 tracts. Fifteen towns have 2 tracts, and the largest town block has 30. The active sum of the row-standardized town matrix is $S_{0,\\mathrm{town}} = 489$.",
    ])
    s7 = None
    for child in body:
        if child.tag != b.q(b.W, "tbl"):
            continue
        header = "".join(t.text or "" for t in child.iter(b.q(b.W, "t")))
        if header.startswith("Path"):
            s7 = child
            break
    if s7 is None:
        raise SystemExit("Table S7 not found")
    append_row(s7, [
        "run_boston.py",
        "Load boston_corrected.txt, build four layers, run 50 training-only splits, fit the Gibbs models, write results/boston/summary.json.",
        "numpy, scipy",
    ])
    append_row(s7, [
        "build_boston_docx.py",
        "Write the standalone Boston note from the saved split files.",
        "python-docx",
    ])
    step = b.require_one(body, "Step 4: Reproduce the complete analysis")
    insert_many(step, [
        "Step 3b: Reproduce the Boston housing analysis",
        "Run python run_boston.py. The script reads data/boston_corrected.txt, screens and estimates on the 405 training tracts of each split, and writes results/boston/split_00.json through split_49.json together with results/boston/summary.json. The reported run used 12 worker processes and took about 2.5 hours, because each proposal factorizes a 405 by 405 filter. python build_boston_docx.py rebuilds the standalone Boston note from those files. Supplementary Figure S9 is written to results/boston/figure_s9.png by build_v14_docx.py.",
    ])
    attach_figure(files, root, body)
    b.store_zip(SUP_OUT, files, root)


def attach_figure(files, root, body):
    rels = etree.fromstring(files["word/_rels/document.xml.rels"])
    ids = [rel.get("Id") for rel in rels]
    number = 1
    while f"rId{number}" in ids:
        number += 1
    rid = f"rId{number}"
    rel = etree.SubElement(rels, "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    rel.set("Id", rid)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", "media/image9.png")
    files["word/media/image9.png"] = FIG_PATH.read_bytes()
    files["word/_rels/document.xml.rels"] = etree.tostring(
        rels, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    template = None
    for child in body:
        if child.tag == b.q(b.W, "p") and any(
            el.tag.endswith("blip") for el in child.iter()
        ):
            template = child
    if template is None:
        raise SystemExit("no drawing paragraph to clone")
    drawing = deepcopy(template)
    office = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    found = False
    for blip in drawing.iter():
        if blip.tag.endswith("blip"):
            blip.set(f"{office}embed", rid)
            found = True
    if not found:
        raise SystemExit("cloned drawing has no blip")
    caption = b.require_one(body, "Supplementary Figure S8.")
    caption.addnext(drawing)
    drawing.addnext(para(
        "Supplementary Figure S9. Boston tract centroids in WGS84 UTM zone 19N. The left panel colors the 506 centroids by municipal code. The StatLib file does not contain digitized town polygons, so legal boundaries are not drawn. The right panel shows the holdout residual, actual minus predicted log value, for split 5 of the screened precision-kriging predictor. That split is the one whose Multi-SEM RMSE is closest to the 50-split median.",
        caption,
    ))


def word_count(path):
    files, root = b.load_zip(path)
    del files
    body = root.find(b.q(b.W, "body"))
    n = 0
    for child in body:
        if child.tag == b.q(b.W, "p") and b.ptext(child).startswith("References"):
            break
        if child.tag == b.q(b.W, "tbl"):
            continue
        n += len(b.ptext(child).split())
    return n


def main():
    make_figure()
    edit_main()
    edit_supplement()
    print("main words before references, excluding tables:", word_count(MAIN_OUT))
    print(MAIN_OUT)
    print(SUP_OUT)


if __name__ == "__main__":
    main()
