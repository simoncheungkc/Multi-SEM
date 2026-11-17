"""Manuscript v23, supplement v21, and the formal response to the reviewers."""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from build_revision_docs import (
    M_NS,
    _assert_s41_clean,
    body_of,
    following_table,
    insert_after,
    replace_paragraph,
)
from build_v10_docx import q, set_cell_text
from build_v12_docx import W, load_zip, require_one, store_zip

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v22.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v23.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v20.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v21.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers.docx"


def _lowercase_network_k(body) -> int:
    changed = 0
    for node in body.iter(f"{{{M_NS}}}t"):
        if node.text and node.text.strip() == "K":
            node.text = "k"
            changed += 1
    if changed != 3:
        raise SystemExit(f"expected to lowercase three network-count K symbols, changed {changed}")
    return changed


def _table3_headers(body) -> None:
    table = following_table(body, "Table 3. Empirical size and rejection rates")
    labels = [
        "Network",
        "Role",
        "n=100 Type I error (null DGP)",
        "n=100 power or false discoveries (active DGP)",
        "n=200 Type I error (null DGP)",
        "n=200 power or false discoveries (active DGP)",
        "n=500 Type I error (null DGP)",
        "n=500 power or false discoveries (active DGP)",
    ]
    cells = table.find(q(W, "tr")).findall(q(W, "tc"))
    if len(cells) != len(labels):
        raise SystemExit(f"Table 3 has {len(cells)} columns")
    for cell, text in zip(cells, labels):
        set_cell_text(cell, text)


def edit_main(body) -> None:
    replace_paragraph(
        body,
        "We evaluate Multi-SEM using two substantive empirical benchmarks.",
        (
            "We evaluate Multi-SEM using two substantive empirical benchmarks. In a spatial hedonic study of "
            r"$n=506$ Boston census tracts across 92 municipalities, Multi-SEM simultaneously captures geographic "
            "distance decay, regional multi-hop spillovers, and municipal boundaries. Across 50 repeated random "
            r"80/20 splits, Multi-SEM reduces mean holdout RMSE by 28.5% relative to OLS (0.1375 versus 0.1921; "
            r"Nadeau–Bengio resampled $t_{49}=7.29$, $p=2.34\times 10^{-9}$, outperforming OLS on 50 of 50 splits) "
            "and by 2.0% relative to a single-layer geographic SEM (0.1403), achieving lower prediction error on "
            r"37 of 50 splits (resampled $t_{49}=1.11$, $p=0.272$). A 63-month longitudinal panel of S&P 100 "
            r"equity returns confirms that the pipeline scales to dynamic, non-geographic networks, outperforming "
            r"OLS in 44 of 63 months (sign test $p=0.0022$; Newey–West $p=0.072$)."
        ),
    )
    insert_after(
        require_one(body, "Higher-order spatial autoregressive and spatial-error specifications have been part of the classical"),
        [
            (
                "Bayesian model averaging across weight matrices (LeSage and Fischer, 2008; Piribauer and "
                "Crespo Cuaresma, 2016) treats competing matrices under a model-selection assumption in which "
                "one network is the true generator. Multi-SEM instead models coexisting, overlapping dependence "
                "channels together, through one additive disturbance."
            )
        ],
        require_one(body, "Higher-order spatial autoregressive and spatial-error specifications have been part of the classical"),
    )
    insert_after(
        require_one(body, "Apply Benjamini–Hochberg control at"),
        [
            (
                r"In Algorithm 1, $k^{*}$ is the Benjamini–Hochberg cutoff rank, "
                r"$k^{*}=\max\{i:p_{(i)}\le (i/k) q^{*}\}$, and $k$ is the number of layers that remain after "
                "structural pruning."
            )
        ],
        require_one(body, "Apply Benjamini–Hochberg control at"),
    )
    _lowercase_network_k(body)
    _table3_headers(body)
    replace_paragraph(
        body,
        "The screened Multi-SEM with precision kriging achieves a mean holdout RMSE of",
        (
            "Screened Multi-SEM with precision kriging achieves a mean holdout RMSE of 0.1375, compared with "
            "0.1921 for OLS, a 28.5% reduction. Multi-SEM achieves a lower RMSE than OLS on 50 of 50 splits. "
            r"The Nadeau and Bengio (2003) corrected resampled test, which accounts for training overlap by "
            r"inflating the variance of the mean loss gap by $1/R+n_*/n_t=0.269$, yields "
            r"$t_{49}=7.29$, $p=2.34\times 10^{-9}$ (corrected 95% interval $[0.0131, 0.0230]$). Against the "
            "single-layer geographic SEM (0.1403), Multi-SEM achieves an incremental 2.0% RMSE reduction and "
            r"wins on 37 of 50 splits. The corrected resampled test gives $t_{49}=1.11$, $p=0.272$ "
            "(corrected 95% interval $[-0.0006, 0.0021]$). The split-level win count is 37 of 50. The "
            "corrected confidence interval for the mean squared-error gap spans zero."
        ),
    )
    replace_paragraph(
        body,
        "Screened Multi-SEM achieves a lower holdout RMSE than the geographic SEM on",
        (
            "Supplementary Table S9 isolates the candidate layers on the same 50 splits. The geographic SEM "
            "alone has mean RMSE 0.1403. The resolvent alone, a higher-order filter on that graph, has mean "
            r"RMSE 0.1377. Geographic contiguity together with municipal boundaries, $A_{\mathrm{geo}}+A_{\mathrm{town}}$, "
            "lowers mean RMSE to 0.1384 and has the lower RMSE on 33 of 50 splits, without the resolvent. "
            "Geography together with the resolvent has mean RMSE 0.1383. Screened Multi-SEM has the lowest "
            "mean RMSE, 0.1375. Municipal boundaries change holdout error relative to the geographic SEM alone. "
            "They do not account for more of that point estimate than the resolvent filter. None of these "
            "contrasts with the geographic SEM is significant at the 5% level after the Nadeau–Bengio correction. "
            "The smallest such $p$-value is 0.060, for geography plus the resolvent."
        ),
    )
    replace_paragraph(
        body,
        "Supplementary Table S9 isolates the candidate networks on the same 50 splits.",
        (
            "The same conclusion is in Table S9: the municipal increment and the resolvent increment are both "
            "visible in the point estimates, and the corrected intervals against the geographic SEM include zero."
        ),
    )


def edit_supplement(body) -> None:
    _assert_s41_clean(body)
    replace_paragraph(
        body,
        "This revision does not add a Zenodo DOI",
        (
            "Data and code availability. The replication code and the frozen inputs "
            "data/boston_corrected.txt and data/sp100_clean_snapshot.csv are the files used for this revision. "
            "The intended public archive is https://github.com/simoncheungkc/Multi-SEM, with a Zenodo deposit "
            "cited as 10.5281/zenodo.XXXXXXX. On 11 November 2026 that GitHub address returned 404, and the "
            "Zenodo suffix is a placeholder rather than an issued identifier. Until both exist, the numerical "
            "results are reproduced from the files in this project, not from a live Yahoo Finance download."
        ),
    )


def write_letter() -> None:
    from docx import Document
    from docx.shared import Pt

    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)

    def heading(text, level):
        document.add_heading(text, level=level)

    def paragraph(text):
        document.add_paragraph(text)

    heading("Formal Response to the Reviewers", 0)
    paragraph(
        "Manuscript reference: SPASTA-D-26-00479. Journal: Spatial Statistics (Elsevier). "
        "Title: Beyond a Single Spatial Weight Matrix: Multiplex Spatial Error Regression with "
        "Calibrated Screening and Precision Kriging. Authors: King-Chau Cheung and Tommy Cheung."
    )
    paragraph(
        "We thank the Editor and both reviewers. The revisions are in manuscript v23 and "
        "Supplementary Materials v21. Numerical statements below are the values produced by the "
        "frozen workflow: the archived Boston splits, the ablation and sensitivity fits, the "
        "opposite-sign simulation, and the filtered-innovation diagnostics."
    )

    heading("Response to Reviewer 1", 1)
    heading("Comment 1. Higher-order literature and model averaging", 2)
    paragraph(
        "Comment. The introduction does not describe the higher-order literature. Relevant work "
        "includes Anselin (1988), Lacombe (2004), Elhorst, Lacombe and Piras (2012), Lee and Liu "
        "(2010), Lam and Souza (2020), LeSage and Fischer (2008), and Piribauer and Crespo Cuaresma (2016)."
    )
    paragraph(
        "Response. Section 1 and the new Section 2.1 cite those papers. Multi-SEM is presented as a "
        "pipeline, not as a new model class. Section 2.1 also states the contrast with Bayesian model "
        "averaging: that literature treats competing weight matrices as alternative generators, whereas "
        "the additive disturbance lets overlapping channels enter the same error process."
    )
    heading("Comment 2. The l1 bound and precision prediction", 2)
    paragraph(
        "Comment. The l1 condition should be related to Elhorst et al. (2012). Precision conditioning "
        "is GMRF material (Rue and Held, 2005) and is discussed for spatial prediction by Goulard, "
        "Laurent and Thomas-Agnan (2017)."
    )
    paragraph(
        "Response. Section 2.1 and Section 3.2 now state that the open l1 ball is a geometry-free "
        "sufficient condition, used because the exact stability region of Elhorst, Lacombe and Piras "
        "(2012) depends on the joint spectrum of non-commuting matrices. Section 2.1 links the "
        "predictor to Goldberger (1962), Goulard, Laurent and Thomas-Agnan (2017), and Rue and Held (2005)."
    )
    heading("Comment 3. Exact Moran null and Monte Carlo", 2)
    paragraph(
        "Comment. The null depends only on the spectrum of B_s = H' A_s* H. An exact p-value costs one "
        "eigendecomposition. Conditional Gaussian Monte Carlo simulates that distribution. It is not a "
        "bootstrap or a permutation test."
    )
    paragraph(
        "Response. Section 4.2 now cites Tiefelsdorf (2000) and Imhof (1961), states the O(d^3) cost of "
        "the eigenvalues, and describes the B = 499 spherical Monte Carlo at cost O(B d^2). The text "
        "does not call that procedure a bootstrap or a permutation test."
    )
    heading("Comment 4. Kappa, psi, and the industry-layer p-value", 2)
    paragraph(
        "Comment. The switching rule is not tied to the Kelejian–Prucha column-sum condition. Psi stays "
        "below 0.056, so its threshold is untested. Kappa is knife-edge: the Boston town layer is near "
        "3 and the S&P industry layer has kappa = 3.07, while the reported Gaussian p-value is 0.0038."
    )
    paragraph(
        "Response. Section 4.2 ties kappa to the Kelejian and Prucha (2001) requirement that absolute "
        "row and column sums stay bounded. After row standardization the row sums equal one, isolates "
        "excepted, so degree heterogeneity appears in the column sums. Kappa greater than 3 and psi "
        "greater than 0.20 are pre-specified operational rules, motivated by the size simulations in "
        "Section 7.2. For weighted graphs, Section 8.1 states that kappa measures unstandardized "
        "row-sum heterogeneity. In Section 8.2 the industry layer has kappa = 3.07, so the operational "
        "rule selects the Monte Carlo branch. The archived screen stored the Gaussian tail, p = 0.0038. "
        "The Monte Carlo probability was not stored. We therefore do not replace 0.0038 by 0.0040. "
        "A B = 499 reference has grid spacing 0.002, and 0.004 is a neighboring grid point, not a "
        "recomputed tail probability."
    )
    heading("Comment 5. Marginal screening of overlapping layers", 2)
    paragraph(
        "Comment. Residual Moran's I is marginal. Overlapping layers reject together. A joint "
        "multi-weight LM test would address that."
    )
    paragraph(
        "Response. Section 4 states that the screen is an inclusive discovery step under "
        "Benjamini–Hochberg control, not a partial-coefficient identification step. Exact duplicates "
        "are removed when the off-diagonal correlation is at least 0.99. The Boston geographic layer "
        "and its resolvent have correlation 0.975 and are both retained. Their separation is left to "
        "the covariance-adapted Metropolis proposal. On Split 5 the frozen proposal correlation between "
        "those two coefficients is -0.90."
    )
    heading("Comment 6. Induced subgraph as a pseudo-posterior", 2)
    paragraph(
        "Comment. Persistent bias in lambda means the induced criterion is a pseudo-posterior. Equal "
        "RMSE shows only that point prediction is insensitive."
    )
    paragraph(
        "Response. Section 7.5 states that the induced-subgraph criterion is a pseudo-posterior: "
        "re-row-standardizing the training block changes the operator scale. When the full topology is "
        "observed, the Schur precision is the coherent training likelihood. The induced fit is retained "
        "as a fast surrogate when the target is point imputation. The reported RMSE comparison does not "
        "calibrate predictive intervals."
    )
    heading("Comment 7. Complexity claim", 2)
    paragraph(
        "Comment. The claim that kriging reduces the cost from O(n_t^3) to O(n_*^3) is overstated, "
        "because every proposal still factorizes the training filter."
    )
    paragraph(
        "Response. The abstract and Section 6 now limit that statement to the holdout imputation solve."
    )
    heading("Comment 8. Boston benchmarks, no-kriging fit, and resampled tests", 2)
    paragraph(
        "Comment. (a) OLS is a weak benchmark. (b) Multi-SEM without kriging is worse than OLS. "
        "(c) Overlapping splits make Wilcoxon and sign tests invalid. Use Nadeau and Bengio (2003)."
    )
    paragraph(
        "Response. (a) Supplementary Table S9 reports the resolvent-only model (mean RMSE 0.1377), "
        "municipal cliques (0.1499), geography plus municipal boundaries (0.1384; lower RMSE on 33 of "
        "50 splits), geography plus the resolvent (0.1383), and screened Multi-SEM (0.1375), against "
        "the geographic SEM (0.1403). (b) Section 8.1 explains the no-kriging RMSE of 0.2072. The tax "
        "rate, pupil–teacher ratio, large-lot zoning, and non-retail acreage do not vary within "
        "municipalities and are attenuated by V_lambda, so the slopes are identified mainly from "
        "within-municipality variation. Without the precision correction the filtered mean does not "
        "absorb inter-municipal baselines. (c) The Wilcoxon values 1.78e-15 and 7.50e-5 have been "
        "removed. Against OLS, t_49 = 7.29, p = 2.34e-9, interval [0.0131, 0.0230], with 50 of 50 wins. "
        "Against the geographic SEM, t_49 = 1.11, p = 0.272, interval [-0.0006, 0.0021], with 37 of 50 "
        "wins. The interval contains zero."
    )
    heading("Minor comments", 2)
    paragraph(
        "Notation. Network counts are written with lowercase k, including Algorithm 1. The symbol k* "
        "is defined there as the Benjamini–Hochberg cutoff rank. Regression dimension is described as "
        "p+1 coefficients, and Moran p-values are distinguished from that dimension in Section 4."
    )
    paragraph(
        "Repository. Section S5 records the intended GitHub address and a Zenodo placeholder. The "
        "GitHub URL returned 404 on 11 November 2026, so the statement points to the project files "
        "data/boston_corrected.txt and data/sp100_clean_snapshot.csv."
    )
    paragraph(
        "Boston specification. Section 8.1 and Table 7 state that the geographic layer, its resolvent, "
        "and the town layer are selected on 50 of 50 splits, and the radial layer on 28 splits. Split 5 "
        "retains the first three."
    )
    paragraph(
        "Table 3. Size columns are labeled empirical Type I error under the null DGP. The active-DGP "
        "columns are power for the Poisson d = 8 and power-law gamma = 2.2 layers, and false discoveries "
        "for the inactive layers. The caption reports the Monte Carlo standard error, about 0.0097 at "
        "the nominal 5% size and at most 0.022 for a rejection rate."
    )

    heading("Response to Reviewer 2", 1)
    heading("Comment 1. What is new", 2)
    paragraph(
        "Comment. The additive multi-weight error model, the l1 bound, Cliff–Ord moments, and "
        "precision conditioning are established. The paper should state what is new."
    )
    paragraph(
        "Response. Sections 1 and 2.1 describe the contribution as a pipeline: ex-ante pruning of "
        "affiliation resolvents that reproduce the clique, isolate-scaled Moran moments with a "
        "topology-triggered Monte Carlo tail, a covariance-adapted proposal for collinear layers, and "
        "precision kriging. The municipal-boundary results are reported as predictive comparisons in "
        "Table S9, not as a claim that the town layer is identified separately from the geographic resolvent."
    )
    heading("Comment 2. Prior geometry of L", 2)
    paragraph(
        "Comment. With xi = 1 the Gaussian factor is nearly flat on the l1 ball, so concentration of "
        "L near 1 is largely the volume factor p(L) proportional to L^{k-1}. Please show the prior of "
        "L and refit under a prior on L itself."
    )
    paragraph(
        "Response. Section 7.3 and Supplement S2.3 derive p(L) proportional to L^{k-1}. Figure S10 "
        "uses 200,000 uniform draws in the l1 ball in R^3. Their mean L is 0.750. The Split 5 posterior "
        "under the paper prior has mean 0.852 and histogram mode 0.844, and the refit reproduces the "
        "archived mean. Under L ~ Beta(1, 2), which has zero density at L = 1, the mean is 0.836, the "
        "mode remains 0.844, and the 97.5% quantile moves from 0.971 to 0.921. The S&P 100 fits were "
        "not repeated under the Beta prior. Those layers are a different selection, and the Boston "
        "Split 5 refit is the sensitivity we report."
    )
    heading("Comment 3. Admissible region and opposite signs", 2)
    paragraph(
        "Comment. The admissible set is the connected component containing the origin. For one layer "
        "the lower bound is 1/omega_min, not -1. Draws reach L = 0.999998. The opposite-sign motivation "
        "was not in the simulations."
    )
    paragraph(
        "Response. Section 3.2 keeps the open l1 ball as a sufficient condition and cites Elhorst, "
        "Lacombe and Piras (2012) for the exact region of non-commuting matrices. We did not add a "
        "simulation outside the l1 ball. Table 4b adds lambda = (0.50, -0.30, 0, 0, 0), with signed "
        "sum 0.20 and l1 norm 0.80, over 200 replications at n = 200. The active coefficients are "
        "recovered with the correct signs and are shrunk toward zero: bias -0.106 and RMSE 0.142 for "
        "0.50 (coverage 0.910), and bias +0.066 and RMSE 0.100 for -0.30 (coverage 0.915). The three "
        "inactive coefficients have bias about 0 and coverage 0.970 to 0.975. The largest retained "
        "Boston draw is 0.999877 on Split 5 and 0.999998 across the 50 splits. An earlier draft quoted "
        "0.9842 for Split 5. That figure is not the archived maximum."
    )
    heading("Comment 4. Resolvent as a polynomial filter", 2)
    paragraph(
        "Comment. G_A is a polynomial in one geographic matrix. The layers are highly correlated, the "
        "posterior correlation is -0.90, and the resolvent is nearly dense."
    )
    paragraph(
        "Response. Sections 3.4 and 8.1 describe the Katz resolvent of a continuous graph as a damped "
        "polynomial filter. Affiliation resolvents reproduce the clique and are pruned. Table S9 gives "
        "mean RMSE 0.1403 (geographic SEM), 0.1377 (resolvent alone), 0.1499 (municipal cliques), "
        "0.1384 (geography plus municipal boundaries, lower on 33 of 50 splits), 0.1383 (geography plus "
        "the resolvent), and 0.1375 (screened Multi-SEM). The municipal increment is real as a point "
        "estimate and is not larger than the resolvent-only increment. No contrast with the geographic "
        "SEM has a Nadeau–Bengio p-value below 0.05. The abstract no longer treats the resolvent as a "
        "separate physical network, and Section 8.1 records the proposal correlation of -0.90."
    )
    heading("Comment 5. Training operators versus full-graph kriging", 2)
    paragraph(
        "Comment. Lambda is estimated on re-standardized training blocks, while kriging uses the full "
        "graph. The bias does not vanish with n. Data augmentation would give a full predictive distribution."
    )
    paragraph(
        "Response. Section 7.5 states that re-standardizing the training block changes the scale of "
        "lambda and that the RMSE comparison addresses point prediction. Data augmentation is noted as "
        "an alternative that draws the holdout outcomes inside the sampler. It was not implemented. The "
        "induced subgraph remains the estimator used for the reported point imputations."
    )
    heading("Comment 6. Complexity claim", 2)
    paragraph(
        "Comment. The cubic reduction is measured against a dense inversion that an SEM fit does not "
        "require, and every Metropolis step already factorizes the training filter."
    )
    paragraph(
        "Response. The same qualification as in the response to Reviewer 1: the abstract limits the "
        "claim to the holdout imputation solve."
    )
    heading("Comment 7. Resampled p-values", 2)
    paragraph(
        "Comment. The 50 splits are not independent. The value 1.78e-15 equals 2 times 2^{-50}. The "
        "Wilcoxon p-value for the 2% geographic increment has the same defect."
    )
    paragraph(
        "Response. Both uncorrected p-values were removed. Section 8.1 reports the Nadeau and Bengio "
        "(2003) statistics given under Reviewer 1, Comment 8, including the interval against the "
        "geographic SEM, which contains zero."
    )
    heading("Comment 8. Boston specification, censoring, and the Charles River", 2)
    paragraph(
        "Comment. Sixteen tracts are censored at 50. The Charles River dummy is spatial. The GLS mean "
        "predicts worse than OLS. Random splits are interpolation. State which proposal generated Tables 7 and 9."
    )
    paragraph(
        "Response. Table S10 adds CHAS and, separately, drops the 16 ceiling tracts (n = 490, training "
        "392, holdout 98). With CHAS, mean RMSE is 0.1904 (OLS), 0.1410 (geographic SEM), and 0.1378 "
        "(Multi-SEM): a 27.6% reduction versus OLS (t_49 = 7.07, p = 5.11e-9) and a 2.3% reduction versus "
        "the geographic SEM (p = 0.204). Without the ceiling tracts the means are 0.1806, 0.1297, and "
        "0.1289: a 28.7% reduction versus OLS (t_49 = 6.82, p = 1.26e-8) and a 0.6% reduction versus the "
        "geographic SEM (p = 0.623). The geographic increment is not stable across these checks. "
        "Section 8.1 explains the no-kriging result and states that the random splits assess interpolation "
        "inside the observed metropolitan area. Leave-one-town-out cross-validation was not added. "
        "Tables 8 and 9 use the covariance-adapted kernel, four chains, 8,000 iterations, and 4,000 burn-in."
    )
    heading("Comment 9. Simulation design", 2)
    paragraph(
        "Comment. Only one positive lambda vector was studied. The simulated graphs are nearly orthogonal. "
        "Table 5 uses 30 replications. Table 4 and Table 4a differ in the covariate design. No competing method is included."
    )
    paragraph(
        "Response. Table 4b is the opposite-sign design summarized under Comment 3. Section 8.1 is the "
        "collinear empirical case, with off-diagonal correlation 0.975. Table 3 now states Monte Carlo "
        "standard errors. Section 7.3 states that the regressors in Tables 4 and 4a are independent "
        "standard normals, then column-standardized. They are not drawn with correlation 0.20. Table 5 "
        "compares the screened and oracle fits with OLS and single-network specifications on the "
        "synthetic graphs. We did not add a further competing estimator."
    )
    heading("Comment 10. Reproducibility and filtered innovations", 2)
    paragraph(
        "Comment. The simulations and the S&P fits used the isotropic proposal. There is no frozen "
        "archive, and run_sp100.py downloads live Yahoo prices. Post-fit Moran statistics on the "
        "filtered innovations were not reported."
    )
    paragraph(
        "Response. Section 5.3 still states that the simulation and S&P fits used the isotropic proposal "
        "and were not rerun. Boston uses the covariance-adapted kernel. run_sp100.py now has "
        "USE_STATIC_CACHE = True and reads data/sp100_clean_snapshot.csv, written from the existing "
        "price cache: 100 equities, 2,187 adjusted closes from 2 January 2018 through 15 September 2026, "
        "and 2,186 daily returns, together with the monthly covariates. It does not call Yahoo Finance "
        "on that path. Table S11 reports Moran's I for the OLS training residuals and for "
        "epsilon_hat = V_lambda_hat (Y_t - X_t beta_hat). On the geographic layer the mean OLS statistic "
        "is 0.442 and the mean filtered statistic is -0.021. On Split 5 the filtered statistic is -0.012 "
        "(p = 0.755). The share of splits with p > 0.30 is 0.80, 0.94, 0.92, and 0.96 on the four "
        "candidate matrices. The filtered geographic statistics range from -0.077 to 0.007, so the "
        "result is a mean near zero, not a claim that every split lies in [-0.015, 0.008]."
    )
    heading("Minor comments", 2)
    paragraph(
        "The quadratic form in the Metropolis ratio is script S. Q is reserved for the precision. "
        "Network counts use k. Section 2 cites Bivand, Gómez-Rubio and Rue (2014) for single-weight "
        "Bayesian comparisons. Benjamini and Hochberg (1995) is cited in the bibliography and used in "
        "Section 4. Kappa is described as row-sum heterogeneity for weighted graphs. Section S5 contains "
        "the data and code statement."
    )
    document.save(LETTER_OUT)


def main() -> None:
    files, root = load_zip(MAIN_IN)
    edit_main(body_of(root))
    store_zip(MAIN_OUT, files, root)
    print(MAIN_OUT.name)
    files, root = load_zip(SUP_IN)
    edit_supplement(body_of(root))
    store_zip(SUP_OUT, files, root)
    print(SUP_OUT.name)
    write_letter()
    print(LETTER_OUT.name)


if __name__ == "__main__":
    main()
