"""Manuscript v24, supplement v22, and the second response to the reviewers.

The August industry Monte Carlo probability is the seed-2026 result in
results/revision/august_industry_mc.json. It is not 0.0040.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.shared import Pt

from build_revision_docs import body_of, following_table, replace_paragraph
from build_v10_docx import q, set_cell_text
from build_v12_docx import W, load_zip, require_one, store_zip

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v23.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v24.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v21.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v22.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers_2.docx"

ABSTRACT = (
    "Spatial dependence in empirical systems often operates simultaneously across continuous "
    "physical distance, administrative jurisdictions, and functional networks. Conventional spatial "
    "error models condition on a single spatial weight matrix, potentially overlooking overlapping "
    "relational channels and distorting inference. This paper develops Multiplex Spatial Error "
    "Regression (Multi-SEM), an integrated econometric pipeline combining ex-ante structural pruning, "
    "topology-calibrated residual screening, stability-constrained Bayesian estimation, and "
    "precision-form spatial kriging. An additive disturbance specification preserves linear "
    r"conditional-mean interpretability ($\mathbb{E}[Y \mid X] = X\beta$) under exogeneity, while the "
    r"parameter restriction $\Vert\lambda\Vert_1 = \sum_{s=1}^{k} \vert\lambda_s\vert < 1$ establishes "
    "a uniform sufficient condition for spectral stability across row-substochastic layers. We derive "
    "exact design-conditional null moments for residual Moran's $I$ under isolate scaling and "
    "introduce topology-triggered Monte Carlo calibration to correct finite-sample size distortions "
    "on hub-dominated networks prior to false discovery rate control. For out-of-sample prediction, "
    r"formulating spatial kriging through the full-graph precision matrix ($Q = \sigma^{-2} V_\lambda' V_\lambda$) "
    r"reduces the computational complexity of the holdout solve from $\mathcal{O}(n_t^3)$ to "
    r"$\mathcal{O}(n_*^3)$. In an empirical application to $n = 506$ census tracts across 92 "
    "municipalities in Greater Boston, covariance-adapted Multi-SEM jointly represents candidate "
    "spatial dependence associated with continuous geographic proximity, a multi-hop polynomial "
    "spatial filter, and discrete municipal jurisdictions. Across 50 repeated random 80/20 splits, "
    "Multi-SEM reduces mean holdout RMSE by 28.5% relative to OLS ($0.1375$ vs. $0.1921$; "
    r"Nadeau–Bengio resampled $t_{49} = 7.29$, $p = 2.34 \times 10^{-9}$, outperforming OLS on 50 of "
    "50 splits) and achieves a 2.0% incremental error reduction over a single-layer geographic SEM "
    r"($0.1403$), attaining lower error on 37 of 50 splits (conservative resampled $t_{49} = 1.11$, "
    r"$p = 0.272$). Across 63 monthly S&P 100 equity cross-sections, Multi-SEM achieved lower holdout "
    r"RMSE than OLS in 44 months (sign test $p = 0.0022$), although autocorrelation-robust inference "
    r"for the monthly loss differential was less decisive (Newey–West $p = 0.072$)."
)

INTRO = (
    "Tobler's first law characterizes spatial dependence through continuous distance decay, whereby "
    "nearby observations tend to be more strongly related than distant ones (Tobler, 1970). Yet "
    "administrative, legal, and functional boundaries often intersect with this geographic gradient. "
    "A residential property in Greater Boston has contiguous geographic neighbours, but also belongs "
    "to a municipal jurisdiction that determines property tax assessment, zoning bylaws, and public "
    "school provision (Harrison and Rubinfeld, 1978; Gilley and Pace, 1996). The same intersection "
    "of affiliation and co-movement arises among firms. Standard spatial regressions traditionally "
    r"condition on a single weight matrix $W$ (Bivand et al., 2014; Bivand and Wong, 2018). While "
    "higher-order spatial autoregressive and error models have been analyzed since Anselin (1988) "
    "and Lacombe (2004), empirical practice remains constrained by the challenge of testing, "
    "stabilizing, and estimating multiple non-commuting spatial matrices. Multi-SEM resolves this "
    r"limitation through a unified pipeline combining structural pruning, calibrated screening, "
    r"$\ell_1$-constrained Bayesian estimation, and precision-domain spatial interpolation."
)

NADEAU = (
    "Screened Multi-SEM with precision kriging achieves a mean holdout RMSE of 0.1375, compared with "
    "0.1921 for OLS, a 28.5% reduction. Multi-SEM achieves a lower RMSE than OLS on 50 of 50 splits. "
    "The Nadeau and Bengio (2003) corrected resampled test, which accounts for training overlap by "
    r"inflating the variance of the mean loss gap by $1/R + n_*/n_t = 0.269$, yields "
    r"$t_{49} = 7.29$, $p = 2.34 \times 10^{-9}$ (corrected 95% interval $[0.0131, 0.0230]$). "
    "Against the single-layer geographic SEM (0.1403), Multi-SEM achieves an incremental 2.0% RMSE "
    r"reduction and wins on 37 of 50 splits. The corrected resampled test gives $t_{49} = 1.11$, "
    r"$p = 0.272$ (corrected 95% interval $[-0.0006, 0.0021]$). The split-level win count is 37 of "
    "50. Both 95% intervals are intervals for the mean reduction in mean squared error, not in root "
    "mean squared error. The interval against the geographic SEM spans zero."
)

INDUSTRY = (
    "The four archived $p$-values do not lie on the Monte Carlo grid $0.002, 0.004, \\ldots$, because "
    "the archived screening routine reports the Gaussian tail. For the correlation, resolvent, and "
    r"sector layers, $\kappa$ and $\psi$ are below the thresholds, so the Gaussian tail is the "
    r"operational $p$-value. The industry layer has $\kappa = 3.07$. Recomputing the conditional "
    "Gaussian Monte Carlo on that same full cross-section of 100 equities, with $B = 499$ spherical "
    r"draws and seed 2026, produces 2 standardized draws at least as extreme as the observed statistic, "
    r"so $p_{\mathrm{MC}} = (1+2)/500 = 0.006$. The same draws produce 6 exceedances of the absolute "
    r"Moran statistic, or $0.014$. The industry layer remains inside the Benjamini–Hochberg selection "
    "under either probability. The 80-firm training block from the same split has $\\kappa = 2.46$, "
    "below the Monte Carlo trigger, and a Gaussian tail of $0.028$. That training block is not the "
    r"screen in Table 10, and it does not yield $p_{\mathrm{MC}} = 0.0040$. Each exceedance moves the "
    "add-one probability by $0.002$, so another Monte Carlo seed can print $0.004$. The probability "
    r"recorded here is the seed-2026 result on the cross-section that has $\kappa = 3.07$."
)

CAPTION = (
    "Table 10. Residual Moran's $I$ for the August 2026 showcase. The archived screen used all "
    r"$n = 100$ equities, before the 80/20 split. $\kappa$ is the maximum degree divided by the mean "
    r"degree on the off-diagonal adjacency. $\psi$ is the leading singular-value share of the "
    "row-standardized matrix. The stored probabilities are analytical Gaussian tails. For the industry "
    r"layer, $\kappa = 3.07$, so Algorithm 1 calls for conditional Gaussian Monte Carlo. On this same "
    r"full cross-section, $B = 499$ spherical draws with seed 2026 give "
    r"$p_{\mathrm{MC}} = (1+2)/500 = 0.006$ for the standardized statistic. The industry cell reports "
    "the Gaussian tail and that Monte Carlo probability together."
)


def delete_paragraph(body, snippet):
    paragraph = require_one(body, snippet)
    paragraph.getparent().remove(paragraph)


def replace_runs(body, old, new, expected):
    count = 0
    for node in body.iter(q(W, "t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)
            count += 1
    if count != expected:
        raise SystemExit(f"{old!r} replaced in {count} runs, expected {expected}")


def edit_main(body) -> None:
    replace_paragraph(body, "Many empirical systems are inherently multiplex", ABSTRACT)
    delete_paragraph(body, "Multi-SEM addresses four methodological challenges.")
    delete_paragraph(body, "We evaluate Multi-SEM using two substantive empirical benchmarks.")
    replace_paragraph(body, "Tobler's first law characterizes spatial dependence", INTRO)
    replace_paragraph(
        body,
        "Higher-order spatial autoregressive and spatial-error models have been studied since Anselin",
        "Section 2.1 places this pipeline in the higher-order literature and distinguishes simultaneous "
        "multi-channel dependence from Bayesian model averaging over competing weight matrices.",
    )
    replace_paragraph(
        body,
        "treats competing matrices under a model-selection assumption",
        "Recent literature has addressed spatial weight-matrix uncertainty by Bayesian model averaging "
        "(LeSage and Fischer, 2008; Piribauer and Crespo Cuaresma, 2016) and by penalized selection "
        "(Lam and Souza, 2020). Model averaging seeks one dominant matrix from a set of competing "
        "alternatives. Multi-SEM addresses a different empirical setting: simultaneous multi-channel "
        "dependence, in which continuous physical space and discrete administrative jurisdictions can "
        "operate together in the disturbance. That joint representation is not the same as separate "
        "identification of each channel. In the Boston application the geographic coefficient and its "
        "resolvent remain strongly collinear.",
    )
    replace_runs(
        body,
        "Mild shrinkage toward zero subject to the sufficient stability restriction",
        "Weak shrinkage toward zero subject to the sufficient stability restriction",
        1,
    )
    replace_paragraph(
        body,
        "Screened Multi-SEM with precision kriging achieves a mean holdout RMSE of 0.1375",
        NADEAU,
    )
    replace_paragraph(body, "The four archived", INDUSTRY)
    replace_paragraph(body, "Table 10. Residual Moran's", CAPTION)
    table = following_table(body, "Table 10. Residual Moran's")
    row = table.findall(q(W, "tr"))[4]
    cells = row.findall(q(W, "tc"))
    set_cell_text(cells[5], "Gaussian; MC")
    set_cell_text(cells[6], "0.0038; 0.006")


def edit_supplement(body) -> None:
    replace_runs(
        body,
        "the residual bootstrap or permutation calibration described in the main text",
        "the conditional Gaussian Monte Carlo calibration described in the main text",
        2,
    )
    replace_runs(
        body,
        "a residual parametric bootstrap or permutation procedure",
        "the conditional Gaussian Monte Carlo calibration procedure",
        1,
    )
    replace_runs(
        body,
        "On 11 November 2026 that GitHub address returned 404",
        "On 12 November 2026 that GitHub address still did not resolve",
        1,
    )


def write_letter() -> None:
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
        "We thank the Editor and both reviewers. The revisions are in the manuscript "
        "REGRESSION WITH MULTIPLE NETWORKS v24.docx and in Supplementary Materials v22. "
        "The numerical statements below are the values produced by the frozen Boston splits, "
        "the ablation and sensitivity fits, the opposite-sign simulation, the filtered-innovation "
        "diagnostics, and a recomputed August 2026 industry-layer Monte Carlo."
    )

    heading("Response to Reviewer 1", 1)
    heading("Comment 1. Higher-order literature and model averaging", 2)
    paragraph(
        "Comment. The introduction does not describe the higher-order literature. Relevant work "
        "includes Anselin (1988), Lacombe (2004), Elhorst, Lacombe and Piras (2012), Lee and Liu "
        "(2010), Lam and Souza (2020), LeSage and Fischer (2008), and Piribauer and Crespo Cuaresma (2016)."
    )
    paragraph(
        "Response. Section 1 now places Anselin (1988) and Lacombe (2004) in the opening motivation, "
        "and Section 2.1 reviews that lineage together with Lee and Liu (2010) and Lam and Souza (2020). "
        "Multi-SEM is presented as a pipeline, not as a new model class. Section 2.1 contrasts it with "
        "Bayesian model averaging: model averaging seeks one dominant matrix, whereas the additive "
        "disturbance can contain more than one channel at the same time. The Boston geographic "
        "coefficient and its resolvent remain collinear, so the text does not claim that those two "
        "operators are separately identified."
    )
    heading("Comment 2. The l1 bound and precision prediction", 2)
    paragraph(
        "Comment. The l1 condition should be related to Elhorst et al. (2012). Precision conditioning "
        "is standard Gaussian Markov random-field material (Rue and Held, 2005) and is discussed for "
        "spatial prediction by Goulard, Laurent and Thomas-Agnan (2017)."
    )
    paragraph(
        "Response. Section 2.1 states that the open l1 ball is a geometry-free sufficient condition. "
        "The exact stability region of Elhorst, Lacombe and Piras (2012) depends on the joint spectrum "
        "of non-commuting matrices, and that spectrum is not recomputed at every proposal. Section 2.1 "
        "links the predictor to Goldberger (1962), Goulard, Laurent and Thomas-Agnan (2017), and Rue "
        "and Held (2005). No simulation was run outside the l1 ball."
    )
    heading("Comment 3. Exact Moran null and Monte Carlo", 2)
    paragraph(
        "Comment. The null depends only on the spectrum of B_s = H' A_s* H. An exact p-value costs one "
        "eigendecomposition. Conditional Gaussian Monte Carlo simulates that distribution. It is not a "
        "bootstrap or a permutation test."
    )
    paragraph(
        "Response. Section 4.2 cites Tiefelsdorf (2000) and Imhof (1961), states the cubic cost of the "
        "eigenvalues, and describes the B = 499 spherical Monte Carlo. Supplement Sections S1.2, S1.3, "
        "and S3.2 now use the same name. The phrases \"residual bootstrap\" and \"permutation calibration\" "
        "have been removed from those sections."
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
        "greater than 0.20 are pre-specified operational rules from the size simulations in Section 7.2. "
        "For weighted graphs, kappa measures unstandardized row-sum heterogeneity."
    )
    paragraph(
        "The industry contradiction is in the archived full cross-section, not in the 80-firm training "
        "block. On 31 August 2026 that full screen has kappa = 3.070, Moran I = 0.263, z = 2.894, and "
        "Gaussian tail p = 0.00381, which is the archived 0.0038. Conditional Gaussian Monte Carlo with "
        "B = 499 and seed 2026 gives 2 standardized draws at least as extreme as the observed statistic, "
        "so p_MC = (1+2)/500 = 0.006. The same draws give 6 exceedances of |I|, so the absolute-statistic "
        "probability is 0.014. Table 10 and Section 8.2 now report 0.0038 and 0.006 together. The value "
        "0.0040 is one step away on the add-one grid and is not the seed-2026 result."
    )
    paragraph(
        "The 80-firm training block from the same seed-2026 split has kappa = 2.46. Algorithm 1 does "
        "not send that block to Monte Carlo. Its Gaussian tail is 0.0277, and the seed-2026 Monte Carlo "
        "probabilities on that block are 0.034 for the standardized statistic and 0.054 for |I|. "
        "We therefore do not replace the Table 10 probability by 0.0040."
    )
    heading("Comment 5. Marginal screening of overlapping layers", 2)
    paragraph(
        "Comment. Residual Moran's I is marginal. Overlapping layers reject together. A joint "
        "multi-weight score test would address that."
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
        "Response. The abstract and Section 6 limit that statement to the holdout imputation solve."
    )
    heading("Comment 8. Boston benchmarks, no-kriging fit, and resampled tests", 2)
    paragraph(
        "Comment. (a) OLS is a weak benchmark. (b) Multi-SEM without kriging is worse than OLS. "
        "(c) Overlapping splits make Wilcoxon and sign tests invalid. Use Nadeau and Bengio (2003) "
        "and report confidence intervals."
    )
    paragraph(
        "Response. (a) Supplementary Table S9 reports, on the same 50 splits, geographic SEM 0.1403, "
        "resolvent only 0.1377, town only 0.1499, geography plus town 0.1384 (lower RMSE on 33 of 50 "
        "splits; corrected p = 0.515), geography plus resolvent 0.1383 (corrected p = 0.060), and "
        "screened Multi-SEM 0.1375 (corrected p = 0.272 against the geographic SEM). Adding municipal "
        "boundaries lowers the point estimate without the resolvent. The resolvent-only model is closer "
        "to the full model than geography plus town, and none of the contrasts with the geographic SEM "
        "is significant at 5% after the overlap correction. (b) Section 8.1 explains the no-kriging "
        "RMSE of 0.2072, which is worse than OLS at 0.1921. Tax, the pupil–teacher ratio, zoning, and "
        "industrial acreage do not vary within municipalities and are attenuated by the spatial filter, "
        "so the slopes and the kriging correction are one prediction system. (c) The uncorrected "
        "Wilcoxon probabilities have been removed. Against OLS the corrected test is t_49 = 7.29, "
        "p = 2.34e-9, with a 95% interval for the mean MSE reduction of [0.0131, 0.0230], and Multi-SEM "
        "wins 50 of 50 splits. Against the geographic SEM the corrected test is t_49 = 1.11, p = 0.272, "
        "with interval [-0.0006, 0.0021], and Multi-SEM wins 37 of 50 splits. Section 8.1 states that "
        "these intervals are on the mean-squared-error scale, not the RMSE scale. The interval against "
        "the geographic SEM includes zero."
    )
    heading("Minor comments", 2)
    paragraph(
        "Notation uses k for the layer count, p+1 for the regression dimension, and p_s for a Moran "
        "probability. In Algorithm 1, k* is the Benjamini–Hochberg cutoff rank. The replication files "
        "are data/boston_corrected.txt and data/sp100_clean_snapshot.csv. Section S5.1 states where "
        "they sit. Table 7 and Section 8.1 state that geography, the resolvent, and the town layer are "
        "retained on 50 of 50 splits and the radial layer on 28 of 50. Split 5 retains the first three. "
        "Table 3 separates Type I error under the null design from power and false discoveries under "
        "the active design, and its caption reports the Monte Carlo standard errors."
    )

    heading("Response to Reviewer 2", 1)
    heading("Comment 1. What is new", 2)
    paragraph(
        "Comment. The additive multi-weight error is a higher-order spatial error model. The l1 bound, "
        "the Cliff–Ord moments, and precision conditioning are established ingredients. The manuscript "
        "should say what the pipeline adds."
    )
    paragraph(
        "Response. Sections 1 and 2.1 state the contribution as a pipeline: ex-ante removal of affiliation "
        "resolvents that reproduce the base clique to 1.4e-17, isolate-scaled residual Moran moments with "
        "a topology trigger for Monte Carlo calibration, a covariance-adapted proposal for collinear "
        "operators, and the empirical comparison of municipal boundaries with continuous distance. The "
        "individual bounds and matrix identities are not claimed as new theorems."
    )
    heading("Comment 2. Prior geometry of L", 2)
    paragraph(
        "Comment. With xi = 1 the Gaussian factor barely varies on the l1 ball, so concentration of "
        "L near the boundary is mostly prior volume. Please show the prior distribution of L, and refit "
        "under a prior that penalizes the boundary."
    )
    paragraph(
        "Response. For k = 3 a uniform distribution on the ball induces p(L) proportional to L^2, with "
        "prior mean 0.750. Supplementary Figure S10 compares 200,000 such draws with the Split 5 posterior. "
        "The posterior mean is 0.852 and the histogram mode is 0.844, with central 95% interval "
        "[0.770, 0.971]. A boundary-penalizing prior, L ~ Beta(1, 2) with a uniform Dirichlet direction, "
        "gives posterior mean 0.836 and the same mode, 0.844, with upper quantile 0.921. The mode stays "
        "away from zero. This refit is for Boston Split 5 only. The S&P application was not refit under "
        "the Beta prior. The largest retained Split 5 draw is L = 0.999877, inside the open ball."
    )
    heading("Comment 3. Admissible region and opposite signs", 2)
    paragraph(
        "Comment. The admissible set can extend beyond the l1 ball, and the constraint is active. "
        "Please simulate coefficients of opposite sign, and coefficients outside the ball but inside "
        "the admissible region."
    )
    paragraph(
        "Response. Table 4b reports 200 replications at n = 200 with lambda = (0.50, -0.30, 0, 0, 0), "
        "signed sum 0.20 and l1 norm 0.80. Both active signs are recovered. The Poisson degree-8 "
        "coefficient has bias -0.106, RMSE 0.142, and 91.0% coverage. The power-law coefficient has "
        "bias +0.066, RMSE 0.100, and 91.5% coverage. The three inactive coefficients stay near zero, "
        "with coverage 97.0% to 97.5%. The largest R-hat in the ensemble is 1.042. The coefficients are "
        "shrunk toward zero; the table does not show unbiased recovery. No design outside the l1 ball "
        "was run. Section 3.2 states the ball as a sufficient condition and as a limitation relative "
        "to the non-convex region in Elhorst, Lacombe and Piras (2012)."
    )
    heading("Comment 4. Resolvent and ablation", 2)
    paragraph(
        "Comment. The Katz resolvent is a polynomial filter in one weight matrix. The geographic layer "
        "and the resolvent are highly collinear, and the resolvent is dense."
    )
    paragraph(
        "Response. Sections 3.4 and 8.1 describe the resolvent as a geometrically damped polynomial "
        "filter, not as a second physical network. The ablation in Table S9 is the comparison requested. "
        "Geography plus municipal boundaries has mean RMSE 0.1384, against 0.1403 for geography alone, "
        "and wins 33 of 50 splits, with corrected p = 0.515. The resolvent alone has mean RMSE 0.1377. "
        "Screened Multi-SEM has mean RMSE 0.1375. Municipal boundaries change the point estimate without "
        "the resolvent, and they do not account for more of that estimate than the resolvent. The "
        "abstract does not claim that the three channels are separately identified."
    )
    heading("Comment 5. Induced subgraph and data augmentation", 2)
    paragraph(
        "Comment. Lambda is estimated on a re-standardized training block, while kriging uses the "
        "full-graph precision. The bias does not vanish with n. Data augmentation is an alternative."
    )
    paragraph(
        "Response. Section 7.5 acknowledges the change of scale and discusses data augmentation. The "
        "induced subgraph is retained because the target in the reported tables is point imputation. "
        "The RMSE comparison is not a calibration of predictive intervals."
    )
    heading("Comment 6. Imputation complexity", 2)
    paragraph(
        "Comment. The cubic reduction is measured against a dense inversion that an SEM does not need, "
        "and it should not be an abstract contribution."
    )
    paragraph(
        "Response. The abstract and Section 6.2 now restrict the statement to the holdout imputation solve."
    )
    heading("Comment 7. Overlapping splits", 2)
    paragraph(
        "Comment. The headline Wilcoxon probability equals the smallest two-sided value attainable when "
        "all 50 differences are positive. It is not a valid significance level. The same objection "
        "applies to the comparison with the geographic SEM."
    )
    paragraph(
        "Response. Both uncorrected probabilities have been removed. The corrected results are those "
        "in the response to Reviewer 1, Comment 8. The 2.0% RMSE increment over the geographic SEM is "
        "not reported as statistically significant."
    )
    heading("Comment 8. Censoring, the Charles River dummy, and extrapolation", 2)
    paragraph(
        "Comment. Sixteen tracts are censored at 50. The Charles River dummy is spatial. The "
        "no-kriging fit is worse than OLS. Random splits assess interpolation. Please say which "
        "Metropolis kernel produced Tables 7 and 9."
    )
    paragraph(
        "Response. Table S10 reruns the 50 splits with the Charles River dummy and, separately, drops "
        "the 16 ceiling tracts (n = 490, training size 392). With the dummy, mean RMSE is 0.1904 for "
        "OLS, 0.1410 for the geographic SEM, and 0.1378 for Multi-SEM: 27.6% below OLS "
        "(t_49 = 7.07, p = 5.11e-9, 50 of 50) and 2.3% below the geographic SEM "
        "(t_49 = 1.29, p = 0.204, 38 of 50). Without the ceiling tracts, the means are 0.1806, 0.1297, "
        "and 0.1289: 28.7% below OLS (t_49 = 6.82, p = 1.26e-8, 50 of 50) and 0.6% below the geographic "
        "SEM (t_49 = 0.49, p = 0.623, 35 of 50). The geographic increment does not remain under the "
        "censoring check. Section 8.1 gives the no-kriging explanation and states that the random splits "
        "assess interpolation inside the observed city. Leave-one-town-out cross-validation was not "
        "added. Section 5.3 and Section 8.1 state that Tables 8 and 9 use the covariance-adapted kernel."
    )
    heading("Comment 9. Simulation design", 2)
    paragraph(
        "Comment. Only one lambda vector was studied. The candidate graphs are nearly orthogonal. "
        "Table 5 uses 30 replications. The covariate design differs across tables and is not described. "
        "No competing method is included."
    )
    paragraph(
        "Response. Table 4b is the opposite-sign design described above. Section 8.1 uses the Boston "
        "application as the collinear case, with off-diagonal correlation 0.975. Table 3 reports Monte "
        "Carlo standard errors of at most 0.0097 for size and 0.022 for power. Tables 4 and 4a both use "
        "an intercept and five independent standard normal regressors, then column-standardize them. "
        "They do not use two covariates in one table and five in the other, and they do not use a mutual "
        "correlation of 0.20. Table 5 compares OLS, single-network SEMs, screened Multi-SEM, and an "
        "oracle Multi-SEM on the same synthetic graphs."
    )
    heading("Comment 10. Reproducibility and filtered innovations", 2)
    paragraph(
        "Comment. The simulations and the S&P fits used the isotropic proposal. There is no frozen "
        "archive, and run_sp100.py downloads live Yahoo prices. Post-fit Moran statistics on the "
        "filtered innovations were not reported."
    )
    paragraph(
        "Response. Section 5.3 still states that the simulation and S&P fits used the isotropic proposal "
        "and were not rerun. Boston uses the covariance-adapted kernel. run_sp100.py sets "
        "USE_STATIC_CACHE = True and reads data/sp100_clean_snapshot.csv: 100 equities and 2,187 adjusted "
        "closes from 2 January 2018 through 15 September 2026, with 2,186 daily returns and the monthly "
        "covariates. That path does not call Yahoo Finance. The intended public address "
        "https://github.com/simoncheungkc/Multi-SEM still did not resolve on 12 November 2026, and "
        "10.5281/zenodo.XXXXXXX remains a placeholder. The numerical results are reproduced from the "
        "project files. Table S11 reports Moran's I on the filtered innovations. On the geographic layer "
        "the mean OLS statistic is 0.442 and the mean filtered statistic is -0.021. On Split 5 the "
        "filtered statistic is -0.012 (p = 0.755). The share of splits with p > 0.30 is 0.80, 0.94, "
        "0.92, and 0.96 on the four candidate matrices. The filtered geographic statistics range from "
        "-0.077 to 0.007."
    )
    heading("Minor comments", 2)
    paragraph(
        "The residual quadratic form is script S. Q is reserved for the precision. Bivand, Gómez-Rubio "
        "and Rue (2014) is cited for single-weight Bayesian comparisons. Benjamini and Hochberg (1995) "
        "is cited where the false-discovery rule is used. Repeated computational-cost sentences were "
        "shortened. Kappa on a weighted graph is row-sum heterogeneity. Section S5.1 is the data and "
        "code statement. Equations (S1.5) and (S2.7), and the early-rejection step of Algorithm S1, "
        "already render the l1 constraint with absolute-value bars; those bars were checked in the "
        "equation markup and were not rewritten. There was no separate drafting note in Section S5.1 "
        "telling the authors to switch the repository to public."
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
