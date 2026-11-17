"""Manuscript v34, supplement v32, and response letter 9.

Reads version 33, supplement version 31, and letter 8. Does not overwrite them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from build_revision_docs import following_table, insert_after, make_grid_table, replace_paragraph
from build_v10_docx import q
from build_v12_docx import W, load_zip, require_one, store_zip
from build_v26_docs import upright_greek
from build_v29_docs import assert_word_safe, replace_text

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v33.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v34.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v31.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v32.docx"
LETTER_IN = ROOT / "Response_to_Reviewers_8.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers_9.docx"

LM_PARAGRAPH = (
    "Comparison with joint and robust partial LM diagnostics. While Algorithm 1 screens "
    "candidate layers marginally to preserve discovery power before Bayesian regularization, "
    "omnibus dependence across all $k'$ pruned operators can also be evaluated with the "
    "Gaussian multi-weight Lagrange multiplier statistic "
    "$\\mathrm{LM}_{\\mathrm{joint}}=g'\\mathcal{I}^{-1}g$, referred to $\\chi^2_{k'}$, "
    "where $g_s=(e'A_s e)/\\hat{\\sigma}^2$ and "
    "$\\mathcal{I}_{rs}=\\mathrm{tr}((A_r+A_r')A_s)$ (Anselin, 1988; Lee and Liu, 2010), "
    "together with the local-alternative robust partial statistics "
    "$\\mathrm{LM}^*_s=(\\mathcal{I}^{-1}g)_s^2/(\\mathcal{I}^{-1})_{ss}$, referred to "
    "$\\chi^2_1$ (Anselin et al., 1996). Because $\\mathrm{LM}^*_s$ orthogonalizes score "
    "components only locally around $\\lambda_{-s}=0$, it can lose power when global "
    "dependence is strong and candidate layers are highly collinear, as in Greater Boston, "
    "where $r_{\\mathrm{vec}}(A_{\\mathrm{geo}},G_{A_{\\mathrm{geo}}})=0.975$ and "
    "$L\\approx 0.85$. We therefore report $\\mathrm{LM}_{\\mathrm{joint}}$ and "
    "$\\mathrm{LM}^*_s$ as complementary pre-estimation diagnostics in Supplement Section "
    "S4.10 (Table S16), while retaining Algorithm 1's marginal Benjamini–Hochberg screen "
    "as an inclusive discovery step prior to joint Bayesian estimation."
)

SECTION_732 = (
    "Table 4c evaluates the two active networks from Table 2 with five column-standardized "
    "Gaussian covariates and $\\lambda_{\\mathrm{true}}=(0.75,-0.55)'$, which has "
    "$\\ell_1$ norm $1.30$. Across the 200 realized graph pairs, the combined operator "
    "$B(\\lambda_{\\mathrm{true}})=0.75A_1-0.55A_2$ has mean spectral radius $0.693$ and "
    "maximum spectral radius $0.741$, remaining strictly below $1$ in all 200 replications. "
    "For every realized graph pair, homogeneity of the spectral radius implies "
    "$\\rho(B(t\\lambda_{\\mathrm{true}}))=t\\,\\rho(B(\\lambda_{\\mathrm{true}}))<1$ "
    "for all $t\\in[0,1]$, so the entire linear segment from the origin lies inside the "
    "spectral-stability region and hence within the origin-connected invertibility component "
    "where $\\det V(\\lambda)\\neq 0$. Each replication fits both the $\\ell_1$-constrained "
    "sampler and the spectral-stability sampler using four covariance-adapted chains of "
    "6,000 iterations (3,000 burn-in), attaining maximum split-$\\hat{R}$ values of "
    "$1.027$ and $1.011$, respectively. Out-of-sample RMSE is evaluated on a separate 20% "
    "node holdout using induced-subgraph estimation and full-graph precision kriging, and "
    "Figure S11 illustrates the Replication 1 posterior draws against the $\\ell_1$ diamond "
    "and the spectral-radius boundary. Maximum split-$\\hat{R}$ and holdout RMSE remain "
    "separate columns of Table 4c: the $\\ell_1$ fit has maximum split-$\\hat{R}$ $1.02719$ "
    "and holdout RMSE $1.02721$, and the spectral-stability fit has maximum "
    "split-$\\hat{R}$ $1.011$ and holdout RMSE $1.014$."
)

CONCLUSION = [
    (
        "Multiplex Spatial Error Regression (Multi-SEM) provides an integrated Bayesian "
        "workflow for cross-sectional regression when disturbances propagate across multiple "
        "overlapping networks. Rather than introducing a new model class or new "
        "matrix-conditioning identities, the framework operationalizes the established "
        "additive higher-order spatial error specification "
        "$u=\\sum_{s=1}^{k}\\lambda_s A_s u+\\epsilon$, which preserves "
        "$\\mathbb{E}(Y\\mid X)=X\\beta$ under exogeneity, by combining ex-ante redundancy "
        "pruning, design-conditional residual screening with topology-triggered Monte Carlo "
        "calibration, covariance-adapted MCMC estimation on the sufficient $\\ell_1$ "
        "stability ball, and precision-form spatial kriging."
    ),
    (
        "The simulation and diagnostic experiments clarify both the strengths and the "
        "boundaries of this workflow. Topology-calibrated screening corrects finite-sample "
        "size distortions on hub-dominated graphs prior to Benjamini–Hochberg control, while "
        "covariance adaptation navigates narrow posterior ridges induced by strongly collinear "
        "layers. Restricting estimation to $\\Vert\\lambda\\Vert_1<1$ guarantees spectral "
        "invertibility without proposal-level eigendecompositions, but the outside-ball "
        "experiment in Table 4c shows that this sufficient bound truncates parameter inference "
        "when true opposite-signed dependence exceeds unit $\\ell_1$ norm. Induced-subgraph "
        "estimation is a fast surrogate when evaluating the full-graph Schur precision "
        "$P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}$ at every MCMC step is computationally "
        "prohibitive. Its credible intervals pertain to the re-standardized training operator "
        "(Section 7.5, Table S13b), and individual channel magnitudes remain sensitive to "
        "boundary prior geometry even when predictive scores are stable (Section S2.3, Table S4)."
    ),
    (
        "Across the Greater Boston housing and S&P 100 equity applications, Multi-SEM achieves "
        "substantial out-of-sample gains over non-spatial OLS that arise primarily from "
        "contemporaneous precision-form kriging rather than shifts in the conditional "
        "regression mean $X\\beta$. Relative to well-specified single-network spatial "
        "benchmarks, incremental improvements are modest and depend on the validation design: "
        "Multi-SEM performs best under within-region interpolation (Table 9) and "
        "leave-one-town-out holdouts, reverses ranking against a municipal-clique SEM under "
        "spatial-cluster extrapolation (Table S14), and yields modest, autocorrelation-sensitive "
        "gains over an industry SEM in the rolling equity panel (Table 13)."
    ),
    (
        "Two methodological boundaries merit emphasis for applied work. First, marginal "
        "residual Moran screening functions as an inclusive discovery step under "
        "false-discovery-rate control rather than partial-channel identification. Strongly "
        "correlated operators, such as a spatial graph and its Katz resolvent, represent "
        "joint multi-scale dependence rather than separately identified physical mechanisms. "
        "The joint and robust partial Lagrange multiplier diagnostics in Table S16 make that "
        "distinction explicit at the OLS null. Second, extending the pipeline to dynamic "
        "spatio-temporal panels, endogenous spatial lags, and censored or non-Gaussian "
        "likelihoods remains a natural direction for future research."
    ),
]

S410_HEADING = "S4.10 Joint multi-weight and robust partial Lagrange multiplier diagnostics"
S410 = (
    "Table S16 reports the omnibus statistic "
    "$\\mathrm{LM}_{\\mathrm{joint}}=g'\\mathcal{I}^{-1}g$ on $\\chi^2_4$ and the "
    "local-alternative robust partial statistics $\\mathrm{LM}^*_s$ on $\\chi^2_1$, "
    "both evaluated at the OLS null on training residuals. In Greater Boston the mean "
    "joint statistic is $393.58$, and the null is rejected on 50 of 50 splits "
    "(every $p<10^{-16}$). The geographic and resolvent partial means are $88.95$ and "
    "$152.36$, about half the corresponding one-network LM statistics $197.79$ and "
    "$276.14$, consistent with an information correlation of $0.973$ between those "
    "operators. Both partial tests remain significant on all 50 splits. The town partial "
    "mean falls from a marginal LM of $153.53$ to $17.56$ and is below $0.05$ on 49 of "
    "50 splits. The radial partial mean is $21.16$, larger than its marginal LM of "
    "$1.62$. On the August 2026 S&P 100 training block ($n_t=80$) the joint statistic "
    "is $8.11$ ($p=0.088$). The sector robust partial $p$-value is $0.255$, compared "
    "with a marginal Moran $p$-value of $0.062$, while the industry partial test is "
    "$3.94$ ($p=0.047$). These diagnostics are computed by run_joint_lm.py from the "
    "archived Boston split seeds and the offline S&P snapshot, without refitting the "
    "Gibbs sampler."
)

S410_CAPTION = (
    "Table S16. Joint multi-weight LM ($\\chi^2_4$) and local-alternative robust partial "
    "LM ($\\chi^2_1$) diagnostics on training-sample OLS residuals. Boston Moran $z$ "
    "values are means of the archived Proposition 1 statistics. Boston partial $p$-values "
    "are medians across the 50 splits; geographic and resolvent medians are below "
    "$10^{-16}$. The S&P $z$ statistics are the Gaussian residual Moran statistics on the "
    "80-firm training block. The radial Moran test is below $0.05$ on 28 of 50 Boston "
    "splits, while its robust partial test is below $0.05$ on 50 of 50."
)

TABLE_S16 = [
    ["Dataset and layer", "Moran z", "Partial LM", "Partial p", "Joint LM", "Joint p"],
    ["Boston, 50-split mean (n = 405)", "—", "—", "—", "393.58", "< 1e-16"],
    ["Geographic 6-NN", "15.03", "88.95", "< 1e-16", "—", "—"],
    ["Geographic resolvent", "18.19", "152.36", "< 1e-16", "—", "—"],
    ["Town jurisdiction", "13.23", "17.56", "2.0e-5", "—", "—"],
    ["Radial highway", "2.13", "21.16", "6.4e-6", "—", "—"],
    ["S&P 100, August 2026 (n = 80)", "—", "—", "—", "8.11", "0.088"],
    ["Trailing return correlation", "-0.28", "0.58", "0.446", "—", "—"],
    ["Boosted correlation resolvent", "0.15", "0.06", "0.811", "—", "—"],
    ["Yahoo Finance sector", "1.87", "1.30", "0.255", "—", "—"],
    ["Yahoo Finance industry", "2.20", "3.94", "0.047", "—", "—"],
]

LETTER = (
    "We thank the reviewer for pointing out that marginal residual Moran tests can reject "
    "overlapping layers together when both correlate with a common dependence source, and "
    "for suggesting a joint multi-weight Lagrange multiplier test. Section 4.2 and "
    "Supplementary Section S4.10 (Table S16) now report the omnibus statistic "
    "LM_joint = g' I^{-1} g on chi-squared with k' degrees of freedom and the "
    "local-alternative robust partial statistics on chi-squared with 1 degree of freedom "
    "(Anselin et al., 1996). On the 50 Boston training splits the mean joint statistic is "
    "393.58 and every p-value is below 10^{-16}. The geographic and resolvent partial "
    "means are 88.95 and 152.36, about half the corresponding one-network LM statistics, "
    "and both remain significant on every split. The August 2026 S&P 100 training block "
    "has joint statistic 8.11 (p = 0.088). Because the robust partial statistics are local "
    "and can lose power when global dependence is strong, Algorithm 1 remains the inclusive "
    "discovery step. Separation of retained collinear layers stays with the joint Bayesian "
    "posterior, the ablation in Table S9, and the high-collinearity simulation in Table S12."
)


def supplement(body):
    anchor = require_one(body, "Table S15. Script, seed")
    template = following_table(body, "Table S14. Equal-weight")
    heading_style = require_one(body, "S4.9 Spatial extrapolation")
    prose_style = require_one(body, "Table S14 pools one prediction")
    blocks = [
        (S410_HEADING, heading_style),
        (S410, prose_style),
        (S410_CAPTION, prose_style),
    ]
    nodes = [make_grid_table(template, TABLE_S16)]
    from build_v10_docx import make_p
    from build_v12_docx import blocks_from
    for text, style in reversed(blocks):
        nodes.insert(0, make_p(blocks_from(text), style))
    for node in nodes:
        anchor.addprevious(node)


def conclusion(body):
    heading = require_one(body, "9. Conclusion")
    style = require_one(body, "Multi-SEM provides a framework for cross-sectional")
    insert_after(heading, CONCLUSION, style)
    for snippet in (
        "Multi-SEM provides a framework for cross-sectional",
        "The four pipeline components are an integrated Bayesian",
        "The simulation results show that the oracle specification",
        "Three additional analyses evaluate the sensitivity",
        "On the August 2026 panel the induced-subgraph posterior means",
    ):
        body.remove(require_one(body, snippet))


def letter(body):
    replace_text(
        body,
        "The revised manuscript is version 33 and the supplement is version 31.",
        "The revised manuscript is version 34 and the supplement is version 32.",
    )
    replace_text(
        body,
        "A joint multi-weight Lagrange-multiplier test is not added. ",
        "Table S16 now reports the joint and robust partial LM diagnostics beside that screen. ",
    )
    anchor = require_one(body, "2. Resolved by Clarification")
    from build_v10_docx import make_p
    from build_v12_docx import blocks_from
    anchor.addprevious(make_p(blocks_from(LETTER), anchor))


def manifest():
    path = ROOT / "MANIFEST.sha256"
    lines = path.read_text(encoding="utf-8").splitlines()
    digest = hashlib.sha256((ROOT / "run_joint_lm.py").read_bytes()).hexdigest()
    row = f"{digest}  run_joint_lm.py  # Table S16 joint and robust partial LM diagnostics"
    lines = [line for line in lines if "run_joint_lm.py" not in line]
    lines.append(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    main_files, main_root = load_zip(MAIN_IN)
    main_body = main_root.find(q(W, "body"))
    screen = require_one(main_body, "frozen correlation between the geographic coefficient")
    insert_after(screen, [LM_PARAGRAPH], screen)
    replace_paragraph(main_body, "Table 4c uses the two active networks", SECTION_732)
    conclusion(main_body)
    anselin = require_one(main_body, "Anselin, L. (1988). Spatial Econometrics")
    insert_after(
        anselin,
        [
            "Anselin, L., Bera, A. K., Florax, R., & Yoon, M. J. (1996). Simple diagnostic "
            "tests for spatial dependence. Regional Science and Urban Economics, 26(1), 77–104."
        ],
        anselin,
    )
    upright_greek(main_body)
    assert_word_safe(main_root, "v34")
    store_zip(MAIN_OUT, main_files, main_root)

    sup_files, sup_root = load_zip(SUP_IN)
    sup_body = sup_root.find(q(W, "body"))
    supplement(sup_body)
    upright_greek(sup_body)
    assert_word_safe(sup_root, "supplement v32")
    store_zip(SUP_OUT, sup_files, sup_root)

    letter_files, letter_root = load_zip(LETTER_IN)
    letter(letter_root.find(q(W, "body")))
    assert_word_safe(letter_root, "letter 9")
    store_zip(LETTER_OUT, letter_files, letter_root)
    manifest()
    print("wrote", MAIN_OUT.name, SUP_OUT.name, LETTER_OUT.name)


if __name__ == "__main__":
    main()
