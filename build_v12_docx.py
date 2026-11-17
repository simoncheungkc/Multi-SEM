"""Build v12, supplement v10, and the EcoSta cover letter.

Does not overwrite v11 or Supplementary_Materials_MultiSEM_v9.
"""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path

from lxml import etree

from build_v10_docx import convert_n_star, make_p, omath, q, replace_with
from mnreg.latex_omml import el

ROOT = Path(__file__).resolve().parent
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
REL = "http://schemas.openxmlformats.org/package/2006/relationships"
OFFICE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
A = "http://schemas.openxmlformats.org/drawingml/2006/main"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
IMAGE_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"

MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v11.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v9.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v12.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v10.docx"
LETTER_OUT = ROOT / "Cover_Letter_MultiSEM_EcoSta.docx"


def ptext(p):
    return "".join(t.text or "" for t in p.iter(q(W, "t")))


def blocks_from(s: str):
    parts = re.split(r"\$([^$]+)\$", s)
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part:
                out.append(("t", part))
        else:
            out.append(("m", part))
    return out


def paras(*strings):
    return [blocks_from(s) for s in strings]


def find_hits(body, snippet):
    return [c for c in body if c.tag == q(W, "p") and snippet in ptext(c)]


def require_one(body, snippet):
    hits = find_hits(body, snippet)
    if len(hits) != 1:
        preview = " | ".join(ptext(h)[:70] for h in hits[:4])
        raise SystemExit(f"{snippet!r} matched {len(hits)}: {preview}")
    return hits[0]


def remove_containing(body, snippet):
    body.remove(require_one(body, snippet))


def replace_containing(body, snippet, text):
    replace_with(require_one(body, snippet), blocks_from(text))


def span_nodes(body, start_snippet, end_snippet):
    start = require_one(body, start_snippet)
    nodes = []
    seen = False
    end = None
    for c in list(body):
        if c is start:
            seen = True
        if not seen:
            continue
        if c is not start and c.tag == q(W, "p") and ptext(c).startswith(end_snippet):
            end = c
            break
        nodes.append(c)
    if end is None:
        raise SystemExit(f"no endpoint {end_snippet!r} after {start_snippet!r}")
    return nodes, end


def replace_span(body, start_snippet, end_snippet, paragraph_strings):
    nodes, end = span_nodes(body, start_snippet, end_snippet)
    source = nodes[0]
    for n in nodes:
        body.remove(n)
    for text in paragraph_strings:
        end.addprevious(make_p(blocks_from(text), source))
    return len(nodes)


def insert_before_snippet(body, snippet, paragraph_strings):
    anchor = require_one(body, snippet)
    for text in paragraph_strings:
        anchor.addprevious(make_p(blocks_from(text), anchor))


def load_zip(path: Path):
    files = {}
    with zipfile.ZipFile(path) as zin:
        for item in zin.infolist():
            files[item.filename] = zin.read(item.filename)
    root = etree.fromstring(files["word/document.xml"])
    return files, root


def store_zip(path: Path, files, root):
    files = dict(files)
    files["word/document.xml"] = etree.tostring(
        root, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)
    os.replace(tmp, path)


def word_count_root(root, stop_at=None):
    n = 0
    body = root.find(q(W, "body"))
    for child in body:
        if child.tag == q(W, "p") and stop_at and ptext(child).startswith(stop_at):
            break
        if child.tag == q(W, "tbl"):
            continue
        n += len(ptext(child).split())
    return n


def edit_main(body):
    # Phase 1.2
    node = require_one(body, "is the only specification that improves on OLS")
    for t in node.iter(q(W, "t")):
        if t.text and "Neither beats OLS" in t.text:
            t.text = t.text.split("Neither beats OLS")[0] + (
                "While Sector SEM (0.0588) and Industry SEM (0.0609) also outperform "
                "baseline OLS (0.0637), the joint Multi-SEM achieves the lowest overall "
                "error (0.0576). In contrast, the non-network benchmarks—industry fixed "
                "effects (0.0653) and one-factor shrinkage (0.0697)—fail to beat OLS."
            )

    # Phase 1.4
    node = require_one(body, "Because this parameter adjustment compensates")
    for t in node.iter(q(W, "t")):
        if t.text and "Because this parameter adjustment compensates" in t.text:
            t.text = (
                "In the simulated configurations examined here, this parameter adjustment "
                "offset the modified graph scale, resulting in negligible predictive loss "
                "(0.31% at n = 200 and 0.08% at n = 500). Consequently, the induced-subgraph "
                "estimator acts as a fast empirical surrogate when cross-block links are "
                "unavailable ex-ante, while the Schur marginal precision "
            )
        elif t.text and t.text.strip().startswith("provides the strictly coherent"):
            t.text = " remains the exact likelihood criterion when full network topology is observed."

    # Phase 1.3
    remove_containing(body, "recomputed over")

    # Phase 2 taxonomy
    gics_node = require_one(body, "The GICS industry network")
    for t in gics_node.iter(q(W, "t")):
        if t.text and "GICS industry network" in t.text:
            t.text = t.text.replace("GICS industry network", "Yahoo Finance industry network")

    # Phase 2 section headings
    for old, new in [
        ("6.1 Partitioning the Multiplex System", "6.1 Partitioning"),
        ("6.2 Precision-Form Contemporaneous Kriging", "6.2 Precision Kriging"),
        ("6.3 Fully Bayesian Predictive Inference vs. Plug-in Estimation", "6.3 Bayesian Predictive Sampling"),
        ("6.4 Methodological Distinction: Contemporaneous Kriging vs. Temporal Forecasting", "6.4 Kriging vs. Forecasting"),
        ("6.5 Predictive Evaluation Metrics and Baseline Models", "6.5 Metrics and Benchmarks"),
    ]:
        replace_containing(body, old, new)

    # Phase 4 replacements
    replace_span(
        body,
        "Many empirical systems are multiplex",
        "Keywords:",
        [
            "Empirical entities frequently participate in multiple overlapping networks, allowing residual dependence to propagate across several relational channels simultaneously. This paper develops Multiplex Spatial Error Regression (Multi-SEM) as an integrated econometric pipeline combining network screening, collinearity pruning, stability-constrained Bayesian estimation, and precision-form prediction.",
            "Multi-SEM addresses four key methodological challenges: (i) an additive disturbance specification preserves linear conditional-mean interpretability ($\\mathbb{E}[Y \\mid X] = X\\beta$) under exogeneity without spatial-lag endogeneity; (ii) an $\\ell_1$-norm bound, $\\Vert \\lambda \\Vert_1 = \\sum_{s=1}^{k} \\vert \\lambda_s \\vert < 1$, provides a uniform sufficient condition for spectral stability and filter invertibility across row-substochastic layers; (iii) candidate networks are screened via residual Moran’s $I$ with exact design-conditional moments, using conditional Gaussian Monte Carlo recalibration for hub-dominated graphs prior to false discovery rate control; and (iv) unobserved outcomes are imputed via precision-form kriging ($Q = \\sigma^{-2} V_\\lambda' V_\\lambda$), reducing computational complexity from $\\mathcal{O}(n_t^3)$ to $\\mathcal{O}(n_*^3)$.",
            "In synthetic benchmarks, Multi-SEM eliminates boundary estimation instabilities and achieves an 8.6% holdout RMSE reduction (6.1% under feasible screening). A 63-month longitudinal study of S&P 100 returns demonstrates that Multi-SEM reliably captures contemporaneous residual co-movement, outperforming OLS in 44 of 63 months (two-sided sign test $p = 0.0022$).",
        ],
    )
    replace_containing(
        body,
        "Keywords:",
        "Keywords: Multiple Networks; Spatial Error Model; Moran’s $I$; Gibbs Sampler; Spatial Kriging; S&P 100.",
    )

    replace_span(
        body,
        "1. Introduction",
        "2. Related Work",
        [
            "1. Introduction",
            "In interconnected economic, financial, and social systems, cross-sectional entities are rarely independent. Latent shocks, omitted contextual variables, and common environments routinely generate cross-sectional correlation across entities. Under strictly exogenous regressors ($\\mathbb{E}[u \\mid X] = 0$), ordinary least squares (OLS) remains consistent but is statistically inefficient, standard hypothesis tests are distorted, and out-of-sample predictions discard informative residual dependence.",
            "Spatial econometrics models such dependence primarily through Spatial Autoregressive (SAR) lag models or Spatial Error Models (SEM). Traditional formulations, however, assume dependence operates through a single weight matrix $W$. In modern empirical applications, entities are embedded in multiplex networks, participating simultaneously in multiple overlapping relational structures. Publicly listed firms, for instance, are simultaneously linked through industrial sectors, sub-industry niches, supply-chain flows, and return co-movements. Restricting analysis to a single network risks omitted network dependence, introduces arbitrary definitions of neighborhood proximity, and overlooks severe multicollinearity across overlapping relational channels.",
            "This paper develops Multiplex Spatial Error Regression (Multi-SEM) as an integrated inferential and computational pipeline combining four linked components:",
            "Additive Error Propagation. Formulating disturbance dependence as $u = \\sum_{s=1}^{k} \\lambda_s A_s u + \\epsilon$ preserves linear conditional-mean interpretability ($\\mathbb{E}[Y \\mid X] = X\\beta$) under exogeneity while bypassing the simultaneity bias of spatial lag models.",
            "Uniform $\\ell_1$ Spectral Stability. We establish that $\\Vert \\lambda \\Vert_1 = \\sum_{s=1}^{k} \\vert \\lambda_s \\vert < 1$ provides a uniform sufficient condition ensuring invertibility of the multiplex filter $V_\\lambda = I_n - \\sum_{s=1}^{k} \\lambda_s A_s$ across row-substochastic layers, preventing MCMC exploration of unstable boundary regions where large signed coefficients cancel.",
            "Topology-Calibrated Residual Screening. Candidate layers are screened against the global uncorrelated null using residual Moran’s $I$ with exact design-conditional moments and isolate scaling. While Gaussian calibration is accurate for homogeneous graphs, hub-dominated topologies trigger an ex-ante conditional Monte Carlo calibration before Benjamini–Hochberg FDR control. Structurally redundant transforms (for example, clique resolvents with $\\vert r_{\\mathrm{vec}} \\vert \\ge 0.99$) are pruned ex-ante.",
            "Dual Precision-Form Kriging. Unobserved entities are imputed via the disturbance precision matrix $Q = \\sigma^{-2} V_\\lambda' V_\\lambda$, which shifts the computational burden from an $\\mathcal{O}(n_t^3)$ dense training covariance inversion to an $\\mathcal{O}(n_*^3)$ holdout-block factorization.",
            "Across synthetic graphs and a 63-month panel of S&P 100 equity returns, Multi-SEM eliminates boundary estimation instabilities, achieves an 8.6% holdout RMSE reduction in simulations (6.1% under feasible screening), and outperforms OLS in 44 of 63 months (sign test $p = 0.0022$). Complete proofs, prior sensitivity analyses, and replication code are compiled in the Supplementary Materials.",
        ],
    )

    replace_span(
        body,
        "2. Related Work",
        "3. The Multiplex Spatial Error Model",
        [
            "2. Related Work and Methodological Background",
            "Spatial econometrics distinguishes between dependence in observable outcomes (SAR; Anselin, 2009; LeSage & Pace, 2009) and dependence in unobserved disturbances (SEM; Kelejian & Prucha, 2010). In SAR models, endogenous spatial lags $AY$ model behavioral contagion and peer effects, but OLS estimation yields inconsistent estimates due to simultaneity. Conversely, SEM models unobserved spatial confounders or omitted regional variables in the disturbance $u$, preserving standard regression interpretability under exogeneity: $\\mathbb{E}[Y \\mid X] = X\\beta$.",
            "When expanding to multiple relational layers, multiplicative spatial models $(I - \\lambda_1 A_1)(I - \\lambda_2 A_2)u = \\epsilon$ generate non-commutative cross-products $A_1 A_2$ that lose structural interpretability after row-normalization. This motivates the additive formulation $u = \\sum_{s=1}^{k} \\lambda_s A_s u + \\epsilon$ (Florax & Nijkamp, 2005; Bivand et al., 2014).",
            "However, existing multiplex regressions face three critical issues. First, screening networks via auxiliary OLS regressions ($Y = X\\beta + \\lambda_s A_s Y + \\epsilon$) introduces endogeneity and tests outcome dependence rather than disturbance dependence (Manski, 1993; Lee, 2002). Residual Moran screening aligns testing with the SEM error structure. Second, earlier multi-network models rely on the signed-sum condition $\\vert \\sum_{s=1}^{k} \\lambda_s \\vert < 1$, which is insufficient to guarantee non-singularity when $k \\ge 2$ because large opposing coefficients can produce singular filters. The $\\ell_1$ bound $\\Vert \\lambda \\Vert_1 < 1$ provides a uniform sufficient condition. Third, higher-order Katz/resolvent transforms of dense affiliation cliques belong to $\\operatorname{span}\\{J, I\\}$; after row-standardization, they duplicate base networks and must be pruned to avoid collinearity.",
        ],
    )

    replace_span(
        body,
        "3.3 Structural Identification",
        "4. Hypothesis Testing",
        [
            "3.3 Intercept Attenuation Near the Stability Boundary",
            "For fully row-stochastic network layers ($A_s 1_n = 1_n$), the spatial filter contracts the constant vector:",
            "$V_\\lambda 1_n = \\left(1 - \\sum_{s=1}^{k} \\lambda_s\\right) 1_n$",
            "As $\\sum_{s=1}^{k} \\lambda_s \\to 1$, $V_\\lambda 1_n \\to 0_n$, attenuating the effective intercept regressor and inflating the posterior variance of $\\beta_0$. If the non-intercept regressors $X_c$ are mean-centered and candidate networks are symmetric, $1_n' V_\\lambda X_c = 0$, shielding slope estimates from this specific information loss. When networks contain isolate rows (as with the 17 isolated firms in the empirical industry network), contraction is non-uniform, resulting in variance inflation rather than global rank collapse. Complete derivations appear in Supplement Section S1.",
            "3.4 Affiliation Clique Resolvents and Structural Redundancy",
            "Multi-step paths are often modeled via Katz resolvents $G_A = A(I - \\alpha A)^{-1}$. For an affiliation clique $C_m = \\frac{1}{n_m - 1}(J_{n_m} - I_{n_m})$, every positive integer power satisfies $C_m^l \\in \\operatorname{span}\\{J_{n_m}, I_{n_m}\\$. Higher-order return walks create positive diagonal entries. Purging these self-loops to maintain zero diagonals ($A_{s,ii} = 0$) followed by row-standardization yields an exact identity: $\\widetilde{G}(C_m) \\equiv C_m$, verified numerically to machine precision (about $10^{-16}$). Resolvent boosting is therefore redundant for affiliation cliques and is reserved for non-clique networks (for example, return correlations).",
            "3.5 Likelihood and Computational Complexity",
            "Under Gaussian innovations $\\epsilon \\sim \\mathcal{N}(0, \\sigma^2 I_n)$, the concentrated profile log-likelihood for $\\lambda$ is",
            "$\\ln L_c(\\lambda) = -\\frac{n}{2}[1 + \\ln(2\\pi)] - \\frac{n}{2}\\ln[\\hat{\\sigma}^{2}(\\lambda)] + \\ln\\vert\\det V_\\lambda\\vert$",
            "where $\\hat{\\sigma}^{2}(\\lambda) = \\frac{1}{n}(Y - X\\hat{\\beta}_\\lambda)' V_\\lambda' V_\\lambda (Y - X\\hat{\\beta}_\\lambda)$ and $\\hat{\\beta}_\\lambda = (X' V_\\lambda' V_\\lambda X)^{-1} X' V_\\lambda' V_\\lambda Y$. For $n \\le 500$, $\\ln\\vert\\det V_\\lambda\\vert$ is evaluated via dense partial-pivoting LU decomposition at cost $\\mathcal{O}(n^3)$. For large-scale sparse networks, polynomial or stochastic trace estimators are documented in Supplement Section S5.2.",
        ],
    )

    replace_span(
        body,
        "The second-order term illustrates",
        "3.2 Invertibility",
        [],
    )

    replace_span(
        body,
        "Fitting Equation (29)",
        "4.2 Residual Moran",
        [
            "Under the global null the model is Gaussian linear regression. OLS residuals are $e = My$ with residual maker $M = I - X(X'X)^{-1}X'$. Exact design-conditional moments of residual Moran’s $I$, including isolate scaling, are stated in Proposition 1. The algebra is in Supplement Section S1.2.",
        ],
    )

    replace_span(
        body,
        "4.3 Exact Finite-Sample Moments",
        "Proposition 1",
        [
            "4.3 Exact Finite-Sample Moments",
            "Proposition 1 records the design-conditional null moments. The proof, including the spherical representation and isolate scaling, is Supplement Section S1.2.",
        ],
    )

    replace_span(
        body,
        "Remark 1. Finite-Sample Distribution",
        "Corollary 1",
        [],
    )

    # Algorithm 1, placed with the ex-ante pruning rule
    insert_before_snippet(
        body,
        "active subset of networks retained",
        [
            "Algorithm 1. Ex-ante pruning and residual Moran screening",
            "Step 1. Compute the OLS residual vector from $Y = X\\beta + u$.",
            "Step 2. For each pair $(A_r, A_s)$ in $\\{A_1, \\ldots, A_K\\}$, compute the off-diagonal correlation $r_{\\mathrm{vec}}(A_r, A_s)$. If $\\vert r_{\\mathrm{vec}} \\vert \\ge 0.99$, drop the structurally redundant transform ex-ante. Denote the non-redundant candidate set by $\\{A_1, \\ldots, A_{K'}\\}$.",
            "Step 3. For $s = 1, \\ldots, K'$, compute residual Moran’s $I$ on $A_s$ using the exact design-conditional moments. Use the Section 4.4 reference: the analytical normal tail, or conditional Gaussian Monte Carlo with $B = 499$ when $\\kappa_s > 3$ or $\\psi_s > 0.20$.",
            "Step 4. Apply Benjamini–Hochberg FDR control to these $K'$ p-values and retain the layers that survive. Family size is $K'$, fixed before any outcome is used in a test. There is no post-FDR collinearity step and no rule that keeps the matrix with the smaller Moran p-value.",
        ],
    )

    replace_span(
        body,
        "5.3 Full Conditional Distributions",
        "5.5 Evaluation of the Spatial Filter",
        [
            "5.3 Conditional Posterior Distributions",
            "Conditional on $\\lambda$, Multi-SEM reduces to standard linear regression on filtered variables $\\tilde{Y} = V_\\lambda Y$ and $\\tilde{X} = V_\\lambda X$. Under conjugate priors $\\beta \\sim \\mathcal{N}(\\mu_0, \\Sigma_0)$ and $\\sigma^2 \\sim \\mathrm{Inv\\text{-}Gamma}(\\alpha_0, \\gamma_0)$, the full conditionals are Gaussian and inverse-gamma:",
            "$\\beta \\mid Y, X, \\sigma^2, \\lambda \\sim \\mathcal{N}(\\mu_\\beta, \\Sigma_\\beta),\\quad \\Sigma_\\beta = (\\Sigma_0^{-1} + \\sigma^{-2} X' V_\\lambda' V_\\lambda X)^{-1},\\quad \\mu_\\beta = \\Sigma_\\beta(\\Sigma_0^{-1}\\mu_0 + \\sigma^{-2} X' V_\\lambda' V_\\lambda Y)$",
            "$\\sigma^2 \\mid Y, X, \\beta, \\lambda \\sim \\mathrm{Inv\\text{-}Gamma}\\left(\\alpha_0 + \\frac{n}{2},\\ \\gamma_0 + \\frac{1}{2}(Y - X\\beta)' V_\\lambda' V_\\lambda (Y - X\\beta)\\right)$",
            "Full algebraic completions of the square appear in Supplement Section S2.1.",
            "5.4 Adaptive Metropolis Step on the $\\ell_1$ Ball",
            "Updating $\\lambda$ proceeds via random-walk Metropolis with candidate draws $\\lambda^* \\sim \\mathcal{N}(\\lambda^{(t-1)}, \\tau_t^2 I_k)$. If $\\Vert \\lambda^* \\Vert_1 \\ge 1$, the proposal is rejected immediately without evaluating matrix determinants. Otherwise, $\\lambda^*$ is accepted with probability $\\alpha = \\min(1, \\exp(\\Delta \\ln \\pi))$, where",
            "$\\Delta \\ln \\pi = \\ln\\vert\\det V_{\\lambda^*}\\vert - \\ln\\vert\\det V_{\\lambda^{(t-1)}}\\vert - \\frac{Q_{\\lambda^*} - Q_{\\lambda^{(t-1)}}}{2\\sigma^2} - \\frac{\\xi}{2}(\\Vert \\lambda^* \\Vert_2^2 - \\Vert \\lambda^{(t-1)} \\Vert_2^2)$",
            "Proposal scale $\\tau_t$ is adapted every 50 iterations during burn-in to target an acceptance rate of 0.30, and held fixed post-burn-in to ensure a time-homogeneous Markov transition kernel (Roberts & Rosenthal, 2009).",
        ],
    )

    replace_span(
        body,
        "5.6 MCMC Implementation",
        "6. Out-of-Sample Prediction",
        [
            "5.6 MCMC Implementation and Convergence Diagnostics",
            "Four chains are started from overdispersed values inside the $\\ell_1$ ball. Convergence is assessed with the classical split-$\\hat{R}$ (each chain is split in half; this is not the rank-normalized statistic of Vehtari et al.) and with the effective sample size. The operational threshold is split-$\\hat{R} \\le 1.025$. The boundary statistic on draw $m$ is $L^{(m)} = \\sum_{s=1}^{k} \\vert \\lambda_s^{(m)} \\vert$, which is distinct from the signed sum $S^{(m)} = \\sum_{s=1}^{k} \\lambda_s^{(m)}$ in the intercept identity.",
        ],
    )

    replace_span(
        body,
        "To evaluate predictive performance on the holdout set",
        "Benchmark Models",
        [
            "Holdout accuracy is RMSE, MAE, and the Pearson correlation between predicted and actual outcomes.",
        ],
    )

    # Section 7 prose that restates tables or describes moved figures
    for snippet in [
        "To systematically evaluate the statistical",
        "Finite-sample performance of the residual Moran",
        "Parameter recovery of the constrained Bayesian",
        "Out-of-sample predictive performance. We evaluate the accuracy",
        "Together, these experiments evaluate the complete Multi-SEM workflow",
        "which restricts posterior sampling to the prescribed",
        "Only two of the five candidate networks are active",
        "The true network autoregressive parameters satisfy",
        "The regression design matrix is",
        "and the innovation variance is",
        "The response vector is then generated from the Multi-SEM",
        "This simulation design therefore provides a controlled",
        "We evaluate the residual Moran",
        "The difference is consistent with the spectral behaviour",
        "With 500 Monte Carlo replications, the standard error",
        "For relatively homogeneous networks, the analytical",
        "Statistical Power",
        "Under the two-network alternative DGP",
        "The inactive-network rejection frequencies",
        "Interpretation of Figures 1",
        "Figures 1–4 show the same size",
        "We first examine the parameter-recovery performance",
        "while their  norm is approximately",
        "Thus, the posterior mean vector lies within",
        "Because the stability restriction applies to every posterior draw",
        "MCMC Convergence and Mixing",
        "All split-",
        "Figure 5 displays the post-burn-in",
        "Figure 6 presents the corresponding",
        "Monte Carlo Parameter Recovery",
        "The preceding experiment examines posterior recovery",
        "Table 4a reports the results",
        "Several mechanisms can contribute to the observed shrinkage",
        "Intercept Information Near the Stability Boundary",
        "The single-dataset five-network experiment",
        "For the fitted five-network model",
        "The corresponding filtered constant direction",
        "subject to the qualifications for any zero rows",
        "Overall, the simulation results show that the constrained",
        "To evaluate whether modelling multiple network layers improves out-of-sample",
        "The Multi-SEM predictors are compared",
        "Multi-SEM versus OLS.",
        "Thus, jointly modelling the two active network-error",
        "Multi-SEM versus single-network SEMs.",
        "Impact of inactive networks.",
        "Moran-selected Multi-SEM.",
        "Figure 7 illustrates predicted versus actual",
        "Finally, Figure 8 compares",
        "Taken together, the results in Table 5",
        "To evaluate the practical performance of the Multiplex",
        "The empirical study focuses on three aspects",
        "Network screening and selection. We examine whether",
        "Bayesian estimation and stability.",
        "Out-of-sample contemporaneous prediction. We evaluate whether",
        "Taken together, the 63-month empirical analysis",
        "and examines whether modelling multiple forms of residual",
    ]:
        hits = find_hits(body, snippet)
        if snippet == "MCMC Convergence and Mixing" and len(hits) == 2:
            body.remove(hits[0])
            continue
        if len(hits) != 1:
            raise SystemExit(f"delete {snippet!r} matched {len(hits)}")
        body.remove(hits[0])

    insert_before_snippet(
        body,
        "7.1 Simulation Design",
        [
            "The experiments check screening size and power, posterior recovery inside the $\\ell_1$ ball, and holdout prediction. Design details are in Table 2. Size and power are in Table 3, with plots in Supplementary Figures S1–S4. Recovery is in Tables 4 and 4a. Holdout error is in Table 5; the corresponding charts are Supplementary Figures S5 and S6.",
        ],
    )
    insert_before_snippet(
        body,
        "Table 5.  Mean out-of-sample RMSE",
        [
            "On the n = 200, 20% node-holdout design, the oracle Multi-SEM mean RMSE is 1.025, against 1.122 for OLS, an 8.6% reduction. The Moran-selected specification has mean RMSE 1.054, a 6.1% reduction relative to OLS. Supplementary Figures S5 and S6 show one holdout scatter and the mean RMSE by model.",
        ],
    )

    replace_span(
        body,
        "Lagged Monthly Return",
        "Within each estimation window",
        [
            "The four covariates, all dated before the target month and then standardized, are the lagged monthly return, realized volatility over the preceding 30 days, 12-minus-1 momentum, and the lagged log share price. The log price is not market equity: it depends on shares outstanding and on splits.",
        ],
    )
    replace_containing(
        body,
        "Figure 10 presents scatter",
        "Supplementary Figure S7 plots predicted against actual August 2026 holdout returns.",
    )
    replace_containing(
        body,
        "Figure 12 summarizes",
        "Supplementary Figure S8 reports average holdout RMSE by model over the 63 months. The month-by-month path is Figure 11.",
    )

    replace_span(
        body,
        "9. Conclusion",
        "References",
        [
            "9. Conclusion",
            "This paper develops Multiplex Spatial Error Regression (Multi-SEM) as a unified statistical pipeline for cross-sectional regression and spatial imputation when residual dependence operates across multiple overlapping networks. By situating multiplex dependence in the disturbance rather than in endogenous outcome lags, Multi-SEM preserves linear conditional-mean interpretability under exogeneity.",
            "The pipeline resolves four applied challenges. First, the parameter restriction $\\Vert \\lambda \\Vert_1 = \\sum_{s=1}^{k} \\vert \\lambda_s \\vert < 1$ provides a uniform sufficient condition ensuring spectral stability across row-substochastic layers, preventing MCMC algorithms from exploring explosive regions where large opposing coefficients offset. Second, candidate networks are screened via residual Moran’s $I$ using exact design-conditional null moments with isolate scaling; hub-dominated degree structures trigger an ex-ante Gaussian Monte Carlo calibration before Benjamini–Hochberg FDR control. Third, near-collinear clique resolvents are pruned ex-ante. Fourth, out-of-sample prediction is formulated in precision form, replacing dense $\\mathcal{O}(n_t^3)$ covariance inversions with $\\mathcal{O}(n_*^3)$ holdout-block factorizations.",
            "Simulation experiments confirm that Multi-SEM eliminates boundary instability and achieves an 8.6% holdout RMSE reduction (6.1% under feasible screening). Comparisons of induced training subgraphs with coherent Schur-complement marginal precisions show that the induced subgraph is an accurate, computationally fast surrogate when cross-block ties are missing. In a 63-month longitudinal study of S&P 100 returns, Multi-SEM outperforms OLS in 44 of 63 months (two-sided sign test $p = 0.0022$). The predictive gain arises primarily from contemporaneous peer-residual kriging rather than from regression mean shifts.",
            "Promising directions for future research include extending Multi-SEM to dynamic spatial panel models with time-varying network topologies and accommodating generalized linear models for discrete and count network outcomes.",
        ],
    )

    # Phase 1.5
    tables = [c for c in body if c.tag == q(W, "tbl")]
    last = tables[-1]
    blob = "".join(t.text or "" for t in last.iter(q(W, "t")))
    if "mnreg" not in blob and "Path" not in blob:
        raise SystemExit("last table is not the replication manifest")
    body.remove(last)

def retitle_pointers(body):
    # Captions of moved figures have already been removed.
    # Match the full figure number so Figure 11 is left unchanged.
    mapping = {"1": "S1", "2": "S2", "3": "S3", "4": "S4", "7": "S5", "8": "S6", "10": "S7", "12": "S8"}

    def repl_text(text):
        text = text.replace("Figures 1–4", "Supplementary Figures S1–S4")
        text = text.replace("Figures 1-4", "Supplementary Figures S1–S4")
        text = text.replace("Figures 7–8", "Supplementary Figures S5–S6")
        text = text.replace("Figures 7-8", "Supplementary Figures S5–S6")

        def one(match):
            number = match.group(1)
            if number not in mapping:
                return match.group(0)
            return "Supplementary Figure " + mapping[number]

        return re.sub(r"Figure (\d+)", one, text)

    for child in body:
        if child.tag != q(W, "p"):
            continue
        for t in child.iter(q(W, "t")):
            if t.text:
                t.text = repl_text(t.text)


MOVED = [
    ("Figure 1.", "Supplementary Figure S1."),
    ("Figure 2.", "Supplementary Figure S2."),
    ("Figure 3.", "Supplementary Figure S3."),
    ("Figure 4.", "Supplementary Figure S4."),
    ("Figure 7.", "Supplementary Figure S5."),
    ("Figure 8.", "Supplementary Figure S6."),
    ("Figure 10.", "Supplementary Figure S7."),
    ("Figure 12.", "Supplementary Figure S8."),
]


def collect_figures(body):
    children = list(body)
    blocks = []
    for i, c in enumerate(children):
        if c.tag != q(W, "p"):
            continue
        s = ptext(c).strip()
        for old, new in MOVED:
            if s.startswith(old):
                parts = []
                if i and children[i - 1].find(".//" + q(W, "drawing")) is not None:
                    parts.append(children[i - 1])
                parts.append(c)
                blocks.append((old, new, parts))
                break
    found = [b[0] for b in blocks]
    expected = [m[0] for m in MOVED]
    if found != expected:
        raise SystemExit(f"figure captions found {found}, expected {expected}")
    return blocks


def move_figures(main_files, main_root, sup_files, sup_root):
    main_body = main_root.find(q(W, "body"))
    sup_body = sup_root.find(q(W, "body"))
    blocks = collect_figures(main_body)
    rels = etree.fromstring(main_files["word/_rels/document.xml.rels"])
    rel_by_id = {rel.get("Id"): rel.get("Target") for rel in rels}
    sup_rels = etree.fromstring(sup_files["word/_rels/document.xml.rels"])
    used_ids = {rel.get("Id") for rel in sup_rels}
    next_id = 1
    copied_targets = {}

    def new_rid():
        nonlocal next_id
        while f"rIdM{next_id}" in used_ids:
            next_id += 1
        rid = f"rIdM{next_id}"
        used_ids.add(rid)
        next_id += 1
        return rid

    sect = sup_body.find(q(W, "sectPr"))
    intro = [
        "S6. Figures Moved from the Main Text",
        "These figures were removed from the main manuscript and are cited there as Supplementary Figures S1–S8. S1–S4 are the Moran size and power plots (main-text Figures 1–4). S5 and S6 are the simulation holdout charts (main-text Figures 7 and 8). S7 and S8 are the August 2026 scatter and the 63-month average RMSE chart (main-text Figures 10 and 12).",
    ]
    source = sup_body.find(q(W, "p"))
    for text in intro:
        sect.addprevious(make_p(blocks_from(text), source))

    for _old, new_caption, parts in blocks:
        copied = []
        for part in parts:
            node = etree.fromstring(etree.tostring(part))
            for blip in node.iter(q(A, "blip")):
                old_rid = blip.get(q(OFFICE_REL, "embed"))
                target = rel_by_id[old_rid]
                if target not in copied_targets:
                    rid = new_rid()
                    copied_targets[target] = rid
                    media = main_files["word/" + target.replace("\\", "/")]
                    sup_files["word/" + target.replace("\\", "/")] = media
                    rel = etree.SubElement(sup_rels, q(REL, "Relationship"))
                    rel.set("Id", rid)
                    rel.set("Type", IMAGE_REL)
                    rel.set("Target", target)
                blip.set(q(OFFICE_REL, "embed"), copied_targets[target])
            if node.tag == q(W, "p") and ptext(node).strip().startswith(_old):
                replaced = False
                for t in node.iter(q(W, "t")):
                    if t.text and t.text.strip().startswith(_old):
                        t.text = new_caption + t.text.strip()[len(_old):]
                        replaced = True
                        break
                if not replaced:
                    raise SystemExit(f"could not retitle {_old}")
            copied.append(node)
        for node in copied:
            sect.addprevious(node)

    for _old, _new, parts in blocks:
        for part in parts:
            if part.getparent() is main_body:
                main_body.remove(part)

    sup_files["word/_rels/document.xml.rels"] = etree.tostring(
        sup_rels, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    # Drop image parts no longer referenced by the main document.
    still = set()
    for blip in main_root.iter(q(A, "blip")):
        rid = blip.get(q(OFFICE_REL, "embed"))
        if rid in rel_by_id:
            still.add(rel_by_id[rid])
    for rel in list(rels):
        target = rel.get("Target")
        if rel.get("Type") == IMAGE_REL and target not in still:
            rels.remove(rel)
            main_files.pop("word/" + target.replace("\\", "/"), None)
    main_files["word/_rels/document.xml.rels"] = etree.tostring(
        rels, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    types = etree.fromstring(sup_files["[Content_Types].xml"])
    have = {el.get("Extension") for el in types if el.tag == q(CT, "Default")}
    for ext, mime in (("jpeg", "image/jpeg"), ("png", "image/png"), ("jpg", "image/jpeg")):
        if ext not in have:
            node = etree.Element(q(CT, "Default"))
            node.set("Extension", ext)
            node.set("ContentType", mime)
            types.insert(0, node)
    sup_files["[Content_Types].xml"] = etree.tostring(
        types, xml_declaration=True, encoding="UTF-8", standalone=True
    )


def audit(root, label):
    bare = 0
    for om in root.iter(q(M, "oMath")):
        for child in om:
            if etree.QName(child).localname == "e":
                bare += 1
    nstar = 0
    for ssup in root.iter(q(M, "sSup")):
        base = "".join(t.text or "" for t in ssup.find(q(M, "e")).iter(q(M, "t"))) if ssup.find(q(M, "e")) is not None else ""
        sup = "".join(t.text or "" for t in ssup.find(q(M, "sup")).iter(q(M, "t"))) if ssup.find(q(M, "sup")) is not None else ""
        if base == "n" and "*" in sup:
            nstar += 1
    gics = []
    body = root.find(q(W, "body"))
    for child in body:
        if child.tag == q(W, "p") and "GICS" in ptext(child):
            gics.append(ptext(child)[:140])
    print(label, "bare m:e", bare, "n-superscript-star", nstar, "GICS paragraphs", len(gics))
    for line in gics:
        print("  GICS:", line)


def write_letter():
    from docx import Document

    doc = Document()
    doc.add_paragraph("Econometrics and Statistics (EcoSta)")
    doc.add_paragraph("Part A: Econometrics")
    doc.add_paragraph("")
    doc.add_paragraph("Dear Co-Editors,")
    paragraphs = [
        "We submit “Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation” for consideration in Econometrics and Statistics, Part A: Econometrics. The paper is an econometric method for cross-sectional regression when residual dependence runs through several networks at once.",
        "Four results organize the contribution. First, an additive spatial error, rather than an endogenous spatial lag, keeps the conditional mean linear in the regressors under exogeneity. Second, the restriction ‖λ‖₁ = Σ |λₛ| < 1 is a uniform sufficient condition for invertibility of the multiplex filter on row-substochastic weights. It is stronger than a bound on the signed sum, which can be small while opposing coefficients make the filter singular. Third, candidate layers are screened with residual Moran’s I, using exact design-conditional moments. A pre-data rule switches to conditional Gaussian Monte Carlo (B = 499) when degree heterogeneity or spectral concentration is high, and structurally duplicate transforms are dropped before any test so that the multiple-testing family does not depend on the outcome. Fourth, held-out outcomes are imputed from the disturbance precision, which factors a holdout block of order n* rather than the full training covariance.",
        "In the simulations, the oracle specification reduces holdout RMSE by 8.6% relative to OLS, and the feasible Moran-screened specification by 6.1%. On 63 monthly S&P 100 cross-sections, Multi-SEM has lower RMSE than OLS in 44 months (two-sided sign test p = 0.0022). A separate large-sample comparison shows that re-standardizing the induced training subgraph shifts the autoregressive estimates, while holdout RMSE changes by 0.31% at n = 200 and by 0.08% at n = 500. The induced subgraph is therefore an empirical surrogate when cross-block links are unavailable; the Schur marginal precision remains the exact likelihood when the full network is observed.",
        "The supplementary file contains the proofs, the Gibbs derivations, prior sensitivity, and the replication manifest. The versioned Python package (mnreg, together with run_simulation.py, run_sp100.py, run_csda_revision.py, and run_nonnetwork_benchmarks.py) reproduces the synthetic and empirical results from the cached cross-sections. We would be glad to provide any further material the editorial office needs.",
        "Sincerely,",
        "Simon K. C. Cheung",
        "Tommy K. Y. Cheung",
    ]
    for text in paragraphs:
        doc.add_paragraph(text)
    doc.save(LETTER_OUT)
    print("wrote", LETTER_OUT.name)


def main():
    main_files, main_root = load_zip(MAIN_IN)
    sup_files, sup_root = load_zip(SUP_IN)
    before = word_count_root(main_root, stop_at="References")
    edit_main(main_root.find(q(W, "body")))
    move_figures(main_files, main_root, sup_files, sup_root)
    retitle_pointers(main_root.find(q(W, "body")))
    nstar = convert_n_star(main_root)
    print("n-star conversions", nstar)
    after = word_count_root(main_root, stop_at="References")
    full = len(" ".join(t.text or "" for t in main_root.iter(q(W, "t"))).split())
    print(f"prose words before References, excluding tables: {before} -> {after}")
    print(f"all w:t words including references and tables: {full}")
    audit(main_root, "main")
    audit(sup_root, "supplement")
    # bare m:e already counted; also confirm Algorithm 1 and the deleted claims
    blob = " ".join(ptext(p) for p in main_root.find(q(W, "body")) if p.tag == q(W, "p"))
    for needle in [
        "only specification that improves",
        "recomputed over",
        "compensates for the modified graph scale",
        "Algorithm 1",
        "B = 499",
        "6.5 Metrics and Benchmarks",
        "Supplementary Figure S8",
    ]:
        print(f"contains {needle!r}: {needle in blob or needle.replace('B = 499','499') in blob}")
    if "mnreg/model.py" in blob:
        raise SystemExit("replication table text still in the main narrative")
    store_zip(MAIN_OUT, main_files, main_root)
    store_zip(SUP_OUT, sup_files, sup_root)
    write_letter()
    print("wrote", MAIN_OUT.name, MAIN_OUT.stat().st_size)
    print("wrote", SUP_OUT.name, SUP_OUT.stat().st_size)


if __name__ == "__main__":
    main()
