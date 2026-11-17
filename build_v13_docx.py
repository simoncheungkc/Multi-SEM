"""Build manuscript v13 and supplement v11. Does not overwrite v12 or v10."""

from __future__ import annotations

import copy

from lxml import etree

import build_v12_docx as b

MAIN_OUT = b.ROOT / "REGRESSION WITH MULTIPLE NETWORKS v13.docx"
SUP_OUT = b.ROOT / "Supplementary_Materials_MultiSEM_v11.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def math_text(p):
    bits = []
    for node in p.iter():
        if node.text and etree.QName(node).localname == "t":
            bits.append(node.text)
    return "".join(bits)


def remove_short(body, snippet, max_chars=120):
    hits = []
    for c in list(body):
        if c.tag != b.q(b.W, "p"):
            continue
        blob = math_text(c)
        if snippet in blob and len(blob) <= max_chars:
            hits.append(c)
    if len(hits) != 1:
        raise SystemExit(f"short remove {snippet!r} matched {len(hits)}")
    body.remove(hits[0])


def set_cell(tc, text):
    texts = list(tc.iter(b.q(b.W, "t")))
    if not texts:
        return
    texts[0].text = text
    for extra in texts[1:]:
        extra.text = None


def add_cell(tr, text):
    template = tr.find(b.q(b.W, "tc"))
    tc = copy.deepcopy(template)
    set_cell(tc, text)
    tr.append(tc)


def expand_table6(body):
    caption = b.require_one(body, "Table 6.  Residual Moran")
    table = caption.getnext()
    if table.tag != b.q(b.W, "tbl"):
        raise SystemExit("Table 6 caption is not followed by a table")
    grid = table.find(b.q(b.W, "tblGrid"))
    prototype = grid[0]
    for _ in range(3):
        grid.append(copy.deepcopy(prototype))
    rows = [
        ["Candidate network", "Moran's I", "Z", "kappa", "psi", "Reference", "Two-sided p"],
        ["Trailing return correlation (5-NN)", "0.018", "0.83", "2.49", "0.035", "Gaussian", "0.406"],
        ["Boosted correlation resolvent", "0.027", "1.45", "1.00", "0.053", "Gaussian", "0.146"],
        ["Yahoo Finance sector", "0.122", "2.70", "1.43", "0.045", "Gaussian", "0.0070"],
        ["Yahoo Finance industry", "0.263", "2.89", "3.07", "0.018", "Gaussian tail", "0.0038"],
    ]
    for tr, values in zip(table.findall(b.q(b.W, "tr")), rows):
        cells = tr.findall(b.q(b.W, "tc"))
        if len(cells) != 4:
            raise SystemExit(f"expected 4 cells, found {len(cells)}")
        for tc, value in zip(cells, values[:4]):
            set_cell(tc, value)
        for value in values[4:]:
            add_cell(tr, value)
    b.replace_with(
        caption,
        b.blocks_from(
            "Table 6. Residual Moran's $I$ for the August 2026 showcase. "
            "The archived screen used all $n = 100$ equities, before the 80/20 split "
            "($d = n - p - 1 = 95$). "
            "$\\kappa$ is the maximum degree divided by the mean degree on the off-diagonal adjacency. "
            "$\\psi$ is the leading singular-value share of the row-standardized matrix. "
            "The $p$-values are the analytical Gaussian tails stored by the screening routine."
        ),
    )


STABILITY = (
    "enforces spectral invertibility across the parameter space and eliminates "
    "the non-invertibility pitfalls of signed-sum restrictions"
)


def edit_main(body):
    # Preserve Table 1 before the Section 5 span is removed.
    caps = [c for c in body if c.tag == b.q(b.W, "p") and b.ptext(c).startswith("Table 1.")]
    if len(caps) != 1:
        raise SystemExit(f"Table 1 caption matched {len(caps)}")
    cap = caps[0]
    tbl = cap.getnext()
    saved = [copy.deepcopy(cap), copy.deepcopy(tbl)]

    b.replace_span(
        body,
        "Empirical entities often participate",
        "1. Introduction",
        [
            "Empirical entities frequently interact across multiple overlapping networks, allowing residual dependence to propagate through several relational channels simultaneously. This paper develops Multiplex Spatial Error Regression (Multi-SEM) as an integrated econometric pipeline combining ex-ante network screening, stability-constrained Bayesian estimation, and precision-form imputation.",
            "Multi-SEM addresses four methodological challenges: (i) an additive disturbance specification preserves linear conditional-mean interpretability ($\\mathbb{E}[Y \\mid X] = X\\beta$) under exogeneity; (ii) the restriction $\\Vert \\lambda \\Vert_1 = \\sum_{s=1}^{k} \\vert \\lambda_s \\vert < 1$ is a uniform sufficient condition for spectral stability of the filter on row-substochastic layers; (iii) candidate networks are screened with residual Moran's $I$, using exact design-conditional moments, with conditional Gaussian Monte Carlo calibration when a layer is hub-dominated, and then Benjamini–Hochberg control; and (iv) unobserved outcomes are imputed from the precision $Q = \\sigma^{-2} V_\\lambda' V_\\lambda$, at cost $\\mathcal{O}(n_*^3)$ rather than $\\mathcal{O}(n_t^3)$.",
            f"In synthetic simulations, Multi-SEM {STABILITY}. It achieves an 8.6% holdout RMSE reduction (6.1% under feasible screening). In a 63-month study of S&P 100 equity returns, Multi-SEM outperforms OLS in 44 of 63 months (two-sided sign test $p = 0.0022$; Newey–West $p = 0.072$).",
            "Keywords: Multiple Networks; Spatial Error Model; Moran's $I$; Gibbs Sampler; Spatial Kriging; S&P 100.",
        ],
    )
    b.replace_span(
        body,
        "1. Introduction",
        "2. Related Work",
        [
            "1. Introduction",
            "In interconnected economic and financial systems, cross-sectional observations are rarely independent. Latent shocks, omitted environmental variables, and common industry factors induce cross-sectional residual correlation. Under strictly exogenous regressors ($\\mathbb{E}[u \\mid X] = 0$), ordinary least squares remains consistent but inefficient, standard errors are distorted, and out-of-sample predictions ignore informative residual dependence.",
            "Spatial econometrics models such dependence through spatial lag models or spatial error models. Conventional formulations use a single weight matrix $W$. In the applications considered here, the same entities sit in several networks at once: sector, industry, and return co-movement. Using only one of those layers omits the others and treats overlapping layers as if they were distinct regressors.",
            "This paper develops Multiplex Spatial Error Regression (Multi-SEM). The disturbance is $u = \\sum_{s=1}^{k} \\lambda_s A_s u + \\epsilon$, so $\\mathbb{E}[Y \\mid X] = X\\beta$ under exogeneity. The bound $\\Vert \\lambda \\Vert_1 = \\sum_{s=1}^{k} \\vert \\lambda_s \\vert < 1$ keeps the filter $V_\\lambda = I_n - \\sum_{s=1}^{k} \\lambda_s A_s$ invertible on row-substochastic layers, including cases in which large coefficients of opposite sign cancel in the signed sum. Layers are screened with residual Moran's $I$. Structurally redundant transforms, including clique resolvents with $\\vert r_{\\mathrm{vec}} \\vert \\ge 0.99$, are removed before the tests. Holdout outcomes are imputed from $Q = \\sigma^{-2} V_\\lambda' V_\\lambda$.",
            f"On synthetic graphs, Multi-SEM {STABILITY} and reduces holdout RMSE by 8.6% relative to OLS (6.1% when the active layers are chosen by the screen). On 63 months of S&P 100 returns it beats OLS in 44 months (sign test $p = 0.0022$). Proofs, prior checks, and the replication code are in the Supplementary Materials.",
        ],
    )
    b.replace_span(
        body,
        "2. Related Work",
        "3. The Multiplex Spatial Error Model",
        [
            "2. Related Work and Methodological Background",
            "Spatial econometrics distinguishes dependence in outcomes (SAR; Anselin, 2009; LeSage & Pace, 2009) from dependence in disturbances (SEM; Kelejian & Prucha, 2010). A spatial lag $AY$ makes OLS inconsistent. A spatial error leaves $\\mathbb{E}[Y \\mid X] = X\\beta$ under exogeneity.",
            "A multiplicative filter $(I - \\lambda_1 A_1)(I - \\lambda_2 A_2)u = \\epsilon$ produces cross-products $A_1 A_2$ that are hard to interpret after row-normalization. The additive filter $u = \\sum_{s=1}^{k} \\lambda_s A_s u + \\epsilon$ avoids that product (Florax & Nijkamp, 2005; Bivand et al., 2014).",
            "Three gaps matter for the multiplex case. An auxiliary regression of $Y$ on $A_s Y$ is an endogenous lag test, not a test of the disturbance (Manski, 1993; Lee, 2002); residual Moran's $I$ matches the SEM. A bound on the signed sum $\\vert \\sum_s \\lambda_s \\vert < 1$ does not stop two large coefficients of opposite sign from making $V_\\lambda$ singular. Katz resolvents of affiliation cliques lie in $\\operatorname{span}\\{J, I\\}$ and, after the diagonal is removed and the rows are standardized, duplicate the base clique.",
        ],
    )
    b.replace_span(
        body,
        "3. The Multiplex Spatial Error Model",
        "4. Hypothesis Testing",
        [
            "3. The Multiplex Spatial Error Model (Multi-SEM)",
            "3.1 Model Formulation",
            "Let $Y \\in \\mathbb{R}^n$ be the outcomes on a common vertex set, and let $X = [1_n, X_c]$ contain an intercept and standardized exogenous covariates. Each adjacency matrix has a zero diagonal and is row-standardized: a positive row sum produces $A_{s,ij} = W_{s,ij} / \\sum_j W_{s,ij}$, and a zero row sum stays a zero row. Thus $\\Vert A_s \\Vert_\\infty \\le 1$. The model is",
            "$Y = X\\beta + u, \\qquad u = \\sum_{s=1}^{k} \\lambda_s A_s u + \\epsilon, \\qquad \\epsilon \\sim \\mathcal{N}(0, \\sigma^2 I_n)$",
            "With $V_\\lambda = I_n - \\sum_{s=1}^{k} \\lambda_s A_s$, the reduced form is $u = V_\\lambda^{-1}\\epsilon$, so $\\mathbb{E}[Y \\mid X] = X\\beta$ and $\\operatorname{Cov}(Y \\mid X) = \\sigma^2 (V_\\lambda' V_\\lambda)^{-1}$.",
            "3.2 Invertibility and the $\\ell_1$ Stability Bound",
            "Non-singularity of $V_\\lambda$ requires $\\rho(\\sum_s \\lambda_s A_s) < 1$. The induced infinity norm gives",
            "$\\rho\\left(\\sum_{s=1}^{k} \\lambda_s A_s\\right) \\le \\sum_{s=1}^{k} \\vert \\lambda_s \\vert \\, \\Vert A_s \\Vert_\\infty \\le \\sum_{s=1}^{k} \\vert \\lambda_s \\vert$",
            "so $\\Vert \\lambda \\Vert_1 = \\sum_{s=1}^{k} \\vert \\lambda_s \\vert < 1$ is a uniform sufficient condition on row-stochastic or row-substochastic layers. A bound on the signed sum is not. The pair $\\lambda_1 = 1.32$, $\\lambda_2 = -1.78$ has $\\vert \\lambda_1 + \\lambda_2 \\vert = 0.46 < 1$, while $\\Vert \\lambda \\Vert_1 = 3.10$ and the filter need not be invertible. Draws are restricted to the open $\\ell_1$ ball.",
            "3.3 Intercept Attenuation Near the Stability Boundary",
            "If every layer is fully row-stochastic, $V_\\lambda 1_n = (1 - \\sum_{s=1}^{k} \\lambda_s) 1_n$. As the signed sum approaches 1, the filtered intercept shrinks and the posterior variance of $\\beta_0$ rises. Mean-centered covariates and symmetric layers keep $1_n' V_\\lambda X_c = 0$, so the slopes do not absorb that particular loss. Isolate rows, as in the Yahoo Finance industry network, make the contraction non-uniform: the intercept variance inflates, and the column of ones does not drop rank globally.",
            "3.4 Affiliation Clique Resolvents and Structural Redundancy",
            "A Katz resolvent is $G_A = A(I - \\alpha A)^{-1}$. For a zero-diagonal clique $C_m = \\frac{1}{n_m-1}(J_{n_m} - I_{n_m})$, every positive integer power lies in $\\operatorname{span}\\{J_{n_m}, I_{n_m}\\}$. Zeroing the resolvent diagonal and row-standardizing returns $C_m$ exactly. Affiliation layers are therefore not resolvent-boosted. The return-correlation layer is.",
            "3.5 Likelihood Evaluation",
            "The concentrated profile log-likelihood is",
            "$\\ln L_c(\\lambda) = -\\frac{n}{2}[1 + \\ln(2\\pi)] - \\frac{n}{2}\\ln[\\hat\\sigma^2(\\lambda)] + \\ln\\vert\\det V_\\lambda\\vert$",
            "with $\\hat\\sigma^2(\\lambda) = n^{-1}(Y - X\\hat\\beta_\\lambda)' V_\\lambda' V_\\lambda (Y - X\\hat\\beta_\\lambda)$ and $\\hat\\beta_\\lambda = (X' V_\\lambda' V_\\lambda X)^{-1} X' V_\\lambda' V_\\lambda Y$. For $n \\le 500$ the log absolute determinant is a dense LU factorization, at cost $\\mathcal{O}(n^3)$. Large sparse approximations are in Supplement Section S5.2.",
        ],
    )
    b.replace_span(
        body,
        "4. Hypothesis Testing",
        "5. Bayesian Estimation",
        [
            "4. Hypothesis Testing and Multiplex Network Screening",
            "4.1 Residual Moran's $I$",
            "Under $H_0: \\lambda_1 = \\cdots = \\lambda_K = 0$ the model is OLS. Residuals are $e = MY$ with $M = I - X(X'X)^{-1}X'$. For layer $A_s$,",
            "$I_s = \\frac{n}{S_{0,s}} \\frac{e' A_s e}{e' e}, \\qquad S_{0,s} = \\sum_{i,j} A_{s,ij}$",
            "Proposition 1. Under a Gaussian null, let $d = n - p - 1$ and let $A_s^*$ be the symmetrized layer. Then $I_s$ is a scale factor times a quadratic form in a uniform direction on the unit sphere in $\\mathbb{R}^d$, independent of $e'e$. The exact conditional moments are",
            "$\\mathbb{E}[I_s \\mid X] = c_s \\frac{\\operatorname{tr}(M A_s)}{d}, \\quad \\operatorname{Var}(I_s \\mid X) = c_s^2 \\frac{2}{d(d+2)}\\left[\\operatorname{tr}((M A_s^* M)^2) - \\frac{(\\operatorname{tr}(M A_s^*))^2}{d}\\right]$",
            "with $c_s = n / S_{0,s}$. The standardized score $Z_s$ does not depend on $\\sigma^2$. Its finite-sample law depends on the spectrum of the projected layer (Supplement S1.2). Under regular degree bounds, $Z_s$ is asymptotically standard normal.",
            "4.2 Operational Rule and Algorithm 1",
            "Hub-dominated layers make the Gaussian tail liberal. Before any residual is formed, compute $\\kappa_s = \\max_i d_{s,i} / \\bar d_s$ on the off-diagonal adjacency and $\\psi_s = \\theta_{s,1} / \\sum_j \\theta_{s,j}$ from the singular values of the row-standardized matrix. If $\\kappa_s > 3$ or $\\psi_s > 0.20$, the $p$-value is a conditional Gaussian Monte Carlo probability with $B = 499$ draws. Otherwise it is the analytical tail $2(1 - \\Phi(|Z_s|))$.",
            "At each rolling month, the OLS residuals and the Moran statistics are computed on the $n_t = 80$ training rows only. Holdout outcomes $Y_*$ are not used in the screen or in the parameter update. The August showcase table in Section 8.2 is the exception: the archived call used all 100 equities, and that fact is stated with the table.",
            "Algorithm 1. Ex-ante structural pruning and residual Moran screening",
            "Step 1. For every pair in $\\{A_1,\\ldots,A_K\\}$, compute the off-diagonal Pearson correlation $r_{\\mathrm{vec}}(A_r, A_s)$. If $\\vert r_{\\mathrm{vec}} \\vert \\ge 0.99$, drop the redundant transform. Call the retained set $\\{A_1,\\ldots,A_{K'}\\}$. This step does not use $Y$.",
            "Step 2. On the training rows, compute $e = M Y_t$ with $M = I_{n_t} - X_t(X_t' X_t)^{-1} X_t'$.",
            "Step 3. For $s = 1,\\ldots,K'$, compute $Z_s$ from the moments in Proposition 1. If $\\kappa_s > 3$ or $\\psi_s > 0.20$, compute $p_s$ from $B = 499$ conditional Gaussian draws. Otherwise set $p_s = 2(1 - \\Phi(|Z_s|))$.",
            "Step 4. Apply Benjamini–Hochberg control at $q^* = 0.05$ to these $K'$ $p$-values and retain $S = \\{s : p_s \\le p_{(k^*)}\\}$. There is no later step that re-prunes inside $S$ by comparing Moran $p$-values.",
            "Estimation is conditional on $S$. The archived rolling-month program uses this training-only residual, and if $S$ is empty it keeps layers with $p < 0.10$ and, if that set is also empty, the smallest $p$-value. Across the 63 months the largest off-diagonal $\\vert r_{\\mathrm{vec}} \\vert$ is 0.972, so the 0.99 prune removes nothing.",
        ],
    )
    b.replace_span(
        body,
        "5. Bayesian Estimation",
        "6. Out-of-Sample Prediction",
        [
            "5. Bayesian Estimation via Constrained Gibbs Sampling",
            "5.1 Prior Specification",
            "On the selected layers the prior is independent across blocks: $\\beta \\sim \\mathcal{N}(0, 100 I)$, $\\sigma^2 \\sim \\mathrm{Inv\\text{-}Gamma}(0.001, 0.001)$, and",
            "$p(\\lambda) \\propto \\exp\\left(-\\frac{1}{2}\\lambda'\\lambda\\right) \\mathbb{I}\\left(\\sum_{s=1}^{k} \\vert \\lambda_s \\vert < 1\\right)$",
            "so the Gaussian factor on $\\lambda$ has precision $\\xi = 1$. These are the values in Table 1 and in Supplement Table S4. They are not varied in the reported fits.",
            "5.2 Conditional Posterior Distributions",
            "Given $\\lambda$, the filter $V_\\lambda$ reduces the model to a Gaussian linear regression in $\\tilde Y = V_\\lambda Y$ and $\\tilde X = V_\\lambda X$. The full conditionals are",
            "$\\beta \\mid Y, X, \\sigma^2, \\lambda \\sim \\mathcal{N}(\\mu_\\beta, \\Sigma_\\beta), \\quad \\Sigma_\\beta = (\\Sigma_0^{-1} + \\sigma^{-2} X' V_\\lambda' V_\\lambda X)^{-1}$",
            "$\\mu_\\beta = \\Sigma_\\beta(\\Sigma_0^{-1}\\mu_0 + \\sigma^{-2} X' V_\\lambda' V_\\lambda Y)$",
            "$\\sigma^2 \\mid Y, X, \\beta, \\lambda \\sim \\mathrm{Inv\\text{-}Gamma}\\left(\\alpha_0 + n/2,\\ \\gamma_0 + \\frac12 (Y - X\\beta)' V_\\lambda' V_\\lambda (Y - X\\beta)\\right)$",
            "The completions of the square are in Supplement Section S2.1.",
            "5.3 Adaptive Metropolis Step on the $\\ell_1$ Ball",
            "The $\\lambda$ update is a random-walk Metropolis step, $\\lambda^* \\sim \\mathcal{N}(\\lambda^{(t-1)}, \\tau_t^2 I_k)$. If $\\Vert \\lambda^* \\Vert_1 \\ge 1$, the proposal is rejected before any determinant is computed. Otherwise it is accepted with probability $\\min(1, \\exp(\\Delta \\ln \\pi))$, where",
            "$\\Delta \\ln \\pi = \\ln\\vert\\det V_{\\lambda^*}\\vert - \\ln\\vert\\det V_{\\lambda^{(t-1)}}\\vert - \\frac{Q_{\\lambda^*} - Q_{\\lambda^{(t-1)}}}{2\\sigma^2} - \\frac{\\xi}{2}(\\Vert \\lambda^* \\Vert_2^2 - \\Vert \\lambda^{(t-1)} \\Vert_2^2)$",
            "Every 50 burn-in iterations, $\\tau_t$ is multiplied by 0.85 or 1.15 when the running acceptance rate is more than 0.08 away from 0.30. After burn-in, $\\tau$ is held fixed (Roberts & Rosenthal, 2009).",
            "5.4 Algorithm 2",
            "Algorithm 2. Adaptive Metropolis-within-Gibbs sampler",
            "Step 1. Propose $\\lambda^*$. Reject immediately if $\\Vert \\lambda^* \\Vert_1 \\ge 1$.",
            "Step 2. Otherwise accept or reject with the ratio in Section 5.3.",
            "Step 3. Draw $\\beta$ from the Gaussian conditional in Section 5.2, using the current $\\lambda$ and $\\sigma^2$.",
            "Step 4. Draw $\\sigma^2$ from the inverse-gamma conditional, then refresh the $\\lambda$ posterior ordinate.",
            "Step 5. Apply the burn-in adaptation of $\\tau$ described above. The chain used for reporting is the post-burn-in draws.",
        ],
    )
    anchor = b.require_one(body, "6. Out-of-Sample Prediction")
    for node in saved:
        anchor.addprevious(node)

    b.replace_span(
        body,
        "6. Out-of-Sample Prediction",
        "7. Simulation Studies",
        [
            "6. Out-of-Sample Prediction and Contemporaneous Spatial Kriging",
            "6.1 Training Likelihood",
            "Order the $n_*$ holdout units before the $n_t$ training units, and partition $Q = \\sigma^{-2} V_\\lambda' V_\\lambda$ the same way. The marginal precision of the training disturbances is the Schur complement",
            "$P_T = Q_{tt} - Q_{t*} Q_{**}^{-1} Q_{*t}$",
            "and the coherent training model is $Y_t \\mid X, \\theta \\sim \\mathcal{N}(X_t \\beta, P_T^{-1})$. The replication archive instead fits $\\lambda$ on the induced training blocks after those blocks are row-standardized again, then imputes from the full-graph precision. Section 7.5 compares the two criteria. Evaluating $P_T$ inside every proposal adds an $n_* \\times n_*$ factorization.",
            "6.2 Precision-Form Kriging",
            "Gaussian conditioning gives $u_* \\mid u_t \\sim \\mathcal{N}(-Q_{**}^{-1} Q_{*t} u_t,\\ Q_{**}^{-1})$. The holdout predictor is",
            "$\\mathbb{E}[Y_* \\mid Y_t, X, \\theta] = X_*\\beta - Q_{**}^{-1} Q_{*t}(Y_t - X_t \\beta), \\qquad \\operatorname{Cov}(Y_* \\mid Y_t, X, \\theta) = Q_{**}^{-1}$",
            "Supplement Section S1.4 shows that this is the same predictor as $X_*\\beta + \\Omega_{*t}\\Omega_{tt}^{-1} u_t$. The linear solve in the holdout block is $\\mathcal{O}(n_*^3)$.",
            "6.3 Kriging versus a Time-Ahead Forecast",
            "The predictor conditions on training residuals from the same date. A forecast from date $t-1$ to date $t$ cannot use those residuals. Under this model the time-ahead conditional mean is $X_*\\beta$. Feeding same-date peer residuals into a trading forecast would be look-ahead.",
        ],
    )

    # Orphaned displays left by earlier trims.
    remove_short(body, "0.10,-0.20", 80)
    remove_short(body, "1.122-1.025", 40)
    remove_short(body, "screening → estimation", 80)
    for snippet in ("=0.82", "≈0.82", "0.181", "1-0.82"):
        hits = [
            c
            for c in list(body)
            if c.tag == b.q(b.W, "p") and snippet in math_text(c) and len(b.ptext(c).split()) <= 2
        ]
        if len(hits) > 1:
            raise SystemExit(f"orphan {snippet!r} matched {len(hits)}: {[math_text(h)[:60] for h in hits]}")
        for h in hits:
            body.remove(h)
    # The standalone filtered-intercept display, if it is still a math-only paragraph.
    hits = [
        c
        for c in list(body)
        if c.tag == b.q(b.W, "p")
        and "1-" in math_text(c)
        and "1n" in math_text(c).replace(" ", "")
        and len(b.ptext(c).split()) <= 2
        and len(math_text(c)) < 40
    ]
    for h in hits:
        body.remove(h)

    # Table 4a undercoverage. The rates are the Monte Carlo column of that table.
    cap4 = b.require_one(body, "Table 4a.")
    table4 = cap4.getnext()
    note = b.make_p(
        b.blocks_from(
            "Central 95% credible intervals for the two active coefficients cover the truth in 87.5% and 89.5% of the 200 replications at $n = 200$ (Table 4a). "
            "That undercoverage is the finite-sample shrinkage of those coefficients toward zero when several operators compete for the same residual variance inside the joint $\\ell_1$ ball. "
            "The inactive-layer intervals cover zero at 94% or above in the same experiment."
        ),
        cap4,
    )
    table4.addnext(note)

    # Section 7.5 wording.
    node = b.require_one(body, "marginal-likelihood biases diminish")
    for t in node.iter(b.q(b.W, "t")):
        if t.text and "predictive equivalence becomes even tighter" in t.text:
            t.text = t.text.replace(
                "predictive equivalence becomes even tighter",
                "the two holdout RMSE values stay close",
            )
        if t.text and "In the simulated configurations examined here" in t.text:
            t.text = (
                " is a consequence of re-standardizing the training subgraph, which changes the local scale of the network operator. "
                "In the simulated configurations examined here, this parameter adjustment offset the modified graph scale, "
                "yielding negligible differences in holdout RMSE (0.31% at $n = 200$ and 0.08% at $n = 500$). "
                "Thus, the induced-subgraph estimator acts as a computationally fast surrogate when cross-block ties are missing, "
                "while the Schur marginal precision "
            )
        if t.text and "remains the exact likelihood criterion" in t.text:
            t.text = " provides the strictly coherent objective when the full topology is observed ex ante."
    formula = b.make_p(
        b.blocks_from("The Schur marginal precision in that comparison is $P_T = Q_{tt} - Q_{t*} Q_{**}^{-1} Q_{*t}$."),
        node,
    )
    node.addnext(formula)

    # Section 8.2 narrative and Table 6.
    b.replace_containing(
        body,
        "Table 6 reports August 2026",
        "Table 6 is the archived August 2026 screen. It was computed from OLS residuals on all $n = 100$ equities, and those residuals use the 20 holdout outcomes. "
        "The industry and sector layers are the two that pass Benjamini–Hochberg control on that full-sample screen, which is why Table 7 estimates those two layers. "
        "The same split, restricted to the $n_t = 80$ training rows ($d = 75$), does not reproduce that selection: industry $I = 0.208$ ($z = 2.20$, $p = 0.028$) and sector $I = 0.091$ ($z = 1.87$, $p = 0.062$), "
        "and with four tests the Benjamini–Hochberg threshold for the smallest $p$-value is 0.0125, so neither layer is retained. "
        "On the training block every $\\kappa_s$ is at most 2.61 and every $\\psi_s$ is at most 0.056, so those training $p$-values are the Gaussian branch of Algorithm 1.",
    )
    expand_table6(body)
    # Place the diagnostic sentence after the table.
    cap6 = None
    for c in body:
        if c.tag == b.q(b.W, "p") and math_text(c).startswith("Table 6. Residual Moran"):
            cap6 = c
            break
    if cap6 is None:
        raise SystemExit("new Table 6 caption not found")
    tbl6 = cap6.getnext()
    diag = b.make_p(
        b.blocks_from(
            "The four archived $p$-values are not on the Monte Carlo grid $\\{0.002, 0.004, \\ldots\\}$. "
            "The screening routine returns the Gaussian tail. For the correlation, resolvent, and sector layers, $\\kappa \\le 3$ and $\\psi \\le 0.20$, so that tail is the operational $p$-value. "
            "The industry layer has $\\kappa = 3.07$, so Algorithm 1 would have replaced $p = 0.0038$ with a $B = 499$ conditional Gaussian Monte Carlo probability before FDR. "
            "That Monte Carlo probability was not stored. The printed $p = 0.0038$ is the Gaussian tail."
        ),
        cap6,
    )
    tbl6.addnext(diag)

    b.replace_containing(
        body,
        "Each month uses the same 20 held-out tickers",
        "At each rolling month, OLS residuals and Moran statistics are evaluated on the $n_t = 80$ training observations only. Holdout outcomes $Y_*$ are unobserved during screening and during parameter estimation. "
        "Covariates and the correlation layers use only information dated before that month. "
        "If no layer passes Benjamini–Hochberg control at 0.05, the program keeps layers with $p < 0.10$ and, if that set is empty, the smallest $p$-value. "
        "The selected layers are re-estimated on the induced training subgraph, and holdout returns are imputed from the full-graph precision. "
        "A no-kriging fit uses the same coefficients but predicts from the regression mean alone.",
    )

    node = b.require_one(body, "lowest overall error")
    for t in node.iter(b.q(b.W, "t")):
        if t.text and "While Sector SEM" in t.text:
            head, _sep, _tail = t.text.partition("While Sector SEM")
            t.text = head + (
                "Multi-SEM achieves the lowest overall error (0.0576). "
                "While Sector SEM (0.0588) and Industry SEM (0.0609) also improve upon OLS (0.0637), "
                "the non-network benchmarks—industry fixed effects (0.0653) and one-factor shrinkage (0.0697)—"
                "fail to outperform OLS."
            )

    b.replace_span(
        body,
        "9. Conclusion",
        "References",
        [
            "9. Conclusion",
            "Multi-SEM is a pipeline for cross-sectional regression when the disturbance runs through more than one network. Putting that dependence in the disturbance keeps $\\mathbb{E}[Y \\mid X] = X\\beta$ under exogeneity.",
            f"Four pieces are used in the reported fits. The bound $\\Vert \\lambda \\Vert_1 < 1$ {STABILITY}. Residual Moran's $I$ uses the exact design-conditional moments, and a layer with $\\kappa > 3$ or $\\psi > 0.20$ is referred to a $B = 499$ conditional Gaussian Monte Carlo draw before Benjamini–Hochberg control. Clique resolvents that match a base layer at off-diagonal correlation 0.99 are dropped before those tests. Holdout imputation solves a system in the $n_*$ holdout block of the precision rather than factoring the $n_t$ training covariance.",
            "In the simulations the oracle specification reduces holdout RMSE by 8.6% relative to OLS, and the screened specification by 6.1%. Active-coefficient intervals at $n = 200$ undercover, at 87.5% and 89.5%. The induced-subgraph fit is a fast surrogate when cross-block ties are missing; the Schur precision $P_T$ is the coherent training likelihood when they are observed. On 63 months of S&P 100 returns, Multi-SEM beats OLS in 44 months (sign test $p = 0.0022$). The gain is in the same-date kriging correction. The Newey–West test of the monthly loss differential has $p = 0.072$. Extensions to dynamic panels and to non-Gaussian outcomes are left open.",
        ],
    )

    # Catch any leftover slogan.
    for t in body.iter(b.q(b.W, "t")):
        if t.text and "eliminates boundary" in t.text:
            t.text = t.text.replace("eliminates boundary estimation instabilities", STABILITY)
            t.text = t.text.replace("eliminates boundary instability", STABILITY)


def edit_supplement(body):
    title = b.require_one(body, "Regression with Multiple Networks")
    b.replace_with(
        title,
        b.blocks_from(
            "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation"
        ),
    )
    moved = b.require_one(body, "The coherent training model is the marginal")
    anchor = b.require_one(body, "S1.5 Near-Collinearity")
    heading = b.make_p(
        b.blocks_from(
            "S1.4.1 Implication for Training Likelihood: Coherent Marginal vs. Induced Subgraph"
        ),
        anchor,
    )
    anchor.addprevious(heading)
    anchor.addprevious(moved)
    b.replace_containing(
        body,
        "The implementation then row-standardizes the resolvent",
        "Two matrices have to be distinguished. The unpurged resolvent keeps the diagonal self-loops. It lies in $\\operatorname{span}\\{J, I\\}$, and its off-diagonal correlation with the base clique $C_m$ is about 0.999. "
        "The purged resolvent zeroes that diagonal before row-standardization. For an affiliation clique the purged matrix equals $C_m$; on unequal blocks, including a singleton, the largest absolute entrywise difference is on the order of $10^{-16}$. "
        "The model uses the purged matrix, so an affiliation resolvent is not entered as a second layer. The unpurged matrix would also be removed by the off-diagonal correlation screen.",
    )


def main():
    main_files, main_root = b.load_zip(b.MAIN_IN)
    # v13 is built from v12, which is b.MAIN_OUT of the previous script.
    main_files, main_root = b.load_zip(b.MAIN_OUT)
    sup_files, sup_root = b.load_zip(b.ROOT / "Supplementary_Materials_MultiSEM_v10.docx")
    before = b.word_count_root(main_root, stop_at="References")
    edit_main(main_root.find(b.q(b.W, "body")))
    edit_supplement(sup_root.find(b.q(b.W, "body")))
    after = b.word_count_root(main_root, stop_at="References")
    full = len(" ".join(t.text or "" for t in main_root.iter(b.q(b.W, "t"))).split())
    print(f"prose excluding tables, before references: {before} -> {after}")
    print(f"all w:t words: {full}")
    bare = 0
    for om in main_root.iter(b.q(M, "oMath")):
        for ch in om:
            if etree.QName(ch).localname == "e":
                bare += 1
    print("bare m:e", bare)
    blob = " ".join(b.ptext(p) for p in main_root.find(b.q(b.W, "body")) if p.tag == b.q(b.W, "p"))
    for needle in [
        "only specification that improves",
        "eliminates boundary",
        "predictive equivalence",
        "Algorithm 1",
        "Algorithm 2",
        "n = 100",
        "87.5%",
        "kappa = 3.07",
    ]:
        print(f"  {needle!r}: {needle in blob}")
    if bare:
        raise SystemExit("bare m:e under oMath")
    b.store_zip(MAIN_OUT, main_files, main_root)
    b.store_zip(SUP_OUT, sup_files, sup_root)
    print("wrote", MAIN_OUT.name, MAIN_OUT.stat().st_size)
    print("wrote", SUP_OUT.name, SUP_OUT.stat().st_size)


if __name__ == "__main__":
    main()
