"""Build the Multi-SEM re-analysis Word report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "results" / "figures"
EQ = ROOT / "results" / "_eq"
OUT = ROOT / "MultiSEM_Reanalysis_Report.docx"
EQ.mkdir(parents=True, exist_ok=True)


def shade_header(row, color="1F4E79"):
    for cell in row.cells:
        tc = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:fill"), color)
        shd.set(qn("w:val"), "clear")
        tc.append(shd)
        for p in cell.paragraphs:
            for run in p.runs:
                run.font.color.rgb = RGBColor(255, 255, 255)
                run.bold = True


def set_cell_font(cell, size=10, bold=False, color=None, align="left"):
    for p in cell.paragraphs:
        if align == "center":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        elif align == "right":
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.name = "Times New Roman"
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
            run.bold = bold
            if color:
                run.font.color.rgb = color


def add_table(doc, headers, rows, col_widths=None, highlight_last=False):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        table.rows[0].cells[j].text = h
        set_cell_font(table.rows[0].cells[j], size=10, bold=True, align="center")
    shade_header(table.rows[0])
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            table.rows[i + 1].cells[j].text = str(val)
            align = "left" if j == 0 else "center"
            bold = highlight_last and i == len(rows) - 1
            set_cell_font(table.rows[i + 1].cells[j], size=10, bold=bold, align=align)
    if col_widths:
        for row in table.rows:
            for j, w in enumerate(col_widths):
                row.cells[j].width = Inches(w)
    return table


def caption(doc, text, kind="Table"):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    r.italic = True
    r.font.size = Pt(11)
    r.font.name = "Times New Roman"
    return p


def body(doc, text, first_line_indent=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(8)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if first_line_indent:
        p.paragraph_format.first_line_indent = Cm(1.0)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    r.font.name = "Times New Roman"
    r.font.size = Pt(12)
    return p


def add_heading_custom(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(31, 78, 121)
        run.font.name = "Times New Roman"
    return h


def render_eq(latex, name, fontsize=16, figsize=(6.4, 0.85)):
    path = EQ / f"{name}.png"
    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.5, f"${latex}$", ha="center", va="center", fontsize=fontsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white", pad_inches=0.08)
    plt.close(fig)
    return path


def add_eq(doc, latex, name, number, figsize=(6.4, 0.9), fontsize=16):
    path = render_eq(latex, name, fontsize=fontsize, figsize=figsize)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(2)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(5.8))
    n = doc.add_paragraph()
    n.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    n.paragraph_format.space_after = Pt(8)
    rr = n.add_run(f"({number})")
    rr.font.name = "Times New Roman"
    rr.font.size = Pt(12)


def add_figure(doc, path, caption_text, width=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(8)
    run = p.add_run()
    run.add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(10)
    r = cap.add_run(caption_text)
    r.italic = True
    r.font.size = Pt(11)
    r.font.name = "Times New Roman"


def set_run_font(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Cm(2.54)
    section.bottom_margin = Cm(2.54)
    section.left_margin = Cm(2.54)
    section.right_margin = Cm(2.54)

    styles = doc.styles
    styles["Normal"].font.name = "Times New Roman"
    styles["Normal"].font.size = Pt(12)
    styles["Normal"]._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")

    # Title
    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    t.paragraph_format.space_after = Pt(6)
    r = t.add_run("REGRESSION WITH MULTIPLE NETWORKS")
    set_run_font(r, 16, bold=True)
    r.font.color.rgb = RGBColor(31, 78, 121)

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    st.paragraph_format.space_after = Pt(4)
    r = st.add_run("Re-analysis under a Unified Multiplex Spatial Error Model")
    set_run_font(r, 13, italic=True)

    au = doc.add_paragraph()
    au.alignment = WD_ALIGN_PARAGRAPH.CENTER
    au.paragraph_format.space_after = Pt(18)
    r = au.add_run(
        "Simon K. C. Cheung  ·  Tommy K. Y. Cheung\n"
        "The Chinese University of Hong Kong and Swinburne University of Technology"
    )
    set_run_font(r, 11)

    add_heading_custom(doc, "Abstract", 1)
    body(
        doc,
        "This report redoes the data analysis of the multi-network regression study under a "
        "single multiplex spatial error model (Multi-SEM). The original draft mixed a spatial "
        "autoregressive (lag) testing section with a spatial error Gibbs sampler, plotted test "
        "statistics as p-values, and reported network coefficients outside the invertibility "
        "region. The analysis below uses residual Moran’s I with two-sided p-values in [0, 1], "
        "a stationarity-constrained Gibbs sampler, and kriging-style out-of-sample predictions "
        "with OLS and single-network baselines. Out-of-sample results evaluate contemporaneous "
        "cross-sectional kriging (predicting held-out nodes in period t from peer residuals in "
        "the same period). They are not time-ahead forecasts (t−1 → t) of the kind usual in "
        "financial econometrics. In simulations, Moran’s I recovers a Poisson network and a "
        "heavy-tailed power-law network among five candidates, and Multi-SEM reduces out-of-sample "
        "RMSE by 8.6% relative to OLS. On S&P 100 monthly returns, industry and sector affiliation "
        "networks are selected in the showcase month; posterior network coefficients remain inside "
        "the stable region; and running 63-month holdout RMSE is 0.0657 versus 0.0685 for OLS.",
        first_line_indent=False,
    )

    k = doc.add_paragraph()
    k.paragraph_format.space_after = Pt(12)
    r = k.add_run("Keywords: ")
    set_run_font(r, 12, italic=True)
    r = k.add_run(
        "Multiple networks; Moran’s I; spatial error model; Gibbs sampler; "
        "out-of-sample prediction; S&P 100."
    )
    set_run_font(r, 12)

    # ------------------------------------------------------------------
    add_heading_custom(doc, "1.  Unified model", 1)
    body(
        doc,
        "For n entities, k row-standardized networks As, and covariates X (an intercept plus "
        "standardized explanatory columns), the multiplex network autoregressive error model is",
    )
    add_eq(
        doc,
        r"Y = X\beta + u,\qquad u=\sum_{s=1}^{k}\lambda_s A_s u+\varepsilon,"
        r"\quad \varepsilon\sim N(0,\sigma^2 I_n)",
        "eq1",
        1,
        figsize=(6.6, 0.7),
    )
    body(
        doc,
        "Equivalently V(Y − Xβ) = ε. Two matrices appear throughout and must not be confused. "
        "The spatial filter is V = I − Σs λs As. It depends on λ and maps disturbances to "
        "innovations. The OLS residual maker, used only in Moran screening, is "
        "M = I − X(X′X)⁻¹X′. It does not depend on λ. For nonnegative row-stochastic As, "
        "ρ(As) ≤ 1. A sufficient invertibility condition, via the triangle inequality, is",
    )
    add_eq(
        doc,
        r"\sum_{s=1}^{k}|\lambda_s| < 1",
        "eq2",
        2,
        figsize=(3.8, 0.5),
        fontsize=18,
    )
    body(
        doc,
        "This implies ρ(Σs λs As) ≤ Σs |λs| ρ(As) ≤ Σs |λs| < 1. The weaker signed bound "
        "|Σs λs| < 1 is not enough: large opposing λ can keep the signed sum small while the "
        "spectral radius is at least 1. The Gibbs sampler therefore rejects any proposal with "
        "Σs |λs| ≥ 1. Under positive spatial dependence all λs > 0, so the two bounds coincide. "
        "The error covariance is Ω = σ² (V′V)⁻¹ = σ² KK′ with K = V⁻¹.",
    )
    add_heading_custom(doc, "Remark.  Identification of the intercept", 2)
    body(
        doc,
        "When every As is row-stochastic, As 1 = 1 for all s, so the constant vector 1 is a "
        "common eigenvector with eigenvalue 1. (If some rows are zero because of isolates, "
        "(V1)i = 1 at those nodes and the identity below holds on the non-isolated component.) "
        "Then",
    )
    add_eq(
        doc,
        r"V\mathbf{1}=\left(1-\sum_{s=1}^{k}\lambda_s\right)\mathbf{1}",
        "eq2b",
        "2′",
        figsize=(5.2, 0.55),
        fontsize=16,
    )
    body(
        doc,
        "The filtered intercept is therefore β0 (1 − Σs λs). Equation (2′) is an identity in the "
        "signed sum; invertibility (Equation 2) uses the ℓ1 bound Σs |λs| < 1. The ridge appears "
        "as Σλs → 1 with positive dependence. Slope coefficients on demeaned or standardized "
        "covariates are not subject to the same degeneracy. Section 2.3 illustrates the ridge "
        "(intercept posterior SD 1.19 when five networks are fitted jointly). The S&P 100 fit in "
        "Section 3.3 does not, because Σλ̂ = 0.39 stays well away from the boundary.",
    )

    add_heading_custom(doc, "1.1  Screening with residual Moran’s I", 2)
    body(
        doc,
        "Fit OLS Y = Xβ + e. For each candidate network As compute the residual Moran statistic "
        "and a two-sided normal p-value",
    )
    add_eq(
        doc,
        r"I_s=\frac{e^{\top}A_s e}{e^{\top}e},\quad "
        r"z_s=\frac{I_s-E(I_s)}{\sqrt{\mathrm{Var}(I_s)}},\quad "
        r"p_s=2\left(1-\Phi(|z_s|)\right)",
        "eq3",
        3,
        figsize=(6.6, 0.75),
    )
    body(
        doc,
        "The mean and variance of Is are computed from the residual maker M (Cliff–Ord formulae). "
        "They do not involve the spatial filter V. Networks with Benjamini–Hochberg "
        "FDR q = 0.05 enter the Multi-SEM. Near-duplicate adjacency matrices (vectorized "
        "correlation ≥ 0.99) are dropped so that boosted resolvents of clique-like affiliation "
        "graphs are not entered twice.",
    )

    add_heading_custom(doc, "1.2  Gibbs sampler", 2)
    body(
        doc,
        "Priors used in every simulation and empirical fit are listed in Table 1. Full "
        "conditionals for β and σ² are Normal and Inverse-Gamma. The vector λ is updated by "
        "adaptive random-walk Metropolis–Hastings targeting a 30% acceptance rate. Each proposal "
        "evaluates log|V| with a dense LU factorization (numpy.linalg.slogdet). For n ≤ 500 the "
        "exact determinant is inexpensive. For future multiplexes with n > 2,000, Chebyshev "
        "polynomial approximations or Hutchinson-type stochastic trace estimators (Barry–Pace "
        "splines) can be substituted in mnreg/model.py for log|V| without changing the Gibbs "
        "architecture.",
    )
    caption(doc, "Table 1.  Prior distributions and hyperparameters.")
    add_table(
        doc,
        ["Parameter", "Prior", "Hyperparameters"],
        [
            ["β", "N(μ0, Σ0)", "μ0 = 0,  Σ0 = 100 I"],
            ["σ²", "Inverse-Gamma (shape α0, rate γ0)", "α0 = γ0 = 0.001"],
            ["λ", "N(0, I/ξ), truncated to Σs |λs| < 1", "ξ = 1"],
        ],
    )
    body(
        doc,
        "Simulation recovery uses four chains of 6,000 iterations with 3,000 discarded as "
        "burn-in. The S&P 100 showcase uses four chains of 8,000 iterations with 4,000 burn-in. "
        "Convergence is reported with split-R̂ and effective sample size (ESS).",
        first_line_indent=True,
    )

    add_heading_custom(doc, "1.3  Out-of-sample prediction", 2)
    body(
        doc,
        "Partition nodes into a training set t and a test set *. With Ω = (V′V)⁻¹ formed from "
        "the full n × n networks,",
    )
    add_eq(
        doc,
        r"\hat Y_*=X_*\hat\beta+\Omega_{*t}\Omega_{tt}^{-1}(Y_t-X_t\hat\beta)",
        "eq4",
        4,
        figsize=(6.4, 0.65),
    )
    body(
        doc,
        "Point predictions use posterior means in the showcase fits and concentrated-likelihood "
        "estimates in rolling windows. Equation (4) is spatial kriging of contemporaneous "
        "residuals: it imputes Y* in the same cross-section using observed peer residuals "
        "Yt − Xt β̂. It is not a time-ahead forecast of Yt+1 from information dated t. The "
        "X*β̂ term alone (no Ω*t correction) is reported as a no-kriging baseline so that the "
        "two exercises stay distinct.",
    )

    # ------------------------------------------------------------------
    add_heading_custom(doc, "2.  Simulation study", 1)
    add_heading_custom(doc, "2.1  Design", 2)
    body(
        doc,
        "Five undirected networks are generated on n ∈ {100, 200, 500} nodes and then "
        "row-standardized (Table 2). Only two networks enter the data-generating process: a "
        "Poisson (Erdős–Rényi) graph with mean degree 8 (λ = 0.40) and a power-law configuration "
        "model with exponent γ = 2.2 (λ = 0.35). The remaining three networks are inactive. "
        "The ℓ1 bound on true λ is Σs |λs| = 0.75 < 1.",
    )
    caption(doc, "Table 2.  Simulated networks and true Multi-SEM coefficients.")
    add_table(
        doc,
        ["Network", "Generator", "Role", "λ in DGP"],
        [
            ["Poisson d = 8", "Erdős–Rényi, mean degree 8", "active", "0.40"],
            ["Power-law γ = 2.2", "configuration model, P(k) ∝ k⁻²·²", "active", "0.35"],
            ["Power-law γ = 2.5", "same, γ = 2.5", "inactive", "0"],
            ["Power-law γ = 3.0", "same, γ = 3.0", "inactive", "0"],
            ["Poisson d = 4", "Erdős–Rényi, mean degree 4", "inactive (decoy)", "0"],
        ],
    )
    body(
        doc,
        "Covariates are an intercept plus five independent N(0, 1) columns, with "
        "β = (0.10, −0.20, 0.40, −0.10, 0.25, 0.15) and σ² = 1. For each n, networks and X are "
        "held fixed and Y is redrawn in 500 replications. Size is the rejection rate when "
        "Y = Xβ + ε (all λ = 0). Power is the rejection rate under the two-network DGP. The "
        "nominal level is α = 0.05. Realized mean degrees at n = 200 are 8.0 (Poisson-8, max 14), "
        "4.57 (PL 2.2, max 39), 3.41 (PL 2.5), 3.16 (PL 3.0), and 4.0 (Poisson-4).",
    )

    add_heading_custom(doc, "2.2  Moran’s I: size and power", 2)
    body(
        doc,
        "Table 3 reports unadjusted rejection rates at α = 0.05. Size is near 5% for Poisson "
        "graphs. The heaviest-tailed power-law (γ = 2.2) is slightly liberal (11–12% at n = 100 "
        "and n = 500). Benjamini–Hochberg FDR brings those rates back to about 2–5%. Two-sided "
        "p-values lie in [0, 1] (Figure 2); they are not z-scores.",
    )
    caption(doc, "Table 3.  Empirical size and power of residual Moran’s I (500 replications, α = 0.05).")
    add_table(
        doc,
        ["Network", "Role", "n=100 size", "n=100 power", "n=200 size", "n=200 power", "n=500 size", "n=500 power"],
        [
            ["Poisson d = 8", "active", "0.058", "0.522", "0.050", "0.864", "0.060", "0.998"],
            ["Power-law γ = 2.2", "active", "0.116", "0.734", "0.074", "0.970", "0.120", "1.000"],
            ["Power-law γ = 2.5", "inactive", "0.068", "0.052", "0.078", "0.072", "0.068", "0.064"],
            ["Power-law γ = 3.0", "inactive", "0.060", "0.080", "0.082", "0.080", "0.056", "0.070"],
            ["Poisson d = 4", "inactive", "0.082", "0.108", "0.086", "0.050", "0.066", "0.090"],
        ],
    )
    body(
        doc,
        "At n = 200 and n = 500 the test recovers both active networks and does not treat the "
        "inactive Poisson graph or the other power-law graphs as signal. Power is higher for the "
        "hub-dominated γ = 2.2 graph than for the denser Poisson graph at the same n. On the "
        "Gibbs recovery dataset (n = 200), Moran’s I with FDR selected exactly the two active "
        "networks.",
    )
    add_figure(
        doc,
        FIG / "moran_size_power_n200.png",
        "Figure 1.  Type I error and power of residual Moran’s I at n = 200 (500 replications). "
        "The dashed line is the nominal 5% level. Active networks are Poisson d = 8 and power-law γ = 2.2.",
        width=6.4,
    )
    add_figure(
        doc,
        FIG / "moran_pvalues_n200.png",
        "Figure 2.  Two-sided Moran I p-value histograms at n = 200. Values lie in [0, 1]. "
        "The vertical line is α = 0.05. Top row: null DGP. Bottom row: two-network DGP.",
        width=6.4,
    )
    add_figure(
        doc,
        FIG / "moran_size_power_n100.png",
        "Figure 3.  Size and power at n = 100.",
        width=6.2,
    )
    add_figure(
        doc,
        FIG / "moran_size_power_n500.png",
        "Figure 4.  Size and power at n = 500. Power for the two active networks is essentially one.",
        width=6.2,
    )

    add_heading_custom(doc, "2.3  Gibbs posterior recovery", 2)
    body(
        doc,
        "Four chains, 6,000 iterations and 3,000 burn-in, produced Metropolis acceptances of "
        "0.25–0.31. All split-R̂ values are at most 1.03. ESS for λ is 300–550 (somewhat thin); "
        "regression slopes have ESS above 8,000. Table 4 reports the posterior.",
    )
    caption(doc, "Table 4.  Posterior recovery at n = 200 (four chains).")
    add_table(
        doc,
        ["Parameter", "Truth", "Mean", "95% CI", "R̂"],
        [
            ["λ, Poisson d = 8", "0.40", "0.332", "(0.093, 0.559)", "1.011"],
            ["λ, PL γ = 2.2", "0.35", "0.337", "(0.162, 0.504)", "1.022"],
            ["λ, PL γ = 2.5", "0", "−0.031", "(−0.199, 0.131)", "1.010"],
            ["λ, PL γ = 3.0", "0", "0.082", "(−0.082, 0.244)", "1.003"],
            ["λ, Poisson d = 4", "0", "0.101", "(−0.068, 0.275)", "1.001"],
            ["β1", "−0.20", "−0.345", "(−0.496, −0.191)", "1.001"],
            ["β2", "0.40", "0.483", "(0.338, 0.625)", "1.000"],
            ["σ²", "1.00", "1.160", "(0.943, 1.431)", "1.000"],
        ],
    )
    body(
        doc,
        "Every interval for an active λ covers the truth; every inactive λ interval covers 0. "
        "The signed sum of posterior means is 0.82 and Σs |λ̂s| ≈ 0.88, both strictly less than 1. "
        "Consistent with the intercept remark above, "
        "β0 is weakly identified when all five networks are fitted together (posterior SD 1.19): "
        "Σλ̂ is close enough to 1 that V1 is small. Slope posteriors remain well mixed and cover "
        "the true βj.",
    )
    add_figure(
        doc,
        FIG / "gibbs_traces_recovery.png",
        "Figure 5.  Gibbs traces after burn-in, n = 200. Dashed lines are true values. "
        "Four chains shown in blue, orange, green, and red.",
        width=6.0,
    )
    add_figure(
        doc,
        FIG / "gibbs_lambda_posteriors.png",
        "Figure 6.  Marginal posterior densities of λ. Dashed lines are true values; "
        "solid lines are posterior means.",
        width=6.3,
    )

    add_heading_custom(doc, "2.4  Out-of-sample predictions", 2)
    body(
        doc,
        "Thirty independent samples at n = 200 are split 80/20 by nodes. Training uses the "
        "induced subgraph; prediction uses kriging on the full networks. Table 5 reports mean "
        "test RMSE and MAE. Multi-SEM on the true active pair cuts RMSE by 8.6% versus OLS and "
        "beats every single-network SEM. It had lower RMSE than OLS in 29 of 30 replications. "
        "Inactive single-network SEMs are slightly worse than OLS: they add a near-zero λ and "
        "extra covariance estimation noise.",
    )
    caption(doc, "Table 5.  Mean out-of-sample RMSE and MAE, n = 200, 30 replications, 20% node holdout.")
    add_table(
        doc,
        ["Model", "RMSE", "MAE"],
        [
            ["OLS (no network)", "1.122", "0.905"],
            ["Single SEM, Poisson d = 8", "1.080", "0.875"],
            ["Single SEM, PL γ = 2.2", "1.068", "0.859"],
            ["Single SEM, inactive networks", "1.129–1.131", "0.907–0.912"],
            ["Multi-SEM, Moran-selected", "1.054", "0.845"],
            ["Multi-SEM, true active pair", "1.025", "0.826"],
        ],
        highlight_last=True,
    )
    add_figure(
        doc,
        FIG / "simulation_pred_vs_actual.png",
        "Figure 7.  Predicted versus actual Y on a 20% node holdout (first replication). "
        "The line is predicted = actual.",
        width=6.4,
    )
    add_figure(
        doc,
        FIG / "simulation_oos_rmse.png",
        "Figure 8.  Mean test RMSE by model across 30 replications. Axis labels match Table 5.",
        width=5.6,
    )

    # ------------------------------------------------------------------
    add_heading_custom(doc, "3.  S&P 100 stocks", 1)
    add_heading_custom(doc, "3.1  Data and networks", 2)
    body(
        doc,
        "Adjusted close prices are taken from Yahoo Finance, 2 January 2018 through 15 September "
        "2026, for 100 current S&P 100 tickers (Wikipedia was blocked with HTTP 403; BK had no "
        "Yahoo quote and was dropped). The outcome Y is the monthly return. Predictors, known at "
        "the previous month-end and standardized on the estimation sample, are lagged monthly "
        "return, monthly realized volatility, 12-minus-1 momentum, and log price (a size proxy).",
    )
    body(
        doc,
        "Four candidate networks are considered. Return-based graphs are rebuilt each month from "
        "the prior 252 trading days, with no look-ahead.",
    )
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("1. Correlation (5-NN). ")
    set_run_font(r, 12, bold=True)
    r = p.add_run(
        "For each stock, the five largest absolute correlations of daily returns over the prior "
        "252 trading days. The directed k-NN graph is symmetrized first, W = max(A, A′), then "
        "row-standardized so that every non-isolated row sums to 1 (isolated rows remain 0). "
        "After this step ρ(W) ≤ 1, which is the spectral-radius bound used in Equation (2). "
        "About 362 undirected edges, mean degree 7.2."
    )
    set_run_font(r, 12)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("2. Boosted correlation resolvent. ")
    set_run_font(r, 12, bold=True)
    r = p.add_run(
        "A(I − αA)⁻¹ with α = 0.5. Because A is row-stochastic, the spectral radius is at most 1 "
        "and α ∈ (0, 1) guarantees convergence of the Neumann series. This captures multi-hop "
        "co-movement and is a dense weighted graph after row-normalization."
    )
    set_run_font(r, 12)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run("3. Sector affiliation. ")
    set_run_font(r, 12, bold=True)
    r = p.add_run(
        "An edge if two names share a GICS sector (yfinance). 561 edges, mean degree 11.2."
    )
    set_run_font(r, 12)

    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Cm(1.0)
    p.paragraph_format.space_after = Pt(10)
    r = p.add_run("4. Industry (related companies). ")
    set_run_font(r, 12, bold=True)
    r = p.add_run(
        "An edge if two names share a GICS industry. 114 edges, mean degree 2.3, 17 isolates. "
        "Firms in the same industry are treated as neighbors."
    )
    set_run_font(r, 12)

    body(
        doc,
        "Affiliation graphs are already symmetric. All four candidates—after any "
        "symmetrization—are row-standardized before Moran screening, Gibbs sampling, and "
        "kriging, so non-isolated rows sum to 1 and Equation (2′) holds on that component. "
        "Isolated rows remain 0, so (V1)i = 1 at isolates. Affiliation graphs "
        "are not boosted. For a block-diagonal clique, the resolvent is a "
        "positive multiple of A inside each block, so after row-normalization the boosted "
        "matrix is identical to the original. That collinearity produced |λ| > 1 in the old Hong Kong "
        "stock table. Duplicate matrices with vectorized correlation ≥ 0.99 are dropped "
        "automatically.",
    )

    add_heading_custom(doc, "3.2  Screening, August 2026", 2)
    body(
        doc,
        "Table 6 reports Moran’s I on OLS residuals in the showcase month. FDR keeps sector and "
        "industry. Residual dependence that month is industry/sector clustering, not the "
        "trailing-return correlation graph. Over the 63-month running window the picture is less "
        "static: correlation or its resolvent is selected in 45–51 months, sector in 35, and "
        "industry in 33.",
    )
    caption(doc, "Table 6.  Residual Moran’s I, S&P 100, August 2026.")
    add_table(
        doc,
        ["Network", "I", "z", "Two-sided p"],
        [
            ["Correlation (5-NN)", "0.018", "0.83", "0.406"],
            ["Boosted correlation resolvent", "0.027", "1.45", "0.146"],
            ["Sector affiliation", "0.122", "2.70", "0.0070"],
            ["Industry (related companies)", "0.263", "2.89", "0.0038"],
        ],
    )

    add_heading_custom(doc, "3.3  Gibbs posterior", 2)
    body(
        doc,
        "The model is estimated on 80 training names in August 2026 (Table 7). Four chains of "
        "8,000 iterations with 4,000 burn-in yield acceptances 0.24–0.35. All R̂ ≤ 1.005. ESS "
        "exceeds 1,000 for both λs. At the posterior means Σs |λ̂s| = 0.39 < 1, and every "
        "retained draw satisfies the ℓ1 invertibility bound. The only covariate whose 95% "
        "interval excludes zero is 12-1 momentum (negative: short-term reversal after a long-run "
        "run-up). σ² is on the scale of monthly return variance, not the 10⁻⁶ artefact in the "
        "original NC-SIDS table.",
    )
    caption(doc, "Table 7.  Posterior summaries, S&P 100 Multi-SEM, August 2026 (80 training stocks).")
    add_table(
        doc,
        ["Parameter", "Mean", "Median", "SD", "95% CI", "R̂", "ESS"],
        [
            ["λ sector", "0.209", "0.216", "0.192", "(−0.185, 0.559)", "1.005", "1,026"],
            ["λ industry", "0.181", "0.179", "0.123", "(−0.063, 0.418)", "1.005", "1,516"],
            ["Intercept", "0.028", "0.028", "0.013", "(0.001, 0.053)", "1.000", "13,441"],
            ["Lag return", "−0.005", "−0.005", "0.009", "(−0.023, 0.013)", "1.000", "12,123"],
            ["Volatility", "0.011", "0.011", "0.010", "(−0.009, 0.030)", "1.000", "5,762"],
            ["Momentum 12-1", "−0.025", "−0.025", "0.010", "(−0.044, −0.006)", "1.000", "11,626"],
            ["Log price", "0.007", "0.007", "0.009", "(−0.010, 0.025)", "1.000", "9,835"],
            ["σ²", "0.00485", "0.00477", "0.00083", "(0.00347, 0.00671)", "1.000", "10,907"],
        ],
    )
    add_figure(
        doc,
        FIG / "sp100_gibbs_traces.png",
        "Figure 9.  S&P 100 Gibbs traces for λ after burn-in, August 2026. "
        "Four chains shown in blue, orange, green, and red. All draws satisfy Σs |λs| < 1.",
        width=6.0,
    )

    add_heading_custom(doc, "3.4  Holdout prediction, August 2026", 2)
    body(
        doc,
        "Twenty held-out names are predicted by kriging from the training residuals (Table 8, "
        "Figure 10). Multi-SEM reduces RMSE by 9.7% versus OLS and raises the predicted–actual "
        "correlation from 0.28 to 0.60. The correlation network, which failed the Moran screen, "
        "is worse than OLS.",
    )
    caption(doc, "Table 8.  August 2026 holdout performance (n_test = 20).")
    add_table(
        doc,
        ["Model", "RMSE", "MAE", "corr(pred, actual)"],
        [
            ["OLS", "0.0637", "0.0504", "0.28"],
            ["Correlation SEM", "0.0659", "0.0512", "0.19"],
            ["Boosted correlation SEM", "0.0655", "0.0514", "0.20"],
            ["Sector SEM", "0.0588", "0.0449", "0.52"],
            ["Industry SEM", "0.0609", "0.0472", "0.51"],
            ["Multi-SEM (sector + industry)", "0.0576", "0.0445", "0.60"],
        ],
        highlight_last=True,
    )
    add_figure(
        doc,
        FIG / "sp100_pred_vs_actual.png",
        "Figure 10.  Predicted versus actual monthly return, August 2026 holdout. "
        "The line is predicted = actual.",
        width=6.3,
    )

    add_heading_custom(doc, "3.5  Running out-of-sample predictions, June 2021–August 2026", 2)
    body(
        doc,
        "A fixed 20% ticker holdout is predicted every month from the other names’ "
        "contemporaneous residuals (spatial kriging), with networks and coefficients re-estimated "
        "on that month’s training set. This evaluates cross-sectional spatial spillover in "
        "period t: held-out returns at t are imputed from peer returns at t. It is not a "
        "time-ahead forecast (t−1 → t) of the kind usual in financial econometrics. A second "
        "predictor uses only X*β̂ (no kriging), which is the object that would remain if one "
        "insisted on using only predetermined covariates. Table 9 and Figures 11–12 summarize "
        "63 months from June 2021 through August 2026.",
    )
    caption(doc, "Table 9.  Mean monthly holdout error, S&P 100, June 2021–August 2026 (63 months).")
    add_table(
        doc,
        ["Model", "RMSE", "MAE"],
        [
            ["OLS", "0.06845", "0.05171"],
            ["Correlation SEM", "0.06660", "0.04994"],
            ["Sector SEM", "0.06701", "0.05029"],
            ["Industry SEM", "0.06608", "0.04948"],
            ["Multi-SEM (Moran-selected, kriging)", "0.06568", "0.04918"],
            ["Multi-SEM, Xβ only (no kriging)", "0.06852", "0.05168"],
        ],
        highlight_last=False,
    )
    # bold the Multi-SEM kriging row manually
    tbl = doc.tables[-1]
    for cell in tbl.rows[5].cells:
        for p in cell.paragraphs:
            for run in p.runs:
                run.bold = True

    body(
        doc,
        "Multi-SEM has the best average RMSE and MAE. It beats OLS in 44 of 63 months (70%). "
        "Mean predicted–actual correlation is 0.38 versus 0.25 for OLS. Almost all of the gain "
        "is from residual kriging through the selected networks: dropping the Ω*t term returns "
        "performance to OLS. The industry network is the strongest single network over the full "
        "window; combining it with whatever else Moran selects that month is a small further "
        "improvement.",
    )
    add_figure(
        doc,
        FIG / "sp100_running_rmse.png",
        "Figure 11.  Running monthly holdout RMSE for a fixed 20% ticker holdout, "
        "June 2021–August 2026.",
        width=6.3,
    )
    add_figure(
        doc,
        FIG / "sp100_oos_mean_rmse.png",
        "Figure 12.  Average holdout RMSE by model, June 2021–August 2026.",
        width=5.6,
    )

    # ------------------------------------------------------------------
    add_heading_custom(doc, "4.  What this re-analysis fixes", 1)
    items = [
        (
            "One model. ",
            "Testing, estimation, and prediction are all Multi-SEM (error dependence). "
            "The endogenous lag AY is not treated as a regressor.",
        ),
        (
            "Valid p-values. ",
            "Histograms and tables use 2(1 − Φ(|z|)) ∈ [0, 1], plus size and power at α = 0.05, "
            "instead of 2.5% quantiles of mislabeled z-scores.",
        ),
        (
            "Stationarity. ",
            "Every Metropolis proposal with Σs |λs| ≥ 1 is rejected. That ℓ1 bound is the "
            "sufficient invertibility condition and rules out large opposing λ. S&P 100 posterior "
            "means are 0.21 and 0.18. Collinear boosted copies of affiliation graphs are removed.",
        ),
        (
            "Priors and the Jacobian. ",
            "Hyperparameters are listed in Table 1. log|V| is the exact LU log-determinant.",
        ),
        (
            "Baselines. ",
            "OLS, each single-network SEM, selected Multi-SEM, and (for stocks) a no-kriging Xβ predictor.",
        ),
        (
            "Diagnostics. ",
            "Split-R̂ and ESS are reported. All showcase R̂ values are at most 1.03.",
        ),
        (
            "Scale. ",
            "σ² matches the outcome (simulation ≈ 1; monthly returns ≈ 0.005).",
        ),
    ]
    for i, (lead, rest) in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Cm(0.75)
        p.paragraph_format.first_line_indent = Cm(-0.75)
        p.paragraph_format.space_after = Pt(6)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        r = p.add_run(f"{i}.  {lead}")
        set_run_font(r, 12, bold=True)
        r = p.add_run(rest)
        set_run_font(r, 12)

    add_heading_custom(doc, "5.  Conclusion", 1)
    body(
        doc,
        "The Multi-SEM is now used consistently for screening, posterior inference, and "
        "prediction. Moran’s I p-values lie in [0, 1] and have calibrated size with power that "
        "rises with n; Gibbs draws stay inside Σs |λs| < 1; and multi-network kriging improves on "
        "OLS and on single-network SEMs in both the simulation holdout and the 63-month S&P 100 "
        "node-holdout.",
    )
    body(
        doc,
        "Those S&P 100 numbers measure contemporaneous cross-sectional imputation, not predictive "
        "asset returns. In period t, some names are treated as unobserved and are kriged from "
        "peer residuals in the same month. That is the natural out-of-sample experiment for a "
        "spatial error model. A genuine t−1 → t forecast would drop the Ω*t term and is reported "
        "separately; it does not beat OLS. Table 9 should be read in that light.",
    )
    body(
        doc,
        "The intercept–Σλ ridge is a structural feature of row-stochastic multiplex SEM, not a "
        "sampler defect. It is visible in the five-network simulation recovery and absent in the "
        "S&P 100 fit, where Σλ̂ = 0.39. Exact LU log-determinants are used for the present "
        "n ≤ 500; larger networks can keep the same Gibbs steps with an approximate log|V|.",
    )

    add_heading_custom(doc, "6.  Replication files", 1)
    caption(doc, "Table 10.  Code and output files.")
    add_table(
        doc,
        ["Path", "Contents"],
        [
            ["mnreg/model.py", "Moran’s I, Gibbs sampler, kriging, R̂, ESS"],
            ["mnreg/networks.py", "ER / power-law graphs, affiliation, correlation, resolvent"],
            ["run_simulation.py", "Size/power, posterior recovery, out-of-sample predictions"],
            ["run_sp100.py", "S&P 100 networks, posterior, running predictions"],
            ["results/", "CSV and JSON tables"],
            ["results/figures/", "Size/power, p-value histograms, traces, predicted vs actual"],
            ["data/", "Cached prices and fundamentals"],
        ],
    )
    body(
        doc,
        "Run python run_simulation.py then python run_sp100.py, or python run_all.py. "
        "This document was generated from those outputs. Sections 1–4 can be merged into "
        "Sections 3–5 of the main LaTeX draft, replacing the previous lag-model derivation, "
        "the p-value histograms, and the original Hong Kong stock tables.",
        first_line_indent=True,
    )

    footer = section.footer
    fp = footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = fp.add_run("Multi-SEM re-analysis  ·  Regression with Multiple Networks")
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"
    run.font.color.rgb = RGBColor(89, 89, 89)

    # page numbers
    add_page_number(section)

    doc.save(OUT)
    print("Wrote", OUT)


def add_page_number(section):
    footer = section.footer
    p = footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    fld1 = OxmlElement("w:fldChar")
    fld1.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld2 = OxmlElement("w:fldChar")
    fld2.set(qn("w:fldCharType"), "end")
    run._r.append(fld1)
    run._r.append(instr)
    run._r.append(fld2)
    run.font.size = Pt(9)
    run.font.name = "Times New Roman"


if __name__ == "__main__":
    build()
