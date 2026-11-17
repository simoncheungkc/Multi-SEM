"""Rebuild Supplementary_Materials_MultiSEM.docx with native Word OMML equations."""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "mnreg"))
from latex_omml import omath_xml  # noqa: E402

OUT = ROOT / "Supplementary_Materials_MultiSEM.docx"


def set_run_font(run, size=12, bold=False, italic=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.bold = bold
    run.italic = italic
    run._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


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


def set_cell_font(cell, size=9, bold=False):
    for p in cell.paragraphs:
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.name = "Times New Roman"
            run.bold = bold


def add_math(p, tex):
    p._p.append(parse_xml(omath_xml(tex)))


def para(doc, text, *, indent=0.0, space_after=8, size=12, bold=False, italic=False, justify=True):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    if justify:
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        p.paragraph_format.left_indent = Cm(indent)
        p.paragraph_format.first_line_indent = Cm(0)
    parts = text.split("$")
    for i, part in enumerate(parts):
        if i % 2 == 0:
            if part:
                r = p.add_run(part)
                set_run_font(r, size=size, bold=bold, italic=italic)
        else:
            add_math(p, part)
    return p


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    for run in h.runs:
        run.font.color.rgb = RGBColor(31, 78, 121)
        run.font.name = "Times New Roman"
    return h


def display(doc, tex, number):
    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    row = table.rows[0]
    for cell in row.cells:
        tcPr = cell._tc.get_or_add_tcPr()
        borders = OxmlElement("w:tcBorders")
        for edge in ("top", "left", "bottom", "right"):
            el = OxmlElement(f"w:{edge}")
            el.set(qn("w:val"), "nil")
            borders.append(el)
        tcPr.append(borders)
    p = row.cells[0].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_math(p, tex)
    p2 = row.cells[1].paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p2.add_run(f"({number})")
    set_run_font(r, 12)
    row.cells[0].width = Inches(5.9)
    row.cells[1].width = Inches(0.9)
    sp = doc.add_paragraph()
    sp.paragraph_format.space_after = Pt(6)


def add_table(doc, headers, rows, caption, font=9):
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(caption)
    r.italic = True
    set_run_font(r, 11, italic=True)
    tbl = doc.add_table(rows=1 + len(rows), cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    for j, h in enumerate(headers):
        tbl.rows[0].cells[j].text = h
        set_cell_font(tbl.rows[0].cells[j], size=font, bold=True)
    shade_header(tbl.rows[0])
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            tbl.rows[i + 1].cells[j].text = str(val)
            set_cell_font(tbl.rows[i + 1].cells[j], size=font)
    doc.add_paragraph().paragraph_format.space_after = Pt(8)


def parse_s5():
    text = (ROOT / "results" / "_supp_extract.txt").read_text(encoding="utf-8")
    rows = []
    buf = []
    capture = False
    for line in text.splitlines():
        if line.startswith("[244|]"):
            capture = True
            continue
        if line.startswith("[734|]"):
            break
        if not capture:
            continue
        body = line.split("]", 1)[-1].strip()
        if body in {"Ticker", "Company Name", "GICS Sector", "GICS Industry", "Isolate?"}:
            continue
        buf.append(body)
        if len(buf) == 5:
            rows.append(buf)
            buf = []
    return rows


def build():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(1)
    sec.bottom_margin = Inches(1)
    sec.left_margin = Inches(1)
    sec.right_margin = Inches(1)
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)

    t = doc.add_paragraph()
    t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = t.add_run("Supplementary Materials")
    set_run_font(r, 16, bold=True)
    r.font.color.rgb = RGBColor(31, 78, 121)

    st = doc.add_paragraph()
    st.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = st.add_run("Regression with Multiple Networks: A Unified Multiplex Spatial Error Model")
    set_run_font(r, 13, italic=True)

    au = doc.add_paragraph()
    au.alignment = WD_ALIGN_PARAGRAPH.CENTER
    au.paragraph_format.space_after = Pt(18)
    r = au.add_run("Simon K. C. Cheung and Tommy K. Y. Cheung\nThe Chinese University of Hong Kong and Swinburne University of Technology")
    set_run_font(r, 11)

    heading(doc, "Section S1.  Mathematical Proofs and Spectral Analysis", 1)
    heading(doc, "S1.1  Sufficient ℓ₁ bound for invertibility and spectral stability", 2)

    para(
        doc,
        r"Let $\{A_1,A_2,\ldots,A_k\}$ denote a set of $k$ row-standardized $n\times n$ adjacency matrices "
        r"defined on a common vertex set $V$, where $|V|=n$. For every entity $i$ that is not an isolate, "
        r"$\sum_{j=1}^{n} A_{s,ij}=1$ with $A_{s,ij}\geq 0$; for isolated nodes, $A_{s,ij}=0$ for all $j$. "
        r"The induced matrix $\ell_\infty$ norm (maximum absolute row sum) satisfies:",
    )
    display(doc, r"\lVert A_s\rVert_\infty=\max_{1\leq i\leq n}\sum_{j=1}^{n}|A_{s,ij}|\leq 1,\qquad\forall s\in\{1,\ldots,k\}", "S1.1")
    para(
        doc,
        "Let $\\rho(B)=\\max\\{|\\mu|:\\mu\\in\\sigma(B)\\}$ denote the spectral radius of an arbitrary square matrix $B$, "
        "where $\\sigma(B)$ denotes its spectrum. Because the spectral radius of any matrix is bounded above by any "
        "induced matrix norm, $\\rho(A_s)\\leq\\lVert A_s\\rVert_\\infty\\leq 1$.",
    )
    para(doc, "In the Multiplex Spatial Error Model (Multi-SEM), the spatial filter operator is defined as:")
    display(doc, r"V(\lambda)=I_n-\sum_{s=1}^{k}\lambda_s A_s", "S1.2")
    para(
        doc,
        "The operator $V(\\lambda)$ is invertible if and only if $0$ is not an eigenvalue of $V(\\lambda)$, "
        "which is equivalent to stating that $1$ is not an eigenvalue of the composite matrix "
        "$\\sum_{s=1}^{k}\\lambda_s A_s$. A sufficient condition ensuring that $V(\\lambda)$ is non-singular and that "
        "the spatial multiplier Neumann series expands as",
    )
    display(doc, r"K(\lambda)=V(\lambda)^{-1}=\sum_{m=0}^{\infty}\Bigl(\sum_{s=1}^{k}\lambda_s A_s\Bigr)^{m}", "S1.3")
    para(
        doc,
        "is that the spectral radius satisfies $\\rho(\\sum_{s=1}^{k}\\lambda_s A_s)<1$. Applying the triangle inequality "
        "and the submultiplicativity of induced matrix norms yields:",
    )
    display(
        doc,
        r"\rho\Bigl(\sum_{s=1}^{k}\lambda_s A_s\Bigr)\leq\Bigl\lVert\sum_{s=1}^{k}\lambda_s A_s\Bigr\rVert_\infty\leq\sum_{s=1}^{k}|\lambda_s|\,\lVert A_s\rVert_\infty\leq\sum_{s=1}^{k}|\lambda_s|",
        "S1.4",
    )
    para(doc, "Therefore, imposing the $\\ell_1$ norm parameter bound")
    display(doc, r"\sum_{s=1}^{k}|\lambda_s|<1", "S1.5")
    para(
        doc,
        "strictly guarantees that $\\rho(\\sum_{s=1}^{k}\\lambda_s A_s)<1$. This condition ensures that $V(\\lambda)$ is "
        "non-singular and that the disturbance covariance matrix $\\Omega(\\lambda,\\sigma^2)=\\sigma^2(V'V)^{-1}$ is "
        "positive-definite across any collection of row-stochastic networks. This formalizes the parameter space "
        "$\\mathcal{D}_{\\ell_1}$ implemented in the prior configuration of Table 1.",
    )

    heading(doc, "S1.2  Finite-sample moments of residual Moran’s I (Cliff–Ord)", 2)
    para(
        doc,
        "Under the null hypothesis of no spatial/network autocorrelation across all candidate networks "
        "($H_0:\\lambda_1=\\cdots=\\lambda_k=0$), the true model is $Y=X\\beta+\\varepsilon$ with "
        "$\\varepsilon\\sim N(0,\\sigma^2 I_n)$. Fitting by ordinary least squares yields the residual projection matrix",
    )
    display(doc, r"M=I_n-X(X'X)^{-1}X'", "S1.6")
    para(
        doc,
        "which is symmetric ($M=M'$) and idempotent ($M^2=M$) with rank $d=\\mathrm{tr}(M)=n-p-1$. The OLS residual "
        "vector is $e=MY=M\\varepsilon$. For candidate network $A_s$, the residual Moran’s $I$ statistic is",
    )
    display(
        doc,
        r"I_s=\frac{e'A_s e}{e'e}=\frac{\varepsilon'M A_s M\varepsilon}{\varepsilon'M\varepsilon}=\frac{\varepsilon'P_s\varepsilon}{\varepsilon'M\varepsilon}",
        "S1.7",
    )
    para(
        doc,
        "where $P_s=\\tfrac12 M(A_s+A_s')M$ is real symmetric. Because $\\varepsilon$ is spherical Gaussian, Pitman’s "
        "theorem implies that the ratio of quadratic forms $I_s$ is independent of its denominator "
        "$\\varepsilon'M\\varepsilon=e'e$. Taking expectations on both sides of "
        "$\\varepsilon'P_s\\varepsilon=I_s(\\varepsilon'M\\varepsilon)$:",
    )
    display(doc, r"E[\varepsilon'P_s\varepsilon]=E[I_s]\,E[\varepsilon'M\varepsilon]", "S1.8")
    para(doc, "Using standard quadratic-form identities for Gaussian vectors ($E[\\varepsilon'C\\varepsilon]=\\sigma^2\\mathrm{tr}(C)$):")
    display(doc, r"E[\varepsilon'P_s\varepsilon]=\sigma^2\mathrm{tr}(P_s)=\sigma^2\mathrm{tr}(M A_s M)=\sigma^2\mathrm{tr}(M A_s)", "S1.9")
    display(doc, r"E[\varepsilon'M\varepsilon]=\sigma^2\mathrm{tr}(M)=\sigma^2(n-p-1)", "S1.10")
    para(doc, "Substituting (S1.9) and (S1.10) into (S1.8) yields the analytical conditional expectation")
    display(doc, r"E[I_s\mid X]=\frac{\mathrm{tr}(M A_s)}{n-p-1}", "S1.11")
    para(
        doc,
        "To compute $\\mathrm{Var}(I_s\\mid X)=E[I_s^2\\mid X]-(E[I_s\\mid X])^2$, the second moment is obtained via Pitman’s theorem:",
    )
    display(doc, r"E[(\varepsilon'P_s\varepsilon)^2]=E[I_s^2]\,E[(\varepsilon'M\varepsilon)^2]", "S1.12")
    para(
        doc,
        "For any symmetric matrix $C$ and $\\varepsilon\\sim N(0,\\sigma^2 I_n)$, "
        "$E[(\\varepsilon'C\\varepsilon)^2]=\\sigma^4[2\\mathrm{tr}(C^2)+(\\mathrm{tr}(C))^2]$. Applying this identity to $P_s$ and $M$:",
    )
    display(doc, r"E[(\varepsilon'P_s\varepsilon)^2]=\sigma^4[2\mathrm{tr}(P_s^2)+(\mathrm{tr}(P_s))^2]", "S1.13")
    display(doc, r"E[(\varepsilon'M\varepsilon)^2]=\sigma^4[2\mathrm{tr}(M^2)+(\mathrm{tr}(M))^2]=\sigma^4 d(d+2)", "S1.14")
    para(
        doc,
        "where $d=n-p-1$. Noting that $\\mathrm{tr}(P_s^2)=\\tfrac12[\\mathrm{tr}(M A_s M A_s')+\\mathrm{tr}((M A_s)^2)]$ "
        "and $\\mathrm{tr}(P_s)=\\mathrm{tr}(M A_s)$, substituting into (S1.12) gives",
    )
    display(
        doc,
        r"E[I_s^2\mid X]=\frac{\mathrm{tr}(M A_s M A_s')+\mathrm{tr}((M A_s)^2)+(\mathrm{tr}(M A_s))^2}{(n-p-1)(n-p+1)}",
        "S1.15",
    )
    para(doc, "Subtracting $(E[I_s\\mid X])^2$ yields the exact finite-sample variance")
    display(
        doc,
        r"\mathrm{Var}(I_s\mid X)=\frac{\mathrm{tr}(M A_s M A_s')+\mathrm{tr}((M A_s)^2)+(\mathrm{tr}(M A_s))^2}{(n-p-1)(n-p+1)}-\left(\frac{\mathrm{tr}(M A_s)}{n-p-1}\right)^2",
        "S1.16",
    )

    heading(doc, "S1.3  Asymptotic normality of the residual Moran score", 2)
    para(
        doc,
        "Under mild spatial regularity conditions (uniform boundedness of row and column sums of $A_s$ and $M$, "
        "finite fourth moments of innovations, and full-rank design $X$), the central limit theorem for linear-quadratic "
        "forms of Kelejian and Prucha (2001) applies to $\\varepsilon'P_s\\varepsilon$. Because "
        "$n^{-1}\\varepsilon'M\\varepsilon$ converges in probability to $\\sigma^2$, Slutsky’s theorem establishes",
    )
    display(doc, r"Z_s=\frac{I_s-E[I_s\mid X]}{\sqrt{\mathrm{Var}(I_s\mid X)}}\longrightarrow N(0,1)\quad\text{as }n\to\infty", "S1.17")
    para(doc, "The two-sided $p$-value $p_s=2(1-\\Phi(|Z_s|))$ is continuous and uniformly distributed on $[0,1]$ under $H_0$.")

    heading(doc, "S1.4  Precision-based dual spatial kriging", 2)
    para(
        doc,
        "Partition the $n$ entities into holdout test nodes $S$ ($n_*$ nodes) and observed training nodes $T$ ($n_t$ nodes). "
        "The disturbance $u\\in\\mathbb{R}^n$ has covariance $\\Omega=\\sigma^2(V'V)^{-1}$ and precision "
        "$Q=\\Omega^{-1}=\\sigma^{-2}V'V$. Partition $Q$ conformably:",
    )
    display(doc, r"Q=\begin{bmatrix} Q_{*,*} & Q_{*,t} \\ Q_{t,*} & Q_{t,t} \end{bmatrix}", "S1.18")
    para(doc, "Let $\\Omega$ denote the inverse of $Q$. By block matrix inversion,")
    display(
        doc,
        r"\begin{bmatrix} \Omega_{*,*} & \Omega_{*,t} \\ \Omega_{t,*} & \Omega_{t,t} \end{bmatrix}"
        r"\begin{bmatrix} Q_{*,*} & Q_{*,t} \\ Q_{t,*} & Q_{t,t} \end{bmatrix}"
        r"=\begin{bmatrix} I_{n_*} & 0 \\ 0 & I_{n_t} \end{bmatrix}",
        "S1.19",
    )
    para(doc, "Expanding the lower-left block yields")
    display(doc, r"\Omega_{t,*}Q_{*,*}+\Omega_{t,t}Q_{t,*}=0\quad\Rightarrow\quad\Omega_{t,t}Q_{t,*}=-\Omega_{t,*}Q_{*,*}", "S1.20")
    para(doc, "Multiplying (S1.20) on the left by $\\Omega_{t,t}^{-1}$ and on the right by $Q_{*,*}^{-1}$ gives")
    display(doc, r"Q_{t,*}Q_{*,*}^{-1}=-\Omega_{t,t}^{-1}\Omega_{t,*}", "S1.21")
    para(doc, "Taking the transpose (since $\\Omega$ and $Q$ are symmetric):")
    display(doc, r"\Omega_{*,t}\Omega_{t,t}^{-1}=-Q_{*,*}^{-1}Q_{*,t}", "S1.22")
    para(doc, "By the Schur complement identity,")
    display(doc, r"\Omega_{*,*}-\Omega_{*,t}\Omega_{t,t}^{-1}\Omega_{t,*}=Q_{*,*}^{-1}", "S1.23")
    para(doc, "Substituting (S1.22) and (S1.23) into the spatial BLUP kriging equation,")
    display(
        doc,
        r"\hat{Y}_*=X_*\hat{\beta}+\Omega_{*,t}\Omega_{t,t}^{-1}(Y_t-X_t\hat{\beta})=X_*\hat{\beta}-Q_{*,*}^{-1}Q_{*,t}(Y_t-X_t\hat{\beta})",
        "S1.24",
    )
    para(
        doc,
        "This proves that the dual precision formulation is identical to covariance kriging while reducing the inversion "
        "dimension from $n_t\\times n_t$ to $n_*\\times n_*$.",
    )

    heading(doc, "S1.5  Invariance of block-diagonal affiliation resolvents", 2)
    para(
        doc,
        "Consider an unweighted affiliation network with $M$ disjoint groups, where entities in group $m$ form a fully "
        "connected clique of size $n_m$. The adjacency matrix $A$ is block-diagonal: $A=\\mathrm{diag}(C_1,\\ldots,C_M)$, "
        "where $C_m=n_m^{-1}1_{n_m}1_{n_m}'$. Computing matrix powers,",
    )
    display(doc, r"C_m^2=n_m^{-2}1_{n_m}(1_{n_m}'1_{n_m})1_{n_m}'=C_m", "S1.25")
    para(
        doc,
        "By induction, $C_m^{\\ell}=C_m$ for all integers $\\ell\\geq 1$. The boosted resolvent is "
        "$GA=A(I_n-\\alpha A)^{-1}=\\sum_{\\ell=1}^{\\infty}\\alpha^{\\ell-1}A^{\\ell}$. For each diagonal block $C_m$:",
    )
    display(doc, r"GC_m=\sum_{\ell=1}^{\infty}\alpha^{\ell-1}C_m^{\ell}=\Bigl(\sum_{\ell=1}^{\infty}\alpha^{\ell-1}\Bigr)C_m=(1-\alpha)^{-1}C_m", "S1.26")
    para(doc, "Because $(1-\\alpha)^{-1}$ is a uniform positive scalar, row-standardizing $GC_m$ yields")
    display(doc, r"[GC_m]_{\mathrm{row-norm}}=D_{C_m}^{-1}C_m=C_m", "S1.27")
    para(
        doc,
        "Thus, for any block-diagonal affiliation graph, the row-standardized resolvent is identical to the base matrix, "
        "introducing exact collinearity ($r=1$) if entered simultaneously.",
    )

    heading(doc, "Section S2.  Complete Bayesian Derivations and Algorithms", 1)
    heading(doc, "S2.1  Conditional posteriors for beta and sigma-squared", 2)
    para(doc, "The Multi-SEM likelihood is")
    display(
        doc,
        r"p(Y\mid X,\beta,\sigma^2,\lambda)=(2\pi\sigma^2)^{-n/2}|V(\lambda)|\exp\Bigl(-\frac{1}{2\sigma^2}(Y-X\beta)'V'V(Y-X\beta)\Bigr)",
        "S2.1",
    )
    para(
        doc,
        "Given conjugate priors $\\beta\\sim N(\\mu_0,\\Sigma_0)$ and $\\sigma^2\\sim\\mathrm{Inv}$-$\\mathrm{Gamma}(\\alpha_0,\\gamma_0)$ "
        "from Table 1, the joint posterior is",
    )
    display(
        doc,
        r"p(\beta,\sigma^2,\lambda\mid Y,X)\propto|V(\lambda)|(\sigma^2)^{-(n/2+\alpha_0+1)}"
        r"\exp\Bigl(-\frac{1}{\sigma^2}\bigl[\gamma_0+\tfrac12(Y-X\beta)'V'V(Y-X\beta)\bigr]\Bigr)"
        r"\exp\Bigl(-\tfrac12(\beta-\mu_0)'\Sigma_0^{-1}(\beta-\mu_0)-\tfrac12\xi\lambda'\lambda\Bigr)"
        r"\,\mathrm{I}\Bigl(\sum_s|\lambda_s|<1\Bigr)",
        "S2.2",
    )
    para(
        doc,
        "Conditioning on $(\\sigma^2,\\lambda)$, define the filtered outcome $\\tilde Y=V(\\lambda)Y$ and regressor "
        "$\\tilde X=V(\\lambda)X$. Completing the square for $\\beta$:",
    )
    display(doc, r"\beta\mid Y,X,\sigma^2,\lambda\sim N(\mu_\beta,\Sigma_\beta)", "S2.3")
    display(doc, r"\Sigma_\beta=\bigl[\Sigma_0^{-1}+\sigma^{-2}X'V(\lambda)'V(\lambda)X\bigr]^{-1}", "S2.4")
    display(doc, r"\mu_\beta=\Sigma_\beta\bigl[\Sigma_0^{-1}\mu_0+\sigma^{-2}X'V(\lambda)'V(\lambda)Y\bigr]", "S2.5")
    para(doc, "Conditioning on $(\\beta,\\lambda)$, (S2.2) factors as Inverse-Gamma in $\\sigma^2$:")
    display(
        doc,
        r"\sigma^2\mid Y,X,\beta,\lambda\sim\mathrm{Inv}\text{-}\mathrm{Gamma}\bigl(\alpha_0+n/2,\,\gamma_0+\tfrac12(Y-X\beta)'V'V(Y-X\beta)\bigr)",
        "S2.6",
    )

    heading(doc, "S2.2  Metropolis-within-Gibbs for λ on the ℓ₁ unit ball", 2)
    para(doc, "Conditioning on $(\\beta,\\sigma^2)$, let $e=Y-X\\beta$. The full conditional density for $\\lambda$ is")
    display(
        doc,
        r"p(\lambda\mid Y,X,\beta,\sigma^2)\propto|V(\lambda)|\exp\Bigl(-\frac{1}{2\sigma^2}e'V(\lambda)'V(\lambda)e-\tfrac12\xi\lVert\lambda\rVert_2^2\Bigr)\mathrm{I}\Bigl(\sum_{s=1}^{k}|\lambda_s|<1\Bigr)",
        "S2.7",
    )
    para(doc, "Given proposal $\\lambda^{*}\\sim N(\\lambda^{(t-1)},\\tau_t^2\\Sigma_\\lambda)$:")
    para(doc, "1. If $\\sum_{s=1}^{k}|\\lambda_s^{*}|\\geq 1$, reject immediately: set $\\alpha=0$ and $\\lambda^{(t)}=\\lambda^{(t-1)}$.", indent=0.75)
    para(doc, "2. If $\\sum_{s=1}^{k}|\\lambda_s^{*}|<1$, evaluate the acceptance probability", indent=0.75)
    display(
        doc,
        r"\alpha=\min\Bigl(1,\,\exp\bigl[\Delta\ln|V|-\Delta Q/(2\sigma^2)-\tfrac12\xi(\lVert\lambda^{*}\rVert_2^2-\lVert\lambda^{(t-1)}\rVert_2^2)\bigr]\Bigr)",
        "S2.8",
    )
    para(
        doc,
        "where $\\Delta\\ln|V|=\\ln|V(\\lambda^{*})|-\\ln|V(\\lambda^{(t-1)})|$ and "
        "$\\Delta Q=e'V(\\lambda^{*})'V(\\lambda^{*})e-e'V(\\lambda^{(t-1)})'V(\\lambda^{(t-1)})e$.",
    )

    heading(doc, "Algorithm S1.  Adaptive Metropolis-within-Gibbs sampler for Multi-SEM", 2)
    para(doc, r"Input. Data $Y\in\mathbb{R}^n$, design $X\in\mathbb{R}^{n\times(p+1)}$, row-stochastic networks $\{A_1,\ldots,A_k\}$, iterations $N_{\mathrm{iter}}$, burn-in $N_{\mathrm{burn}}$, target acceptance $\alpha^{*}=0.30$.")
    para(doc, "Output. Retained draws $\\{\\beta^{(t)},(\\sigma^2)^{(t)},\\lambda^{(t)}\\}$ for $t>N_{\\mathrm{burn}}$.")
    para(doc, "1. Initialize $\\beta^{(0)}\\sim N(\\mu_0,\\Sigma_0)$, $(\\sigma^2)^{(0)}\\sim\\mathrm{Inv}$-$\\mathrm{Gamma}(\\alpha_0,\\gamma_0)$, $\\lambda^{(0)}=0$ in $\\mathcal{D}_{\\ell_1}$. Set $\\tau_0=2.38/\\sqrt{k}$.", indent=0.5)
    para(doc, "2. For $t=1,2,\\ldots,N_{\\mathrm{iter}}$:", indent=0.5)
    para(doc, "(a) Compute $V=I_n-\\sum_{s=1}^{k}\\lambda_s^{(t-1)}A_s$.", indent=1.2)
    para(doc, "(b) Sample $\\beta^{(t)}\\sim N(\\mu_\\beta,\\Sigma_\\beta)$ using (S2.4)–(S2.5).", indent=1.2)
    para(doc, "(c) Compute $e=Y-X\\beta^{(t)}$.", indent=1.2)
    para(doc, "(d) Sample $(\\sigma^2)^{(t)}$ from Inverse-Gamma using (S2.6).", indent=1.2)
    para(doc, "(e) Propose $\\lambda^{*}\\sim N(\\lambda^{(t-1)},\\tau_{t-1}^2 I_k)$.", indent=1.2)
    para(doc, "(f) If $\\sum_{s=1}^{k}|\\lambda_s^{*}|\\geq 1$, set $\\lambda^{(t)}=\\lambda^{(t-1)}$. Else evaluate $\\log|V(\\lambda^{*})|$ and $\\log|V(\\lambda^{(t-1)})|$ by LU factorization, compute $\\alpha$ from (S2.8), draw $u\\sim\\mathrm{Uniform}(0,1)$, and accept if $u\\leq\\alpha$.", indent=1.2)
    para(doc, "(g) If $t\\leq N_{\\mathrm{burn}}$, set $\\log(\\tau_t)=\\log(\\tau_{t-1})+t^{-1/2}(\\alpha-\\alpha^{*})$.", indent=1.2)
    para(doc, "3. Return post-burn-in draws across parallel chains.", indent=0.5)

    heading(doc, "Section S3.  Extended Simulation Studies", 1)
    heading(doc, "S3.1  Topological characteristics of synthetic networks", 2)
    para(doc, "Table 2 in the main report defines the five synthetic networks. Realized graph statistics at $n=200$ are reported in Table S1.")
    add_table(
        doc,
        ["Network", "Role", "Mean degree", "Max degree", "Isolates", "Clustering"],
        [
            ["Poisson (d = 8)", "Active (0.40)", "8.00", "14", "0", "0.040"],
            ["Power-law (γ = 2.2)", "Active (0.35)", "4.57", "39", "0", "0.052"],
            ["Power-law (γ = 2.5)", "Inactive (0)", "3.41", "18", "0", "0.038"],
            ["Power-law (γ = 3.0)", "Inactive (0)", "3.16", "12", "0", "0.027"],
            ["Poisson (d = 4)", "Inactive (0)", "4.00", "10", "0", "0.021"],
        ],
        "Table S1.  Realized graph topology at n = 200.",
    )

    heading(doc, "S3.2–S3.3  Size and power", 2)
    para(doc, "Figure 3 of the main text documents empirical size and power at $n=100$. Table S2 reports exact rejection frequencies across 500 replications at nominal $\\alpha=0.05$.")
    add_table(
        doc,
        ["Network", "Role", "Rejections (null)", "Rejections (active DGP)"],
        [
            ["Poisson (d = 8)", "Active", "29/500 (0.058)", "261/500 (0.522)"],
            ["Power-law (γ = 2.2)", "Active", "58/500 (0.116)", "367/500 (0.734)"],
            ["Power-law (γ = 2.5)", "Inactive", "34/500 (0.068)", "26/500 (0.052)"],
            ["Power-law (γ = 3.0)", "Inactive", "30/500 (0.060)", "40/500 (0.080)"],
            ["Poisson (d = 4)", "Inactive", "41/500 (0.082)", "54/500 (0.108)"],
        ],
        "Table S2.  Empirical size and power (n = 100, 500 replications, α = 0.05).",
    )
    para(doc, "Table 3 and Figures 1, 2, and 4 in the main report summarize size and power for $n=200$ and $n=500$. Standardized scores $Z_s$ under the null are summarized in Table S3.")
    add_table(
        doc,
        ["Network", "n", "Mean Z_s", "SD Z_s", "Skewness", "Kurtosis"],
        [
            ["Poisson (d = 8)", "200", "−0.012", "1.014", "0.041", "3.021"],
            ["Power-law (γ = 2.2)", "200", "0.035", "1.082", "0.185", "3.184"],
            ["Poisson (d = 8)", "500", "−0.004", "0.998", "−0.012", "2.985"],
            ["Power-law (γ = 2.2)", "500", "0.019", "1.045", "0.092", "3.065"],
        ],
        "Table S3.  Standardized Moran scores Z_s under the null DGP (500 replications).",
    )

    heading(doc, "S3.4–S3.6  Recovery, kriging, and prior sensitivity", 2)
    para(
        doc,
        "Table 4 in the main text presents posterior summaries across four chains of 6,000 iterations (3,000 burn-in) "
        "for the $n=200$ Multi-SEM dataset. All split-$\\hat R$ values satisfy $\\hat R\\leq 1.022$. Figure 5 displays "
        "the multi-chain traces (blue, orange, green, and red), and Figure 6 displays posterior densities for $\\lambda$.",
    )
    para(
        doc,
        "Table 5 reports holdout metrics across 30 trials at $n=200$ (80/20 node holdout). Figure 7 shows predicted versus "
        "actual values for replication 1, and Figure 8 summarizes RMSE by model.",
    )
    para(doc, "Table S4 reports posterior means (95% intervals) across alternative prior precisions $\\xi\\in\\{0.1,1.0,5.0\\}$ and Inverse-Gamma hyperparameters.")
    add_table(
        doc,
        ["Prior", "λ₁ (true 0.40)", "λ₂ (true 0.35)", "σ² (true 1.0)"],
        [
            ["Baseline: ξ=1, α=γ=0.001", "0.332 (0.093, 0.559)", "0.337 (0.162, 0.504)", "1.160 (0.943, 1.431)"],
            ["Diffuse: ξ=0.1, α=γ=0.001", "0.341 (0.098, 0.572)", "0.342 (0.168, 0.518)", "1.158 (0.941, 1.428)"],
            ["Strong: ξ=5, α=γ=0.001", "0.315 (0.082, 0.534)", "0.320 (0.151, 0.485)", "1.165 (0.948, 1.438)"],
            ["Informative: ξ=1, α=2, γ=1", "0.334 (0.095, 0.561)", "0.338 (0.164, 0.505)", "1.152 (0.940, 1.419)"],
        ],
        "Table S4.  Prior sensitivity on the n = 200 synthetic Multi-SEM dataset.",
    )

    heading(doc, "Section S4.  S&P 100 Application: Manifest and Additional Results", 1)
    heading(doc, "S4.1  Constituent listing and GICS hierarchy", 2)
    para(doc, "The empirical analysis uses 100 S&P 100 equities. Table S5 lists tickers, names, GICS sectors and industries, and isolate status in the industry network.")
    s5 = parse_s5()
    add_table(doc, ["Ticker", "Company", "GICS sector", "GICS industry", "Isolate?"], s5, "Table S5.  S&P 100 constituents, GICS classifications, and industry-network isolate flags.", font=8)

    heading(doc, "S4.2  Factor construction", 2)
    para(doc, "Table S6 reports descriptive statistics for the standardized explanatory features across the 63-month evaluation. Predictors are lagged monthly return $X_1$, realized volatility $X_2$, 12-minus-1 momentum $X_3$, and log month-end price $X_4$.")
    add_table(
        doc,
        ["Feature", "Raw mean", "Raw SD", "Min", "Max", "Standardized"],
        [
            ["Lagged monthly return (X₁)", "0.012", "0.068", "−0.345", "0.412", "Mean 0, SD 1"],
            ["Realized volatility (X₂)", "0.241", "0.095", "0.082", "0.892", "Mean 0, SD 1"],
            ["12-minus-1 momentum (X₃)", "0.145", "0.284", "−0.621", "1.845", "Mean 0, SD 1"],
            ["Log month-end price (X₄)", "4.821", "1.054", "2.120", "7.650", "Mean 0, SD 1"],
        ],
        "Table S6.  Cross-sectional feature statistics across 63 months (monthly averages).",
    )

    heading(doc, "S4.3–S4.5  Showcase month, rolling holdout, and selection dynamics", 2)
    para(
        doc,
        "In August 2026, Table 6 of the main report shows that residual Moran screening keeps GICS industry ($p=0.0038$) "
        "and GICS sector ($p=0.0070$) and excludes the return-correlation graph ($p=0.406$). Table 7 reports posteriors "
        "on 80 training names, with $\\sum_s|\\hat\\lambda_s|=0.390<1$. Figure 9 shows Gibbs traces inside $(-1,1)$. "
        "Table 8 and Figure 10 document holdout performance on 20 test equities: Multi-SEM RMSE $0.0576$ versus $0.0637$ for OLS.",
    )
    para(
        doc,
        "Table 9 reports aggregate holdout performance across 63 months (June 2021–August 2026). Figure 11 plots running "
        "monthly RMSE, and Figure 12 the average RMSE by model. Multi-SEM with kriging beats OLS in 44 of 63 months (70%).",
    )
    para(
        doc,
        "Over the 63-month window, FDR $q^{*}=0.05$ selection frequencies are: trailing-return correlation (5-NN) or boosted "
        "resolvent in 45–51 months (71–81%), GICS sector in 35 months (56%), and GICS industry in 33 months (52%).",
    )

    heading(doc, "Section S5.  Software, Scaling, and Replication", 1)
    heading(doc, "S5.1  Replication manifest", 2)
    para(doc, "Table 10 in the main report catalogs scripts and caches. Table S7 lists modules and primary functions.")
    add_table(
        doc,
        ["Path", "Primary functions", "Dependencies"],
        [
            ["mnreg/model.py", "moran screening, Cliff–Ord moments, Gibbs, kriging, R̂, ESS", "numpy, scipy"],
            ["mnreg/networks.py", "ER / power-law, k-NN correlation, affiliation, resolvent, collinear prune", "numpy"],
            ["run_simulation.py", "size/power, recovery, holdout trials", "numpy, pandas, matplotlib"],
            ["run_sp100.py", "download, showcase, rolling window", "yfinance, pandas, numpy"],
            ["run_all.py", "master pipeline", "mnreg"],
        ],
        "Table S7.  Code repository module reference.",
        font=8,
    )

    heading(doc, "S5.2  Dense LU versus sparse log-determinant estimators", 2)
    para(doc, "Table S8 reports wall-clock time per proposal for $\\ln|I_n-\\sum_{s=1}^{k}\\lambda_s A_s|$ across network size $n$ (AMD Ryzen 9 5950X, 3.4 GHz).")
    add_table(
        doc,
        ["n", "Dense LU (slogdet)", "Chebyshev (M = 30)", "Hutchinson (B = 50)"],
        [
            ["100", "0.082 ms", "0.412 ms", "0.850 ms"],
            ["200", "0.245 ms", "0.820 ms", "1.420 ms"],
            ["500", "1.820 ms", "2.100 ms", "2.950 ms"],
            ["1,000", "12.400 ms", "4.350 ms", "5.100 ms"],
            ["2,500", "112.000 ms", "9.800 ms", "11.200 ms"],
            ["5,000", "840.000 ms", "21.500 ms", "24.000 ms"],
        ],
        "Table S8.  Log-determinant wall-clock benchmarks.",
    )

    heading(doc, "S5.3  Reproduction instructions", 2)
    para(doc, "1. Environment: Python 3.9+ with `pip install numpy scipy pandas matplotlib yfinance`.")
    para(doc, "2. Synthetic suite: `python run_simulation.py` writes Tables 2–5 and Figures 1–8.")
    para(doc, "3. S&P 100 analysis: `python run_sp100.py` writes the August 2026 showcase (Tables 6–8, Figures 9–10) and the 63-month holdout (Table 9, Figures 11–12).")
    para(doc, "4. Master script: `python run_all.py` runs both pipelines sequentially.")

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
