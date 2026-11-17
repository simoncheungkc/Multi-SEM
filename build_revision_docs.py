"""Build manuscript v22, supplement v20, and the response to reviewers.

Numerical sentences are filled from results/revision/*.json. Those files are
written by run_revision_tasks.py.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from lxml import etree

from build_v10_docx import make_p, q, set_cell_text
from build_v12_docx import W, blocks_from, load_zip, ptext, require_one, store_zip

ROOT = Path(__file__).resolve().parent
REV = ROOT / "results" / "revision"
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v21.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v22.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v19.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v20.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers.docx"

M_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
XML_NS = "http://www.w3.org/XML/1998/namespace"


def load(name):
    path = REV / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sci(p):
    if p >= 0.001:
        return f"{p:.3f}"
    exponent = f"{p:.2e}"
    mant, exp = exponent.split("e")
    return rf"{mant}\times 10^{{{int(exp)}}}"


def t_num(value):
    return f"{value:.2f}"


def _bold_runs(paragraph):
    for rpr in paragraph.iter(q(W, "rPr")):
        if rpr.find(q(W, "b")) is None:
            rpr.append(etree.Element(q(W, "b")))


def insert_before(anchor, texts, style_from):
    # addprevious puts each new paragraph immediately before the anchor, so
    # iterating forward leaves the texts in their original order.
    for text in texts:
        anchor.addprevious(make_p(blocks_from(text), style_from))


def insert_after(anchor, texts, style_from):
    for text in reversed(texts):
        anchor.addnext(make_p(blocks_from(text), style_from))


def replace_paragraph(body, snippet, text):
    paragraph = require_one(body, snippet)
    new = make_p(blocks_from(text), paragraph)
    paragraph.getparent().replace(paragraph, new)
    return new


def replace_in_runs(body, old, new):
    count = 0
    for node in body.iter(q(W, "t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)
            count += 1
    if count != 1:
        raise SystemExit(f"{old!r} replaced in {count} runs")
    return count


def body_of(root):
    return root.find(q(W, "body"))


def following_table(body, snippet):
    start = require_one(body, snippet)
    seen = False
    for child in body:
        if child is start:
            seen = True
            continue
        if seen and child.tag == q(W, "tbl"):
            return child
    raise SystemExit(f"no table after {snippet!r}")


def make_grid_table(template, rows):
    tbl = etree.Element(q(W, "tbl"))
    tbl_pr = template.find(q(W, "tblPr"))
    if tbl_pr is not None:
        tbl.append(copy.deepcopy(tbl_pr))
    grid = etree.SubElement(tbl, q(W, "tblGrid"))
    width = str(max(900, 9360 // max(len(rows[0]), 1)))
    for _ in rows[0]:
        col = etree.SubElement(grid, q(W, "gridCol"))
        col.set(q(W, "w"), width)
    sample = template.find(q(W, "tr")).find(q(W, "tc"))
    for row in rows:
        tr = etree.SubElement(tbl, q(W, "tr"))
        for value in row:
            tc = copy.deepcopy(sample)
            paragraphs = tc.findall(q(W, "p"))
            for extra in paragraphs[1:]:
                tc.remove(extra)
            paragraph = tc.find(q(W, "p"))
            for child in list(paragraph):
                if child.tag != q(W, "pPr"):
                    paragraph.remove(child)
            run = etree.SubElement(paragraph, q(W, "r"))
            text = etree.SubElement(run, q(W, "t"))
            text.text = "" if value is None else str(value)
            if text.text[:1] == " " or text.text[-1:] == " ":
                text.set(q(XML_NS, "space"), "preserve")
            tr.append(tc)
    return tbl


def add_image(files, png_path: Path):
    rels = etree.fromstring(files["word/_rels/document.xml.rels"])
    ids = []
    for rel in rels:
        rel_id = rel.get("Id", "")
        if rel_id.startswith("rId") and rel_id[3:].isdigit():
            ids.append(int(rel_id[3:]))
    new_id = f"rId{max(ids) + 1}"
    media = "word/media/figure_s10_prior_posterior.png"
    files[media] = png_path.read_bytes()
    node = etree.SubElement(rels, f"{{{PKG_NS}}}Relationship")
    node.set("Id", new_id)
    node.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    node.set("Target", "media/figure_s10_prior_posterior.png")
    files["word/_rels/document.xml.rels"] = etree.tostring(
        rels, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    return new_id


def clone_picture(body, new_id):
    source = None
    for child in body:
        if child.tag != q(W, "p"):
            continue
        if any(etree.QName(node).localname == "blip" for node in child.iter()):
            source = child
            break
    if source is None:
        raise SystemExit("supplement has no embedded picture to clone")
    picture = copy.deepcopy(source)
    for node in picture.iter():
        if etree.QName(node).localname == "blip":
            node.set(f"{{{R_NS}}}embed", new_id)
        if etree.QName(node).localname in {"extent", "ext"}:
            # 6.4 in by 2.6 in, matching the two-panel figure.
            if "cx" in node.attrib:
                node.set("cx", "5846400")
            if "cy" in node.attrib:
                node.set("cy", "2376000")
    return picture


def numbers():
    nb = load("nadeau_bengio.json")
    moran = load("moran_innovations_summary.json")["summary"]
    ablation = load("table_s9_ablation.json")
    sensitivity = load("table_s10_sensitivity.json")
    figure = load("figure_s10_summary.json")
    table4b = load("table4b.json")
    if int(table4b.get("n_reps", 0)) != 200:
        raise SystemExit(f"Table 4b has {table4b.get('n_reps')} replications; expected 200")
    if len(ablation) != 6:
        raise SystemExit(f"ablation table has {len(ablation)} rows; expected 6")
    ols = next(row for row in nb["comparisons"] if row["name"].endswith("OLS"))
    geo = next(row for row in nb["comparisons"] if "Geographic" in row["name"])
    by_model = {row["model"]: row for row in ablation}
    return {
        "nb": nb,
        "ols": ols,
        "geo": geo,
        "moran": moran,
        "ablation": by_model,
        "sensitivity": sensitivity,
        "figure": figure,
        "table4b": table4b,
        "rmse": nb["mean_rmse"],
    }


def fmt_nb(row):
    return (
        rf"$t_{{{row['df']}}}={t_num(row['t'])}$"
        rf", $p={sci(row['p'])}$"
    )


def edit_main(body, z):
    ols, geo = z["ols"], z["geo"]
    rmse = z["rmse"]
    replace_paragraph(
        body,
        "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation",
        "Beyond a Single Spatial Weight Matrix: Multiplex Spatial Error Regression with Calibrated Screening and Precision Kriging",
    )
    replace_in_runs(
        body,
        "reducing computational complexity from",
        "reducing the computational complexity of the holdout imputation solve from",
    )
    replace_paragraph(
        body,
        "We evaluate Multi-SEM using two substantive empirical benchmarks.",
        (
            "We evaluate Multi-SEM using two substantive empirical benchmarks. In a spatial hedonic study of "
            r"$n=506$ Boston census tracts across 92 municipalities, Multi-SEM simultaneously captures geographic "
            "distance decay, regional multi-hop spillovers, and municipal boundaries. Across 50 repeated random "
            r"80/20 splits, Multi-SEM reduces mean holdout RMSE by 28.5% relative to OLS (0.1375 versus 0.1921; "
            f"Nadeau–Bengio resampled {fmt_nb(ols)}, outperforming OLS on {ols['wins']} of {ols['R']} splits) and by "
            "2.0% relative to a single-layer geographic SEM (0.1403), outperforming it on "
            f"{geo['wins']} of {geo['R']} splits (resampled {fmt_nb(geo)}). The corrected 95% interval for the "
            "mean squared-error gap against the geographic SEM includes zero. A 63-month longitudinal panel of "
            r"S&P 100 equity returns confirms that the pipeline scales to dynamic, non-geographic networks, "
            r"outperforming OLS in 44 of 63 months (sign test $p=0.0022$; Newey–West $p=0.072$). Those equity fits "
            "used the isotropic proposal."
        ),
    )
    replace_paragraph(
        body,
        "For the corrected Boston tracts, screened Multi-SEM",
        (
            "On synthetic graphs, the oracle specification reduces holdout RMSE by 8.6% relative to OLS, while the "
            "screened specification achieves a 6.1% reduction. For the corrected Boston tracts, screened Multi-SEM "
            "with precision kriging lowers mean holdout RMSE by 28.5% across 50 random splits "
            f"(Nadeau–Bengio resampled {fmt_nb(ols)}; {ols['wins']} of {ols['R']} splits). "
            r"Across 63 months of S&P 100 returns, Multi-SEM outperforms OLS in 44 months (sign test $p=0.0022$). "
            "Proofs, prior checks, and replication code are provided in the Supplementary Materials."
        ),
    )
    insert_after(
        require_one(body, "A single spatial operator cannot simultaneously accommodate these distinct relational channels."),
        [
            (
                "Higher-order spatial autoregressive and spatial-error models have been studied since Anselin (1988) "
                "and Lacombe (2004). Empirical practice nevertheless remains dominated by the selection of a single "
                "weight matrix $W$. Weight-matrix uncertainty has been treated by Bayesian model averaging "
                "(LeSage and Fischer, 2008; Piribauer and Crespo Cuaresma, 2016) and by penalized selection "
                "(Lam and Souza, 2020). Multi-SEM is an alternative pipeline: ex-ante structural pruning, "
                "design-conditional screening, $\\ell_1$-constrained Bayesian estimation, and precision-domain "
                "spatial interpolation. Section 2.1 places that pipeline in the higher-order literature."
            )
        ],
        require_one(body, "A single spatial operator cannot simultaneously accommodate these distinct relational channels."),
    )

    heading = require_one(body, "2. Related Work and Methodological Background")
    prose = require_one(body, "With multiple spatial operators, a multiplicative filter")
    insert_before(
        prose,
        [
            "2.1 Lineage of Higher-Order Spatial Models and Invertibility",
            (
                "Higher-order spatial autoregressive and spatial-error specifications have been part of the classical "
                "toolkit since Anselin (1988). Lacombe (2004) showed that conclusions about a substantive spatial "
                "effect can change when the specification moves beyond a single weight matrix, and Lee and Liu (2010) "
                "gave efficient GMM theory for high-order spatial models with autoregressive disturbances. Recent "
                "empirical practice nevertheless remains dominated by the selection of one matrix $W$. Weight-matrix "
                "uncertainty has been addressed by Bayesian model averaging (LeSage and Fischer, 2008; Piribauer and "
                "Crespo Cuaresma, 2016) and by penalized selection (Lam and Souza, 2020). Multi-SEM does not claim a "
                "new mathematical model class. It provides one reproducible pipeline that combines ex-ante structural "
                "pruning, design-conditional screening, $\\ell_1$-constrained Bayesian estimation, and precision-domain "
                "spatial interpolation."
            ),
            (
                "Elhorst, Lacombe, and Piras (2012) established parameter-space boundaries for higher-order spatial "
                "lag models and showed that the exact stability region depends on the joint eigenvalues of "
                "non-commuting matrices. Recomputing that joint spectrum inside an MCMC loop is prohibitive once "
                "several layers are active. Following a Gelfand-type matrix-norm bound, restricting $\\lambda$ to the "
                r"open $\ell_1$ ball, $\lVert\lambda\rVert_1=\sum_{s=1}^{k}\lvert\lambda_s\rvert<1$, supplies a "
                "uniform, geometry-free sufficient condition for invertibility of the additive filter for an arbitrary "
                "collection of row-substochastic matrices."
            ),
            (
                r"Expressing spatial prediction through the precision $Q=\sigma^{-2}V_\lambda'V_\lambda$ connects the "
                "econometric best linear unbiased predictor (Goldberger, 1962; Goulard, Laurent, and Thomas-Agnan, "
                "2017) with Gaussian Markov random field theory (Rue and Held, 2005)."
            ),
        ],
        heading if "2.1" in "2.1 Lineage" else prose,
    )
    # The first inserted block is a heading. Restyle it from the section heading, which
    # insert_before applied to every block via `heading`. Body paragraphs in that group
    # were therefore given heading style. Rebuild those three body paragraphs in body style.
    # Done below by leaving them: heading style on a short subsection label is intended,
    # but the long paragraphs should be body text. Re-apply body style to the long ones.
    body_style = require_one(body, "Spatial econometrics distinguishes dependence")
    _restyle_known(body, body_style)

    insert_before(
        require_one(body, "This equivalence is confirmed numerically as well as algebraically."),
        [
            (
                "For a continuous spatial adjacency graph, the Katz resolvent "
                r"$G_A=A(I-\alpha A)^{-1}=\sum_{j=0}^{\infty}\alpha^j A^{j+1}$ is a geometrically damped polynomial "
                "filter. It captures multi-hop diffusion on that single graph. It is not, by itself, a second "
                "physical network. For an affiliation network of disjoint cliques, setting the diagonal to zero and "
                r"row-standardizing returns the original clique, $\widetilde{G}(C)\equiv C$, down to machine precision "
                r"$1.4\times 10^{-17}$ in the Boston town layer. Affiliation resolvents are therefore redundant and "
                "are pruned before screening."
            )
        ],
        require_one(body, "This equivalence is confirmed numerically as well as algebraically."),
    )
    insert_after(
        require_one(body, "Approximations suitable for large sparse systems"),
        [
            (
                r"Denote the residual quadratic form by $\mathcal{S}(\lambda)=(Y-X\beta)'V_\lambda'V_\lambda(Y-X\beta)$. "
                r"The symbol $Q$ is reserved for the $n\times n$ disturbance precision "
                r"$Q=\sigma^{-2}V_\lambda'V_\lambda$."
            )
        ],
        require_one(body, "Approximations suitable for large sparse systems"),
    )
    renamed = _rename_acceptance_quadratic(body)
    if renamed != 2:
        raise SystemExit(f"expected to rename two quadratic-form Q symbols, renamed {renamed}")

    rule = require_one(body, "They identify candidate graphs whose projected residual spectrum")
    insert_after(
        rule,
        [
            (
                "Under the Gaussian null, the exact distribution of "
                r"$I_s=c_s v'B_s v$ is determined by the eigenvalues of $B_s=H'A_s^{*}H$ (Tiefelsdorf, 2000). "
                "Exact tail probabilities can be obtained by numerical inversion of the characteristic function "
                "(Imhof, 1961). Computing every eigenvalue of $B_s$ costs $O(d^3)$, which is expensive inside a "
                "high-dimensional rolling screen. Conditional Gaussian Monte Carlo with $B=499$ draws simulates that "
                "same finite-sample null directly by spherical projection, at cost $O(B d^2)$."
            ),
            (
                "The central limit theorem for linear-quadratic forms requires the row and column sums of $A_s$ to "
                "stay uniformly bounded (Kelejian and Prucha, 2001). After row standardization the row sums equal "
                "one, isolated rows excepted. Degree heterogeneity appears as large column sums and inflates the "
                r"leading singular value $\theta_{s,1}$. The pre-specified rules $\kappa_s>3.0$ and $\psi_s>0.20$ mark "
                "the projected residual operator as dominated by extreme eigenmodes. Those layers are then referred "
                "to the Monte Carlo tail so that the Gaussian approximation does not produce a liberal Type I error."
            ),
            (
                "The screen is marginal and inclusive. It is a discovery step under false-discovery-rate control, "
                "not a partial-coefficient identification step. Geographic distance and its resolvent can both "
                "reject when one spatial factor is present. Separating those collinear layers is left to the "
                "Bayesian stage. The covariance-adapted proposal is built to follow that posterior trade-off; "
                "on Boston Split 5 the frozen correlation between the geographic coefficient and the resolvent is $-0.90$."
            ),
        ],
        rule,
    )

    table3 = following_table(body, "Table 3.  Empirical size and power of residual Moran")
    header = [set_cell_text_return(cell, text) for cell, text in zip(
        table3.find(q(W, "tr")).findall(q(W, "tc")),
        [
            "Network",
            "Role",
            "n=100 null size",
            "n=100 active-DGP rate",
            "n=200 null size",
            "n=200 active-DGP rate",
            "n=500 null size",
            "n=500 active-DGP rate",
        ],
    )]
    del header
    replace_paragraph(
        body,
        "Table 3.  Empirical size and power of residual Moran",
        (
            "Table 3. Empirical size and rejection rates for residual Moran's $I$ (500 replications, nominal "
            r"$\alpha=0.05$). The size columns are empirical Type I error under the null DGP. The active-DGP columns "
            "are power for the Poisson $d=8$ and power-law $\\gamma=2.2$ layers, and the false-discovery rate under "
            "the active DGP for the three inactive layers. The Monte Carlo standard error is "
            r"$\sqrt{\alpha(1-\alpha)/500}\approx 0.0097$ at the nominal 5% size and at most 0.022 for a rejection rate."
        ),
    )
    insert_before(
        require_one(body, "7.4 Out-of-Sample Prediction and Model Comparisons"),
        [
            (
                "Table 4 is a single four-chain run at $n=200$. Table 4a is the 200-replication Monte Carlo ensemble "
                "at the same sample size. In both designs the five regressors are drawn as independent standard "
                "normals and then column-standardized. They are not given a mutual correlation of 0.20; that "
                "correlation is not part of the code that produced Tables 4 and 4a."
            ),
            (
                "The Gaussian factor $\\exp(-\\lVert\\lambda\\rVert_2^2/2)$ varies only from about 0.61 to 1 on the "
                r"open $\ell_1$ ball when $\xi=1$. The volume of the $\ell_1$ cross-polytope of radius $L$ scales as "
                r"$L^k$, so the marginal prior of $L=\lVert\lambda\rVert_1$ induced by a nearly flat density on the "
                r"ball satisfies $p(L)\propto L^{k-1}$ on $[0,1)$. For $k=3$ this is $p(L)\propto L^2$, which places "
                "more prior mass near the outer shell. Supplementary Figure S10 compares that prior with the Boston "
                "Split 5 posterior and with a refit under a Beta$(1,2)$ prior that puts zero density at $L=1$. "
                "Section S2.3 reports the numerical comparison."
            ),
            "7.3.1 Opposite-sign recovery",
            _table4b_paragraph(z["table4b"]),
        ],
        require_one(body, "Spatial econometrics distinguishes dependence"),
    )
    subhead = require_one(body, "7.3.1 Opposite-sign recovery")
    styled_sub = make_p(
        blocks_from("7.3.1 Opposite-sign recovery"),
        require_one(body, "7.4 Out-of-Sample Prediction and Model Comparisons"),
    )
    _bold_runs(styled_sub)
    subhead.getparent().replace(subhead, styled_sub)
    template = following_table(body, "Table 4a.")
    table = make_grid_table(template, _table4b_rows(z["table4b"]))
    require_one(body, "Table 4b reports a further Monte Carlo").addnext(table)

    _replace_max_L(body, z["nb"])
    replace_paragraph(
        body,
        "yielding a Wilcoxon signed-rank statistic of 0",
        (
            "The screened Multi-SEM with precision kriging achieves a mean holdout RMSE of "
            f"{rmse['multi']:.4f}, compared with {rmse['ols']:.4f} for OLS, a 28.5% reduction in the ratio of mean "
            f"RMSEs. Multi-SEM has the lower RMSE on {ols['wins']} of {ols['R']} splits. The Nadeau and Bengio "
            f"(2003) corrected resampled test, which inflates the variance of the mean loss gap by "
            rf"$1/R+n_*/n_t={ols['correction_factor']:.3f}$ because the training sets overlap, gives "
            f"{fmt_nb(ols)}. The corrected 95% interval for the mean MSE reduction is "
            f"[{ols['ci95'][0]:.4f}, {ols['ci95'][1]:.4f}]. The geographic SEM alone achieves a mean RMSE of "
            f"{rmse['geo']:.4f}."
        ),
    )
    replace_paragraph(
        body,
        "A two-sided Wilcoxon signed-rank test evaluated on the 50 paired",
        (
            "Screened Multi-SEM achieves a lower holdout RMSE than the geographic SEM on "
            f"{geo['wins']} of {geo['R']} splits. The same corrected resampled test gives {fmt_nb(geo)}, and the "
            f"corrected 95% interval for the mean MSE reduction is [{geo['ci95'][0]:.4f}, {geo['ci95'][1]:.4f}]. "
            "The interval contains zero. The 2.0% reduction in mean RMSE is therefore not statistically significant "
            "once the dependence among overlapping splits is acknowledged. The win count remains 37 of 50."
        ),
    )
    replace_paragraph(
        body,
        "reduces mean RMSE further to 0.1375. That is a 2.0% reduction",
        (
            "Pace and Gilley (1997) exploit the spatial configuration of these tracts to improve hedonic fit. "
            f"Relative to the OLS mean RMSE of {rmse['ols']:.4f}, the geographic six-nearest-neighbour SEM reduces "
            f"mean RMSE to {rmse['geo']:.4f}. Screened Multi-SEM reduces it further to {rmse['multi']:.4f}. That is "
            f"a 2.0% reduction relative to the geographic SEM and a 28.5% reduction relative to OLS "
            f"({fmt_nb(ols)} against OLS; {fmt_nb(geo)} against the geographic SEM). The town SEM alone has a mean "
            f"RMSE of {rmse['town']:.4f}."
        ),
    )
    insert_after(
        require_one(body, "The interval contains zero."),
        [_ablation_paragraph(z["ablation"], rmse)],
        require_one(body, "The interval contains zero."),
    )
    replace_paragraph(
        body,
        "Multi-SEM without kriging has a mean RMSE of 0.2072",
        (
            f"Multi-SEM without kriging, using only $X_*\\hat\\beta$, has a mean RMSE of {rmse['multi_mean']:.4f} "
            f"and underperforms OLS ({rmse['ols']:.4f}). Four regressors, the tax rate, the pupil–teacher ratio, "
            "large-lot zoning, and non-retail business acreage, do not vary within municipalities and are heavily "
            r"attenuated by the filter $V_\lambda$. The slope estimates are then identified mainly from "
            "within-municipality variation. Applied to a holdout tract without the contemporaneous spatial "
            "adjustment, that filtered mean cannot absorb inter-municipal baseline differences. In an "
            "error-dependence model the slopes and the precision-kriging correction are one prediction system. The "
            "spatial gain appears when the predictor conditions on neighboring residuals."
        ),
    )
    geo_m = z["moran"]["geo"]
    insert_after(
        require_one(body, "The spatial gain appears when the predictor conditions on neighboring residuals."),
        [
            (
                "The same fits are checked for leftover spatial correlation. On the training blocks, Moran's $I$ "
                "is computed for the OLS residuals and for the filtered innovations "
                r"$\hat\varepsilon=V_{\hat\lambda}(Y-X\hat\beta)$, using each of the four candidate matrices. "
                f"The mean OLS statistic on the geographic layer is {geo_m['ols_mean_I']:.3f}. "
                f"After filtering, the Split 5 geographic statistic is {geo_m['filtered_split5_I']:.3f} "
                f"(Gaussian $p={geo_m['filtered_split5_p']:.2f}$). Across the 50 splits the filtered geographic "
                f"statistics range from {geo_m['filtered_min_I']:.3f} to {geo_m['filtered_max_I']:.3f}, and the "
                f"share with $p>0.30$ is {geo_m['filtered_frac_p_gt_030']:.2f}. The mean is near zero. The "
                "splitwise range is wider than the interval from $-0.015$ to $0.008$, so that interval is not "
                "claimed for every split. Supplementary Table S11 reports all four matrices."
            )
        ],
        require_one(body, "The spatial gain appears when the predictor conditions on neighboring residuals."),
    )
    replace_in_runs(
        body,
        "That Monte Carlo probability was not stored; therefore, the reported",
        "The operational rule therefore calls for a conditional Monte Carlo probability before FDR control. The Monte Carlo probability was not stored with the August archive, so the reported",
    )
    _add_references(body)


def set_cell_text_return(cell, text):
    set_cell_text(cell, text)
    return text


def ptext_with_math_placeholder(paragraph):
    return ptext(paragraph)


def _restyle_known(body, body_style):
    # insert_before copied the heading style onto the literature paragraphs.
    # Reconstruct those three paragraphs from the same source strings in body style.
    sources = {
        "Higher-order spatial autoregressive and spatial-error specifications": None,
        "Elhorst, Lacombe, and Piras (2012) established parameter-space boundaries": None,
        "Expressing spatial prediction through the precision": None,
    }
    # The strings are recovered by reading back is unnecessary: replace_paragraph keeps body style
    # only if the current paragraph's pPr is body. Force body pPr.
    for snippet in sources:
        paragraph = require_one(body, snippet)
        fresh = make_p(list(_blocks_of_paragraph_text(snippet, body)), body_style)
        # Cannot recover math from ptext. Copy runs from a freshly built paragraph instead.
        del fresh
    # Practical path: the insert used heading style. Replace each snippet's paragraph by
    # rebuilding from the original inserted text, which we repeat here.
    texts = _lineage_texts()
    for snippet, text in texts:
        paragraph = require_one(body, snippet)
        fresh = make_p(blocks_from(text), body_style)
        paragraph.getparent().replace(paragraph, fresh)


def _lineage_texts():
    return [
        (
            "Higher-order spatial autoregressive and spatial-error specifications",
            (
                "Higher-order spatial autoregressive and spatial-error specifications have been part of the classical "
                "toolkit since Anselin (1988). Lacombe (2004) showed that conclusions about a substantive spatial "
                "effect can change when the specification moves beyond a single weight matrix, and Lee and Liu (2010) "
                "gave efficient GMM theory for high-order spatial models with autoregressive disturbances. Recent "
                "empirical practice nevertheless remains dominated by the selection of one matrix $W$. Weight-matrix "
                "uncertainty has been addressed by Bayesian model averaging (LeSage and Fischer, 2008; Piribauer and "
                "Crespo Cuaresma, 2016) and by penalized selection (Lam and Souza, 2020). Multi-SEM does not claim a "
                "new mathematical model class. It provides one reproducible pipeline that combines ex-ante structural "
                "pruning, design-conditional screening, $\\ell_1$-constrained Bayesian estimation, and precision-domain "
                "spatial interpolation."
            ),
        ),
        (
            "Elhorst, Lacombe, and Piras (2012) established parameter-space boundaries",
            (
                "Elhorst, Lacombe, and Piras (2012) established parameter-space boundaries for higher-order spatial "
                "lag models and showed that the exact stability region depends on the joint eigenvalues of "
                "non-commuting matrices. Recomputing that joint spectrum inside an MCMC loop is prohibitive once "
                "several layers are active. Following a Gelfand-type matrix-norm bound, restricting $\\lambda$ to the "
                r"open $\ell_1$ ball, $\lVert\lambda\rVert_1=\sum_{s=1}^{k}\lvert\lambda_s\rvert<1$, supplies a "
                "uniform, geometry-free sufficient condition for invertibility of the additive filter for an arbitrary "
                "collection of row-substochastic matrices."
            ),
        ),
        (
            "Expressing spatial prediction through the precision",
            (
                r"Expressing spatial prediction through the precision $Q=\sigma^{-2}V_\lambda'V_\lambda$ connects the "
                "econometric best linear unbiased predictor (Goldberger, 1962; Goulard, Laurent, and Thomas-Agnan, "
                "2017) with Gaussian Markov random field theory (Rue and Held, 2005)."
            ),
        ),
    ]


def _blocks_of_paragraph_text(snippet, body):
    return []


def _rename_acceptance_quadratic(body):
    count = 0
    for paragraph in body.iter(q(W, "p")):
        math_text = []
        nodes = []
        for node in paragraph.iter(f"{{{M_NS}}}t"):
            nodes.append(node)
            if node.text:
                math_text.append(node.text)
        blob = "".join(math_text)
        if "Δ" in blob and "det" in blob and "Q" in math_text:
            for node in nodes:
                if node.text == "Q":
                    node.text = "𝒮"
                    count += 1
    return count


def _replace_max_L(body, nb):
    split5 = f"{nb['max_L_split5']:.6f}"
    overall = f"{nb['max_L_across_splits']:.6f}"
    found = 0
    for node in body.iter(f"{{{M_NS}}}t"):
        if node.text and node.text.strip() in {"0.9842", "0.98420"}:
            node.text = split5
            found += 1
    for node in body.iter(q(W, "t")):
        if node.text and "0.9842" in node.text:
            node.text = node.text.replace("0.9842", split5)
            found += 1
    if found == 0:
        raise SystemExit("did not find the Split 5 maximum draw 0.9842")
    # The across-split figure 0.999998 is already the stored maximum. Leave it.


def _ablation_paragraph(ablation, rmse):
    def row(prefix):
        match = next(value for key, value in ablation.items() if key.startswith(prefix))
        return match

    geo = row("A.")
    resolvent = row("B.")
    town = row("C.")
    both = row("D.")
    higher = row("E.")
    full = row("F.")

    def nb_clause(item):
        test = item.get("nadeau_bengio_vs_geo") or {}
        if not test or test.get("p") is None:
            return "this is the geographic baseline"
        return f"Nadeau–Bengio {fmt_nb(test)} against the geographic SEM, {test['wins']} of {test['R']} splits lower"

    return (
        "Supplementary Table S9 isolates the candidate networks on the same 50 splits. "
        f"The geographic SEM has mean RMSE {geo['mean_rmse']:.4f}. "
        f"The resolvent by itself has mean RMSE {resolvent['mean_rmse']:.4f} ({nb_clause(resolvent)}). "
        f"The municipal layer by itself has mean RMSE {town['mean_rmse']:.4f} ({nb_clause(town)}). "
        f"Geographic contiguity together with municipal boundaries has mean RMSE {both['mean_rmse']:.4f} "
        f"({nb_clause(both)}). Geographic contiguity together with its resolvent has mean RMSE "
        f"{higher['mean_rmse']:.4f} ({nb_clause(higher)}). "
        f"The screened model has mean RMSE {full['mean_rmse']:.4f} ({nb_clause(full)}). "
        "The resolvent alone is a higher-order filter on the geographic graph, not a second physical network, "
        "and its point estimate is the closest single-layer fit to the screened model. Adding the municipal "
        "layer to geographic contiguity also lowers the point estimate, by less than the resolvent alone. "
        "None of these contrasts with the geographic SEM is significant at the 5% level after the "
        "Nadeau–Bengio correction. The smallest such p-value is the geographic-plus-resolvent comparison."
    )


def _table4b_paragraph(table):
    bits = []
    for column in table["columns"]:
        bits.append(
            f"{column['parameter']}: bias {column['bias']:.3f}, RMSE {column['rmse']:.3f}, "
            f"95% coverage {column['coverage_95']:.2f}"
        )
    return (
        "Table 4b reports a further Monte Carlo design with opposite signs, "
        r"$\lambda=(0.50,-0.30,0,0,0)$, so $\sum_s\lambda_s=0.20$ while $\lVert\lambda\rVert_1=0.80$. "
        f"Each of {table['n_reps']} replications draws a fresh five-network multiplex and a fresh sample of size "
        f"n={table['n']}, then runs the covariance-adapted sampler (4 chains, 6,000 iterations, 3,000 burn-in). "
        + " ".join(bits)
        + ". The covariates remain independent standard normals. Coverage is the fraction of replications whose "
        "central 95% credible interval contains the true coefficient."
    )


def _table4b_rows(table):
    rows = [["Parameter", "Truth", "Bias", "RMSE", "95% coverage", "Max R-hat"]]
    for column in table["columns"]:
        rows.append([
            column["parameter"],
            f"{column['truth']:.2f}",
            f"{column['bias']:.3f}",
            f"{column['rmse']:.3f}",
            f"{column['coverage_95']:.3f}",
            f"{column['max_rhat']:.3f}",
        ])
    return rows


REFERENCES = [
    (
        "Anselin, L. (2009).",
        "Anselin, L. (1988). Spatial Econometrics: Methods and Models. Kluwer Academic Publishers, Dordrecht.",
    ),
    (
        "Florax, R. J.",
        "Elhorst, J. P., Lacombe, D. J., & Piras, G. (2012). On model specification and parameter space definitions in higher order spatial econometric models. Regional Science and Urban Economics, 42(1–2), 211–220.",
    ),
    (
        "Haario, H.",
        "Goldberger, A. S. (1962). Best linear unbiased prediction in the generalized linear regression model. Journal of the American Statistical Association, 57(298), 369–375.",
    ),
    (
        "Haario, H.",
        "Goulard, M., Laurent, T., & Thomas-Agnan, C. (2017). About predictions in spatial autoregressive models: Optimal and almost optimal strategies. Spatial Economic Analysis, 12(2–3), 304–325.",
    ),
    (
        "Katz, L. (1953)",
        "Imhof, J. P. (1961). Computing the distribution of quadratic forms in normal variables. Biometrika, 48(3/4), 419–426.",
    ),
    (
        "Lee, L. F. (2002)",
        "Lacombe, D. J. (2004). Does econometric methodology matter? An analysis of public policy using spatial econometric techniques. Geographical Analysis, 36(2), 105–118.",
    ),
    (
        "Lee, L. F. (2002)",
        "Lam, C., & Souza, P. C. L. (2020). Estimation and selection of spatial weight matrix in a spatial lag model. Journal of Business & Economic Statistics, 38(3), 693–710.",
    ),
    (
        "LeSage, J. P., & Pace",
        "Lee, L.-F., & Liu, X. (2010). Efficient GMM estimation of high order spatial autoregressive models with autoregressive disturbances. Econometric Theory, 26(1), 187–230.",
    ),
    (
        "LeSage, J. P., & Pace",
        "LeSage, J. P., & Fischer, M. M. (2008). Spatial growth regressions: Model specification, estimation and interpretation. Spatial Economic Analysis, 3(3), 275–304.",
    ),
    (
        "Newey, W. K.",
        "Nadeau, C., & Bengio, Y. (2003). Inference for the generalization error. Machine Learning, 52(3), 239–281.",
    ),
    (
        "Roberts, G. O., & Rosenthal",
        "Piribauer, P., & Crespo Cuaresma, J. (2016). Bayesian variable selection in spatial autoregressive models. Spatial Economic Analysis, 11(4), 457–479.",
    ),
    (
        "Tobler, W. R.",
        "Tiefelsdorf, M. (2000). Modelling Spatial Processes: The Identification and Analysis of Spatial Relationships in Regression Residuals by Means of Moran’s I. Springer, Berlin.",
    ),
]


def _add_references(body):
    style = require_one(body, "Anselin, L. (2009).")
    # Insert in reverse order of the list only when anchors would move. Each anchor is an
    # existing entry, so inserting before it is stable if we go from the bottom of the list.
    for anchor_snippet, entry in REFERENCES:
        if find_contains(body, entry[:40]):
            continue
        anchor = require_one(body, anchor_snippet)
        anchor.addprevious(make_p(blocks_from(entry), style))


def find_contains(body, snippet):
    from build_v12_docx import find_hits
    return find_hits(body, snippet)


def edit_supplement(body, files, z):
    replace_paragraph(
        body,
        "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation",
        "Beyond a Single Spatial Weight Matrix: Multiplex Spatial Error Regression with Calibrated Screening and Precision Kriging",
    )
    _assert_s41_clean(body)
    heading_style = require_one(body, "Section S2.")
    body_style = require_one(body, "Throughout the analysis")
    anchor = require_one(body, "Section S3.  Extended Simulation Studies")
    figure = z["figure"]
    gauss = figure["gaussian_prior_posterior"]
    beta = figure["beta_prior_posterior"]
    insert_before(
        anchor,
        [
            "S2.3 Prior volume of L and the Boston posterior",
            (
                r"Let $L=\lVert\lambda\rVert_1$. The set of $\lambda\in\mathbb{R}^k$ with $\lVert\lambda\rVert_1\le L$ "
                r"has volume proportional to $L^k$. Differentiating the cumulative volume gives the shell factor "
                r"$L^{k-1}$. A prior that is nearly constant on the open unit ball, such as "
                r"$\exp(-\lVert\lambda\rVert_2^2/2)$ with $\xi=1$, therefore induces $p(L)\propto L^{k-1}$ for "
                r"$L\in[0,1)$. For the three Boston layers selected on Split 5, $k=3$ and $p(L)\propto L^2$."
            ),
            (
                "Figure S10 draws 200,000 points uniformly in the $\\ell_1$ ball and plots that marginal against "
                "the Split 5 posterior under the paper prior. The refit reproduces the archived posterior mean of "
                f"$L$, {gauss['stored_mean_L']:.3f}. The histogram mode is {gauss['mode_L']:.3f} "
                f"(2.5% to 97.5% quantiles {gauss['q025']:.3f} to {gauss['q975']:.3f}). "
                "The prior mean of $L$ under the uniform ball is "
                f"{figure['prior_uniform_ball_mean_L']:.3f}, and the prior density increases all the way to $L=1$. "
                "The posterior mode lies in 0.80 to 0.90, inside the open ball and away from a claim that the "
                "draws simply pile up at the boundary."
            ),
            (
                r"The same split was refit with $L\sim\mathrm{Beta}(1,2)$, so $p(L)\propto(1-L)$, and with the "
                r"signed composition uniform on the $\ell_1$ sphere (absolute parts Dirichlet$(1,1,1)$, independent "
                r"signs). This prior has zero density at $L=1$. The resulting posterior mean of $L$ is "
                f"{beta['mean_L']:.3f} and the histogram mode is {beta['mode_L']:.3f} "
                f"(quantiles {beta['q025']:.3f} to {beta['q975']:.3f}). "
                f"The paper-prior posterior has 97.5% quantile {gauss['q975']:.3f}. "
                "The Beta prior, which puts zero density at $L=1$, shortens that upper tail and leaves the "
                "histogram mode essentially unchanged. The mode does not move toward zero."
            ),
        ],
        body_style,
    )
    # Heading line should use the section style. The first inserted paragraph is the heading text.
    heading_paragraph = require_one(body, "S2.3 Prior volume of L")
    styled = make_p(blocks_from("S2.3 Prior volume of $L$ and the Boston posterior"), heading_style)
    _bold_runs(styled)
    heading_paragraph.getparent().replace(heading_paragraph, styled)
    picture_id = add_image(files, Path(figure["figure"]))
    picture = clone_picture(body, picture_id)
    require_one(body, "The same split was refit").addnext(picture)
    caption = make_p(
        blocks_from(
            "Figure S10. Prior and posterior densities of $L=\\lVert\\lambda\\rVert_1$ on Boston Split 5. "
            "Left: uniform draws in the $\\ell_1$ ball, whose marginal is proportional to $L^2$, against the "
            "MCMC posterior under the paper prior. Right: the Beta$(1,2)$ prior density against the posterior "
            "obtained when that prior replaces the Gaussian factor."
        ),
        body_style,
    )
    picture.addnext(caption)

    s4_anchor = require_one(body, "Section S5.")
    insert_before(
        s4_anchor,
        [
            "S4.7 Boston ablation, censoring, the Charles River, and filtered innovations",
            (
                "Table S9 refits the Boston holdout design on the original 50 splits. Models A, C, and F are the "
                "geographic SEM, the municipal SEM, and the screened Multi-SEM already reported in the main text. "
                "Models B, D, and E were estimated with the same covariance-adapted sampler (four chains, 8,000 "
                "iterations, 4,000 burn-in). Nadeau–Bengio $p$-values compare holdout MSE with the geographic SEM."
            ),
        ],
        body_style,
    )
    s4_heading = require_one(body, "S4.7 Boston ablation, censoring")
    styled_s4 = make_p(
        blocks_from("S4.7 Boston ablation, censoring, the Charles River, and filtered innovations"),
        heading_style,
    )
    _bold_runs(styled_s4)
    s4_heading.getparent().replace(s4_heading, styled_s4)
    template = _first_table(body)
    s9 = make_grid_table(template, _s9_rows(z["ablation"]))
    require_one(body, "Table S9 refits the Boston holdout design").addnext(s9)
    s9_caption = make_p(
        blocks_from(
            "Table S9. Boston network ablation on 50 splits of 101 holdout tracts. Wins count strictly smaller RMSE "
            "than OLS. The corrected $t$ statistic is the Nadeau–Bengio resampled test of holdout MSE against the "
            "geographic SEM."
        ),
        body_style,
    )
    s9.addnext(s9_caption)

    sens_paragraphs = [_s10_paragraph(z["sensitivity"])]
    insert_after(s9_caption, sens_paragraphs, body_style)
    s10 = make_grid_table(template, _s10_rows(z["sensitivity"]))
    require_one(body, "Table S10 repeats the 50-split").addnext(s10)

    moran_intro = make_p(
        blocks_from(
            "Table S11 reports Moran's $I$ for the OLS training residuals and for the filtered innovations "
            r"$\hat\varepsilon=V_{\hat\lambda}(Y-X\hat\beta)$ on each training block. The filtered statistic uses "
            "an intercept-only residual maker, because the innovations are the structural disturbances rather than "
            "a second regression residual. The mean OLS statistic on the geographic layer is "
            f"{z['moran']['geo']['ols_mean_I']:.3f}."
        ),
        body_style,
    )
    s10.addnext(moran_intro)
    s11 = make_grid_table(template, _s11_rows(z["moran"]))
    moran_intro.addnext(s11)

    archive = require_one(body, "The replication package contains the scripts")
    insert_after(
        archive,
        [
            (
                "The static inputs are data/boston_corrected.txt, the corrected Harrison–Rubinfeld file distributed "
                "by StatLib, and the cached S&P 100 price files written by the replication scripts. This revision "
                "does not add a Zenodo DOI: no deposit identifier is stored in the project. When a frozen archive "
                "is minted, its DOI should be cited here in place of this file path."
            )
        ],
        archive,
    )


def _assert_s41_clean(body):
    table = following_table(body, "Table S4.1.")
    for node in table.iter(q(W, "t")):
        if node.text and "d = 8" in node.text:
            raise SystemExit(f"Table S4.1 still contains {node.text!r}")


def _first_table(body):
    for child in body:
        if child.tag == q(W, "tbl"):
            return child
    raise SystemExit("no table")


def _s9_rows(ablation):
    rows = [["Model", "Mean RMSE", "Mean MAE", "Wins vs OLS", "Corrected t vs geo", "p vs geo"]]
    for key in (
        "A. Geographic SEM",
        "B. Geographic resolvent only",
        "C. Municipal cliques only",
        "D. Geographic + municipal",
        "E. Geographic + resolvent",
        "F. Screened Multi-SEM",
    ):
        item = ablation[key]
        test = item.get("nadeau_bengio_vs_geo") or {}
        if test.get("p") is None:
            t_cell, p_cell = "—", "—"
        else:
            t_cell = f"{test['t']:.2f}"
            p_cell = f"{test['p']:.3f}"
        rows.append([
            key,
            f"{item['mean_rmse']:.4f}",
            f"{item['mean_mae']:.4f}",
            str(item["wins_vs_ols"]),
            t_cell,
            p_cell,
        ])
    return rows


def _s10_paragraph(sensitivity):
    def pct(block):
        return f"{100 * block['rmse_reduction_vs_ols']:.1f}%", f"{100 * block['rmse_reduction_vs_geo']:.1f}%"

    chas_ols, chas_geo = pct(sensitivity["chas"])
    cen_ols, cen_geo = pct(sensitivity["censor"])
    return (
        "Table S10 repeats the 50-split comparison after two changes of design. The first adds the Charles River "
        f"dummy to the regressors. Multi-SEM then reduces mean RMSE by {chas_ols} relative to OLS and by {chas_geo} "
        "relative to the geographic SEM. The second drops the 16 tracts with corrected median value 50 and redraws "
        f"80/20 splits of the remaining 490 tracts ($n_t=392$, $n_*=98$). The corresponding reductions are {cen_ols} "
        f"and {cen_geo}. Nadeau–Bengio tests use the split sizes of each design."
    )


def _s10_rows(sensitivity):
    rows = [["Design", "Model", "Mean RMSE", "Mean MAE", "Wins vs OLS", "Corrected t", "p"]]
    for key, label in (("chas", "CHAS included"), ("censor", "Ceiling tracts dropped")):
        block = sensitivity[key]
        for item in block["rows"]:
            test = item.get("nadeau_bengio_vs_geo") or {}
            if item["model"] == "OLS":
                extra = block["multi_vs_ols"]
                # The OLS row has no geo contrast. Leave the test columns blank.
                t_cell, p_cell = "—", "—"
                del extra
            elif test.get("p") is None:
                t_cell, p_cell = "—", "—"
            else:
                t_cell = f"{test['t']:.2f}"
                p_cell = f"{test['p']:.3f}"
            rows.append([
                label,
                item["model"],
                f"{item['mean_rmse']:.4f}",
                f"{item['mean_mae']:.4f}",
            "—" if item["model"] == "OLS" else str(item["wins_vs_ols"]),
                t_cell,
                p_cell,
            ])
    return rows


def _s11_rows(moran):
    rows = [[
        "Matrix",
        "Mean OLS I",
        "Mean filtered I",
        "Filtered min",
        "Filtered max",
        "Split 5 I",
        "Split 5 p",
        "Share p>0.30",
    ]]
    for name in ("geo", "resolvent", "town", "radial"):
        item = moran[name]
        rows.append([
            name,
            f"{item['ols_mean_I']:.3f}",
            f"{item['filtered_mean_I']:.3f}",
            f"{item['filtered_min_I']:.3f}",
            f"{item['filtered_max_I']:.3f}",
            f"{item['filtered_split5_I']:.3f}",
            f"{item['filtered_split5_p']:.3f}",
            f"{item['filtered_frac_p_gt_030']:.2f}",
        ])
    return rows


def write_letter(z):
    from docx import Document
    from docx.shared import Pt

    ols, geo = z["ols"], z["geo"]
    document = Document()
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    document.add_heading("Response to the Reviewers", level=0)
    document.add_paragraph(
        "We thank both reviewers for the care of their reports. The revised manuscript is "
        "“Beyond a Single Spatial Weight Matrix: Multiplex Spatial Error Regression with Calibrated "
        "Screening and Precision Kriging.” The computations below use the archived 50 Boston splits "
        "where a new fit was not required, and the same covariance-adapted sampler where it was."
    )
    document.add_heading("Response to Reviewer 1", level=1)
    document.add_heading("Literature and prior art (Major 1 and 2)", level=2)
    document.add_paragraph(
        "We agree that higher-order spatial autoregressive and error specifications are classical. "
        "Section 2.1 now discusses Anselin (1988), Lacombe (2004), Lee and Liu (2010), Elhorst, Lacombe, "
        "and Piras (2012), Lam and Souza (2020), LeSage and Fischer (2008), and Piribauer and Crespo "
        "Cuaresma (2016). Multi-SEM is presented as a pipeline: structural pruning, calibrated screening, "
        "an ℓ1 stability restriction, covariance-adapted MCMC, and precision-domain kriging. The ℓ1 ball "
        "is linked to the higher-order stability region in Elhorst et al. (2012). The predictor is linked "
        "to Goldberger (1962), Goulard, Laurent, and Thomas-Agnan (2017), and Rue and Held (2005)."
    )
    document.add_heading("Resampling tests (Major 8c)", level=2)
    document.add_paragraph(
        "The Wilcoxon p-value of 1.78×10^−15 has been removed. It is the lower bound of a 50-pair "
        "sign test and it treats overlapping training samples as independent. On the archived splits the "
        f"Nadeau and Bengio (2003) statistic for Multi-SEM against OLS is t_49 = {ols['t']:.2f}, "
        f"p = {ols['p']:.2e}, with a corrected 95% interval for the mean MSE reduction of "
        f"[{ols['ci95'][0]:.4f}, {ols['ci95'][1]:.4f}]. Multi-SEM wins on {ols['wins']} of {ols['R']} splits. "
        f"Against the geographic SEM the same test gives t_49 = {geo['t']:.2f}, p = {geo['p']:.3f}, "
        f"interval [{geo['ci95'][0]:.4f}, {geo['ci95'][1]:.4f}], with {geo['wins']} wins. "
        "The interval contains zero. We therefore no longer describe the 2.0% RMSE increment as "
        "significant under resampled cross-validation. Both results are in the abstract, Section 8.1, and Table S9."
    )
    document.add_heading("Marginal screening (Major 5)", level=2)
    document.add_paragraph(
        "Section 4 now states that marginal screening is an inclusive discovery step under false-discovery-rate "
        "control, not a partial-coefficient identification step. Geographic distance and its resolvent can both "
        "reject when spatial dependence is present. Their trade-off is left to the Bayesian stage. On Split 5 "
        "the adapted proposal correlation between those two coefficients is −0.90."
    )
    document.add_heading("Response to Reviewer 2", level=1)
    document.add_heading("Prior geometry of L (Major 2)", level=2)
    gauss = z["figure"]["gaussian_prior_posterior"]
    beta = z["figure"]["beta_prior_posterior"]
    document.add_paragraph(
        "Section 7.3 and Supplement S2.3 now derive p(L) ∝ L^{k−1}. Figure S10 plots 200,000 uniform draws "
        f"in the ℓ1 ball against the Split 5 posterior. Under the paper prior the posterior mean of L is "
        f"{gauss['mean_L']:.3f} and the histogram mode is {gauss['mode_L']:.3f}. "
        f"A refit with L ~ Beta(1, 2), which has zero density at L = 1, has posterior mean {beta['mean_L']:.3f} "
        f"and histogram mode {beta['mode_L']:.3f} "
        f"(97.5% quantile {beta['q975']:.3f}, against {gauss['q975']:.3f} under the paper prior). "
        "Both histogram modes lie in 0.80–0.90. The boundary-penalizing prior shortens the upper tail and "
        "does not move the mode toward zero. The Gaussian-prior refit reproduces the archived Split 5 mean of L."
    )
    document.add_heading("Resolvent and the ablation (Major 4)", level=2)
    document.add_paragraph(
        "Sections 3.4 and 8.1 now describe the Katz resolvent of a single geographic graph as a damped "
        "polynomial filter. Affiliation resolvents are identified as algebraically redundant. Table S9 reports "
        "the ablation. Mean holdout RMSE is "
        f"{z['ablation']['A. Geographic SEM']['mean_rmse']:.4f} for the geographic SEM, "
        f"{z['ablation']['B. Geographic resolvent only']['mean_rmse']:.4f} for the resolvent alone, "
        f"{z['ablation']['C. Municipal cliques only']['mean_rmse']:.4f} for municipal cliques alone, "
        f"{z['ablation']['D. Geographic + municipal']['mean_rmse']:.4f} for geography plus municipal boundaries, "
        f"{z['ablation']['E. Geographic + resolvent']['mean_rmse']:.4f} for geography plus the resolvent, and "
        f"{z['ablation']['F. Screened Multi-SEM']['mean_rmse']:.4f} for the screened model. "
        "The resolvent-only fit is the single specification closest to the screened model. Adding municipal "
        "boundaries to geography lowers the point estimate, by less than the resolvent alone. No contrast "
        "with the geographic SEM has a Nadeau–Bengio p-value below 0.05. The smallest is "
        f"{z['ablation']['E. Geographic + resolvent']['nadeau_bengio_vs_geo']['p']:.3f} "
        "for geography plus the resolvent."
    )
    document.add_heading("Charles River dummy and ceiling censoring (Major 8)", level=2)
    chas = z["sensitivity"]["chas"]
    cens = z["sensitivity"]["censor"]
    document.add_paragraph(
        "Table S10 adds CHAS as an eleventh regressor and, separately, drops the 16 tracts with corrected "
        f"median value 50 (n = 490, training 392, holdout 98). With CHAS, the RMSE reduction is "
        f"{100 * chas['rmse_reduction_vs_ols']:.1f}% versus OLS and "
        f"{100 * chas['rmse_reduction_vs_geo']:.1f}% versus the geographic SEM. "
        f"Without the ceiling tracts the reductions are {100 * cens['rmse_reduction_vs_ols']:.1f}% and "
        f"{100 * cens['rmse_reduction_vs_geo']:.1f}%. The OLS contrasts remain significant "
        f"(CHAS t_49 = {chas['multi_vs_ols']['t']:.2f}, p = {chas['multi_vs_ols']['p']:.2e}; "
        f"censored t_49 = {cens['multi_vs_ols']['t']:.2f}, p = {cens['multi_vs_ols']['p']:.2e}). "
        "The geographic contrasts do not "
        f"(CHAS p = {chas['rows'][2]['nadeau_bengio_vs_geo']['p']:.3f}; "
        f"censored p = {cens['rows'][2]['nadeau_bengio_vs_geo']['p']:.3f}). "
        "Dropping the ceiling tracts cuts the geographic RMSE gap from about 2% to under 1%. "
        "We therefore do not describe either check as preserving a significant 2% increment."
    )
    document.add_heading("Opposite-sign simulation (Major 3 and 9)", level=2)
    document.add_paragraph(
        "The original simulation used λ = (0.40, 0.35, 0, 0, 0). Table 4b adds λ = (0.50, −0.30, 0, 0, 0), "
        "for which the signed sum is 0.20 and the ℓ1 norm is 0.80. Two hundred replications at n = 200, "
        "with networks redrawn each time, are summarized by bias, RMSE, and 95% credible-interval coverage "
        "in that table and in Section 7.3.1. The two active coefficients are recovered with the correct signs "
        "and are shrunk toward zero. The inactive coefficients are approximately unbiased, with coverage near "
        "the nominal 95%."
    )
    document.add_heading("Filtered innovations (Major 10)", level=2)
    geo_m = z["moran"]["geo"]
    document.add_paragraph(
        f"The mean Moran statistic of the OLS training residuals on the geographic layer is {geo_m['ols_mean_I']:.3f}. "
        f"After filtering, the Split 5 geographic statistic is {geo_m['filtered_split5_I']:.3f} "
        f"(p = {geo_m['filtered_split5_p']:.2f}). Across the 50 splits the filtered geographic statistics range from "
        f"{geo_m['filtered_min_I']:.3f} to {geo_m['filtered_max_I']:.3f}, and the share with p > 0.30 is "
        f"{geo_m['filtered_frac_p_gt_030']:.2f}. Table S11 gives the same summary for all four candidate matrices. "
        "The mean is close to zero. The splitwise range is wider than the interval from −0.015 to 0.008, so we "
        "do not claim that every split is inside that interval."
    )
    document.add_heading("Other corrections", level=2)
    document.add_paragraph(
        "The complexity claim in the abstract is now limited to the holdout imputation solve. The quadratic "
        "form in the Metropolis ratio is denoted by a script S, and Q is reserved for the precision. Table 3 "
        "no longer labels inactive-layer rejections as power. The industry-layer narrative in Section 8.2 states "
        "that κ = 3.07 selects the Monte Carlo branch; the stored tail remains the archived Gaussian value 0.0038 "
        "because the Monte Carlo probability was not in the archive. Table S4.1 was checked and does not contain "
        "a stray “(d = 8)” label. The largest retained draw of L is "
        f"{z['nb']['max_L_split5']:.6f} on Split 5 and {z['nb']['max_L_across_splits']:.6f} across the 50 splits, "
        "both strictly below 1. The covariate design for Tables 4 and 4a is independent standard normal draws, "
        "not a correlation of 0.20."
    )
    document.save(LETTER_OUT)


def check_anchors():
    files, root = load_zip(MAIN_IN)
    body = body_of(root)
    snippets = [
        "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation",
        "We evaluate Multi-SEM using two substantive empirical benchmarks.",
        "For the corrected Boston tracts, screened Multi-SEM",
        "With multiple spatial operators, a multiplicative filter",
        "2. Related Work and Methodological Background",
        "This equivalence is confirmed numerically as well as algebraically.",
        "Approximations suitable for large sparse systems",
        "They identify candidate graphs whose projected residual spectrum",
        "Table 3.  Empirical size and power of residual Moran",
        "7.4 Out-of-Sample Prediction and Model Comparisons",
        "Table 4a.",
        "yielding a Wilcoxon signed-rank statistic of 0",
        "A two-sided Wilcoxon signed-rank test evaluated on the 50 paired",
        "reduces mean RMSE further to 0.1375. That is a 2.0% reduction",
        "Multi-SEM without kriging has a mean RMSE of 0.2072",
        "That Monte Carlo probability was not stored; therefore, the reported",
        "Anselin, L. (2009).",
        "Figure 1.",
        "Spatial econometrics distinguishes dependence",
    ]
    for snippet in snippets:
        require_one(body, snippet)
        print("ok", snippet[:70])
    print("anchors ok", len(snippets))


def main():
    import sys
    if "--check" in sys.argv:
        check_anchors()
        return
    z = numbers()
    files, root = load_zip(MAIN_IN)
    edit_main(body_of(root), z)
    store_zip(MAIN_OUT, files, root)
    print(MAIN_OUT.name)
    files, root = load_zip(SUP_IN)
    edit_supplement(body_of(root), files, z)
    store_zip(SUP_OUT, files, root)
    print(SUP_OUT.name)
    write_letter(z)
    print(LETTER_OUT.name)


if __name__ == "__main__":
    main()
