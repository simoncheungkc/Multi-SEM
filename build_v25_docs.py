"""Manuscript v25 and supplement v23.

Text and typesetting revisions from the pre-submission list. Does not
overwrite v24 or supplement v22. Does not replace the Section 5.3 sampler
disclaimer: Tables 4, 4a, 5, and 11–13 were not recomputed with the
covariance-adapted proposal. Does not invent a Zenodo DOI.
"""

from __future__ import annotations

import json
from pathlib import Path

from lxml import etree

from build_revision_docs import insert_after, replace_paragraph
from build_v10_docx import make_p, q
from build_v12_docx import W, blocks_from, load_zip, ptext, require_one, store_zip
from mnreg.latex_omml import el

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v24.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v25.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v22.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v23.docx"
BETA = ROOT / "results" / "revision" / "august_beta_l1.json"

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

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
    "on hub-dominated networks prior to false discovery rate control. For contemporaneous imputation, "
    "the conditional predictor is formulated directly through the holdout block of the full-graph "
    "precision matrix, avoiding explicit construction and inversion of the training covariance. "
    "In an empirical application to $n = 506$ census tracts across 92 "
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
    "stabilizing, and estimating multiple non-commuting spatial matrices. Multi-SEM addresses this "
    "empirical challenge through a unified workflow combining structural pruning, calibrated screening, "
    r"$\ell_1$-constrained Bayesian estimation, and precision-domain spatial interpolation."
)

OUTCOME = (
    "The spatial application uses the corrected Harrison–Rubinfeld data (Harrison and Rubinfeld, 1978; "
    "Gilley and Pace, 1996; Pace and Gilley, 1997). The data contain 506 census tracts nested within "
    "92 municipalities. The outcome is $Y=$ ln(cmedv), where cmedv is measured in thousands of dollars. "
    "Sixteen tracts remain at the reporting ceiling of 50, so their outcomes are censored at ln(50)."
)

DESIGN = (
    "The design matrix includes an intercept and ten regressors: rooms in levels, age, crime, "
    "large-lot zoning, non-retail business acreage, ln(lstat), the property-tax rate, the "
    "pupil–teacher ratio, nitric oxides, and ln(dis). Charles River adjacency and the "
    "racial-composition variable are excluded. For each train–holdout split, the ten regressors are "
    "standardized using the training-sample means and population standard deviations, and the same "
    "constants are then applied to the holdout observations. Longitude and latitude are projected to "
    "WGS84 UTM zone 19N before distances are calculated. Easting ranges from 310,852 m to 350,236 m, "
    "and northing from 4,654,767 m to 4,693,945 m."
)

AUGUST = (
    "Because the exploratory screen for this archived snapshot used all $n = 100$ equities, the "
    "subsequent 80/20 partition serves solely as an in-sample-screened illustration of MCMC mixing "
    "and precision-kriging mechanics; all out-of-sample predictive validation is based exclusively "
    "on the training-only 63-month rolling panel."
)

SCOPE = (
    "Two methodological scope boundaries merit note. First, inference is deliberately restricted to "
    r"the geometry-free sufficient $\ell_1$ stability ball rather than the larger, non-convex exact "
    "admissible region (Elhorst et al., 2012), avoiding costly joint-eigenvalue checks inside each "
    "Metropolis step. Second, because the induced-subgraph estimator yields a pseudo-posterior for "
    "the re-standardized training operator, our empirical evaluation focuses on point imputation; "
    "extending the framework to fully coherent posterior predictive intervals and proper scoring rules "
    "via Schur-complement likelihoods or missing-data augmentation remains a promising avenue for "
    "future work."
)

AVAILABILITY = (
    "Data and code availability. The replication code and the frozen inputs data/boston_corrected.txt "
    "and data/sp100_clean_snapshot.csv are the files used for this revision. The numerical results "
    "are reproduced from those project files, not from a live Yahoo Finance download."
)

AUGUST_SUP = (
    "Thus, the August results illustrate posterior concentration of the network coefficients inside "
    "the stability ball and the mechanics of contemporaneous precision-kriging imputation. The "
    "exploratory screen for this snapshot used all 100 equities, so the 20-firm comparison is an "
    "in-sample-screened illustration. Out-of-sample predictive validation is the training-only "
    "63-month panel."
)


def replace_runs(body, old, new, expected):
    count = 0
    for node in body.iter(q(W, "t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)
            count += 1
    for node in body.iter(q(M, "t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)
            count += 1
    if count != expected:
        raise SystemExit(f"{old!r} replaced in {count} runs, expected {expected}")


def delete_paragraph(body, snippet):
    paragraph = require_one(body, snippet)
    paragraph.getparent().remove(paragraph)


def math_concat(paragraph):
    return "".join((node.text or "") for node in paragraph.iter(q(M, "t")))


def set_plain(math_run):
    rpr = math_run.find(q(M, "rPr"))
    if rpr is None:
        rpr = el("rPr")
        math_run.insert(0, rpr)
    sty = rpr.find(q(M, "sty"))
    if sty is None:
        rpr.insert(0, el("sty", val="p"))
    else:
        sty.set(q(M, "val"), "p")


def upright_gamma(body):
    count = 0
    for node in list(body.iter(q(M, "t"))):
        if node.text and "γ" in node.text:
            set_plain(node.getparent())
            count += 1
    for paragraph in list(body.iter(q(W, "p"))):
        if "γ" not in ptext(paragraph):
            continue
        pieces = ptext(paragraph).split("γ")
        new = etree.Element(q(W, "p"))
        props = paragraph.find(q(W, "pPr"))
        if props is not None:
            new.append(props)
        for index, piece in enumerate(pieces):
            if piece:
                sample = make_p([("t", piece)], paragraph)
                run = sample.find(q(W, "r"))
                new.append(run)
            if index < len(pieces) - 1:
                new.append(upright_gamma_math())
        paragraph.getparent().replace(paragraph, new)
        count += 1
    return count


def upright_gamma_math():
    math = el("oMath")
    run = el("r")
    props = el("rPr")
    props.append(el("sty", val="p"))
    run.append(props)
    text = el("t")
    text.text = "γ"
    run.append(text)
    math.append(run)
    return math


def upright_caption_greek(body):
    caption = require_one(body, "Table 10. Residual Moran's")
    count = 0
    for node in caption.iter(q(M, "t")):
        if node.text and ("κ" in node.text or "ψ" in node.text):
            set_plain(node.getparent())
            count += 1
    if count < 3:
        raise SystemExit(f"Table 10 caption has {count} kappa/psi runs")
    return count


def italic_s_not_script(body):
    """Drop the script alphabet on S next to lambda. Cambria Math script S renders like delta."""
    count = 0
    for math in body.iter(q(M, "oMath")):
        letters = [node.text or "" for node in math.iter(q(M, "t"))]
        if not any("λ" in letter for letter in letters):
            continue
        for node in math.iter(q(M, "t")):
            if node.text != "S":
                continue
            run = node.getparent()
            props = run.find(q(M, "rPr"))
            if props is None:
                continue
            script = props.find(q(M, "scr"))
            if script is None or script.get(q(M, "val")) != "script":
                continue
            props.remove(script)
            count += 1
    if count < 1:
        raise SystemExit("no script S next to lambda was found")
    return count


def replace_no_kriging_predictor(body):
    hits = []
    for child in body:
        if child.tag != q(W, "p"):
            continue
        compact = math_concat(child).replace(" ", "").replace("\u2061", "")
        if compact in {"Y*=X*β,", "Y*=X*β"}:
            hits.append(child)
    if len(hits) != 1:
        raise SystemExit(f"no-kriging predictor matched {len(hits)} paragraphs")
    new = make_p(blocks_from(r"$Y_{*}=X_{*}\hat{\beta},$"), hits[0])
    hits[0].getparent().replace(hits[0], new)


def edit_main(body):
    replace_paragraph(body, "Spatial dependence in empirical systems often operates simultaneously", ABSTRACT)
    replace_paragraph(body, "Tobler's first law characterizes spatial dependence", INTRO)
    replace_runs(body, "bootstrap/permutation\xa0calibration", "conditional Gaussian Monte Carlo calibration", 1)
    replace_paragraph(body, "The data contain 506 census tracts nested within 92 municipalities", OUTCOME)
    replace_paragraph(body, "The design matrix includes an intercept and ten regressors", DESIGN)
    delete_paragraph(body, "stored by the screening routine")
    upright_caption_greek(body)
    replace_runs(body, "August 2026 Holdout Performance", "August 2026 Illustrative Imputation Results", 1)
    replace_runs(body, "August 2026 holdout performance", "August 2026 illustrative imputation results", 1)
    replace_runs(
        body,
        "therefore supports a dynamic multiplex specification",
        "is therefore consistent with a modest predictive advantage from dynamically selected additional network structure",
        1,
    )
    heading = require_one(body, "August 2026 Archived Illustrative Snapshot")
    style = require_one(body, "To examine MCMC parameter mixing")
    insert_after(heading, [AUGUST], style)
    conclusion = require_one(body, "Extensions to dynamic panel settings and non-Gaussian outcomes")
    insert_after(conclusion, [SCOPE], conclusion)
    replace_no_kriging_predictor(body)
    gamma = upright_gamma(body)
    script = italic_s_not_script(body)
    if "have not been rerun" not in "".join(ptext(p) for p in body.iter(q(W, "p"))):
        raise SystemExit("the isotropic-sampler disclosure was removed")
    print(f"main: upright gamma touches {gamma}; script S converted {script}")


def august_beta_paragraph():
    payload = json.loads(BETA.read_text(encoding="utf-8"))
    mean = payload["mean_L"]
    mode = payload["mode_L"]
    lower = payload["q025_L"]
    upper = payload["q975_L"]
    if abs(mean - 0.430) < 0.02:
        raise SystemExit("Beta refit unexpectedly matched 0.430; rewrite the sentence before saving")
    return (
        "The August 2026 showcase was refit on the same 80-firm training block and with the same "
        r"isotropic proposal, replacing the paper prior by $L \sim \mathrm{Beta}(1,2)$. The archived "
        "paper-prior posterior has mean 0.430 and histogram mode 0.494. The Beta(1, 2) refit has "
        f"posterior mean {mean:.3f} and histogram mode {mode:.3f} (2.5% to 97.5% quantiles "
        f"{lower:.3f} to {upper:.3f}). With two active layers the induced density is proportional to "
        r"$(1-L)/L$ on the open ball, which pulls $L$ inward even though the paper-prior mean lies "
        "well below the boundary. No retained draw exceeds 0.95."
    )


def edit_supplement(body):
    anchor = require_one(body, "The same split was refit with")
    insert_after(anchor, [august_beta_paragraph()], anchor)
    replace_paragraph(body, "Thus, the August results provide two complementary forms of evidence", AUGUST_SUP)
    replace_paragraph(body, "Data and code availability.", AVAILABILITY)


def main():
    files, root = load_zip(MAIN_IN)
    edit_main(root.find(q(W, "body")))
    store_zip(MAIN_OUT, files, root)
    files, root = load_zip(SUP_IN)
    edit_supplement(root.find(q(W, "body")))
    store_zip(SUP_OUT, files, root)
    print(f"wrote {MAIN_OUT.name} and {SUP_OUT.name}")


if __name__ == "__main__":
    main()
