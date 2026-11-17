"""Manuscript v32, supplement v30, and the response letter.

Reads v31 and supplement v29. Does not overwrite them.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from build_revision_docs import following_table, replace_paragraph
from build_v10_docx import make_p, omath, q, set_cell_text
from build_v12_docx import W, blocks_from, load_zip, ptext, store_zip
from build_v26_docs import upright_greek
from build_v28_docs import replace_media
from build_v29_docs import assert_word_safe, replace_text

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v31.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v32.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v29.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v30.docx"
LETTER_IN = ROOT / "Response_to_Reviewers_6.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers_7.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"

INTRO = (
    "Simulation experiments and two empirical studies evaluate this workflow across distinct relational regimes. "
    "Across synthetic multiplex graphs and empirical applications to Greater Boston housing tracts and S&P 100 equity returns, "
    "Multi-SEM substantially reduces out-of-sample imputation error relative to non-spatial OLS, while incremental gains over "
    "strong single-network spatial benchmarks are modest and depend on the prediction design (Sections 7 and 8). "
    "Mathematical proofs, prior-volume sensitivity checks, network ablations, spatial extrapolation designs, "
    "post-fit innovation diagnostics, and replication materials are provided in the Supplementary Materials."
)

TABLE5 = (
    " Paired RMSE differences are means over the 30 replications, with Monte Carlo standard errors SD(Δ_r)/sqrt(30) in parentheses. "
    "Moran-selected minus OLS: −0.056 (0.014). Moran-selected minus Single SEM, PL γ = 2.2: −0.000 (0.011). "
    "True active pair minus OLS: −0.091 (0.013). True active pair minus Single SEM, PL γ = 2.2: −0.035 (0.008)."
)

ESS = (
    " The reported minimum of 137 is the minimum, over the 200 replications and the network coefficients, "
    "of the sum of the four within-chain effective sample sizes. It is not a bulk effective sample size computed on the pooled chain."
)

SCHUR = (
    " Marginal credible intervals from the induced estimator describe the re-standardized training-subgraph operator. "
    "Table S13b checks predictive coverage and interval calibration under the coherent Schur likelihood."
)

BETA = (
    " Predictive RMSE and CRPS remain stable under the Beta(1, 2) radial prior, while the individual channel estimates "
    "move inward (the S&P 100 sector and industry coefficients, and λ1 and λ2 in Table S4), so those magnitudes are sensitive "
    "to the boundary geometry of the prior."
)


def _append(body, needle, sentence, required=True):
    found = 0
    for paragraph in body.iter(q(W, "p")):
        if needle in ptext(paragraph):
            nodes = list(paragraph.iter(q(W, "t")))
            nodes[-1].text = (nodes[-1].text or "") + sentence
            found += 1
    if required and not found:
        raise SystemExit(f"paragraph missing: {needle}")
    print("appended", found, needle[:40])


def _ell_region(body):
    for node in body.iter(q(W, "t")):
        if not node.text or "ℓ1 stability region" not in node.text:
            continue
        before, after = node.text.split("ℓ1 stability region", 1)
        node.text = before
        run = node.getparent()
        run.addnext(_plain(" stability region" + after))
        run.addnext(omath(r"\ell_1"))
        return
    raise SystemExit("l1 stability sentence was not found")


def _plain(text):
    run = etree.Element(q(W, "r"))
    node = etree.SubElement(run, q(W, "t"))
    node.text = text
    if text[:1] == " " or text[-1:] == " ":
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return run


def main_text(body):
    replace_paragraph(body, "Simulation experiments and two empirical studies evaluate the behaviour", INTRO)
    replace_text(body, "This undercoverage reflects a combination of", "This undercoverage is consistent with a combination of")
    replace_text(
        body,
        "when cross-block ties are unavailable",
        "when evaluating the full-graph Schur complement at every MCMC step is computationally prohibitive",
        minimum=2,
    )
    _append(body, "induced-subgraph estimator provides a computationally efficient surrogate when evaluating", SCHUR)
    replace_text(
        body,
        "Thus, the sign test and conventional paired comparison provide statistically significant evidence of improved monthly predictive performance, whereas the autocorrelation-robust comparison provides weaker evidence at conventional significance levels.",
        "The autocorrelation-robust Newey–West comparison (p = 0.072) is the primary inferential benchmark for the monthly loss differentials. The sign-test win count, 44 of 63 months (p = 0.0022), is reported alongside it as a descriptive tally.",
    )
    replace_text(
        body,
        ", indicating weaker evidence once serial dependence in the monthly losses is taken into account.",
        ". The autocorrelation-robust Newey–West comparison is the primary inferential benchmark for those monthly loss differentials, alongside the descriptive sign-test count of 44 of 63 months.",
    )
    _append(
        body,
        "The same fits are checked for leftover spatial correlation.",
        " These analytical Moran p-values on the post-fit filtered innovations are descriptive residual diagnostics, not calibrated frequentist hypothesis tests.",
    )
    _append(body, "The simulation results show that the oracle specification", BETA)
    replace_text(body, "power or false discoveries (active DGP)", "power or spurious rejections (active DGP)", minimum=3)
    replace_text(
        body,
        "the false-discovery rate under the active DGP",
        "the spurious rejection rate under the active DGP",
    )
    _append(body, "The minimum network ESS across all 200 replications is 137.", ESS)
    _append(body, "Table 5.  Mean out-of-sample RMSE and MAE", TABLE5)
    table = following_table(body, "Table 13.")
    for row in table.findall(q(W, "tr")):
        cells = row.findall(q(W, "tc"))
        label = "".join(node.text or "" for node in cells[0].iter(q(W, "t")))
        if "strict BH" in label:
            set_cell_text(cells[2], "0.04921")
            return
    raise SystemExit("strict BH row was not found")


def supplement(files, body):
    replace_text(
        body,
        "Run python run_boston.py, python plot_boston_split5.py, and python build_boston_docx.py. ",
        "Run python run_boston.py --adaptive, python plot_boston_split5.py, and python build_boston_docx.py "
        "(without --adaptive the script writes legacy isotropic chains in results/boston that are excluded from all reported tables and figures). ",
    )
    replace_text(
        body,
        "Run python run_boston.py --adaptive. Running without --adaptive writes legacy isotropic chains in results/boston that are excluded from all reported tables and figures. ",
        "The adaptive command ",
    )
    for node in body.iter(q(W, "t")):
        if node.text and node.text.strip() == "Tables 6-10":
            row = node
            while etree.QName(row).localname != "tr":
                row = row.getparent()
            cells = row.findall(q(W, "tc"))
            set_cell_text(cells[1], "run_boston.py --adaptive")
            break
    else:
        raise SystemExit("Table S15 Boston row was not found")
    _ell_region(body)
    replace_text(
        body,
        "(1−k) ln L + ln(1−L)",
        "(1−k) times the logarithm of L, plus the logarithm of (1−L)",
    )
    _append(
        body,
        "gives means 0.132 and 0.152",
        " Predictive RMSE and CRPS stay nearly unchanged under this prior, while the sector and industry estimates move inward relative to the truncated-Gaussian fit.",
    )
    _append(
        body,
        "posterior-mean filtered innovations",
        " These Moran statistics are descriptive post-fit residual diagnostics, not calibrated frequentist hypothesis tests.",
        required=False,
    )
    png = ROOT / "results" / "figures" / "figure_s11_boundary.png"
    if png.exists():
        replace_media(files, body, "Figure S11.", png)


def write_letter():
    files, root = load_zip(LETTER_IN)
    body = root.find(q(W, "body"))
    style = next(child for child in body if child.tag == q(W, "p") and ptext(child).strip())
    sect = body.find(q(W, "sectPr"))
    for child in list(body):
        if child is not sect:
            body.remove(child)
    paragraphs = [
        "Response to the referees and the Associate Editor",
        "The revised manuscript is version 32 and the supplement is version 30.",
        "1. Resolved by New Analysis",
        "The literature placement, the radial Beta(1, 2) sensitivity, the outside-ℓ1-ball stress test, the opposite-sign recovery, the high-collinearity simulation, the coherent Schur coverage and CRPS, the Nadeau–Bengio comparison, universal kriging, the CHAS and censoring checks, the 92-fold leave-one-town-out design, the 10-fold spatial-cluster design, the leak-free S&P 100 screen, the strict BH-or-OLS comparison, the post-fit innovation diagnostics, and the unified adapt_covariance=True sampler are reported in the manuscript and the supplement.",
        "Table 13 now gives the strict BH-or-OLS rule both an RMSE of 0.06558 and an MAE of 0.04921. The MAE uses Multi-SEM in the 45 months in which Benjamini–Hochberg at q* = 0.05 selects a layer and OLS in the other 18 months. Table 5 reports paired RMSE differences over the 30 replications. Moran-selected minus OLS is −0.056 (Monte Carlo SE 0.014). Moran-selected minus the power-law γ = 2.2 single SEM is −0.000 (SE 0.011), which is why those two rows share the rounded mean 1.033. The true active pair minus OLS is −0.091 (SE 0.013).",
        "Table 4a extended 3 of 200 replications from 4,000 to 8,000 iterations. The final split-R-hat has median 1.021, 95th percentile 1.040, and maximum 1.049. The minimum of 137 is the sum of the four within-chain effective sample sizes, then minimized over coefficients and replications. It is not a bulk ESS of the pooled chain.",
        "2. Resolved by Clarification and Terminology Refinement",
        "The Cliff–Ord spectral null is distinguished from the conditional Monte Carlo calibration. G_{A_geo} is a polynomial filter of the geographic adjacency. Precision conditioning is an algebraic identity; the MCMC cost is the training-filter factorization. The origin-connected invertibility component and the spectral-stability region are named separately.",
        "Boundary statements are empirical frequencies: 0 of 16,000 retained draws exceed 0.95. Table 3 now says spurious rejections rather than false discoveries, so that column is not confused with Benjamini–Hochberg FDR. The Newey–West comparison (p = 0.072) is the primary inferential benchmark for the monthly loss differentials. The sign-test count of 44 of 63 months (p = 0.0022) is descriptive. Moran statistics on post-fit filtered innovations are descriptive residual diagnostics.",
        "3. Methodological Scope Boundaries Retained and Justified",
        "Default estimation stays inside the geometry-free sufficient ℓ1 ball. Table 4c quantifies the truncation when the truth lies outside that ball. The induced-subgraph estimator remains the fast surrogate used when the full-graph Schur complement is too costly at every MCMC step. Its credible intervals refer to the re-standardized training subgraph. Table S13b keeps the coherent Schur likelihood beside it.",
        "4. Not Undertaken with Reasons",
        "The screen remains a layer-by-layer Moran test with Benjamini–Hochberg control. A joint multi-weight Lagrange-multiplier test is not added. The thresholds κ_s > 3.0 and ψ_s > 0.20 remain operational triggers for the conditional Gaussian Monte Carlo reference. They are not conditions for a central limit theorem.",
        "The scripts read data/boston_corrected.txt and data/sp100_clean_snapshot.csv. A missing S&P snapshot raises FileNotFoundError unless --download is passed. The address simoncheungkc/Multi-SEM does not resolve, and the authenticated account is not that owner, so no repository was created under a different account and no Zenodo DOI has been minted.",
    ]
    for text in paragraphs:
        paragraph = make_p(blocks_from(text), style)
        sect.addprevious(paragraph)
    upright_greek(body)
    assert_word_safe(root, "letter 7")
    store_zip(LETTER_OUT, files, root)


def main():
    main_files, main_root = load_zip(MAIN_IN)
    main_body = main_root.find(q(W, "body"))
    sup_files, sup_root = load_zip(SUP_IN)
    sup_body = sup_root.find(q(W, "body"))
    main_text(main_body)
    upright_greek(main_body)
    assert_word_safe(main_root, "v32")
    store_zip(MAIN_OUT, main_files, main_root)
    supplement(sup_files, sup_body)
    upright_greek(sup_body)
    assert_word_safe(sup_root, "supplement v30")
    store_zip(SUP_OUT, sup_files, sup_root)
    write_letter()
    print("wrote", MAIN_OUT.name, SUP_OUT.name, LETTER_OUT.name)


if __name__ == "__main__":
    main()
