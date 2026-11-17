"""Manuscript v31, supplement v29, and the four-part response letter.

Reads v30 and supplement v28. Does not overwrite them.
"""

from __future__ import annotations

from pathlib import Path

from lxml import etree

from build_revision_docs import following_table
from build_v10_docx import make_p, omath, q, set_cell_text
from build_v12_docx import W, blocks_from, load_zip, ptext, require_one, store_zip
from build_v26_docs import upright_greek
from build_v29_docs import assert_word_safe, replace_text

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v30.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v31.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v28.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v29.docx"
LETTER_IN = ROOT / "Response_to_Reviewers_5.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers_6.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _fill(cell, text):
    paragraph = cell.find(q(W, "p"))
    for child in list(paragraph):
        if etree.QName(child).localname != "pPr":
            paragraph.remove(child)
    built = make_p(blocks_from(text))
    for child in list(built):
        if etree.QName(child).localname != "pPr":
            paragraph.append(child)


def _cell_text(cell):
    plain = "".join(node.text or "" for node in cell.iter(q(W, "t")))
    maths = "".join(node.text or "" for node in cell.iter(q(M, "t")))
    return plain + maths


def rewrite_step4(body):
    for node in body.iter(q(W, "t")):
        if not node.text or "refresh the residual vector " not in node.text:
            continue
        node.text = " from its inverse-gamma full conditional in Section 5.2, conditional on "
        run = node.getparent()
        old_math = run.getnext()
        period = old_math.getnext()
        pieces = [
            omath(r"\lambda^{(t)}"),
            _plain(" and "),
            omath(r"\beta^{(t)}"),
            _plain(", and refresh the residual vector "),
            omath(r"e = Y - X\beta^{(t)}"),
        ]
        for piece in pieces:
            old_math.addprevious(piece)
        old_math.getparent().remove(old_math)
        period_text = next(period.iter(q(W, "t")))
        period_text.text = " for the next iteration."
        return
    raise SystemExit("Algorithm 2 step 4 was not found")


def _plain(text):
    run = etree.Element(q(W, "r"))
    node = etree.SubElement(run, q(W, "t"))
    node.text = text
    if text[:1] == " " or text[-1:] == " ":
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return run


def section_53(body):
    replace_text(
        body,
        " > 1.05 to 8,000 iterations (4,000 burn-in); after extension, the maximum split-R̂ "
        "across all 200 replications is 1.049. The S&P panel reaches 1.013, Table 4 reaches 1.012, "
        "and the spectral estimator in Table 4c reaches 1.011. The ",
        " > 1.05 to 8,000 iterations (4,000 burn-in); across all 200 replications, the final "
        "split-R̂ has median 1.021, 95th percentile 1.040, and maximum 1.049. The S&P panel reaches "
        "1.013, Table 4 reaches 1.012, and the spectral-stability estimator in Table 4c reaches 1.011. The ",
    )


def terminology(body):
    replace_text(body, "admissible-region simulations", "spectral-stability simulations")
    replace_text(
        body,
        "the combined operator has mean spectral radius",
        "the combined operator B(λ_true) has mean spectral radius",
    )
    replace_text(body, "The spectral constraint recovers", "The spectral-stability estimator recovers")
    replace_text(
        body,
        "the spectral constraint has maximum split-",
        "the spectral-stability estimator has maximum split-",
    )
    replace_text(
        body,
        "No retained draw exceeds 0.95, so the posterior probability is 0, and none reaches the stability boundary at 1.",
        "0 of the 16,000 retained draws exceed 0.95 (empirical frequency 0.0000), and none reaches the stability boundary at 1.",
    )


def table_notes(body):
    for node in body.iter(q(W, "t")):
        if node.text == "0 / 16,000 draws (0.0000)":
            node.text = "Freq(L > 0.95) = 0 / 16,000"
    for node in body.iter(q(W, "t")):
        if node.text == "Multi-SEM (screened + non-FDR safeguard, kriging)":
            node.text = "Multi-SEM (screened / safeguard, kriging)"
    caption = require_one(body, "Table 11.  Posterior")
    table = following_table(body, "Table 11.  Posterior")
    nxt = table.getnext()
    if nxt is None or "zero observed Monte Carlo" not in ptext(nxt):
        note = make_p(
            blocks_from(
                "Note. Freq(L > 0.95) = 0/16,000 denotes zero observed Monte Carlo exceedances "
                "in the retained draws, rather than a zero theoretical tail probability."
            ),
            caption,
        )
        table.addnext(note)


def supplement(body):
    replace_text(
        body,
        "The admissible parameter region used in the Bayesian prior",
        "The constrained parameter domain used in the Bayesian prior",
    )
    replace_text(
        body,
        "within the admissible parameter region",
        "within the ℓ1 stability region",
    )
    replace_text(body, "into the admissible lobe", "into the spectral-stability lobe")
    replace_text(
        body,
        "0 / 16,000 draws (0.0000)",
        "0 of 16,000 retained draws exceed 0.95 (empirical frequency 0.0000)",
    )
    for math in list(body.iter(q(M, "oMath"))):
        bits = "".join(node.text or "" for node in math.iter(q(M, "t")))
        if bits.replace(" ", "") == "PL>0.95Y=0":
            run = _plain("0 of 16,000 retained draws exceed 0.95 (empirical frequency 0.0000)")
            math.addnext(run)
            math.getparent().remove(math)
    for cell in body.iter(q(W, "tc")):
        joined = _cell_text(cell).replace(" ", "")
        if "P(L>0.95" in joined and "Freq" not in joined:
            _fill(cell, r"Freq$(L > 0.95 \mid Y)$")
    table = following_table(body, "Table S4b.")
    for row in table.findall(q(W, "tr")):
        cells = row.findall(q(W, "tc"))
        if cells and "S&P" in _cell_text(cells[0]):
            for cell in cells:
                if _cell_text(cell).strip() == "0":
                    set_cell_text(cell, "0.0000 (0/16,000)")
    caption = require_one(body, "Table S4. Prior sensitivity")
    if "Cartesian" not in ptext(caption):
        nodes = list(caption.iter(q(W, "t")))
        nodes[-1].text = (nodes[-1].text or "") + (
            " The radial row uses the Cartesian log-density (1−k) ln L + ln(1−L) on 0 < L < 1."
        )


def write_letter():
    files, root = load_zip(LETTER_IN)
    body = root.find(q(W, "body"))
    style = None
    for child in body:
        if child.tag == q(W, "p") and ptext(child).strip():
            style = child
            break
    sect = body.find(q(W, "sectPr"))
    for child in list(body):
        if child is not sect:
            body.remove(child)
    paragraphs = [
        "Response to the referees and the Associate Editor",
        "The revised manuscript is version 31 and the supplement is version 29. The points below are grouped into the four requested sections.",
        "1. Resolved by new analysis and quantitative evidence",
        "Sections 1 and 2.1 place the additive higher-order error specification with Anselin (1988) and Lee and Liu (2010). Florax and Nijkamp (2005) is retained for misspecification of a single weight matrix. The abstract describes the Moran moments as an application of the Cliff–Ord moments with isolate scaling.",
        "The radial Beta(1, 2) prior has Cartesian log-density (1−k) ln L + ln(1−L) on 0 < L < 1. Table S4 reports that prior on the n = 200 synthetic design, together with the maximum absolute posterior mean among inactive coefficients, E(L | Y), and the empirical frequency Freq(L > 0.95 | Y). Table S4b reports the same prior for Boston Split 5 and the August 2026 S&P 100 panel.",
        "Table 4c and Figure S11 are the outside-ℓ1-ball stress test. At the true parameter vector the combined operator has mean spectral radius 0.693 and maximum 0.741, below 1 in 200 of 200 replications. Maximum split-R-hat and holdout RMSE are separate columns: 1.027 and 1.027 for the ℓ1 estimator (1.02719 and 1.02721 before rounding), and 1.011 and 1.014 for the spectral-stability estimator. The ℓ1 interval for the first coefficient covers the truth in 10.0% of replications; the spectral-stability interval covers it in 96.0%.",
        "Table 4b is the opposite-sign recovery, generated by run_simulation.py with seed [202611, rep], four chains of 6,000 iterations, 3,000 burn-in, and adapt_covariance=True. Table S12 is the high-collinearity comparison. Tables S13, S13a, and S13b report predictive coverage and CRPS for the induced subgraph and the coherent Schur complement. Table 9 contains the Nadeau–Bengio corrected comparison and the universal-kriging row. Tables S10 and S11 contain the CHAS, censoring, and post-fit innovation checks. Table S14 contains the 92-fold leave-one-town-out design and the 10-fold spatial-cluster design.",
        "Tables 10 and 12 contain the leak-free August screen. Benjamini–Hochberg at q* = 0.05 selects no layer on the 80-firm training block; the non-FDR safeguard retains sector and industry. Table 13 contains both the screened safeguard row (RMSE 0.06568) and the strict BH-or-OLS row (RMSE 0.06558, from the archived value 0.06557895). Every reported simulation and empirical fit uses adapt_covariance=True.",
        "Table 4a starts at four chains of 4,000 iterations with 2,000 burn-in. Three of the 200 replications had maximum split-R-hat above 1.05 and were extended to 8,000 iterations with 4,000 burn-in. Across all 200 replications the final split-R-hat has median 1.021, 95th percentile 1.040, and maximum 1.049. The minimum network ESS is 137.",
        "On the August 2026 panel, 0 of the 16,000 retained draws exceed L = 0.95 (empirical frequency 0.0000). Table 11 states that this count is an observed Monte Carlo frequency, not a claim that the theoretical tail probability is zero. The value 0.0576 in Table 12 is the plug-in posterior-mean prediction; 0.0578 in Tables S4b and S13b is the posterior predictive mean over 2,000 draws and 20 Gaussian replicates.",
        "2. Resolved by clarification and terminology refinement",
        "Section 4.2 distinguishes the Cliff–Ord spectral null from the topology-triggered conditional Monte Carlo calibration. G_{A_geo} is described as a polynomial filter of the geographic adjacency. Section 6.2 states that precision conditioning is an algebraic identity and that the MCMC cost is the training-filter factorization.",
        "The origin-connected invertibility component is the maximal component of the set on which det V(λ) is nonzero that contains the origin. The spectral-stability region is the set on which the spectral radius of B(λ) is below 1. The abstract now says spectral-stability simulations, and Section 5.3 names the spectral-stability estimator.",
        "Boundary statements for the S&P 100 panel are empirical frequencies. Section 8.2, Table 11, Section S2.3, Table S4b, and Section S4.4 report 0 of 16,000 retained draws, written as Freq(L > 0.95 | Y) = 0.0000 (0/16,000) where that column is displayed.",
        "3. Methodological scope boundaries retained and justified",
        "Default estimation remains inside the geometry-free sufficient ℓ1 ball. Table 4c quantifies the truncation risk when the truth lies outside that ball. The spectral-stability sampler is the comparison in that table, not the default.",
        "The induced-subgraph estimator remains the fast imputation surrogate. Table S13b keeps it beside the coherent Schur marginal likelihood and states which intervals belong to which operator.",
        "4. Not undertaken, with reasons",
        "The screen remains a layer-by-layer Moran test with Benjamini–Hochberg control. A joint multi-weight Lagrange-multiplier test is not added. Separation of correlated layers is left to the joint posterior and to the ablation in Table S9.",
        "The thresholds κ_s > 3.0 and ψ_s > 0.20 remain pragmatic computational triggers for switching from the analytical normal tail to the conditional Gaussian Monte Carlo reference. They are not conditions for a central limit theorem.",
        "The replication scripts read data/boston_corrected.txt and data/sp100_clean_snapshot.csv. With USE_STATIC_CACHE = True, a missing S&P snapshot raises FileNotFoundError unless --download is passed. The address simoncheungkc/Multi-SEM does not resolve, and the authenticated account is not that owner, so the repository was not created under a different account and no Zenodo DOI has been minted. Table S15 and the data-availability statement do not contain a DOI.",
    ]
    for text in paragraphs:
        paragraph = make_p(blocks_from(text), style)
        if sect is not None:
            sect.addprevious(paragraph)
        else:
            body.append(paragraph)
    upright_greek(body)
    assert_word_safe(root, "letter 6")
    store_zip(LETTER_OUT, files, root)


def main():
    main_files, main_root = load_zip(MAIN_IN)
    main_body = main_root.find(q(W, "body"))
    sup_files, sup_root = load_zip(SUP_IN)
    sup_body = sup_root.find(q(W, "body"))

    section_53(main_body)
    rewrite_step4(main_body)
    terminology(main_body)
    table_notes(main_body)
    upright_greek(main_body)
    assert_word_safe(main_root, "v31")
    store_zip(MAIN_OUT, main_files, main_root)

    supplement(sup_body)
    upright_greek(sup_body)
    assert_word_safe(sup_root, "supplement v29")
    store_zip(SUP_OUT, sup_files, sup_root)
    write_letter()
    print("wrote", MAIN_OUT.name, SUP_OUT.name, LETTER_OUT.name)


if __name__ == "__main__":
    main()
