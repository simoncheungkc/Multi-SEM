"""Manuscript v29, supplement v27, and the reorganized response letter.

Reads v28 and supplement v26. Does not overwrite them.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

from lxml import etree

from build_revision_docs import following_table, insert_after, replace_paragraph
from build_v10_docx import XML, make_p, omath, q, set_cell_text
from build_v12_docx import W, blocks_from, load_zip, ptext, require_one, store_zip
from build_v26_docs import upright_greek
from mnreg.latex_omml import run

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v28.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v29.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v26.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v27.docx"
LETTER_IN = ROOT / "Response_to_Reviewers_3.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers_4.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
BAR = {"|", "∣", "‖", "∥"}


def load_json(path):
    file = Path(path)
    if not file.exists():
        raise SystemExit(f"missing {file}")
    return json.loads(file.read_text(encoding="utf-8"))


def signed(value, digits=3):
    text = f"{float(value):.{digits}f}"
    return "−" + text[1:] if text.startswith("-") else text


def replace_text(body, old, new, minimum=1):
    count = 0
    for node in body.iter(q(W, "t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)
            count += 1
    if count < minimum:
        raise SystemExit(f"{old!r} replaced in {count} runs")
    print(f"text {count}: {old[:60]}")
    return count


def assert_word_safe(root, label):
    bad = nested = 0
    for node in root.iter(q(M, "rPr")):
        for child in node:
            if etree.QName(child).localname == "rPr":
                bad += 1
    for node in root.iter(q(M, "e")):
        for child in node:
            if etree.QName(child).localname == "e":
                nested += 1
    if bad or nested:
        raise SystemExit(f"{label} invalid OMML: w:rPr inside m:rPr={bad}, nested m:e={nested}")


def strip_invisible_math(body):
    for node in body.iter(q(M, "t")):
        if not node.text:
            continue
        cleaned = "".join(ch for ch in node.text if unicodedata.category(ch) != "Cf")
        if cleaned != node.text:
            node.text = cleaned


def _bar_value(node):
    if node is None:
        return ""
    for key, value in node.attrib.items():
        if etree.QName(key).localname == "val":
            return value or ""
    return ""


def bars_to_glyphs(body):
    """Turn absolute-value and norm delimiters into ordinary math characters.

    PDF export drops the begChr/endChr glyphs of an m:d delimiter, which is
    why the stability formula loses its bars. A character in m:t is exported
    with the rest of the equation.
    """
    converted = 0
    for delim in reversed(list(body.iter(q(M, "d")))):
        props = delim.find(q(M, "dPr"))
        if props is None:
            continue
        begin = _bar_value(props.find(q(M, "begChr")))
        end = _bar_value(props.find(q(M, "endChr")))
        if begin not in BAR and end not in BAR:
            continue
        element = delim.find(q(M, "e"))
        pieces = []
        if begin:
            pieces.append(run(begin, sty="p"))
        if element is not None:
            pieces.extend(list(element))
        if end:
            pieces.append(run(end, sty="p"))
        parent = delim.getparent()
        index = list(parent).index(delim)
        parent.remove(delim)
        for offset, piece in enumerate(pieces):
            parent.insert(index + offset, piece)
        converted += 1
    print("bar delimiters converted", converted)
    return converted


def retarget_probability(body):
    found = 0
    for node in body.iter(q(M, "t")):
        if node.text != "=0.00025":
            continue
        node.text = "=0"
        math = node.getparent()
        while math is not None and etree.QName(math).localname != "oMath":
            math = math.getparent()
        math.addnext(omath(r"\ (<6.3\times 10^{-5})"))
        found += 1
    if found != 1:
        raise SystemExit(f"probability token updated {found} times")
    replace_text(body, "L0.95=0.661; P=0", "L0.95=0.661", minimum=0)


def hat_r_headers(body):
    changed = 0
    for table in body.iter(q(W, "tbl")):
        header = table.find(q(W, "tr"))
        if header is None:
            continue
        for cell in header.findall(q(W, "tc")):
            maths = list(cell.iter(q(M, "oMath")))
            plain = "".join(node.text or "" for node in cell.iter(q(W, "t"))).strip()
            math_text = "".join(
                node.text or "" for math in maths for node in math.iter(q(M, "t"))
            ).strip()
            if plain in {"", "R"} and math_text == "R" and len(maths) == 1:
                maths[0].getparent().replace(maths[0], omath(r"\hat{R}"))
                changed += 1
    print("hat R headers", changed)


def update_table4a(body, payload):
    old = (
        "Across the 200 replications the median split-R-hat is 1.021 and the 95th percentile "
        "is 1.041; the maximum is 1.103."
    )
    lengths = sorted(set(int(value) for value in payload["n_iter_by_rep"] if int(value) > 4000))
    length_text = ", ".join(f"{value:,}" for value in lengths) or "8,000"
    new = (
        f"Replications with maximum split-R-hat above 1.05 after 4,000 iterations were rerun at "
        f"{length_text} iterations. Across all 200 replications the median split-R-hat is "
        f"{payload['rhat_median']:.3f}, the 95th percentile is {payload['rhat_p95']:.3f}, and the "
        f"maximum is {payload['rhat_max']:.3f}."
    )
    replace_text(body, old, new)
    table = following_table(body, "Table 4a.")
    rows = payload["lambda"] + payload["beta"]
    data = table.findall(q(W, "tr"))[1:]
    if len(data) < len(rows):
        raise SystemExit(f"Table 4a has {len(data)} rows for {len(rows)} parameters")
    for row, item in zip(data, rows):
        cells = row.findall(q(W, "tc"))
        set_cell_text(cells[2], signed(item["bias"]))
        set_cell_text(cells[3], f"{item['rmse']:.3f}")
        set_cell_text(cells[4], f"{item['coverage']:.3f} ({item['coverage_mcse']:.3f})")


def edit_main(files, root, table4a, strict):
    body = root.find(q(W, "body"))
    strip_invisible_math(body)
    bars_to_glyphs(body)
    retarget_probability(body)
    hat_r_headers(body)
    update_table4a(body, table4a)
    replace_text(
        body,
        "We derive exact design-conditional null moments for residual Moran’s ",
        "We apply exact design-conditional Cliff–Ord moments for residual Moran’s ",
    )
    replace_text(
        body,
        " under isolate scaling and introduce topology-triggered ",
        " with isolate scaling and pair them with topology-triggered ",
    )
    insert_after(
        require_one(body, "mark the projected residual operator"),
        [
            r"The $\kappa_s>3.0$ trigger is empirically supported by the heavy-tailed power-law "
            r"experiment ($\kappa=6.36$) in Section 7.2, whereas $\psi_s>0.20$ is retained as a "
            "conservative structural safeguard for concentrated singular spectra and is not "
            "calibrated by the present simulation.",
        ],
        require_one(body, "mark the projected residual operator"),
    )
    replace_text(body, "inside the origin's admissible component", "inside the origin-connected invertibility component")
    replace_text(body, "inside the admissible component", "inside the origin-connected invertibility component")
    insert_after(
        require_one(body, "combines induced-subgraph estimation with full-graph imputation"),
        [
            r"The marginal credible intervals from the induced-subgraph estimator apply to the "
            r"re-standardized training subgraph operator. The Schur complement "
            r"$P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}$ is the coherent full-graph marginal likelihood, "
            "and it is the likelihood scored in Table S13b.",
        ],
        require_one(body, "combines induced-subgraph estimation with full-graph imputation"),
    )
    insert_after(
        require_one(body, "We evaluate the finite-sample performance of the induced-subgraph estimator"),
        [
            "Those induced-subgraph intervals describe the re-standardized training operator. "
            r"Table S13b evaluates the coherent Schur likelihood $P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}$ "
            "on the same full-graph parameters.",
        ],
        require_one(body, "We evaluate the finite-sample performance of the induced-subgraph estimator"),
    )
    table8 = following_table(body, "Table 8.")
    table8.addnext(
        make_p(
            blocks_from(
                r"Because $r_{\mathrm{vec}}(A_{\mathrm{geo}},G_{A_{\mathrm{geo}}})=0.975$ and the "
                r"95% credible interval for $\lambda_{\mathrm{geo}}$, $(-0.059,0.504)$, spans zero, "
                r"with posterior correlation $-0.90$ between $\lambda_{\mathrm{geo}}$ and the resolvent "
                r"coefficient, the Boston estimates describe joint geographic-plus-municipal disturbance "
                r"dependence. They do not separately identify one-hop decay from multi-hop decay. "
                r"Model E in Table S9, $A_{\mathrm{geo}}+G_{A_{\mathrm{geo}}}$ with mean RMSE 0.1383, "
                "is the higher-order single-geographic-network comparator."
            ),
            require_one(body, "Table 8."),
        )
    )
    for cell in body.iter(q(W, "tc")):
        plain = "".join(node.text or "" for node in cell.iter(q(W, "t"))).strip()
        if plain == "Multi-SEM (Moran-selected, kriging)":
            set_cell_text(cell, "Multi-SEM (screened / safeguard, kriging)")
        if plain == "—":
            row = cell.getparent()
            label = "".join(node.text or "" for node in row.find(q(W, "tc")).iter(q(W, "t")))
            if "training-only screen" in label:
                cells = row.findall(q(W, "tc"))
                set_cell_text(cells[-1], "0.60")
    table13 = following_table(body, "Table 13.")
    table13.addnext(
        make_p(
            blocks_from(
                "The screened row uses Benjamini–Hochberg control at $q^*=0.05$ in "
                f"{45} months and the non-FDR safeguard in 18 months. Mean RMSE on the 45 "
                "BH months is 0.0670 for Multi-SEM and 0.0710 for OLS. On the 18 safeguard "
                "months the means are 0.0617 and 0.0621. A strict rule that uses Multi-SEM only "
                "when BH is nonempty, and OLS otherwise, has 63-month mean RMSE "
                f"{strict['strict_bh_or_ols_rmse']:.4f}."
            ),
            require_one(body, "Table 13."),
        )
    )
    upright_greek(body)
    assert_word_safe(root, "v29")


def edit_supplement(files, root):
    body = root.find(q(W, "body"))
    strip_invisible_math(body)
    bars_to_glyphs(body)
    retarget_probability(body)
    hat_r_headers(body)
    insert_after(
        require_one(body, "Figure S10 compares the Split 5 posterior"),
        [
            r"The Cartesian log-density $(1-k)\ln\Vert\lambda\Vert_1+\ln(1-\Vert\lambda\Vert_1)$ "
            r"has an exact $\mathrm{Beta}(1,2)$ marginal on $L=\Vert\lambda\Vert_1$ because the "
            r"change of variables from $(L,w)$ to $\lambda\in\mathbb{R}^k$ has volume element "
            r"$d\lambda\propto L^{k-1}\,dL\,dw$. Multiplication by $L^{1-k}$ cancels the shell "
            r"factor $L^{k-1}$ for every dimension $k\ge 2$, leaving $p_L(L)=2(1-L)$.",
        ],
        require_one(body, "Figure S10 compares the Split 5 posterior"),
    )
    replace_text(body, "exact curved spectral-radius boundary", "spectral-radius boundary")
    replace_text(body, "exact spectral-radius boundary", "spectral-radius boundary")
    replace_text(body, "exact-admissible benchmark", "spectral-stability benchmark")
    replace_text(
        body,
        "The earlier isotropic fits remain in results/boston. ",
        "All tables and figures reported in the manuscript are generated from the covariance-adapted "
        "outputs in results/boston_adaptive (python run_boston.py --adaptive). Legacy isotropic runs "
        "in results/boston are excluded from all reported displays. ",
    )
    replace_text(
        body,
        "4 x 4000, adapt_covariance=True",
        "4 x 4000, extended to 8000 when split-R-hat exceeds 1.05, adapt_covariance=True",
    )
    replace_text(
        body,
        "The intended repository is https://github.com/simoncheungkc/Multi-SEM. No live Zenodo DOI is listed because that address did not resolve.",
        "The frozen replication scripts, pinned dependency manifest, and static data snapshots "
        "(data/boston_corrected.txt, data/sp100_clean_snapshot.csv) will be archived on Zenodo "
        "with a permanent DOI prior to final publication.",
    )
    upright_greek(body)
    assert_word_safe(root, "supplement v27")


def write_letter(strict, table4a):
    _files, root = load_zip(LETTER_IN)
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
        "The revised manuscript is version 29 and the supplement is version 27. The changes below are grouped so that each point can be checked against a table, a figure, or an explicit scope statement.",
        "1. Resolved by new analysis and quantitative evidence",
        "Sections 1 and 2.1 place the additive higher-order error specification with Anselin (1988) and Lee and Liu (2010). Florax and Nijkamp (2005) is retained for the misspecification of a single weight matrix. The abstract now describes the Moran moments as an application of the Cliff–Ord moments with isolate scaling.",
        "Figure S10 compares the Split 5 posterior with 200,000 draws from the exact truncated-Gaussian prior and with the analytical shell. The radial Beta(1, 2) prior was checked in code: its Cartesian log-density is (1−k) ln L + ln(1−L), the factor L^{1−k} cancels the shell volume L^{k−1}, and the marginal of L is Beta(1, 2) for every k ≥ 2. Table S4b reports both priors for Boston Split 5 and for the August 2026 S&P 100 panel, including posterior means, credible intervals, E(L | Y), L_0.95, P(L > 0.95 | Y), holdout RMSE, and CRPS.",
        "Table 4c and Figure S11 are the outside-ℓ1-ball stress test. The spectral radius at the truth has mean 0.693 and maximum 0.741, and it is below 1 in 200 of 200 replications. The ℓ1 constraint covers the first coefficient in 10.0% of replications. The spectral-stability constraint covers it in 96.0%. Table 4b reports the opposite-sign design. Table S12 reports the high-collinearity design: covariance adaptation raises minimum effective sample size from 100.3 to 442.5 and lowers maximum split-R-hat from 1.187 to 1.040, while parameter and holdout RMSE stay essentially unchanged.",
        "Tables S13, S13a, and S13b report predictive intervals and CRPS. On Boston splits 0–9 the induced-subgraph and coherent Schur fits have mean RMSE 0.141 and 0.142. Table 9 contains the Nadeau–Bengio corrected comparison and the universal-kriging row. Tables S10 and S11 contain the CHAS, censoring, and post-fit innovation checks. Table S14 contains the 92-fold leave-one-town-out design and the 10-fold spatial-cluster design. Nine of the ten clusters induce a connected geographic subgraph.",
        "Tables 10 and 12 contain the leak-free August screen. Benjamini–Hochberg at q* = 0.05 selects no layer on the 80-firm training block; the non-FDR safeguard retains sector and industry. The training-only Multi-SEM row has holdout correlation 0.60. Every reported simulation and empirical fit uses adapt_covariance=True.",
        f"Table 4a starts at 4 chains of 4,000 iterations. The {table4a['extended_reps']} replications with split-R-hat above 1.05 were rerun at double length. The median is {table4a['rhat_median']:.3f}, the 95th percentile is {table4a['rhat_p95']:.3f}, and the maximum is {table4a['rhat_max']:.3f}.",
        "The August 2026 adapted chains give λ_sec = 0.206 and λ_ind = 0.186, with E(L | Y) = 0.430 and L_0.95 = 0.661. None of the 16,000 retained draws exceeds 0.95, so Table 11 and Section S4.4 now report 0 (less than 6.3 × 10^−5).",
        "2. Resolved by clarification and terminology refinement",
        "Section 4.2 distinguishes the Cliff–Ord spectral null from the topology-triggered Monte Carlo calibration. The κ_s > 3 trigger is tied to the power-law experiment with κ = 6.36 in Section 7.2. The threshold ψ_s > 0.20 is described as a conservative safeguard for concentrated singular spectra, not as a calibrated critical value and not as a Kelejian–Prucha central-limit condition.",
        "Section 3.4 treats G_{A_geo} as a polynomial filter of the geographic adjacency. Section 6.2 states that precision conditioning is an algebraic identity and that the MCMC cost is the training-filter factorization; the illustrative cubic ratios have been removed. Section 8.2 and Table 13 label the 18 months in which no layer passes Benjamini–Hochberg as a non-FDR safeguard. Forty-five months are BH-controlled (mean RMSE 0.0670 versus 0.0710 for OLS) and 18 are safeguard months (0.0617 versus 0.0621). "
        f"A strict rule that reverts to OLS when BH is empty has 63-month mean RMSE {strict['strict_bh_or_ols_rmse']:.4f}.",
        "Section 7.3.2 uses “origin-connected invertibility component” for the set on which the determinant is nonzero, and “spectral-stability region” for the sufficient subset on which the spectral radius is below 1. The same wording is used in Section S3.7 and Figure S11.",
        "3. Methodological scope boundaries retained and justified",
        "Default sampling remains inside the geometry-free ℓ1 ball. That ball guarantees invertibility without a proposal-level eigendecomposition. Table 4c quantifies the truncation: when the truth lies outside the ball, the ℓ1 interval for the first coefficient covers the truth in 10.0% of replications. The spectral-stability sampler is reported there as the comparison, not as the default.",
        "The induced-subgraph estimator remains the fast point-imputation surrogate. Section 7.5 and Table S13b state that its credible intervals refer to the re-standardized training operator, while the Schur complement is the coherent full-graph marginal likelihood.",
        "4. Not undertaken, with reasons",
        "The screen remains a layer-by-layer Moran test with Benjamini–Hochberg control. A joint multi-weight Lagrange-multiplier test is not added. Separation of correlated layers is left to the joint posterior and to the ablation in Table S9, where geography plus its resolvent is the single-network higher-order comparator (mean RMSE 0.1383) and the two layers have off-diagonal correlation 0.975.",
        "The thresholds κ_s > 3.0 and ψ_s > 0.20 remain pragmatic computational triggers for switching from the analytical normal tail to the conditional Gaussian Monte Carlo reference. They are not claimed to be conditions for a central limit theorem.",
        "The replication scripts read data/boston_corrected.txt and data/sp100_clean_snapshot.csv. With the default USE_STATIC_CACHE = True, a missing S&P snapshot now raises FileNotFoundError instead of downloading a new Yahoo Finance series. The public repository address has not resolved, so no DOI is claimed. Table S15 states that the frozen scripts, the pinned manifest, and the static snapshots will be archived on Zenodo before final publication.",
    ]
    anchor = sect if sect is not None else None
    for text in paragraphs:
        paragraph = make_p(blocks_from(text), style)
        if anchor is not None:
            anchor.addprevious(paragraph)
        else:
            body.append(paragraph)
    assert_word_safe(root, "letter")
    store_zip(LETTER_OUT, _files, root)


def refresh_figure():
    from run_postprocess_revision import figure_s11, verify_radial_prior

    verify_radial_prior()
    figure_s11()


def main():
    table4a = load_json(ROOT / "results" / "unified" / "table4a.json")
    if "rhat_median" not in table4a or table4a.get("rhat_max", 2) > 1.05:
        raise SystemExit("Table 4a extension has not produced a maximum split-R-hat at or below 1.05")
    strict = load_json(ROOT / "results" / "v28" / "strict_bh.json")
    refresh_figure()
    main_files, main_root = load_zip(MAIN_IN)
    edit_main(main_files, main_root, table4a, strict)
    store_zip(MAIN_OUT, main_files, main_root)
    sup_files, sup_root = load_zip(SUP_IN)
    edit_supplement(sup_files, sup_root)
    store_zip(SUP_OUT, sup_files, sup_root)
    # Replace the supplement figure after the text edit so the new legend is stored.
    sup_files, sup_root = load_zip(SUP_OUT)
    from build_v28_docs import replace_media

    replace_media(
        sup_files,
        sup_root.find(q(W, "body")),
        "Figure S11.",
        ROOT / "results" / "figures" / "figure_s11_boundary.png",
    )
    store_zip(SUP_OUT, sup_files, sup_root)
    write_letter(strict, table4a)
    print("wrote", MAIN_OUT.name, SUP_OUT.name, LETTER_OUT.name)


if __name__ == "__main__":
    main()
