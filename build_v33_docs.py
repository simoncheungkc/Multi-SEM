"""Manuscript v33, supplement v31, and response letter 8.

Reads version 32, supplement version 30, and letter 7. Does not overwrite them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from build_v10_docx import omath, q
from build_v12_docx import W, load_zip, ptext, require_one, store_zip
from build_v26_docs import upright_greek
from build_v29_docs import assert_word_safe, replace_text

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v32.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v33.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v30.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v31.docx"
LETTER_IN = ROOT / "Response_to_Reviewers_7.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers_8.docx"

SCHUR_TAIL = (
    " Marginal credible intervals from the induced estimator describe the "
    "re-standardized training-subgraph operator. Table S13b checks predictive "
    "coverage and interval calibration under the coherent Schur likelihood."
)
STEP3B_OLD = (
    "Run python run_boston.py --adaptive, python plot_boston_split5.py, and "
    "python build_boston_docx.py (without --adaptive the script writes legacy "
    "isotropic chains in results/boston that are excluded from all reported "
    "tables and figures). "
)
STEP3B_NEW = (
    "Run python run_boston.py --adaptive, python plot_boston_split5.py, and "
    "python build_boston_docx.py. The --adaptive option calls gibbs_multi_sem "
    "in mnreg/model.py with adapt_covariance=True (8,000 iterations, 4,000 "
    "burn-in draws, four chains; Haario, Saksman, and Tamminen, 2001; Roberts "
    "and Rosenthal, 2009) and writes all reported outputs to "
    "results/boston_adaptive/. Running run_boston.py without --adaptive writes "
    "legacy isotropic chains to results/boston/ that are excluded from all "
    "reported tables and figures. "
)
REGIMES = (
    " Three August 2026 objects stay distinct: (a) the archived full-sample "
    "exploratory screen in Table 10; (b) the training-only safeguard that "
    "keeps a top layer when its p-value is below 0.10, reported in Table 12 "
    "at RMSE 0.0576; and (c) the strict BH-or-OLS rule, which selects OLS "
    "because the 80-firm BH set at q* = 0.05 is empty."
)


def _plain(text):
    from lxml import etree

    run = etree.Element(q(W, "r"))
    node = etree.SubElement(run, q(W, "t"))
    node.text = text
    if text[:1] == " " or text[-1:] == " ":
        node.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return run


def _once(body, old):
    hits = [node for node in body.iter(q(W, "t")) if node.text and old in node.text]
    if len(hits) != 1:
        raise SystemExit(f"{old[:80]!r} found in {len(hits)} runs")
    return hits[0]


def repair_schur(body):
    marker = "computationally prohibitive, whereas the Schur marginal precision Marginal"
    node = _once(body, marker)
    head, _tail = node.text.split(marker, 1)
    node.text = head + "prohibitive, whereas the Schur marginal precision "
    run = node.getparent()
    run.addnext(
        _plain(
            " defines the coherent full-graph training likelihood when the full "
            "network topology is observed ex ante. Marginal credible intervals "
            "from the induced estimator describe the re-standardized "
            "training-subgraph operator, while Table S13b checks predictive "
            "coverage and interval calibration under the coherent Schur likelihood."
        )
    )
    run.addnext(omath(r"P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}"))
    replace_text(body, SCHUR_TAIL, "")


def repair_radius(body):
    old = (
        "the mean radius keeps the segment from the origin inside the "
        "origin-connected invertibility component."
    )
    node = _once(body, old)
    head, tail = node.text.split(old, 1)
    node.text = head + "Because "
    run = node.getparent()
    run.addnext(_plain(tail))
    run.addnext(
        _plain(" inside the origin-connected invertibility component.")
    )
    run.addnext(omath(r"\lambda_{\mathrm{true}}"))
    run.addnext(_plain(" for all t in [0, 1], placing "))
    run.addnext(
        omath(r"\rho(B(t\lambda_{\mathrm{true}}))=t\rho(B(\lambda_{\mathrm{true}}))<1")
    )
    run.addnext(
        _plain(
            " in every replication (maximum 0.741), homogeneity of the "
            "spectral radius implies "
        )
    )
    run.addnext(omath(r"\rho(B(\lambda_{\mathrm{true}}))<1"))
    replace_text(body, "For the l1 estimator", "For the ℓ₁ estimator")
    replace_text(body, "SD(Δ_r)/sqrt(30)", "SD(Δ_r)/√30")
    node = _once(body, "B(λ_true)")
    head, tail = node.text.split("B(λ_true)", 1)
    node.text = head
    run = node.getparent()
    run.addnext(_plain(tail))
    run.addnext(omath(r"B(\lambda_{\mathrm{true}})"))
    replace_text(body, "λ1 and λ2", "the two active simulation coefficients")


def supplement(body):
    replace_text(body, STEP3B_OLD, STEP3B_NEW)
    broken = require_one(body, "The adaptive command That option")
    broken.getparent().remove(broken)
    hits = [p for p in body.iter(q(W, "p")) if "Across the retained draws" in ptext(p)]
    if len(hits) != 1:
        raise SystemExit(f"S4.4 paragraph count {len(hits)}")
    hits[0].append(_plain(REGIMES))


def letter(body):
    replace_text(
        body,
        "The revised manuscript is version 32 and the supplement is version 30.",
        "The revised manuscript is version 33 and the supplement is version 31. "
        "Version 33 places the Schur marginal precision inside the Section 7.5 "
        "likelihood sentence and states the spectral-radius homogeneity argument "
        "in Section 7.3.2. python run_all.py calls the adaptive Boston fit and "
        "the offline S&P 100 snapshot.",
    )


def manifest():
    rows = [
        ("run_simulation.py", "Tables 2, 3, 4, 4b, 5; Figures for Moran size/power, Gibbs traces, OOS RMSE"),
        ("run_boston.py", "Tables 6-10 and Boston split archives; reads data/boston_corrected.txt"),
        ("run_sp100.py", "Tables 11-13; missing snapshot raises FileNotFoundError unless --download"),
        ("run_all.py", "Calls run_simulation.py, run_boston.py --adaptive, and run_sp100.py on the offline snapshot"),
        ("run_unified_revision.py", "Tables 4c, S12, S13, S14 and Figure S11"),
        ("run_all_unified.py", "Recomputes Tables 4, 4a, 5 and S&P, then Experiments A-D"),
        ("run_revision_diagnostics.py", "Table 4a Monte Carlo recovery, including network ESS"),
        ("run_table4a_long.py", "Extends replications with split-R-hat above 1.05 and records minimum ESS"),
        ("run_table_s4_prior.py", "Table S4 prior grid, including the radial Beta(1, 2) prior"),
        ("data/boston_corrected.txt", ""),
        ("data/sp100_clean_snapshot.csv", ""),
    ]
    lines = []
    for name, note in rows:
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        comment = f"  # {note}" if note else ""
        lines.append(f"{digest}  {name}{comment}")
    (ROOT / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("manifest", len(lines))


def main():
    main_files, main_root = load_zip(MAIN_IN)
    main_body = main_root.find(q(W, "body"))
    repair_schur(main_body)
    repair_radius(main_body)
    upright_greek(main_body)
    assert_word_safe(main_root, "v33")
    store_zip(MAIN_OUT, main_files, main_root)

    sup_files, sup_root = load_zip(SUP_IN)
    sup_body = sup_root.find(q(W, "body"))
    supplement(sup_body)
    upright_greek(sup_body)
    assert_word_safe(sup_root, "supplement v31")
    store_zip(SUP_OUT, sup_files, sup_root)

    letter_files, letter_root = load_zip(LETTER_IN)
    letter(letter_root.find(q(W, "body")))
    assert_word_safe(letter_root, "letter 8")
    store_zip(LETTER_OUT, letter_files, letter_root)
    manifest()
    print("wrote", MAIN_OUT.name, SUP_OUT.name, LETTER_OUT.name)


if __name__ == "__main__":
    main()
