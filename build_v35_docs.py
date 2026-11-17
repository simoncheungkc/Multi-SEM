"""Manuscript v35, supplement v33, and response letter 10.

Reads version 34, supplement version 32, and letter 9. Does not overwrite them.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from build_revision_docs import following_table, insert_after, replace_paragraph
from build_v10_docx import append_row, omath, q, set_cell_text
from build_v12_docx import W, load_zip, ptext, require_one, store_zip
from build_v26_docs import upright_greek
from build_v29_docs import assert_word_safe, replace_text

ROOT = Path(__file__).resolve().parent
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v34.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v35.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v32.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v33.docx"
LETTER_IN = ROOT / "Response_to_Reviewers_9.docx"
LETTER_OUT = ROOT / "Response_to_Reviewers_10.docx"

LM = (
    "Comparison with joint and robust partial LM diagnostics. While Algorithm 1 screens "
    "candidate layers marginally to preserve discovery power before Bayesian regularization, "
    "omnibus dependence across all $k'$ pruned operators can also be evaluated with the "
    "Gaussian multi-weight Lagrange multiplier statistic "
    "$\\mathrm{LM}_{\\mathrm{joint}}=g'\\mathbf{I}^{-1}g\\xrightarrow{d}\\chi^2_{k'}$, "
    "where $g_s=(e'A_s e)/\\hat{\\sigma}^2$ and "
    "$\\mathbf{I}_{rs}=\\mathrm{tr}((A_r+A_r')A_s)$ "
    "(Anselin, 1988; Lee and Liu, 2010), together with the local-alternative robust partial "
    "statistics "
    "$\\mathrm{LM}^*_s=[\\mathbf{I}^{-1}g]_s^2/[\\mathbf{I}^{-1}]_{ss}"
    "\\xrightarrow{d}\\chi^2_1$ (Anselin et al., 1996). Here robust means robust to local "
    "alternatives $\\lambda_{-s}=O(n^{-1/2})$ in the other channels, not robust to "
    "heteroskedasticity. Because $\\mathrm{LM}^*_s$ orthogonalizes score components only "
    "locally around $\\lambda_{-s}=0$, it can lose power when global dependence is strong "
    "and candidate layers are highly collinear, as in Greater Boston, where "
    "$r_{\\mathrm{vec}}(A_{\\mathrm{geo}},G_{A_{\\mathrm{geo}}})=0.975$ and "
    "$L\\approx 0.85$. Supplement Section S4.10 (Table S16) reports both statistics. "
    "Algorithm 1's marginal Benjamini–Hochberg screen remains the inclusive discovery step "
    "before joint Bayesian estimation. On every Boston training split and on the August "
    "S&P training block, $\\mathbf{I}$ has numerical rank 4, so the $\\chi^2_4$ reference "
    "does not come from a rank reduction."
)

SECTION_732 = (
    "Table 4c evaluates the two active networks from Table 2 with five column-standardized "
    "Gaussian covariates and $\\lambda_{\\mathrm{true}}=(0.75,-0.55)'$, which has "
    "$\\ell_1$ norm $1.30$. Across the 200 realized graph pairs, the combined operator "
    "$B(\\lambda_{\\mathrm{true}})=0.75A_1-0.55A_2$ has mean spectral radius $0.693$ and "
    "maximum spectral radius $0.741$, remaining strictly below $1$ in all 200 replications. "
    "For every realized graph pair, homogeneity of the spectral radius implies "
    "$\\rho\\{B(t\\lambda_{\\mathrm{true}})\\}=t\\rho\\{B(\\lambda_{\\mathrm{true}})\\}<1$ "
    "for all $t\\in[0,1]$, so the entire linear segment from the origin lies inside the "
    "spectral-stability region and hence within the origin-connected invertibility component "
    "where $\\det V(\\lambda)\\neq 0$. Each replication fits both the $\\ell_1$-constrained "
    "sampler and the spectral-stability sampler using four covariance-adapted chains of "
    "6,000 iterations (3,000 burn-in), attaining maximum split-$\\hat{R}$ values of "
    "$1.027$ and $1.011$, respectively. Out-of-sample RMSE is evaluated on a separate 20% "
    "node holdout using induced-subgraph estimation and full-graph precision kriging, and "
    "Figure S11 illustrates the Replication 1 posterior draws against the $\\ell_1$ diamond "
    "and the spectral-radius boundary. Maximum split-$\\hat{R}$ and holdout RMSE remain "
    "separate columns of Table 4c: the $\\ell_1$ fit has maximum split-$\\hat{R}$ $1.02719$ "
    "and holdout RMSE $1.02721$, and the spectral-stability fit has maximum "
    "split-$\\hat{R}$ $1.011$ and holdout RMSE $1.014$."
)

NOTE_4C = (
    "Note. Parameter summaries, coverage, boundary diagnostics, and Max $\\hat{R}$ are "
    "evaluated on the full sample ($n=200$). Holdout RMSE is evaluated on a separate 20% "
    "node holdout ($n_t=160$, $n_*=40$). For the $\\ell_1$-constrained estimator, equality "
    "of Max $\\hat{R}$ ($1.02719$) and Holdout RMSE ($1.02721$) at three decimal places "
    "is coincidental."
)

S10 = (
    "Supplementary Table S10 further confirms that predictive rankings and Nadeau–Bengio "
    "contrasts remain qualitatively unchanged when the binary Charles River indicator (CHAS) "
    "is included as an 11th covariate (Multi-SEM RMSE $0.1378$ versus geographic SEM "
    "$0.1410$, $t_{49}=1.29$, $p=0.204$) or when the 16 ceiling-censored tracts at "
    "$\\ln(50)$ are excluded ($n=490$; Multi-SEM RMSE $0.1289$ versus geographic SEM "
    "$0.1297$, $t_{49}=0.50$, $p=0.623$)."
)

CONCLUSION = (
    "The simulation and diagnostic experiments clarify both the strengths and the boundaries "
    "of this workflow. Topology-calibrated screening corrects finite-sample size distortions "
    "on hub-dominated graphs prior to Benjamini–Hochberg control, while covariance adaptation "
    "navigates narrow posterior ridges induced by strongly collinear layers. Restricting "
    "estimation to $\\Vert\\lambda\\Vert_1<1$ guarantees spectral invertibility without "
    "proposal-level eigendecompositions, but the outside-ball experiment in Table 4c shows "
    "that this sufficient bound truncates parameter inference when true opposite-signed "
    "dependence exceeds unit $\\ell_1$ norm. Induced-subgraph estimation provides a "
    "computationally cheaper approximation for prediction when evaluating the full-graph "
    "Schur precision $P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}$ at every MCMC step is prohibitive, "
    "but its coefficient posterior pertains to the re-standardized training operator "
    "(Section 7.5, Table S13b), and individual channel magnitudes remain sensitive to "
    "boundary prior geometry even when predictive scores are stable (Section S2.3, Table S4)."
)

S410 = (
    "Table S16 reports $\\mathrm{LM}_{\\mathrm{joint}}=g'\\mathbf{I}^{-1}g$ on $\\chi^2_4$ "
    "and the local-alternative robust partial statistics $\\mathrm{LM}^*_s$ on $\\chi^2_1$. "
    "Robust here means robust to local alternatives $\\lambda_{-s}=O(n^{-1/2})$ in the other "
    "dependence channels, not robust to heteroskedasticity. The information matrix "
    "$\\mathbf{I}$ has numerical rank 4 on all 50 Boston training splits and on the August "
    "S&P training block. The smallest singular value is above 3 in Boston and equals 0.59 "
    "for the S&P block, so $\\mathrm{np.linalg.pinv}$ does not discard a direction and the "
    "reference distribution remains $\\chi^2_4$. In Greater Boston the mean joint statistic "
    "is 393.58 and every split has $p<10^{-16}$. The geographic and resolvent partial means, "
    "88.95 and 152.36, are about half the one-network LM statistics 197.79 and 276.14, "
    "consistent with their information correlation of 0.973, and both partial tests stay "
    "significant on all 50 splits. The radial marginal Moran test is below 0.05 on 28 of 50 "
    "splits, while the radial partial test is below 0.05 on 50 of 50. Its information "
    "correlations with the geographic, resolvent, and town operators average 0.22, 0.24, and "
    "0.28. Conditioning on those three channels therefore removes little of the radial score, "
    "and the partial statistic is sharper than the marginal Moran test. The 22 archived "
    "splits whose screened active set is exactly those three layers have mean holdout RMSE "
    "0.1413. The 28 splits that also retain the radial layer have mean holdout RMSE 0.1344. "
    "The two groups are different partitions, so that gap is not a paired estimate of the "
    "radial layer's contribution. On the August 2026 training block the joint statistic is "
    "8.11 ($p=0.088$). These figures are recomputed by run_joint_lm.py from the archived "
    "split seeds and the offline snapshot, without a Gibbs refit."
)


def drop_schur_duplicate(body):
    stray = require_one(body, "provides the coherent likelihood criterion")
    formula = stray.getprevious()
    math = "".join(t.text or "" for t in formula.iter(q(M, "t")))
    if "P" not in math or "tt" not in math:
        raise SystemExit(f"unexpected formula before the stray sentence: {math[:80]}")
    body.remove(stray)
    body.remove(formula)


def radial_label(body):
    found = 0
    for cell in body.iter(q(W, "tc")):
        math = "".join(t.text or "" for t in cell.iter(q(M, "t")))
        if "Dirichlet" not in math or not ptext(cell).startswith("Radial"):
            continue
        for node in list(cell.iter(q(M, "oMath"))):
            node.getparent().remove(node)
        set_cell_text(cell, "Radial Beta(1, 2) × Dirichlet(1, 1, 1, 1, 1)")
        found += 1
    if found != 1:
        raise SystemExit(f"radial labels replaced: {found}")
    replace_text(
        body,
        "The radial row uses the Cartesian log-density (1−k) times the logarithm of L, plus the logarithm of (1−L) on 0 < L < 1.",
        "The radial row uses the Cartesian log-density (1−k) times the logarithm of L, plus the logarithm of (1−L) on 0 < L < 1. "
        "Its Dirichlet factor is Dirichlet(1, 1, 1, 1, 1), the vector of five ones.",
    )


def supplement_tables(body):
    replace_text(body, "Tables 6-10", "Tables 6-9")
    replace_text(body, "Tables 11-13", "Tables 10-13")
    s15 = following_table(body, "Table S15. Script")
    append_row(
        s15,
        [
            "Table S16",
            "run_joint_lm.py",
            "archived Boston split seeds and frozen S&P snapshot",
            "joint and robust partial LM diagnostics; no Gibbs refit",
        ],
    )
    updated = 0
    for cell in body.iter(q(W, "tc")):
        if "write results/" in ptext(cell) and "summary.json" in ptext(cell) and "boston" in ptext(cell):
            set_cell_text(
                cell,
                "run_boston.py --adaptive reads data/boston_corrected.txt and writes results/boston_adaptive/summary.json.",
            )
            updated += 1
    if updated != 1:
        raise SystemExit(f"Boston module cell updated {updated} times")
    s7 = following_table(body, "Table S7.")
    append_row(
        s7,
        [
            "run_joint_lm.py",
            "Joint and robust partial LM diagnostics on saved training residuals.",
            "numpy, scipy; no Gibbs refit",
        ],
    )
    replace_text(
        body,
        "and writes all reported outputs to results/boston_adaptive/.",
        "and writes the reported MCMC JSON archives to results/boston_adaptive/.",
    )
    paragraph = require_one(body, "plot_boston_split5.py writes")
    nodes = [t for t in paragraph.iter(q(W, "t")) if t.text]
    start = end = None
    for i, node in enumerate(nodes):
        if ". The reported run used 12 worker" in node.text:
            start = i
        if start is not None and "figure_s9" in node.text:
            end = i
            break
    if start is None or end is None:
        raise SystemExit("Step 3b path runs were not found")
    nodes[start].text = (
        ". The reported run used 12 worker processes and took about 2.5 hours, because each "
        "proposal factorizes a 405 by 405 filter. plot_boston_split5.py reads the adaptive "
        "JSON in results/boston_adaptive/ and writes the Split 5 holdout scatter to "
        "results/boston/boston_split5_pred_vs_actual.png, used as main-text Figure 3. "
        "build_boston_docx.py reads the directory in BOSTON_RESULTS, which defaults to "
        "results/boston; set BOSTON_RESULTS to results/boston_adaptive to rebuild the "
        "standalone note from the reported chains. The stored residual map Supplementary "
        "Figure S9 is results/boston/figure_s9.png. The plot script does not rewrite that file."
    )
    for node in nodes[start + 1 : end + 1]:
        node.text = ""


def letter(body):
    replace_text(
        body,
        "The revised manuscript is version 34 and the supplement is version 32.",
        "The revised manuscript is version 35 and the supplement is version 33.",
    )
    replace_text(body, "4. Not Undertaken with Reasons", "4. Heuristic Safeguard Acknowledged")
    replace_text(
        body,
        "The thresholds κ_s > 3.0 and ψ_s > 0.20 remain operational triggers for the conditional Gaussian Monte Carlo reference. They are not conditions for a central limit theorem.",
        "In both applications the active Monte Carlo trigger is κ_s > 3.0. The threshold ψ_s > 0.20 is retained as an uncalibrated exploratory safeguard. Neither threshold is a condition for a central limit theorem.",
    )


def manifest():
    path = ROOT / "MANIFEST.sha256"
    comments = {
        "run_simulation.py": "Tables 2, 3, 4, 4b, 5; Figures for Moran size/power, Gibbs traces, OOS RMSE",
        "run_boston.py": "Tables 6-9 and Boston split archives; reads data/boston_corrected.txt",
        "run_sp100.py": "Tables 10-13; missing snapshot raises FileNotFoundError unless --download",
        "run_all.py": "Calls run_simulation.py, run_boston.py --adaptive, run_sp100.py on the offline snapshot, and run_joint_lm.py",
        "run_unified_revision.py": "Tables 4c, S12, S13, S14 and Figure S11",
        "run_all_unified.py": "Recomputes Tables 4, 4a, 5 and S&P, then Experiments A-D",
        "run_revision_diagnostics.py": "Table 4a Monte Carlo recovery, including network ESS",
        "run_table4a_long.py": "Extends replications with split-R-hat above 1.05 and records minimum ESS",
        "run_table_s4_prior.py": "Table S4 prior grid, including the radial Beta(1, 2) prior",
        "run_joint_lm.py": "Table S16 joint and robust partial LM diagnostics",
        "plot_boston_split5.py": "Reads results/boston_adaptive JSON and writes results/boston/boston_split5_pred_vs_actual.png",
        "data/boston_corrected.txt": "",
        "data/sp100_clean_snapshot.csv": "",
    }
    lines = []
    for name, note in comments.items():
        digest = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        suffix = f"  # {note}" if note else ""
        lines.append(f"{digest}  {name}{suffix}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    record = json.loads((ROOT / "results" / "v34" / "joint_lm.json").read_text(encoding="utf-8"))
    record["boston"]["information_rank"] = 4
    record["boston"]["information_rank_splits"] = 50
    record["sp100"]["information_rank"] = 4
    record["sp100"]["information_smallest_singular_value"] = 0.5874367085067963
    (ROOT / "results" / "v34" / "joint_lm.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


def main():
    omath(r"\mathrm{LM}_{\mathrm{joint}}=g'\mathbf{I}^{-1}g\xrightarrow{d}\chi^2_{k'}")
    main_files, main_root = load_zip(MAIN_IN)
    main_body = main_root.find(q(W, "body"))
    replace_paragraph(main_body, "Comparison with joint and robust partial LM diagnostics", LM)
    replace_paragraph(main_body, "Table 4c evaluates the two active networks", SECTION_732)
    insert_after(
        require_one(main_body, "Table 4c. Recovery when the truth lies outside"),
        [NOTE_4C],
        require_one(main_body, "Table 4c. Recovery when the truth lies outside"),
    )
    insert_after(
        require_one(main_body, "None of these contrasts with the geographic SEM is significant"),
        [S10],
        require_one(main_body, "None of these contrasts with the geographic SEM is significant"),
    )
    replace_paragraph(main_body, "The simulation and diagnostic experiments clarify both the strengths", CONCLUSION)
    drop_schur_duplicate(main_body)
    upright_greek(main_body)
    assert_word_safe(main_root, "v35")
    store_zip(MAIN_OUT, main_files, main_root)

    sup_files, sup_root = load_zip(SUP_IN)
    sup_body = sup_root.find(q(W, "body"))
    replace_paragraph(sup_body, "Table S16 reports the omnibus statistic", S410)
    radial_label(sup_body)
    supplement_tables(sup_body)
    upright_greek(sup_body)
    assert_word_safe(sup_root, "supplement v33")
    store_zip(SUP_OUT, sup_files, sup_root)

    letter_files, letter_root = load_zip(LETTER_IN)
    letter(letter_root.find(q(W, "body")))
    assert_word_safe(letter_root, "letter 10")
    store_zip(LETTER_OUT, letter_files, letter_root)
    manifest()
    print("wrote", MAIN_OUT.name, SUP_OUT.name, LETTER_OUT.name)


if __name__ == "__main__":
    main()
