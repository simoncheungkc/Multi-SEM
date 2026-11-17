"""Build the revised manuscript, supplement, and cover letter.

Reads results/*.json written by run_csda_revision.py and
run_nonnetwork_benchmarks.py. Does not overwrite v9 or supplement v8.
"""

from __future__ import annotations

import copy
import json
import os
import zipfile
from pathlib import Path

from lxml import etree

from mnreg.latex_omml import el, latex_nodes

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
XML = "http://www.w3.org/XML/1998/namespace"


def q(ns, tag):
    return f"{{{ns}}}{tag}"


def para_text(p):
    return "".join(t.text or "" for t in p.iter(q(W, "t")))


def body_paras(body):
    return [(i, child) for i, child in enumerate(body) if child.tag == q(W, "p")]


def find_para(body, start):
    hits = [p for _, p in body_paras(body) if para_text(p).startswith(start)]
    if len(hits) != 1:
        raise SystemExit(f"Expected one paragraph starting {start!r}, found {len(hits)}")
    return hits[0]


def find_paras(body, start):
    return [p for _, p in body_paras(body) if para_text(p).startswith(start)]


def make_p(blocks, source=None):
    p = etree.Element(q(W, "p"))
    if source is not None:
        pPr = source.find(q(W, "pPr"))
        if pPr is not None:
            p.append(copy.deepcopy(pPr))
    for kind, content in blocks:
        if kind == "t":
            r = etree.SubElement(p, q(W, "r"))
            rPr = etree.SubElement(r, q(W, "rPr"))
            fonts = etree.SubElement(rPr, q(W, "rFonts"))
            fonts.set(q(W, "ascii"), "Times New Roman")
            fonts.set(q(W, "hAnsi"), "Times New Roman")
            sz = etree.SubElement(rPr, q(W, "sz"))
            sz.set(q(W, "val"), "24")
            szCs = etree.SubElement(rPr, q(W, "szCs"))
            szCs.set(q(W, "val"), "24")
            t = etree.SubElement(r, q(W, "t"))
            if content[:1] == " " or content[-1:] == " ":
                t.set(q(XML, "space"), "preserve")
            t.text = content
        elif kind == "m":
            p.append(omath(content))
        else:
            raise ValueError(kind)
    return p


def omath(tex):
    nodes = latex_nodes(tex)
    om = el("oMath")
    for node in nodes:
        if etree.QName(node).localname == "e":
            for ch in list(node):
                om.append(ch)
        else:
            om.append(node)
    for mr in om.iter(q(M, "r")):
        rPr = mr.find(q(M, "rPr"))
        if rPr is None:
            rPr = el("rPr")
            mr.insert(0, rPr)
        wrpr = etree.Element(q(W, "rPr"))
        fonts = etree.SubElement(wrpr, q(W, "rFonts"))
        fonts.set(q(W, "ascii"), "Cambria Math")
        fonts.set(q(W, "hAnsi"), "Cambria Math")
        # Word accepts w:rPr as the sibling after m:rPr, which is how the
        # existing equations are stored. Nesting it inside m:rPr makes the
        # package fail to open.
        rPr.addnext(wrpr)
    return om


def replace_with(p, blocks):
    new = make_p(blocks, p)
    parent = p.getparent()
    parent.replace(p, new)
    return new


def insert_before(anchor, blocks):
    new = make_p(blocks, anchor)
    anchor.addprevious(new)
    return new


def insert_after(anchor, blocks):
    new = make_p(blocks, anchor)
    anchor.addnext(new)
    return new


def replace_wt_contains(body, old, new, section_start=None):
    """Replace old with new inside w:t nodes. Optionally only after a heading."""
    armed = section_start is None
    n = 0
    for child in body:
        if child.tag == q(W, "p") and section_start and para_text(child).startswith(section_start):
            armed = True
        if not armed:
            continue
        for t in child.iter(q(W, "t")):
            if t.text and old in t.text:
                t.text = t.text.replace(old, new)
                n += 1
    return n


def set_cell_text(tc, text):
    texts = list(tc.iter(q(W, "t")))
    if not texts:
        return
    texts[0].text = text
    if text[:1] == " " or text[-1:] == " ":
        texts[0].set(q(XML, "space"), "preserve")
    for extra in texts[1:]:
        extra.text = ""


def append_row(tbl, values):
    src = tbl.findall(q(W, "tr"))[1]
    row = copy.deepcopy(src)
    cells = row.findall(q(W, "tc"))
    if len(cells) < len(values):
        raise SystemExit(f"row has {len(cells)} cells, needed {len(values)}")
    for tc, val in zip(cells, values):
        set_cell_text(tc, val)
    tbl.append(row)


def tables(body):
    return [c for c in body if c.tag == q(W, "tbl")]


def retag(node, new_local):
    node.tag = q(M, new_local)
    for child in list(node):
        local = etree.QName(child).localname
        mapped = {"sSupPr": "sSubPr", "sup": "sub"}.get(local)
        if mapped and new_local == "sSub":
            child.tag = q(M, mapped)


def _pair_n_star(parent):
    """Replace a child sequence r(n), r(*) by the subscript n_*."""
    changed = 0
    kids = list(parent)
    i = 0
    while i < len(kids) - 1:
        a, b = kids[i], kids[i + 1]
        if etree.QName(a).localname == "r" and etree.QName(b).localname == "r":
            ta = "".join(t.text or "" for t in a.iter(q(M, "t")))
            tb = "".join(t.text or "" for t in b.iter(q(M, "t")))
            if ta == "n" and tb == "*":
                sub = el(
                    "sSub",
                    el("sSubPr"),
                    el("e", el("r", el("t", text="n"))),
                    el("sub", el("r", el("t", text="*"))),
                )
                parent.replace(a, sub)
                parent.remove(b)
                kids = list(parent)
                changed += 1
                continue
        i += 1
    return changed


def convert_n_star(root):
    """Turn a superscript star on n into a subscript. Leave genuine n^3 alone."""
    changed = 0
    for ssup in list(root.iter(q(M, "sSup"))):
        base = ssup.find(q(M, "e"))
        sup = ssup.find(q(M, "sup"))
        if base is None or sup is None:
            continue
        base_txt = "".join(t.text or "" for t in base.iter(q(M, "t")))
        sup_nodes = list(sup.iter(q(M, "t")))
        sup_txt = "".join(t.text or "" for t in sup_nodes)
        if base_txt == "n" and sup_txt == "*":
            retag(ssup, "sSub")
            changed += 1
        elif base_txt == "n" and sup_txt == "*3":
            # (n_*)^3 rather than n^{*3}
            star = sup_nodes[0]
            star.text = "3"
            sub = el(
                "sSub",
                el("sSubPr"),
                el("e", el("r", el("t", text="n"))),
                el("sub", el("r", el("t", text="*"))),
            )
            # rebuild base
            for ch in list(base):
                base.remove(ch)
            base.append(sub)
            changed += 1
    # Anywhere a run "n" is followed by a run "*", including inside R^{n* x n*}.
    for node in list(root.iter()):
        changed += _pair_n_star(node)
    return changed


def load_json(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def edit_main(root):
    body = root.find(q(W, "body"))
    marg = load_json("marginal_vs_induced_summary.json")
    boot = load_json("bootstrap_matched_table3.json")
    bench = load_json("nonnetwork_benchmarks_summary.json")
    show = bench["showcase"]
    h0 = boot["rates"][0]
    h1 = boot["rates"][1]

    title = find_para(body, "REGRESSION WITH MULTIPLE NETWORKS")
    replace_with(
        title,
        [("t", "Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation")],
    )

    # Section 3.4: purged clique resolvent is exactly the base clique.
    clique = find_para(body, "Row normalization preserves this basic structure.")
    replace_with(
        clique,
        [
            ("t", "The code path row-standardizes every candidate after the resolvent is formed. That standardization sets the diagonal to zero before scaling the row. For an affiliation clique the purge is an identity: the normalized resolvent equals the base clique, "),
            ("m", r"\widetilde{G}(C_m)\equiv C_m"),
            ("t", ". A numerical check on unequal blocks, including a singleton, gives a maximum absolute entrywise difference on the order of "),
            ("m", r"10^{-16}"),
            ("t", ". Before the diagonal is removed the resolvent carries positive diagonal mass, and the Pearson correlation of its off-diagonal entries with "),
            ("m", r"C_m"),
            ("t", " is about 0.999. The diagonal-purged matrix that actually enters estimation is not a new layer."),
        ],
    )

    near = find_para(body, "When an affiliation network contains blocks of different sizes")
    replace_with(
        near,
        [
            ("t", "Because the identity holds inside every clique block, unequal block sizes do not create a new affiliation layer once the diagonal has been purged. The unpurged resolvent remains in "),
            ("m", r"\operatorname{span}\{J, I\}"),
            ("t", " and is excluded by the same redundancy. Resolvent features are retained only for non-clique networks, such as the return-correlation graph, where multi-step paths are not a rewrite of the one-step matrix."),
        ],
    )

    # log|det| convention, placed with the likelihood.
    lik = find_para(body, "is the absolute Jacobian determinant associated")
    insert_after(
        lik,
        [
            ("t", "In every likelihood display, "),
            ("m", r"\log\det(V_\lambda)"),
            ("t", " means "),
            ("m", r"\log\lvert\det(V_\lambda)\rvert"),
            ("t", ". The same absolute value is used for the marginal training precision below."),
        ],
    )

    # L versus S.
    boundary = find_para(body, "Because the stability restriction applies to every posterior draw rather than only to the posterior mean, we also examine the posterior distribution of the boundary statistic")
    insert_after(
        boundary,
        [
            ("t", "Write "),
            ("m", r"L^{(m)}=\sum_{s=1}^{k}\lvert\lambda_s^{(m)}\rvert"),
            ("t", " for that boundary statistic on draw "),
            ("m", r"m"),
            ("t", ", and "),
            ("m", r"S^{(m)}=\sum_{s=1}^{k}\lambda_s^{(m)}"),
            ("t", " for the signed sum in the intercept identity. The stability constraint is "),
            ("m", r"L^{(m)}<1"),
            ("t", ". It is not "),
            ("m", r"S^{(m)}<1"),
            ("t", "."),
        ],
    )

    # Section 4.4 operational rule, before 4.5.
    sec45 = find_para(body, "4.5 Comparison with Spatial Lag")
    rule_blocks = [
        [
            ("t", "Operational calibration rule. For each candidate matrix, before any residual is formed, compute the degree heterogeneity of the unstandardized adjacency and the singular-value concentration of the row-standardized weight matrix,"),
        ],
        [
            ("m", r"\kappa_s=\max_i d_{s,i}/\bar{d}_s"),
            ("t", ", "),
            ("m", r"\psi_s=\theta_{s,1}/\sum_{j=1}^{n}\theta_{s,j}"),
            ("t", ", where "),
            ("m", r"\theta_{s,j}"),
            ("t", " are the singular values of the row-standardized matrix in decreasing order. Degrees are taken before row-standardization, because that standardization forces every non-isolated row sum to one. If "),
            ("m", r"\kappa_s>3.0"),
            ("t", " or "),
            ("m", r"\psi_s>0.20"),
            ("t", ", the analytical normal reference is not used. The p-value is the conditional Gaussian Monte Carlo probability from "),
            ("m", r"B=499"),
            ("t", " draws,"),
        ],
        [
            ("m", r"p_s=\dfrac{1+\sum_{b=1}^{B}\mathbb{I}(\lvert Z_s^{(b)}\rvert\ge\lvert Z_s^{\mathrm{obs}}\rvert)}{B+1}"),
            ("t", ", where "),
            ("m", r"Z_s^{(b)}"),
            ("t", " is the standardized Moran statistic on "),
            ("m", r"e^{(b)}=Mz^{(b)}"),
            ("t", " and "),
            ("m", r"z^{(b)}\sim\mathcal{N}(0,I_n)"),
            ("t", ". If "),
            ("m", r"\kappa_s\le 3.0"),
            ("t", " and "),
            ("m", r"\psi_s\le 0.20"),
            ("t", ", use the analytical tail "),
            ("m", r"p_s=2\bigl(1-\Phi(\lvert Z_s\rvert)\bigr)"),
            ("t", ". The indicators "),
            ("m", r"\kappa_s"),
            ("t", " and "),
            ("m", r"\psi_s"),
            ("t", " depend only on the network, so the choice of reference distribution is fixed before "),
            ("m", r"Y"),
            ("t", " is seen. Section 7.2 evaluates the rule on the Table 3 power-law graph."),
        ],
    ]
    for blocks in rule_blocks:
        insert_before(sec45, blocks)

    # Ex-ante collinearity. Replace the post-FDR pruning paragraph and the "not identical" lead-in.
    col_lead = find_para(body, "As shown in Section 3.4, the row-normalized resolvent")
    replace_with(
        col_lead,
        [
            ("t", "Structural redundancy is removed before any Moran statistic is computed, so the multiple-testing family size does not depend on "),
            ("m", r"Y"),
            ("t", ". After the diagonal purge in Equation (5), an affiliation resolvent is exactly the base clique and is not submitted as a second layer. For the remaining candidates the redundancy measure is the Pearson correlation of the off-diagonal entries only."),
        ],
    )
    prune_hits = [
        p
        for _, p in body_paras(body)
        if para_text(p).startswith("If ") and "collinear duplicates" in para_text(p)
    ]
    if len(prune_hits) != 1:
        raise SystemExit(f"prune paragraph count {len(prune_hits)}")
    new_prune = replace_with(
        prune_hits[0],
        [
            ("t", "If "),
            ("m", r"\lvert r_{\mathrm{vec}}\rvert\ge 0.99"),
            ("t", ", one of the two matrices is dropped before the Moran tests. The correlation is"),
        ],
    )
    insert_after(
        new_prune,
        [
            ("m", r"r_{\mathrm{vec}}(A_r,A_s)=\dfrac{\sum_{i\neq j}(A_{r,ij}-\bar a_r)(A_{s,ij}-\bar a_s)}{\sqrt{\sum_{i\neq j}(A_{r,ij}-\bar a_r)^2}\,\sqrt{\sum_{i\neq j}(A_{s,ij}-\bar a_s)^2}}"),
            ("t", ". The sums run over off-diagonal entries only. On the 63 S&P 100 cross-sections the largest such correlation is 0.972, between the return-correlation network and its resolvent, so the threshold does not remove a layer. The screened family stays at four networks, and the selection counts in Section 8.4 are the same as under the previous post-screen filter."),
        ],
    )

    # Matched bootstrap.
    replace_with(
        find_para(body, "To examine whether finite-sample calibration can be improved"),
        [
            ("t", "The Monte Carlo calibration is computed on the same "),
            ("m", r"n=100"),
            ("t", " power-law graph that produces the 11.6% Gaussian rejection rate in Table 3. Replaying the simulation seed recovers that graph exactly: 236 edges, mean degree 4.72, maximum degree 30. Its degree heterogeneity is "),
            ("m", r"\kappa=6.36"),
            ("t", ", so the Section 4.4 rule requires Monte Carlo p-values. The singular-value share is "),
            ("m", r"\psi=0.052"),
            ("t", ", below 0.20, so the degree ratio is the binding trigger. The design matrix is the one drawn with that graph and is held fixed."),
        ],
    )
    replace_with(
        find_para(body, "For these fixed power-law graphs, the asymptotic Gaussian test"),
        [
            ("t", "Five hundred null samples and five hundred alternative samples are drawn on this fixed graph. Each p-value uses "),
            ("m", r"B=499"),
            ("t", " standard normal vectors projected by the residual maker, with the add-one correction in Section 4.4. The Gaussian test has size "),
            ("m", r"0.116"),
            ("t", " and power "),
            ("m", r"0.734"),
            ("t", ". The Monte Carlo test has size "),
            ("m", r"0.062"),
            ("t", " and power "),
            ("m", r"0.632"),
            ("t", "."),
        ],
    )
    replace_with(
        find_para(body, "Thus, the observed bootstrap rejection rates are broadly compatible"),
        [
            ("t", "On the Table 3 graph, Monte Carlo calibration moves the null rejection rate from 11.6% to 6.2%. Power falls from 73.4% to 63.2%. That power loss is the cost of not using the liberal Gaussian tail. The earlier bootstrap, which drew a different power-law graph for each "),
            ("m", r"n"),
            ("t", ", is not used as evidence about Table 3."),
        ],
    )
    replace_with(
        find_para(body, "For the particular "),
        [
            ("t", "The screening protocol is the rule in Section 4.4. Homogeneous graphs in this design have "),
            ("m", r"\kappa\le 3"),
            ("t", " and keep the analytical normal p-value. The Table 3 power-law graph has "),
            ("m", r"\kappa>3"),
            ("t", " and is calibrated by the conditional Gaussian Monte Carlo."),
        ],
    )

    # Section 6 numbering and the marginal likelihood.
    replace_wt_contains(body, "6.4 Fully Bayesian", "6.3 Fully Bayesian")
    replace_wt_contains(body, "6.5 Methodological", "6.4 Methodological")
    replace_wt_contains(body, "6.6 Predictive", "6.5 Predictive")

    mirrors = find_paras(body, "To mirror real-world partial sampling")
    if len(mirrors) != 2:
        raise SystemExit(f"expected 2 induced-subgraph paragraphs, found {len(mirrors)}")
    replace_with(
        mirrors[0],
        [
            ("t", "The coherent observed-data model is the marginal distribution of the training outcomes under the full-graph filter. Partition "),
            ("m", r"Q=\sigma^{-2}V_\lambda'V_\lambda"),
            ("t", " with the holdout block first,"),
        ],
    )
    # re-find because replace removed the node; the second mirror is still there
    mirror2 = find_paras(body, "To mirror real-world partial sampling")
    if len(mirror2) != 1:
        raise SystemExit("second mirror paragraph missing")
    replace_with(
        mirror2[0],
        [
            ("t", "The reported posterior uses Induced Subgraph Estimation with Full-Graph Imputation, defined in Section 6.1. The sampler is given the re-row-standardized training blocks, and the holdout mean is the full-graph precision predictor. Section 7.5 records the difference between this approximation and the coherent marginal likelihood."),
        ],
    )

    sec62 = find_para(body, "6.2 Precision-Form")
    # Insert in reverse order immediately before 6.2 so they read top to bottom.
    blocks_before_62 = [
        [
            ("m", r"Q=\begin{pmatrix} Q_{**} & Q_{*t} \\ Q_{t*} & Q_{tt} \end{pmatrix}"),
            ("t", ". The Schur complement is the marginal precision of the training disturbances,"),
        ],
        [
            ("m", r"P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}"),
            ("t", ", and "),
            ("m", r"Y_t\sim\mathcal{N}(X_t\beta,\, P_T^{-1})"),
            ("t", ". The training log-likelihood therefore uses "),
            ("m", r"\log\lvert\det(P_T)\rvert"),
            ("t", " rather than the Jacobian of an induced-subgraph filter. Forming "),
            ("m", r"P_T"),
            ("t", " at a proposed "),
            ("m", r"\lambda"),
            ("t", " inverts the holdout block. That inversion is "),
            ("m", r"\mathcal{O}(n_{*}^{3})"),
            ("t", " inside the likelihood loop, against "),
            ("m", r"\mathcal{O}(n_t^{3})"),
            ("t", " for a filter that is already restricted to the training nodes."),
        ],
        [
            ("t", "The estimator implemented in the archive is Induced Subgraph Estimation with Full-Graph Imputation. Parameters are fit on the induced training subgraph after its rows are standardized again, and the holdout mean is then computed from the full precision. The two steps do not use the same spatial filter. The split matches applications in which the analyst sees the training topology before the cross-block coupling to future test units is available. When that coupling is known at estimation time, the marginal likelihood above is the coherent training criterion. Section 7.5 compares the two on a common design."),
        ],
    ]
    for blocks in blocks_before_62:
        insert_before(sec62, blocks)

    # Classical split-R-hat label.
    n_rhat = replace_wt_contains(body, "Split Gelman–Rubin statistic", "Classical split-R-hat")
    if n_rhat < 1:
        raise SystemExit("did not find the Gelman–Rubin label")
    rhat_heads = [p for _, p in body_paras(body) if para_text(p).startswith("Classical split-R-hat")]
    if rhat_heads:
        insert_after(
            rhat_heads[0],
            [
                ("t", "The reported value is the classical split-"),
                ("m", r"\hat{R}"),
                ("t", ": each chain is split in half and the square root of the between-within variance ratio is taken. Rank normalization and folding (Vehtari, Gelman, Simpson, Carpenter & Bürkner, 2021) are not applied. The operational threshold remains 1.025."),
            ],
        )

    # Section 7.5 benchmark, before the empirical section.
    sec8 = find_para(body, "8. Empirical Application")
    lam_i = marg["mean_lambda_induced_ml"]
    lam_m = marg["mean_lambda_marginal_ml"]
    section75 = [
        [
            ("t", "7.5 Induced subgraph versus the coherent marginal likelihood"),
        ],
        [
            ("t", "Thirty replications use "),
            ("m", r"n=200"),
            ("t", " nodes, a 20% holdout ("),
            ("m", r"n_{*}=40"),
            ("t", "), and the two active networks from the main design with "),
            ("m", r"\lambda=(0.40,\,0.35)"),
            ("t", ". Both estimators krige the holdout from the full-graph precision. They differ only in the training criterion: the induced, re-row-standardized subgraph, or the Schur marginal precision "),
            ("m", r"P_T"),
            ("t", ". Concentrated likelihoods and a short Gibbs run (800 iterations, 400 discarded, one chain) are reported. Full chains are unnecessary for the comparison, and the 63-month empirical study is not re-estimated under "),
            ("m", r"P_T"),
            ("t", "."),
        ],
        [
            ("t", f"Mean concentrated-likelihood estimates are ({lam_i[0]:.3f}, {lam_i[1]:.3f}) on the induced subgraph and ({lam_m[0]:.3f}, {lam_m[1]:.3f}) under the marginal likelihood, against the true pair (0.40, 0.35). Across replications the median absolute relative difference in the two coefficients is 11.9% and 13.7%. Both estimators shrink "),
            ("m", r"\lambda"),
            ("t", " toward zero; the marginal likelihood shrinks less. Mean holdout RMSE is "),
            ("t", f"{marg['mean_rmse_induced_ml']:.3f} (induced) and {marg['mean_rmse_marginal_ml']:.3f} (marginal), a 0.30% difference in the averages. The median relative RMSE gap on a single replication is 0.75%. A likelihood evaluation on the induced subgraph is {marg['median_likelihood_speedup']:.1f} times faster, and the Gibbs run is {marg['median_mcmc_speedup_marginal_over_induced']:.1f} times faster."),
        ],
        [
            ("t", "The predictive discrepancy is small. The discrepancy in "),
            ("m", r"\lambda"),
            ("t", " is not small enough to describe the induced subgraph as a 1.5% approximation, and the speedup is not sixfold. The archive keeps the induced estimator because the training cross-block to future test nodes is often unavailable when the parameters are fit, and because the holdout RMSE is insensitive to the choice. When those cross-blocks are known, the marginal likelihood is the criterion that matches the kriging model. The replication file is results/marginal_vs_induced.csv."),
        ],
    ]
    for blocks in section75:
        insert_before(sec8, blocks)

    # Section 8 wording.
    n_gics = 0
    for old, new in (
        ("The GICS affiliation networks are constructed from the corresponding sector and industry classifications.",
         "The affiliation networks are built from the Yahoo Finance sector and industry fields. These labels are the industry-level taxonomy supplied by Yahoo Finance, not proprietary 8-digit MSCI/S&P GICS codes."),
        ("GICS Sector Affiliation Network", "Yahoo Finance Sector Affiliation Network"),
        ("GICS Industry Affiliation Network", "Yahoo Finance Industry Affiliation Network"),
        ("resolvent boosting is not applied to the GICS sector or industry networks.",
         "resolvent boosting is not applied to the Yahoo Finance sector or industry networks."),
        ("Although the row-normalized resolvent of such a clique is not mathematically identical to the original network matrix, its structure remains nearly collinear with the base affiliation matrix because all powers of a clique remain within the same low-dimensional matrix span.",
         "After the diagonal is purged, the normalized resolvent of an affiliation clique equals the base clique, so it is not entered as a separate candidate."),
        ("in the GICS affiliation networks. The GICS Industry network produces ",
         "in the Yahoo Finance affiliation networks. The Yahoo Finance industry network produces "),
        ("while the GICS Sector network produces ",
         "while the Yahoo Finance sector network produces "),
        ("the GICS sector network in 35 months, and the GICS industry network in 33 months.",
         "the Yahoo Finance sector network in 35 months (55.6%), and the Yahoo Finance industry network in 33 months (52.4%)."),
        ("the GICS industry network provides the strongest",
         "the Yahoo Finance industry network provides the strongest"),
        ("an 8-digit GICS sub-industry code",
         "an 8-digit MSCI/S&P GICS sub-industry code"),
        ("approximately 45–51 months", "51 of the 63 months (81.0%)"),
        ("approximately 45-51 months", "51 of the 63 months (81.0%)"),
    ):
        n_gics += replace_wt_contains(body, old, new, section_start="8. Empirical Application")
    if n_gics < 5:
        raise SystemExit(f"GICS replacements too few: {n_gics}")

    universe = find_para(body, "Constituent universe and survivorship protocol.")
    insert_after(
        universe,
        [
            ("t", "The sample comprises the 100 corporate constituents of the S&P 100 as of August 31, 2026. For Alphabet Inc., which issues two liquid share classes (GOOG and GOOGL), only GOOGL (Class A) is retained, so the cross-section does not contain two nodes for one balance sheet."),
        ],
    )

    # Benchmark narrative and table rows.
    show_fe = show["IndustryFE_RMSE"]
    show_fac = show["Factor_RMSE"]
    insert_before(
        find_para(body, "Table 8."),
        [
            ("t", f"Two non-network benchmarks use the same August 2026 split. An industry fixed-effects regression, with an intercept for each Yahoo Finance industry observed in the training sample and with the same slope covariates, has holdout RMSE {show_fe:.4f}, MAE {show['IndustryFE_MAE']:.4f}, and correlation {show['IndustryFE_corr']:.2f}. A one-factor imputation predicts each holdout return by its trailing 36-month beta, shrunk halfway toward 1, times the equal-weighted training return in that month. Its RMSE is {show_fac:.4f}, its MAE is {show['Factor_MAE']:.4f}, and its correlation is {show['Factor_corr']:.2f}. Neither beats OLS (RMSE 0.0637). Multi-SEM (RMSE 0.0576) is the only specification that improves on OLS. Industry means track the cross-section, but they do not deliver the kriging gain."),
        ],
    )
    tbls = tables(body)
    # Table index 8 is August, 9 is longitudinal, from the pre-edit document.
    # Insertions do not add tables, so indices hold.
    append_row(
        tbls[8],
        ["Industry fixed effects", f"{show_fe:.4f}", f"{show['IndustryFE_MAE']:.4f}", f"{show['IndustryFE_corr']:.2f}"],
    )
    append_row(
        tbls[8],
        ["One-factor shrinkage", f"{show_fac:.4f}", f"{show['Factor_MAE']:.4f}", f"{show['Factor_corr']:.2f}"],
    )
    append_row(
        tbls[9],
        [
            "Industry fixed effects",
            f"{bench['mean_IndustryFE_RMSE']:.5f}",
            f"{bench['mean_IndustryFE_MAE']:.5f}",
        ],
    )
    append_row(
        tbls[9],
        [
            "One-factor shrinkage",
            f"{bench['mean_Factor_RMSE']:.5f}",
            f"{bench['mean_Factor_MAE']:.5f}",
        ],
    )
    insert_before(
        find_para(body, "Table 9."),
        [
            ("t", f"The same two benchmarks, averaged over the 63 matched months, have RMSE {bench['mean_IndustryFE_RMSE']:.5f} (industry fixed effects) and {bench['mean_Factor_RMSE']:.5f} (one-factor shrinkage), against {bench['mean_OLS_RMSE']:.5f} for OLS, {bench['mean_IndustrySEM_RMSE']:.5f} for the industry SEM, and {bench['mean_MultiSEM_RMSE']:.5f} for Multi-SEM. Industry intercepts are worse than OLS: a saturated group mean fitted on about 80 training names does not travel to the holdout. The industry SEM, which models within-industry residual equicorrelation rather than a mean shift, is the stronger industry benchmark. Multi-SEM’s remaining gain over that SEM is the screened correlation structure, not a relabeling of industry dummies."),
        ],
    )

    nstar = convert_n_star(root)
    print(f"main n-star conversions: {nstar}; GICS/text replacements: {n_gics}")

    # Guard: no bare m:e directly under a new or old oMath.
    bad = 0
    for om in root.iter(q(M, "oMath")):
        for ch in om:
            if etree.QName(ch).localname == "e":
                bad += 1
    if bad:
        raise SystemExit(f"bare m:e under oMath: {bad}")


def edit_supplement(root):
    body = root.find(q(W, "body"))
    n = 0
    for old, new in (
        ("GICS Sector", "Yahoo Finance Sector"),
        ("GICS Industry", "Yahoo Finance Industry"),
        ("GICS sector", "Yahoo Finance sector"),
        ("GICS industry", "Yahoo Finance industry"),
        ("S4.1 Constituent Listing and GICS Classification", "S4.1 Constituent Listing and Yahoo Finance Classification"),
        ("8-digit GICS", "8-digit MSCI/S&P GICS"),
    ):
        n += replace_wt_contains(body, old, new)
    practical = [
        p for _, p in body_paras(body) if para_text(p).startswith("The practical consequence is near-collinearity")
    ]
    if len(practical) != 1:
        raise SystemExit(f"supplement clique paragraph: {len(practical)}")
    replace_with(
        practical[0],
        [
            ("t", "The implementation then row-standardizes the resolvent, which sets the diagonal to zero before scaling each row. For an affiliation clique that step returns the base clique exactly; a numerical check on unequal blocks gives a maximum absolute entrywise difference on the order of 10^{-16}. The unpurged resolvent is the matrix in "),
            ("m", r"\operatorname{span}\{J,I\}"),
            ("t", ". It has a positive diagonal and an off-diagonal correlation of about 0.999 with the clique, so it would also be dropped by the screen in Section 4.6. The matrix that enters the model is the purged one, and it is not a second layer."),
        ],
    )
    mirrors = find_paras(body, "To mirror real-world partial sampling")
    for p in mirrors:
        replace_with(
            p,
            [
                ("t", "The reported fit is Induced Subgraph Estimation with Full-Graph Imputation. Parameters are estimated on the re-row-standardized training blocks, and the holdout mean uses the full-graph precision. The coherent marginal likelihood is the Schur complement stated above in this supplement and compared with the induced estimator in Section 7.5 of the main text."),
            ],
        )
    freq = [
        p for _, p in body_paras(body) if para_text(p).startswith("The network layers retained by the residual Moran")
    ]
    if len(freq) != 1:
        raise SystemExit(f"supplement selection paragraph: {len(freq)}")
    replace_with(
        freq[0],
        [
            ("t", "The network layers retained by residual Moran screening change across the 63 months. The return-correlation network or its resolvent is selected in 51 of the 63 months (81.0%), the Yahoo Finance sector network in 35 months (55.6%), and the Yahoo Finance industry network in 33 months (52.4%). No single layer is selected in every month. These counts are taken from results/sp100_running_oos.csv. The largest off-diagonal correlation among the four candidates over those months is 0.972, below the 0.99 pruning threshold, so the family size is four throughout."),
        ],
    )
    # Branch A sentence if the supplement still says the resolvent is not identical.
    hits = [
        p
        for _, p in body_paras(body)
        if "not mathematically identical" in para_text(p) or "nonzero diagonal" in para_text(p).lower()
    ]
    for p in hits:
        # Only rewrite a pure-prose paragraph. Mixed math paragraphs are left and a new sentence is added.
        if p.find(".//" + q(M, "oMath")) is None and "resolvent" in para_text(p).lower():
            replace_with(
                p,
                [
                    ("t", "After row-standardization the diagonal of the resolvent is set to zero. For an affiliation clique that purge returns the base clique exactly, so the normalized resolvent is not a second parameter. Before the purge, the resolvent has a positive diagonal and an off-diagonal correlation of about 0.999 with the clique."),
                ],
            )
    # Marginal likelihood pointer in S1.4 if a Schur sentence exists.
    schur = [p for _, p in body_paras(body) if "Schur" in para_text(p)]
    if schur:
        insert_after(
            schur[0],
            [
                ("t", "The coherent training model is the marginal "),
                ("m", r"Y_t\sim\mathcal{N}(X_t\beta, P_T^{-1})"),
                ("t", " with "),
                ("m", r"P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}"),
                ("t", " and "),
                ("m", r"Q=\sigma^{-2}V_\lambda'V_\lambda"),
                ("t", ". The archive instead uses induced-subgraph estimation with full-graph imputation. Section 7.5 of the main text compares the two. Evaluating "),
                ("m", r"P_T"),
                ("t", " inside each proposal costs an extra "),
                ("m", r"\mathcal{O}(n_{*}^{3})"),
                ("t", " inversion of the holdout block."),
            ],
        )
    nstar = convert_n_star(root)
    print(f"supplement text replacements: {n}; n-star conversions: {nstar}; resolvent paragraphs rewritten: {len(hits)}")


def save_docx(src: Path, dst: Path, editor):
    xml_bytes = None
    with zipfile.ZipFile(src) as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        editor(root)
        xml_bytes = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        tmp = dst.with_suffix(".tmp.docx")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = xml_bytes if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)
    os.replace(tmp, dst)
    print("wrote", dst.name, dst.stat().st_size)


def word_count(path: Path) -> int:
    root = etree.fromstring(zipfile.ZipFile(path).read("word/document.xml"))
    text = " ".join(t.text or "" for t in root.iter(q(W, "t")))
    return len(text.split())


def write_cover():
    from docx import Document

    marg = load_json("marginal_vs_induced_summary.json")
    boot = load_json("bootstrap_matched_table3.json")
    bench = load_json("nonnetwork_benchmarks_summary.json")
    doc = Document()
    doc.add_paragraph("Computational Statistics & Data Analysis")
    doc.add_paragraph("Section: Statistical Methodology for Data Analysis")
    doc.add_paragraph("")
    doc.add_paragraph("Dear Editors,")
    paras = [
        "We submit “Multiplex Spatial Error Regression: Calibrated Screening, Stable Bayesian Estimation, and Precision Imputation” for consideration as a methodological paper. The contribution is a computational pipeline for partially observed multiplex networks: screen candidate layers, keep the estimator inside an ℓ1 stability region, and impute held-out nodes from the precision matrix. Equity cross-sections are the worked example. The paper is not an empirical finance study.",
        "Three pieces carry the method. First, overlapping row-standardized networks are kept invertible by the constraint ‖λ‖1 = Σ |λs| < 1, which is distinct from the signed sum that appears in the intercept identity. Second, residual Moran tests use design-conditional moments, and a pre-data rule switches to conditional Gaussian Monte Carlo when degree heterogeneity κ exceeds 3 or the leading singular value carries more than 20% of the singular-value mass. On the same n=100 power-law graph that has Gaussian size 11.6% in the size table, that Monte Carlo (B=499) has size 6.2% and power 63.2%, against Gaussian power 73.4%. Third, holdout means are precision-form kriging. Factoring the holdout block is O(n_*^3) rather than a factorization of the training covariance.",
        "Estimation in the archive is induced-subgraph estimation with full-graph imputation. The manuscript now writes the coherent marginal likelihood, PT = Qtt − Qt* Q**^{-1} Q*t, and compares the two criteria in 30 replications at n=200. Mean holdout RMSE differs by 0.30%. The network coefficients do not: the median absolute relative gap is about 12–14%, and the marginal likelihood is less attenuated. A likelihood evaluation on the induced subgraph is "
        f"{marg['median_likelihood_speedup']:.1f} times faster, and the Gibbs comparison is {marg['median_mcmc_speedup_marginal_over_induced']:.1f} times faster. We do not describe the approximation as a 1.5% agreement in λ or as a sixfold speedup. It is the estimator we use when cross-block links to future test nodes are not available at fitting time.",
        f"On the S&P 100 splits already stored in the archive, industry fixed effects and a shrunk one-factor imputation do not beat ordinary least squares. Over 63 months their mean holdout RMSEs are {bench['mean_IndustryFE_RMSE']:.5f} and {bench['mean_Factor_RMSE']:.5f}, against {bench['mean_OLS_RMSE']:.5f} for least squares and {bench['mean_MultiSEM_RMSE']:.5f} for the multiplex model. Industry labels in the paper are Yahoo Finance fields, not 8-digit GICS codes. The return-correlation network or its resolvent is selected in 51 of 63 months, the sector network in 35, and the industry network in 33. Alphabet is represented by GOOGL only.",
        "The replication archive contains the Python package mnreg, the cached cross-sections, run_simulation.py, run_sp100.py, and the two revision scripts that produced the new Monte Carlo and the non-network benchmarks. We hope the paper is suitable for Statistical Methodology for Data Analysis.",
        "Sincerely,",
        "Simon K. C. Cheung and Tommy K. Y. Cheung",
    ]
    for text in paras:
        doc.add_paragraph(text)
    dest = ROOT / "Cover_Letter_MultiSEM_CSDA.docx"
    doc.save(dest)
    print("wrote", dest.name)


def main():
    save_docx(
        ROOT / "REGRESSION WITH MULTIPLE NETWORKS v9.docx",
        ROOT / "REGRESSION WITH MULTIPLE NETWORKS v10.docx",
        edit_main,
    )
    save_docx(
        ROOT / "Supplementary_Materials_MultiSEM_v8.docx",
        ROOT / "Supplementary_Materials_MultiSEM_v9.docx",
        edit_supplement,
    )
    write_cover()
    for name in (
        "REGRESSION WITH MULTIPLE NETWORKS v10.docx",
        "Supplementary_Materials_MultiSEM_v9.docx",
    ):
        path = ROOT / name
        print(name, "w:t words", word_count(path))


if __name__ == "__main__":
    main()
