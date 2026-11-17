"""Corrections from v19 and supplement v17. Does not overwrite those files."""

from __future__ import annotations

import build_v12_docx as b

ROOT = b.ROOT
W = b.W
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v19.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v17.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v20.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v18.docx"
LETTER_IN = ROOT / "Cover_Letter_MultiSEM_EcoSta.docx"
LETTER_OUT = ROOT / "Cover_Letter_MultiSEM_EcoSta_v2.docx"


def replace_once(body, old, new):
    hits = [t for t in body.iter(b.q(W, "t")) if t.text and old in t.text]
    if len(hits) != 1:
        raise SystemExit(f"{old!r} matched {len(hits)}")
    hits[0].text = hits[0].text.replace(old, new, 1)


def replace_all_text(body, old, new, expected):
    hits = [t for t in body.iter(b.q(W, "t")) if t.text and old in t.text]
    if len(hits) != expected:
        raise SystemExit(f"{old!r} matched {len(hits)}, expected {expected}")
    for node in hits:
        node.text = node.text.replace(old, new)


def edit_main(body):
    replace_once(
        body,
        "while guaranteeing an invertible spatial filter.",
        "while enforcing a uniform sufficient condition for invertibility of the spatial filter.",
    )
    replace_once(
        body,
        "using four chains of 6,000 iterations each, with the first 3,000 iterations discarded as burn-in",
        "using four chains of 8,000 iterations each, with the first 4,000 iterations discarded as burn-in",
    )
    replace_all_text(body, "0.1374", "0.1375", 3)
    replace_once(body, "0.1361", "0.1363")
    replace_once(body, "0.1376", "0.1378")
    replace_once(
        body,
        "is 1.000, so every retained draw stays inside the open",
        "is 0.999998, so every retained draw remains strictly inside the open",
    )
    replace_once(body, "Table 7 reports the posterior summaries.", "Table 11 reports the posterior summaries.")
    replace_once(
        body,
        "Table 10 summarizes predictive performance across all 63 monthly cross-sections.",
        "Table 13 summarizes predictive performance across all 63 monthly cross-sections.",
    )
    b.replace_containing(
        body,
        "The empirical analysis considers S&P100 constituent equities",
        "The empirical analysis applies Multi-SEM to two complementary settings: a geographic hedonic "
        "cross-section of Boston census tracts and a repeated cross-sectional panel of S&P 100 equities. "
        "The Boston application (Section 8.1) provides the primary spatial-statistical evaluation, "
        "demonstrating how the pipeline separates continuous physical distance decay from discrete "
        "municipal administrative boundaries. The S&P 100 study (Section 8.2) examines the portability "
        "of the framework to dynamic, non-geographic relational networks across 63 consecutive rolling months.",
    )
    b.remove_containing(body, "The same analytical pipeline is applied to two distinct settings")
    design = b.require_one(body, "Using random seed 2026, we generate 50 train")
    design.addnext(b.make_p(b.blocks_from(
        "The random partitions assess spatial imputation inside the observed metropolitan area. "
        "Each holdout tract is predicted from other tracts on the same map, which is the usual target "
        "of spatial hedonic kriging. Extrapolation to a municipality or region that is absent from the "
        "training sample is a different target and is not claimed here."
    ), design))
    pointer = b.require_one(body, "Section 7.5 compares this approximation")
    pointer.addnext(b.make_p(b.blocks_from(
        "Section 7.5 is the reason the Boston fits keep the induced subgraph. On the simulated networks "
        "the two criteria differ by 0.31% in mean holdout RMSE at $n = 200$ and by 0.08% at $n = 500$. "
        "The induced Gibbs sampler is 2.2 times faster on those $n = 200$ replications (1.67 seconds "
        "against 3.67 seconds), and one likelihood evaluation is 2.6 times faster. Forming the Schur "
        "complement inside every proposal would repeat that extra factorization at each iteration."
    ), pointer))
    rmse = b.require_one(body, "The geographic SEM alone achieves a mean RMSE of 0.1403")
    rmse.addnext(b.make_p(b.blocks_from(
        "Screened Multi-SEM has a lower RMSE than that geographic SEM on 37 of the 50 splits. "
        "The Wilcoxon signed-rank test of the paired squared errors gives $p = 7.50 \\times 10^{-5}$. "
        "The 2.0% reduction in mean RMSE compares the multiplex predictor with the geographic SEM, "
        "whose mean RMSE is 0.1403."
    ), rmse))


def edit_supplement(body):
    b.replace_containing(
        body,
        "Propose .",
        "Propose $\\lambda^* \\sim N(\\lambda^{(t-1)}, \\tau_t^2 \\Sigma_t)$, where $\\Sigma_t = I_k$ "
        "for the first 50 burn-in iterations and "
        "$\\Sigma_t = (2.38^2/k)\\mathrm{Cov}(\\lambda^{(0:t-1)}) + 10^{-5} I_k$ "
        "during the remainder of burn-in. After burn-in, $\\Sigma_t$ and $\\tau_t$ are held fixed.",
    )
    replace_once(
        body,
        "defined in Section 8.1 of the main text",
        "defined in Section 8.2 of the main text",
    )
    replace_once(
        body,
        "The August 2026 cross-section provides a detailed illustration of the complete ",
        "The August 2026 cross-section provides an archived illustrative snapshot of Multi-SEM estimation, posterior inference, and precision-form imputation. ",
    )
    replace_once(body, " procedure. Table 10 in the main text", "Table 10 in the main text")
    showcase = b.require_one(body, "archived illustrative snapshot")
    for node in showcase.iter(b.q(W, "t")):
        if node.text == "Multi-SEM":
            node.text = ""
            break


def edit_letter(body):
    anchor = b.require_one(body, "Sincerely")
    anchor.addprevious(b.make_p(b.blocks_from(
        "The Boston census-tract study is the primary spatial application. Random 80/20 splits assess "
        "imputation within the observed metropolitan area. Extrapolation to an unseen region is left to future work. "
        "Relative to OLS, screened Multi-SEM reduces mean holdout RMSE from 0.1921 to 0.1375 (28.5%; "
        "Wilcoxon $p = 1.78 \\times 10^{-15}$; 50 of 50 splits). Relative to the single-layer geographic "
        "SEM (0.1403) the reduction is 2.0%, and Multi-SEM has the lower RMSE on 37 of 50 splits "
        "(Wilcoxon $p = 7.50 \\times 10^{-5}$). The Boston chains use the covariance-adapted proposal "
        "in Section 5.3, with 8,000 iterations and 4,000 burn-in draws. Section 7.5 reports why the "
        "fits use the induced subgraph rather than the Schur marginal likelihood: holdout RMSE differs "
        "by 0.31% at $n = 200$ and by 0.08% at $n = 500$, while the induced sampler is 2.2 times faster "
        "and one likelihood evaluation is 2.6 times faster. The screening cutoffs $\\kappa_s > 3$ and "
        "$\\psi_s > 0.20$ are pre-specified operational heuristics from the size simulations in Section 7.2."
    ), anchor))


def main():
    files, root = b.load_zip(MAIN_IN)
    edit_main(root.find(b.q(W, "body")))
    b.store_zip(MAIN_OUT, files, root)
    files, root = b.load_zip(SUP_IN)
    edit_supplement(root.find(b.q(W, "body")))
    b.store_zip(SUP_OUT, files, root)
    files, root = b.load_zip(LETTER_IN)
    edit_letter(root.find(b.q(W, "body")))
    b.store_zip(LETTER_OUT, files, root)
    print(MAIN_OUT.name)
    print(SUP_OUT.name)
    print(LETTER_OUT.name)


if __name__ == "__main__":
    main()
