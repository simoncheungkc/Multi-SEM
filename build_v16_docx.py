"""Add the spatial-statistics discussion to the latest manuscript.

Reads v15 and supplement v13. Writes v16 and supplement v14.
Does not overwrite those source files. The Bivand, Rue, and Pace
references already present in v15 are kept; Wong's initials are completed.
"""

from __future__ import annotations

import build_v12_docx as b

ROOT = b.ROOT
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v15.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v13.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v16.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v14.docx"

INTRO = (
    "Spatial regression models traditionally condition on a single weight matrix $W$ representing geographic distance or contiguity. "
    "Bivand and Wong (2018) compare implementations of global and local indicators of spatial association, including Moran's $I$, and show that the numerical result depends on the weights and on the defaults used to construct them. "
    "Bivand, Gómez-Rubio, and Rue (2014) develop approximate Bayesian inference for spatial econometric models that each condition on one weight matrix, and they compare those models with a latent Gaussian Markov random field. "
    "Median housing prices in Boston census tracts are one of their examples. "
    "A single operator still leaves out the other relations in which the same units participate, including physical distance, a municipal border, and a transport corridor. "
    "Multi-SEM tests, screens, and combines those candidate weight matrices in one model whose spatial filter remains invertible."
)
RELATED = (
    "Bivand, Gómez-Rubio, and Rue (2014) fit those single-weight spatial econometric models by integrated nested Laplace approximation, and they compare them with correlated Gaussian random effects. "
    "The random-effect specification is a Gaussian Markov random field (Rue and Held, 2005). "
    "The paper treats one spatial operator. "
    "Several overlapping operators that do not share an eigenspace are a further problem: the coefficient vector has to remain inside $\\lVert \\lambda \\rVert_1 < 1$ if the filter is to stay invertible on row-substochastic layers."
)
KRIGING = (
    "The disturbance precision $Q = \\sigma^{-2} V_\\lambda' V_\\lambda$ is the precision of a Gaussian Markov random field when the filter is local (Rue and Held, 2005). "
    "Bivand, Gómez-Rubio, and Rue (2014) use that latent-field representation as a comparison for single-weight spatial econometric models. "
    "The same precision defines the conditional distribution of the holdout units. "
    "Solving $Q_{**} x = Q_{*t} \\hat{u}_t$ factorizes the $n_* \\times n_*$ block, an $O(n_*^3)$ operation, rather than the $n_t \\times n_t$ training covariance. "
    "Each Metropolis proposal still factorizes $V_\\lambda$ at the training size. "
    "A dense resolvent layer makes $Q$ dense, so the saving on that layer is the smaller block rather than a sparse factorization."
)
BOSTON = (
    "Pace and Gilley (1997) use the spatial configuration of these tracts to improve the hedonic fit. "
    "Table 8 separates that continuous distance channel from the municipal boundary. "
    "The geographic 6-nearest-neighbour spatial error model reduces mean holdout RMSE from 0.1921 to 0.1403, a 27.0% reduction relative to OLS. "
    "Screened Multi-SEM, which also retains the town layer on all 50 splits, reaches 0.1374, a 28.5% reduction. "
    "The Wilcoxon signed-rank test of the paired squared errors against OLS has $p = 1.78 \\times 10^{-15}$. "
    "The town layer is complementary to distance. "
    "The 28.5% figure is the multiplex predictor against OLS, not the marginal effect of adding $A_{\\mathrm{town}}$: the town spatial error model alone has mean RMSE 0.1499."
)

REFERENCES = [
    "References",
    "The following works are cited in the main text for single-weight spatial models, Gaussian Markov random fields, and the Boston hedonic specification.",
    "Bivand, R. S., Gómez-Rubio, V., & Rue, H. (2014). Approximate Bayesian inference for spatial econometrics models. Spatial Statistics, 9, 146–165.",
    "Bivand, R. S., & Wong, D. W. S. (2018). Comparing implementations of global and local indicators of spatial association. TEST, 27(3), 716–748.",
    "Pace, R. K., and Gilley, O. W. (1997). Using the spatial configuration of the data to improve estimation. Journal of Real Estate Finance and Economics, 14(3), 333–340.",
    "Rue, H., & Held, L. (2005). Gaussian Markov Random Fields: Theory and Applications. Chapman & Hall/CRC, Boca Raton.",
]


def insert_after(body, snippet, text):
    anchor = b.require_one(body, snippet)
    anchor.addnext(b.make_p(b.blocks_from(text), anchor))


def replace_text(root, old, new, expected):
    hits = [t for t in root.iter(b.q(b.W, "t")) if t.text and old in t.text]
    if len(hits) != expected:
        raise SystemExit(f"{old!r} matched {len(hits)}, expected {expected}")
    for node in hits:
        node.text = node.text.replace(old, new)


def bare_math(root):
    count = 0
    for el in root.iter(b.q(b.M, "e")):
        parent = el.getparent()
        if parent is not None and parent.tag == b.q(b.M, "oMath"):
            count += 1
    return count


def edit_main():
    files, root = b.load_zip(MAIN_IN)
    body = root.find(b.q(b.W, "body"))
    insert_after(body, "A specification with one weight matrix has to discard one of those channels.", INTRO)
    insert_after(body, "The additive filter", RELATED)
    insert_after(body, "The Metropolis loop still factorizes", KRIGING)
    insert_after(body, "The town SEM keeps", BOSTON)
    replace_text(
        root,
        ", R., & Wong, D. (2018)",
        ", R. S., & Wong, D. W. S. (2018)",
        1,
    )
    if bare_math(root):
        raise SystemExit("bare math in the manuscript")
    b.store_zip(MAIN_OUT, files, root)


def edit_supplement():
    files, root = b.load_zip(SUP_IN)
    body = root.find(b.q(b.W, "body"))
    caption = b.require_one(body, "Supplementary Figure S9. Boston tract centroids")
    made = [b.make_p(b.blocks_from(text), caption) for text in REFERENCES]
    # addnext in reverse so the list reads in the written order.
    node = caption
    for paragraph in made:
        node.addnext(paragraph)
        node = paragraph
    if bare_math(root):
        raise SystemExit("bare math in the supplement")
    b.store_zip(SUP_OUT, files, root)


def main():
    edit_main()
    edit_supplement()
    files, root = b.load_zip(MAIN_OUT)
    print("main words", b.word_count_root(root, stop_at="References"))
    print(MAIN_OUT)
    print(SUP_OUT)


if __name__ == "__main__":
    main()
