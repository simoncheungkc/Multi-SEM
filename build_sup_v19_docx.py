"""Supplement reference list: only works cited in the supplement. Does not overwrite v18."""

from __future__ import annotations

import build_v12_docx as b

ROOT = b.ROOT
W = b.W
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v18.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v19.docx"

ENTRIES = [
    "Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: Series B (Methodological), 57(1), 289–300.",
    "Cliff, A. D., & Ord, J. K. (1973). Spatial Autocorrelation. Pion, London.",
    "Cliff, A. D., & Ord, J. K. (1981). Spatial Processes: Models & Applications. Pion, London.",
    "Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business & Economic Statistics, 13(3), 253–263.",
    "Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. Bernoulli, 7(2), 223–242.",
    "Hastings, W. K. (1970). Monte Carlo sampling methods using Markov chains and their applications. Biometrika, 57(1), 97–109.",
    "Kelejian, H. H., & Prucha, I. R. (2001). On the asymptotic distribution of the Moran I test statistic with applications. Journal of Econometrics, 104(2), 219–257.",
    "LeSage, J. P., & Pace, R. K. (2009). Introduction to Spatial Econometrics. CRC Press, Boca Raton.",
    "Metropolis, N., Rosenbluth, A. W., Rosenbluth, M. N., Teller, A. H., & Teller, E. (1953). Equation of state calculations by fast computing machines. The Journal of Chemical Physics, 21(6), 1087–1092.",
    "Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703–708.",
    "Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC. Journal of Computational and Graphical Statistics, 18(2), 349–367.",
    "Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. The Annals of Applied Probability, 7(1), 110–120.",
]


def edit(body):
    heading = b.require_one(body, "References")
    style = b.require_one(body, "Haario, H., Saksman, E.")
    seen = False
    for child in list(body):
        if child is heading:
            seen = True
            continue
        if seen and child.tag == b.q(W, "p"):
            body.remove(child)
    for text in reversed(ENTRIES):
        heading.addnext(b.make_p(b.blocks_from(text), style))


def main():
    files, root = b.load_zip(SUP_IN)
    edit(root.find(b.q(W, "body")))
    b.store_zip(SUP_OUT, files, root)
    print(SUP_OUT.name)


if __name__ == "__main__":
    main()
