"""Copy v10 to v11 and replace Section 7.5 with the larger-scale comparison."""

from __future__ import annotations

import os
import zipfile
from pathlib import Path

from lxml import etree

from build_v10_docx import make_p, q, W

ROOT = Path(__file__).resolve().parent


def para_text(p):
    return "".join(t.text or "" for t in p.iter(q(W, "t")))


def replace_with(p, blocks):
    new = make_p(blocks, p)
    p.getparent().replace(p, new)
    return new


def main():
    src = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v10.docx"
    dst = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v11.docx"
    with zipfile.ZipFile(src) as zin:
        root = etree.fromstring(zin.read("word/document.xml"))
        body = root.find(q(W, "body"))
        targets = {
            "Thirty replications use": [
                ("t", "The comparison is reported at two sample sizes, with a 20% holdout in each. At "),
                ("m", r"n=200"),
                ("t", " there are 50 replications and a Gibbs run of 2,000 iterations after a 1,000-iteration burn-in. At "),
                ("m", r"n=500"),
                ("t", " there are 20 replications, estimated by concentrated likelihood. Both estimators krige from the full-graph precision. The training criterion is either the induced, re-row-standardized subgraph or the Schur marginal precision "),
                ("m", r"P_T"),
                ("t", ". The 63-month study is not re-estimated under "),
                ("m", r"P_T"),
                ("t", "."),
            ],
            "Mean concentrated-likelihood": [
                ("t", "At "),
                ("m", r"n=200"),
                ("t", ", mean concentrated-likelihood estimates are (0.305, 0.265) on the induced subgraph and (0.339, 0.313) under the marginal likelihood, against the true pair (0.40, 0.35). The biases are (−0.095, −0.085) and (−0.061, −0.037). The median absolute relative gap between the two estimators is 13.7% and 14.9%. Mean holdout RMSE is 1.017 and 1.014, a 0.31% difference. Median relative differences are 3.7% for the slope coefficients and 4.6% for the full coefficient vector. The induced Gibbs run averages 1.67 seconds against 3.67 seconds for the marginal likelihood, a factor of 2.2. A single likelihood evaluation is 2.6 times faster on the induced subgraph. Posterior means from the longer chains are (0.321, 0.268) and (0.367, 0.314)."),
            ],
            "The predictive discrepancy is small": [
                ("t", "At "),
                ("m", r"n=500"),
                ("t", " the marginal-likelihood biases fall to (−0.009, −0.022), while the induced-subgraph biases stay near (−0.080, −0.085). The median relative gap in "),
                ("m", r"\lambda"),
                ("t", " widens to 18.3% and 18.8%. Mean holdout RMSE is 0.9993 and 0.9985, a 0.08% difference, and the median relative gap in the slopes is 2.0%. The discrepancy in "),
                ("m", r"\lambda"),
                ("t", " is a consequence of re-standardizing the training subgraph. It does not shrink as "),
                ("m", r"n"),
                ("t", " grows, and it does not pass into the kriging error. The induced estimator remains appropriate when cross-block links to future test nodes are unavailable at fitting time. When those links are known, the marginal likelihood is the criterion that matches the kriging model. The replication file is results/marginal_vs_induced_large.csv."),
            ],
        }
        found = {k: None for k in targets}
        for child in body:
            if child.tag != q(W, "p"):
                continue
            s = para_text(child)
            for key in targets:
                if s.startswith(key):
                    found[key] = child
        missing = [k for k, v in found.items() if v is None]
        if missing:
            raise SystemExit(f"missing paragraphs: {missing}")
        for key, blocks in targets.items():
            replace_with(found[key], blocks)
        xml = etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)
        tmp = dst.with_suffix(".tmp.docx")
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)
    os.replace(tmp, dst)
    print("wrote", dst.name)


if __name__ == "__main__":
    main()
