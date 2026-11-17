"""Remove duplicated tails left in two v13 paragraphs."""

from lxml import etree

import build_v12_docx as b

files, root = b.load_zip(b.ROOT / "REGRESSION WITH MULTIPLE NETWORKS v13.docx")
body = root.find(b.q(b.W, "body"))

CLEAN_75 = (
    " is a consequence of re-standardizing the training subgraph, which changes the local scale "
    "of the network operator. In the simulated configurations examined here, this parameter "
    "adjustment offset the modified graph scale, yielding negligible differences in holdout RMSE "
    "(0.31% at n = 200 and 0.08% at n = 500). Thus, the induced-subgraph estimator acts as a "
    "computationally fast surrogate when cross-block ties are missing, while the Schur marginal precision "
)

for p in body:
    if p.tag != b.q(b.W, "p"):
        continue
    text = b.ptext(p)
    if "is a consequence of re-standardizing" in text:
        seen = False
        for ch in list(p):
            if etree.QName(ch).localname != "r":
                if seen:
                    p.remove(ch)
                continue
            run_text = "".join(t.text or "" for t in ch.iter(b.q(b.W, "t")))
            if "is a consequence of re-standardizing" in run_text:
                for t in ch.iter(b.q(b.W, "t")):
                    if t.text and "is a consequence of re-standardizing" in t.text:
                        t.text = CLEAN_75
                seen = True
                continue
            if seen:
                p.remove(ch)
        # Restore the symbol and the closing clause.
        p.append(b.omath("P_T"))
        closing = b.make_p(
            [("t", " provides the strictly coherent objective when the full topology is observed ex ante.")],
            p,
        )
        # make_p returns a paragraph; we need only its run.
        for ch in list(closing):
            if etree.QName(ch).localname == "r":
                p.append(ch)
    if "fail to outperform OLS." in text and "also outperform baseline OLS" in text:
        seen = False
        for ch in list(p):
            if etree.QName(ch).localname != "r":
                if seen:
                    p.remove(ch)
                continue
            run_text = "".join(t.text or "" for t in ch.iter(b.q(b.W, "t")))
            if "fail to outperform OLS." in run_text:
                seen = True
                continue
            if seen:
                p.remove(ch)

# Confirm the Schur display paragraph still has the formula.
hits = [p for p in body if p.tag == b.q(b.W, "p") and "Schur marginal precision in that comparison" in b.ptext(p)]
if len(hits) != 1:
    raise SystemExit(f"schur sentence {len(hits)}")
formula = "".join(t.text or "" for t in hits[0].iter(b.q("http://schemas.openxmlformats.org/officeDocument/2006/math", "t")))
print("formula", formula)
if "Q" not in formula:
    raise SystemExit("Schur formula missing")

b.store_zip(b.ROOT / "REGRESSION WITH MULTIPLE NETWORKS v13.docx", files, root)
print("repaired")
