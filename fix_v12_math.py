"""Repair equations that PowerShell escaping broke in the second pass."""

import build_v12_docx as b

files, root = b.load_zip(b.MAIN_OUT)
body = root.find(b.q(b.W, "body"))

b.replace_containing(
    body,
    "The boosted layer is the Katz resolvent",
    r"The boosted layer is the Katz resolvent of that correlation network with damping $\alpha = 0.5$, row-standardized after the resolvent is formed.",
)
b.replace_containing(
    body,
    "Table 6 reports August 2026 residual Moran tests",
    r"Table 6 reports August 2026 residual Moran tests. The Yahoo Finance industry layer ($I = 0.263$, $p = 0.0038$) and the sector layer ($I = 0.122$, $p = 0.0070$) survive Benjamini–Hochberg control at $q = 0.05$. The correlation network ($p = 0.406$) and its resolvent ($p = 0.146$) do not. The showcase model therefore has $k = 2$ layers, sector and industry.",
)
b.replace_containing(
    body,
    "Posterior means are",
    r"Posterior means are $\lambda_{\mathrm{sec}} = 0.209$ and $\lambda_{\mathrm{ind}} = 0.181$. On the 16,000 retained draws (seed 2026), the boundary statistic $L^{(m)} = \vert\lambda_{\mathrm{sec}}^{(m)}\vert + \vert\lambda_{\mathrm{ind}}^{(m)}\vert$ has posterior mean 0.430 and 95th percentile 0.661. Four draws exceed 0.95, and none reach 1.",
)
b.replace_containing(
    body,
    "A no-kriging fit uses the same coefficients",
    "Each month uses the same 20 held-out tickers. Covariates and the correlation layers use only information dated before that month. Layers that survive Algorithm 1 are re-estimated on the induced training subgraph, and holdout returns are imputed from the full-graph precision. A no-kriging fit uses the same coefficients but predicts from the regression mean alone.",
)
b.replace_containing(body, "Parametric Residual Bootstrap", "Conditional Gaussian Monte Carlo")

# No raw backslash left in these repaired paragraphs.
for snippet in ["boosted layer", "Table 6 reports", "Posterior means are", "no-kriging fit"]:
    text = b.ptext(b.require_one(body, snippet))
    if "\\" in text or "$" in text:
        raise SystemExit(f"still broken: {snippet}: {text[:180]}")

prose = b.word_count_root(root, stop_at="References")
full = len(" ".join(t.text or "" for t in root.iter(b.q(b.W, "t"))).split())
print("prose excluding tables, before references:", prose)
print("all w:t words:", full)
b.store_zip(b.MAIN_OUT, files, root)
print("updated")
