"""Add the Split 5 posterior table and place the Boston tables with their text.

Reads v16 and supplement v14. Writes v17 and supplement v15.
The posterior numbers are the retained draws from the archived Split 5
Gibbs seed, checked in recover_split5_posterior.py. The supplied grid
with lambda ESS above 1,000 is not used.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

import build_v12_docx as b
import run_boston as rb
from build_v10_docx import set_cell_text

ROOT = b.ROOT
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v16.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v14.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v17.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v15.docx"
POSTERIOR = ROOT / "results" / "boston" / "split_05_posterior.json"

LABELS = {
    "lambda_geo": "λ, geographic 6-NN",
    "lambda_resolvent": "λ, geographic resolvent",
    "lambda_town": "λ, town jurisdiction",
    "beta0": "Intercept",
    "rm": "Rooms",
    "age": "Age",
    "crim": "Crime",
    "zn": "Large-lot zoning",
    "indus": "Industry acreage",
    "log_lstat": "log lower status",
    "tax": "Property tax",
    "ptratio": "Pupil-teacher ratio",
    "nox": "Nitric oxides",
    "log_dis": "log distance",
    "sigma2": "Innovation variance",
    "L": "L, sum of absolute λ",
}


def replace_count(root, old, new, expected=None):
    hits = [t for t in root.iter(b.q(b.W, "t")) if t.text and old in t.text]
    if expected is not None and len(hits) != expected:
        preview = " | ".join((t.text or "")[:70] for t in hits[:3])
        raise SystemExit(f"{old!r} matched {len(hits)}, expected {expected}: {preview}")
    for node in hits:
        node.text = node.text.replace(old, new)
    return len(hits)


def renumber_tables(root, *, supplement=False):
    replace_count(root, "Table 12", "@@T12@@")
    replace_count(root, "Table 11", "@@T11@@")
    replace_count(root, "Table 10", "@@T10@@")
    replace_count(root, "Table 9", "@@T9@@")
    replace_count(root, "Table 8", "@@T8@@")
    replace_count(root, "@@T12@@", "Table 13")
    replace_count(root, "@@T11@@", "Table 12")
    replace_count(root, "@@T10@@", "Table 11")
    replace_count(root, "@@T9@@", "Table 10")
    replace_count(root, "@@T8@@", "Table 9")
    if supplement:
        done = False
        for token in ("Tables 9–11", "Tables 9-11"):
            if replace_count(root, token, "Tables 10–12"):
                done = True
        if not done:
            raise SystemExit("supplement showcase table range was not found")


def move_after(anchor, *nodes):
    current = anchor
    for node in nodes:
        current.addnext(node)
        current = node


def caption_and_table(caption):
    table = caption.getnext()
    if table is None or table.tag != b.q(b.W, "tbl"):
        raise SystemExit(f"no table after {b.ptext(caption)[:60]!r}")
    return caption, table


def fmt(value, digits):
    text = f"{value:.{digits}f}"
    return text.replace("-", "\u2212")


def ess(value):
    return f"{int(round(value)):,}"


def interval(row, digits):
    return f"({fmt(row['q025'], digits)}, {fmt(row['q975'], digits)})"


def posterior_rows(payload):
    order = list(LABELS)
    by_key = {row["key"]: row for row in payload["rows"]}
    by_key["L"] = payload["L"]
    lines = []
    for key in order:
        row = by_key[key]
        digits = 4 if key == "sigma2" else 3
        lines.append([
            LABELS[key],
            fmt(row["mean"], digits),
            fmt(row["median"], digits),
            fmt(row["sd"], digits),
            interval(row, digits),
            f"{row['rhat']:.3f}",
            ess(row["ess"]),
        ])
    return lines


def make_posterior_table(template, lines):
    table = deepcopy(template)
    rows = table.findall(b.q(b.W, "tr"))
    proto = deepcopy(rows[1])
    for row in rows[1:]:
        table.remove(row)
    for values in lines:
        row = deepcopy(proto)
        cells = row.findall(b.q(b.W, "tc"))
        if len(cells) < len(values):
            raise SystemExit(f"template row has {len(cells)} cells")
        for cell, value in zip(cells, values):
            set_cell_text(cell, value)
        table.append(row)
    return table


def edit_main():
    payload = json.loads(POSTERIOR.read_text(encoding="utf-8"))
    files, root = b.load_zip(MAIN_IN)
    body = root.find(b.q(b.W, "body"))
    replace_count(
        root,
        "Table 8 nevertheless reports posterior means from these original runs; the estimates were not replaced by results from longer chains.",
        "TABLEPOST reports the Split 5 posterior from that original run. The draws were not replaced by a longer chain.",
        1,
    )
    renumber_tables(root)
    replace_count(root, "TABLEPOST", "Table 8", 1)
    tables = [c for c in body if c.tag == b.q(b.W, "tbl")]
    network = b.require_one(body, "Table 6 summarizes the full-sample network topology.")
    move_after(network, *caption_and_table(b.require_one(body, "Table 6. Full-sample")))
    screening = b.require_one(body, "Table 7 summarizes the screening results across the 50 training samples.")
    move_after(screening, *caption_and_table(b.require_one(body, "Table 7. Residual Moran screening")))
    holdout = b.require_one(body, "Table 9 reports predictive performance on the log-value scale.")
    move_after(holdout, *caption_and_table(b.require_one(body, "Table 9. Mean holdout error")))
    figure_caption = b.require_one(body, "Figure 3. Predicted versus actual")
    drawing = figure_caption.getprevious()
    if drawing is None or not any(el.tag.endswith("blip") for el in drawing.iter()):
        raise SystemExit("Figure 3 drawing was not immediately before its caption")
    split5 = b.require_one(body, "Figure 3 illustrates predicted versus realized")
    move_after(split5, drawing, figure_caption)
    tail = payload["L"]
    caption_text = (
        "Table 8. Posterior summaries for the screened Boston Multi-SEM on Split 5 "
        f"(n_t = {payload['n_train']} training tracts). The radial layer was not selected. "
        "The interval is the central 95 percent of the retained draws from the archived Gibbs seed. "
        f"The 95th percentile of L is {tail['q95']:.3f}, and the posterior probability that L exceeds 0.95 is {tail['p_gt_095']:.3f}."
    )
    posterior_anchor = b.require_one(body, "Posterior estimation follows Algorithm 2")
    caption = b.make_p(b.blocks_from(caption_text), posterior_anchor)
    table = make_posterior_table(tables[10], posterior_rows(payload))
    move_after(posterior_anchor, caption, table)
    bare = sum(
        1 for el in root.iter(b.q(b.M, "e"))
        if el.getparent() is not None and el.getparent().tag == b.q(b.M, "oMath")
    )
    if bare:
        raise SystemExit(f"bare math {bare}")
    b.store_zip(MAIN_OUT, files, root)


def moment(value):
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:.1f}"
    if magnitude >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def edit_supplement():
    files, root = b.load_zip(SUP_IN)
    body = root.find(b.q(b.W, "body"))
    renumber_tables(root, supplement=True)
    tables = [c for c in body if c.tag == b.q(b.W, "tbl")]
    template = tables[1]
    data = rb.load_boston()
    raw = data["raw"]
    specs = [
        ("Rooms", "rooms per dwelling"),
        ("Age", "proportion of units built before 1940"),
        ("Crime", "per capita crime rate"),
        ("Large-lot zoning", "percent of land in lots over 25,000 sq ft"),
        ("Industry acreage", "percent non-retail business acres"),
        ("log lower status", "natural log of the percent lower status"),
        ("Property tax", "tax rate per $10,000"),
        ("Pupil-teacher ratio", "pupils per teacher"),
        ("Nitric oxides", "parts per 10 million"),
        ("log distance", "natural log of weighted distance to employment centres"),
    ]
    lines = [["Variable", "Unit", "Mean", "SD"]]
    for index, (name, unit) in enumerate(specs):
        column = raw[:, index]
        lines.append([name, unit, moment(float(column.mean())), moment(float(column.std(ddof=1)))])
    # Rebuild from a four-column template. Header is row 0; overwrite it too.
    table = deepcopy(template)
    rows = table.findall(b.q(b.W, "tr"))
    proto = deepcopy(rows[1])
    for cell, value in zip(rows[0].findall(b.q(b.W, "tc")), lines[0]):
        set_cell_text(cell, value)
    for row in rows[1:]:
        table.remove(row)
    for values in lines[1:]:
        row = deepcopy(proto)
        cells = row.findall(b.q(b.W, "tc"))
        for cell, value in zip(cells, values):
            set_cell_text(cell, value)
        table.append(row)
    anchor = b.require_one(body, "Seventeen towns contain a single tract")
    caption = b.make_p(b.blocks_from(
        "Table S4.1. Full-sample moments of the ten Boston regressors (n = 506). "
        "The standard deviation uses the divisor n − 1. "
        "The regression standardizes each column inside the training sample of each split, so these are the unstandardized moments. "
        "The 17 singleton municipalities are named in the next paragraph."
    ), anchor)
    anchor.addprevious(caption)
    caption.addnext(table)
    b.store_zip(SUP_OUT, files, root)


def main():
    if not POSTERIOR.exists():
        raise SystemExit(f"missing {POSTERIOR}")
    edit_main()
    edit_supplement()
    files, root = b.load_zip(MAIN_OUT)
    print("main words", b.word_count_root(root, stop_at="References"))
    print(MAIN_OUT)
    print(SUP_OUT)


if __name__ == "__main__":
    main()
