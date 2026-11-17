"""Insert the Boston holdout scatter and renumber main-text figures.

Reads v14 and supplement v12. Writes v15 and supplement v13.
Does not overwrite those source files.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from lxml import etree
from PIL import Image

import build_v12_docx as b

ROOT = b.ROOT
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v14.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v12.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v15.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v13.docx"
FIG = ROOT / "results" / "boston" / "boston_split5_pred_vs_actual.png"

CAPTION = (
    "Figure 3. Predicted versus actual $\\log(\\mathrm{cmedv})$ on the $n_* = 101$ holdout census tracts for Split 5 of the Boston housing study. "
    "Left: OLS regression ($\\hat{Y}_* = X_* \\hat{\\beta}_{\\mathrm{OLS}}$, $\\mathrm{RMSE} = 0.1912$, $\\mathrm{MAE} = 0.1408$, $r = 0.867$). "
    "Right: Screened Multi-SEM precision-form spatial kriging ($\\hat{Y}_* = X_* \\hat{\\beta} - Q_{**}^{-1} Q_{*t} \\hat{u}_t$, $\\mathrm{RMSE} = 0.1347$, $\\mathrm{MAE} = 0.0927$, $r = 0.939$). "
    "The solid line is the 45-degree identity line ($y = \\hat{y}$). "
    "Split 5 is the lower of the two central Multi-SEM holdout RMSEs across the 50 random 80/20 partitions."
)
NARRATIVE = (
    "Figure 3 illustrates predicted versus realized log housing values for Split 5 on the $n_* = 101$ holdout tracts. "
    "Ordered by Multi-SEM holdout RMSE, Split 5 ($0.1347$) and Split 8 ($0.1376$) are the two central trials among the 50 partitions. "
    "The sample median is their average, $0.1361$. The 50-split mean in Table 8 is $0.1374$. "
    "Under OLS (left panel), predictions follow the covariates but shrink toward the conditional mean: the standard deviation of the 101 predictions is $0.348$, against $0.375$ for the outcomes, and tracts at the high and low ends fall inside the 45-degree line. "
    "Multi-SEM precision kriging (right panel) conditions on observed training residuals across the layers retained on this split, $A_{\\mathrm{geo}}$, $G_{A_{\\mathrm{geo}}}$, and $A_{\\mathrm{town}}$. "
    "The correction $-Q_{**}^{-1} Q_{*t} \\hat{u}_t$ places the predictions on the 45-degree line. "
    "On this split, Multi-SEM reduces holdout RMSE by 29.5% (from 0.1912 to 0.1347), lowers MAE by 34.1% (from 0.1408 to 0.0927), and raises the Pearson correlation from 0.867 to 0.939."
)


def replace_text(root, old, new, expected):
    hits = [t for t in root.iter(b.q(b.W, "t")) if t.text and old in t.text]
    if len(hits) != expected:
        preview = " | ".join((t.text or "")[:80] for t in hits[:4])
        raise SystemExit(f"{old!r} matched {len(hits)}, expected {expected}: {preview}")
    for node in hits:
        node.text = node.text.replace(old, new)


def renumber_main(root):
    pairs = [
        ("Figure 11 plots", "Figure 5 plots", 1),
        ("path is Figure 11.", "path is Figure 5.", 1),
        ("Figure 11.  Running", "Figure 5.  Running", 1),
        ("Figure 9.  S&P", "Figure 4.  S&P", 1),
        ("Figure 9", "Figure 4", 1),
        ("Figure 6.  Marginal", "Figure 2.  Marginal", 1),
        ("Figure 5.  Gibbs", "Figure 1.  Gibbs", 1),
        ("Table 8 compares", "Table 11 compares", 1),
    ]
    for old, new, expected in pairs:
        replace_text(root, old, new, expected)


def insert_boston_figure(files, root):
    body = root.find(b.q(b.W, "body"))
    anchor = b.require_one(body, "The one-layer geographic and town samplers meet")
    template = None
    for child in body:
        if child.tag == b.q(b.W, "p") and any(el.tag.endswith("blip") for el in child.iter()):
            template = child
    if template is None:
        raise SystemExit("no drawing to clone")
    drawing = deepcopy(template)
    rels = etree.fromstring(files["word/_rels/document.xml.rels"])
    ids = {rel.get("Id") for rel in rels}
    number = 1
    while f"rId{number}" in ids:
        number += 1
    rid = f"rId{number}"
    rel = etree.SubElement(rels, "{http://schemas.openxmlformats.org/package/2006/relationships}Relationship")
    rel.set("Id", rid)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", "media/image5.png")
    files["word/media/image5.png"] = FIG.read_bytes()
    files["word/_rels/document.xml.rels"] = etree.tostring(
        rels, xml_declaration=True, encoding="UTF-8", standalone=True
    )
    office = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    found = False
    for blip in drawing.iter():
        if blip.tag.endswith("blip"):
            blip.set(f"{office}embed", rid)
            found = True
    if not found:
        raise SystemExit("cloned drawing has no blip")
    image = Image.open(FIG)
    width = 6.20
    height = width * image.size[1] / image.size[0]
    cx = str(int(round(width * 914400)))
    cy = str(int(round(height * 914400)))
    for el in drawing.iter():
        if "cx" in el.attrib and "cy" in el.attrib:
            el.set("cx", cx)
            el.set("cy", cy)
        if el.tag.endswith("docPr"):
            el.set("id", "17")
            el.set("name", "Picture 17")
        if el.tag.endswith("cNvPr"):
            el.set("name", "boston_split5_pred_vs_actual.png")
    caption = b.make_p(b.blocks_from(CAPTION), anchor)
    narrative = b.make_p(b.blocks_from(NARRATIVE), anchor)
    anchor.addnext(drawing)
    drawing.addnext(caption)
    caption.addnext(narrative)


def edit_supplement():
    files, root = b.load_zip(SUP_IN)
    body = root.find(b.q(b.W, "body"))
    replace_text(root, "Figure 3", "Supplementary Figure S3", 1)
    replace_text(
        root,
        " in the main text reports empirical size and power for ",
        " reports empirical size and power for ",
        1,
    )
    replace_text(root, "Table 3 and Figures 1, 2, and 4", "Table 3 and Supplementary Figures S1, S2, and S4", 1)
    replace_text(root, " in the main text summarize", " summarize", 1)
    replace_text(root, "Figure 5 presents", "Figure 1 presents", 1)
    replace_text(root, "Figure 6 displays", "Figure 2 displays", 1)
    replace_text(root, "Figure 7 plots", "Supplementary Figure S5 plots", 1)
    replace_text(root, "Figure 8 summarizes", "Supplementary Figure S6 summarizes", 1)
    replace_text(root, "Figure 9 displays", "Figure 4 displays", 1)
    replace_text(root, "Table 8 and Figure 10", "Table 11 and Supplementary Figure S7", 1)
    replace_text(root, "Figure 11 plots", "Figure 5 plots", 1)
    replace_text(
        root,
        "Tables 2–5 and Figures 1–8",
        "Tables 2–5, main-text Figures 1 and 2, and Supplementary Figures S1–S6",
        1,
    )
    replace_text(root, "and Figure 9,", "and Figure 4,", 1)
    replace_text(root, "and Figure 11.", "and Figure 5.", 1)
    b.replace_containing(
        body,
        "These figures were removed from the main manuscript",
        "Supplementary Figures S1–S8 were removed from the main manuscript. S1–S4 are the Moran size and power plots. S5 and S6 are the simulation holdout charts. S7 is the August 2026 predicted-versus-actual scatter. S8 is the 63-month average RMSE chart. The simulation Gibbs traces and posterior densities remain in the main text as Figures 1 and 2. The Boston holdout scatter is main-text Figure 3, and the month-by-month S&P 100 path is main-text Figure 5.",
    )
    b.replace_containing(
        body,
        "Run python run_boston.py. The script reads data/boston_corrected.txt",
        "Step 3b: Run python run_boston.py, python plot_boston_split5.py, and python build_boston_docx.py. "
        "run_boston.py reads data/boston_corrected.txt, screens and estimates on the 405 training tracts of each split, and writes results/boston/split_00.json through split_49.json together with results/boston/summary.json. "
        "The reported run used 12 worker processes and took about 2.5 hours, because each proposal factorizes a 405 by 405 filter. "
        "plot_boston_split5.py writes results/boston/boston_split5_pred_vs_actual.png, the Split 5 holdout scatter used as main-text Figure 3. "
        "build_boston_docx.py rebuilds the standalone Boston note and embeds that scatter. "
        "The spatial residual map is results/boston/figure_s9.png, Supplementary Figure S9.",
    )
    b.replace_containing(
        body,
        "That split is the one whose Multi-SEM RMSE is closest to the 50-split median.",
        "Supplementary Figure S9. Boston tract centroids in WGS84 UTM zone 19N. The left panel colors the 506 centroids by municipal code. The StatLib file does not contain digitized town polygons, so legal boundaries are not drawn. The right panel shows the holdout residual, actual minus predicted log value, for Split 5 of the screened precision-kriging predictor. Split 5 is the lower of the two central Multi-SEM holdout RMSEs. The predicted-versus-actual scatter for the same tracts is main-text Figure 3.",
    )
    s42 = b.require_one(body, "S4.2 Constituent Listing")
    s42.addprevious(b.make_p(b.blocks_from(
        "The tract-level holdout prediction errors for Split 5 are mapped in Supplementary Figure S9, while the corresponding predicted-versus-actual scatter comparisons are shown in main-text Figure 3."
    ), s42))
    b.store_zip(SUP_OUT, files, root)


def word_count(path):
    files, root = b.load_zip(path)
    del files
    return b.word_count_root(root, stop_at="References")


def main():
    if not FIG.exists():
        raise SystemExit(f"missing {FIG}")
    files, root = b.load_zip(MAIN_IN)
    renumber_main(root)
    insert_boston_figure(files, root)
    bare = 0
    for el in root.iter(b.q(b.M, "e")):
        parent = el.getparent()
        if parent is not None and parent.tag == b.q(b.M, "oMath"):
            bare += 1
    if bare:
        raise SystemExit(f"bare m:e count {bare}")
    b.store_zip(MAIN_OUT, files, root)
    edit_supplement()
    print("main words", word_count(MAIN_OUT))
    print(MAIN_OUT)
    print(SUP_OUT)


if __name__ == "__main__":
    main()
