"""Store the supplied Boston posterior grid as its own Word file.

The numbers are copied as supplied. They are not the archived Split 5 posterior.
"""

from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.shared import Inches, Pt

from build_boston_docx import add_mixed, add_table

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "Boston_Split5_posterior_table_supplied.docx"

ROWS = [
    ["λ geo (Geographic 6-NN)", "0.127", "0.124", "0.048", "(0.035, 0.224)", "1.004", "1,420"],
    ["λ res (Spatial resolvent)", "0.514", "0.518", "0.062", "(0.388, 0.631)", "1.008", "1,180"],
    ["λ town (Municipal cliques)", "0.189", "0.186", "0.041", "(0.110, 0.272)", "1.006", "1,650"],
    ["β0 (Intercept)", "3.042", "3.042", "0.018", "(3.007, 3.078)", "1.001", "12,500"],
    ["β rm (Rooms)", "0.082", "0.082", "0.015", "(0.053, 0.111)", "1.000", "14,200"],
    ["β lstat (log lower status)", "−0.218", "−0.218", "0.019", "(−0.255, −0.181)", "1.000", "11,800"],
    ["β crim (Crime)", "−0.074", "−0.074", "0.014", "(−0.102, −0.046)", "1.000", "13,400"],
    ["β tax (Property tax)", "−0.058", "−0.058", "0.021", "(−0.099, −0.017)", "1.000", "9,800"],
    ["β ptratio (Pupil-teacher ratio)", "−0.041", "−0.041", "0.013", "(−0.067, −0.015)", "1.000", "12,100"],
    ["β dis (log distance)", "−0.124", "−0.124", "0.024", "(−0.171, −0.077)", "1.000", "8,900"],
    ["β nox (Nitric oxides)", "−0.062", "−0.062", "0.018", "(−0.097, −0.027)", "1.000", "10,400"],
    ["β age (Age)", "0.009", "0.009", "0.014", "(−0.018, 0.036)", "1.000", "13,100"],
    ["β zn (Zoning)", "0.021", "0.021", "0.015", "(−0.008, 0.051)", "1.000", "11,200"],
    ["β indus (Industry acreage)", "0.012", "0.012", "0.018", "(−0.023, 0.047)", "1.000", "10,600"],
    ["σ² (Innovation variance)", "0.0148", "0.0146", "0.0012", "(0.0126, 0.0174)", "1.000", "11,900"],
    ["L = ||λ||₁", "0.830", "0.828", "0.052", "95th percentile: 0.912", "1.008", "P(L > 0.95) = 0.018"],
]


def main():
    doc = Document()
    section = doc.sections[0]
    section.orientation = WD_ORIENT.LANDSCAPE
    section.page_width = Inches(11)
    section.page_height = Inches(8.5)
    section.left_margin = Inches(0.6)
    section.right_margin = Inches(0.6)
    section.top_margin = Inches(0.6)
    section.bottom_margin = Inches(0.6)
    style = doc.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(11)
    add_mixed(
        doc,
        "Table 8. Posterior parameter estimates for the Boston Housing Multi-SEM "
        "(Split 5, $n_t = 405$ training tracts).",
        size=12,
        bold=True,
        space_after=8,
    )
    add_table(
        doc,
        ["Parameter", "Mean", "Median", "SD", "95% credible interval", "R-hat", "ESS"],
        ROWS,
    )
    add_mixed(
        doc,
        "The radial coefficient was excluded by the screening procedure on Split 5, "
        "as documented in Table 3 of the Boston report.",
        size=11,
        space_after=8,
    )
    add_mixed(
        doc,
        "These entries are stored as supplied. They are not the archived Split 5 posterior. "
        "The archived means, intervals, split-$\\hat R$, and effective sample sizes are "
        "Table 8 of REGRESSION WITH MULTIPLE NETWORKS v17.docx and "
        "results/boston/split_05_posterior.json.",
        size=11,
        space_after=6,
    )
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
