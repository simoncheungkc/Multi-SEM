"""Write Equation (S2.9) as a native Word OMML aligned equation."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn
from docx.shared import Inches, Pt
from lxml import etree

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "EquationS29_Quadratic_Form.docx"

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
etree.register_namespace("m", M)


def el(tag, *children, **attrs):
    node = etree.Element(f"{{{M}}}{tag}")
    for key, val in attrs.items():
        node.set(f"{{{M}}}{key}", str(val))
    for child in children:
        if child is None:
            continue
        if isinstance(child, (list, tuple)):
            for item in child:
                if item is not None:
                    node.append(item)
        else:
            node.append(child)
    return node


def lst(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return [i for i in x if i is not None]
    return [x]


def run(text, sty=None, scr=None, aln=False):
    r = el("r")
    if sty or scr or aln:
        rpr = el("rPr")
        if sty:
            rpr.append(el("sty", val=sty))
        if scr:
            rpr.append(el("scr", val=scr))
        if aln:
            rpr.append(el("aln"))
        r.append(rpr)
    t = el("t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def e(*children):
    return el("e", *children)


def ssup(base, sup):
    return el("sSup", e(*lst(base)), el("sup", *lst(sup)))


def ssub(base, sub):
    return el("sSub", e(*lst(base)), el("sub", *lst(sub)))


def ssubsup(base, sub, sup):
    return el("sSubSup", e(*lst(base)), el("sub", *lst(sub)), el("sup", *lst(sup)))


def frac(num, den):
    return el("f", el("num", *lst(num)), el("den", *lst(den)))


def delim(content, beg="(", end=")"):
    return el(
        "d",
        el("dPr", el("begChr", val=beg), el("endChr", val=end), el("grow", val="1")),
        e(*lst(content)),
    )


def tilde(base):
    return el("acc", el("accPr", el("chr", val="\u0303")), e(*lst(base)))


def aln():
    return run("", aln=True)


def sigma2():
    return ssup(run("σ"), run("2"))


def inv_sigma0():
    return ssubsup(run("Σ"), run("0"), run("−1"))


def mu0():
    return ssub(run("μ"), run("0"))


def ytilde():
    return tilde(run("Y"))


def xtilde():
    return tilde(run("X"))


def inv_sigma_over_sigma2():
    return frac(run("1"), sigma2())


def set_run_font(run_obj, size=12, bold=False, italic=False):
    run_obj.font.name = "Times New Roman"
    run_obj.font.size = Pt(size)
    run_obj.bold = bold
    run_obj.italic = italic
    run_obj._element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")


def set_cell_border(cell):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        element = OxmlElement(f"w:{edge}")
        element.set(qn("w:val"), "nil")
        borders.append(element)
    tcPr.append(borders)


def build_omath():
    yxb = lambda: delim([ytilde(), run(" − "), xtilde(), run("β")])
    bmu = lambda: delim([run("β"), run(" − "), mu0()])

    line1 = e(
        run("Q"),
        delim(run("β")),
        aln(),
        run(" = "),
        inv_sigma_over_sigma2(),
        yxb(),
        run("′"),
        yxb(),
        run(" + "),
        bmu(),
        run("′"),
        inv_sigma0(),
        bmu(),
    )

    line2 = e(
        aln(),
        run(" = "),
        inv_sigma_over_sigma2(),
        delim(
            [
                ytilde(),
                run("′"),
                ytilde(),
                run(" − 2"),
                run("β"),
                run("′"),
                xtilde(),
                run("′"),
                ytilde(),
                run(" + "),
                run("β"),
                run("′"),
                xtilde(),
                run("′"),
                xtilde(),
                run("β"),
            ]
        ),
        run(" + "),
        run("β"),
        run("′"),
        inv_sigma0(),
        run("β"),
        run(" − 2"),
        run("β"),
        run("′"),
        inv_sigma0(),
        mu0(),
        run(" + "),
        mu0(),
        run("′"),
        inv_sigma0(),
        mu0(),
    )

    line3 = e(
        aln(),
        run(" = "),
        run("β"),
        run("′"),
        delim(
            [
                inv_sigma0(),
                run(" + "),
                inv_sigma_over_sigma2(),
                xtilde(),
                run("′"),
                xtilde(),
            ]
        ),
        run("β"),
        run(" − 2"),
        run("β"),
        run("′"),
        delim(
            [
                inv_sigma0(),
                mu0(),
                run(" + "),
                inv_sigma_over_sigma2(),
                xtilde(),
                run("′"),
                ytilde(),
            ]
        ),
        run(" + "),
        ssub(run("C"), run("0")),
    )

    return el(
        "oMathPara",
        el(
            "oMath",
            el("eqArr", el("eqArrPr", el("maxDist", val="0"), el("objDist", val="1")), line1, line2, line3),
        ),
    )


def build():
    doc = Document()
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(0.75)
    section.right_margin = Inches(0.75)
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(16)
    r = title.add_run("Quadratic Form in the Conditional Posterior of β")
    set_run_font(r, 14, bold=True)

    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    row = table.rows[0]
    set_cell_border(row.cells[0])
    set_cell_border(row.cells[1])
    row.cells[0].width = Inches(6.5)
    row.cells[1].width = Inches(0.8)

    p = row.cells[0].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(parse_xml(etree.tostring(build_omath(), encoding="unicode")))

    p2 = row.cells[1].paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p2.add_run("(S2.9)")
    set_run_font(r, 12)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    note.paragraph_format.space_before = Pt(16)
    note.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    r = note.add_run(
        "Completing the square in β, with Ỹ = V(λ)Y and X̃ = V(λ)X. "
        "The term C₀ collects all pieces that do not depend on β."
    )
    set_run_font(r, 12)

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
