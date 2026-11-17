"""Write Equation (S2.6) as a native Word OMML aligned equation."""

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
OUT = ROOT / "EquationS26_Joint_Posterior.docx"

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


def nary_sum(sub, sup, content):
    return el(
        "nary",
        el("naryPr", el("chr", val="∑"), el("limLoc", val="undOvr"), el("grow", val="1")),
        el("sub", *lst(sub)),
        el("sup", *lst(sup)),
        e(*lst(content)),
    )


def lam_bold():
    return run("λ", sty="bi")


def sigma2():
    return ssup(run("σ"), run("2"))


def Vlam():
    return [run("V"), delim(lam_bold())]


def aln():
    return run("", aln=True)


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
    line1 = e(
        run("p"),
        delim(
            [
                run("β"),
                run(", "),
                sigma2(),
                run(", "),
                lam_bold(),
                run(" ∣ "),
                run("Y"),
                run(", "),
                run("X"),
            ]
        ),
        aln(),
        run(" ∝ "),
        delim(Vlam(), "|", "|"),
        run(" "),
        ssup(
            delim(sigma2()),
            [
                run("−"),
                delim(
                    [
                        frac(run("n"), run("2")),
                        run(" + "),
                        ssub(run("α"), run("0")),
                        run(" + 1"),
                    ]
                ),
            ],
        ),
    )

    yxb = lambda: delim([run("Y"), run(" − "), run("X"), run("β")])
    line2 = e(
        aln(),
        run(" × "),
        run("exp", sty="p"),
        delim(
            [
                run("−"),
                frac(run("1"), sigma2()),
                delim(
                    [
                        ssub(run("γ"), run("0")),
                        run(" + "),
                        frac(run("1"), run("2")),
                        yxb(),
                        run("′"),
                        *Vlam(),
                        run("′"),
                        *Vlam(),
                        yxb(),
                    ],
                    "[",
                    "]",
                ),
            ]
        ),
    )

    bmu = lambda: delim([run("β"), run(" − "), ssub(run("μ"), run("0"))])
    line3 = e(
        aln(),
        run(" × "),
        run("exp", sty="p"),
        delim(
            [
                run("−"),
                frac(run("1"), run("2")),
                bmu(),
                run("′"),
                ssubsup(run("Σ"), run("0"), run("−1")),
                bmu(),
                run(" − "),
                frac(run("ξ"), run("2")),
                lam_bold(),
                run("′"),
                lam_bold(),
            ]
        ),
        run(" "),
        run("I", sty="p", scr="double-struck"),
        delim(
            [
                nary_sum(
                    [run("s"), run("="), run("1")],
                    run("k"),
                    delim(ssub(run("λ"), run("s")), "|", "|"),
                ),
                run(" < 1"),
            ]
        ),
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
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    doc.styles["Normal"].font.name = "Times New Roman"
    doc.styles["Normal"].font.size = Pt(12)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title.paragraph_format.space_after = Pt(16)
    r = title.add_run("Joint Posterior of the Multi-SEM Parameters")
    set_run_font(r, 14, bold=True)

    table = doc.add_table(rows=1, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    row = table.rows[0]
    set_cell_border(row.cells[0])
    set_cell_border(row.cells[1])
    row.cells[0].width = Inches(6.3)
    row.cells[1].width = Inches(0.8)

    p = row.cells[0].paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p._p.append(parse_xml(etree.tostring(build_omath(), encoding="unicode")))

    p2 = row.cells[1].paragraphs[0]
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r = p2.add_run("(S2.6)")
    set_run_font(r, 12)

    note = doc.add_paragraph()
    note.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    note.paragraph_format.space_before = Pt(16)
    note.paragraph_format.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
    r = note.add_run(
        "The joint posterior of (β, σ², λ) given (Y, X) is, up to a constant of proportionality, "
        "the product of the Gaussian likelihood, the Inverse-Gamma kernel in σ², the Gaussian prior "
        "on β, the Gaussian prior on λ, and the indicator that enforces the ℓ1 invertibility bound "
        "Σs |λs| < 1. Here V(λ) = I − Σs λs As."
    )
    set_run_font(r, 12)

    doc.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
