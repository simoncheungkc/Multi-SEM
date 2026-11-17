"""LaTeX subset -> Word OMML, used to rebuild the supplementary materials."""

from __future__ import annotations

import re
from lxml import etree

M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
etree.register_namespace("m", M)

GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "varepsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι",
    "kappa": "κ", "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "rho": "ρ", "sigma": "σ", "tau": "τ", "phi": "φ", "varphi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω", "ell": "ℓ",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ", "Xi": "Ξ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

SYMS = {
    "infty": "∞", "times": "×", "cdot": "·", "ast": "*", "pm": "±",
    "leq": "≤", "geq": "≥", "neq": "≠", "approx": "≈", "propto": "∝",
    "sim": "∼", "mid": "∣", "in": "∈", "notin": "∉", "forall": "∀", "exists": "∃",
    "rightarrow": "→", "longrightarrow": "→", "Rightarrow": "⇒",
    "leftarrow": "←", "mapsto": "↦", "subset": "⊂", "subseteq": "⊆",
    "cup": "∪", "cap": "∩", "emptyset": "∅", "partial": "∂",
    "ell": "ℓ", "l1": "ℓ₁",
    "prime": "′", "ldots": "…", "cdots": "⋯", "dots": "…",
}

UPRIGHT = {"tr", "exp", "ln", "log", "det", "diag", "min", "max", "Var", "Cov", "rank", "sup", "inf"}


def el(tag, *children, **attrs):
    node = etree.Element(f"{{{M}}}{tag}")
    for k, v in attrs.items():
        node.set(f"{{{M}}}{k}", str(v))
    for ch in children:
        if ch is None:
            continue
        if isinstance(ch, (list, tuple)):
            for x in ch:
                if x is not None:
                    node.append(x)
        else:
            node.append(ch)
    return node


def run(text, sty=None, scr=None):
    r = el("r")
    if sty or scr:
        rpr = el("rPr")
        if sty:
            rpr.append(el("sty", val=sty))
        if scr:
            rpr.append(el("scr", val=scr))
        r.append(rpr)
    t = el("t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def wrap_e(*nodes):
    flat = []
    for node in nodes:
        if etree.QName(node).localname == "e":
            flat.extend(list(node))
        else:
            flat.append(node)
    return el("e", *flat)


class LatexParser:
    def __init__(self, s: str):
        self.s = s
        self.n = len(s)
        self.i = 0

    def peek(self):
        return self.s[self.i] if self.i < self.n else ""

    def get(self):
        ch = self.peek()
        self.i += 1
        return ch

    def skip_space(self):
        while self.i < self.n and self.s[self.i] in " \t":
            self.i += 1

    def parse_expr(self, stop=""):
        nodes = []
        steps = 0
        while self.i < self.n:
            steps += 1
            if steps > 20000:
                raise RuntimeError(f"parse_expr overflow at i={self.i} rest={self.s[self.i:self.i+20]!r}")
            ch = self.peek()
            if stop and ch in stop:
                break
            if ch in " \t":
                self.i += 1
                continue
            nodes.append(self.parse_atom())
        return nodes

    def parse_atom(self):
        start = self.i
        self.skip_space()
        ch = self.peek()
        if not ch:
            return run("")
        if ch == "\\":
            node = self.parse_command()
        elif ch == "{":
            self.get()
            node = self._group_or_single(self.parse_expr("}"))
            if self.peek() == "}":
                self.get()
        elif ch in "0123456789.":
            node = self.parse_number()
        elif ch.isalpha():
            node = self.parse_ident()
        elif ch == "'":
            self.get()
            node = run("′")
        elif ch in "|&":
            self.get()
            node = run(ch)
        else:
            self.get()
            node = run({"-": "−"}.get(ch, ch))
        node = self.parse_scripts(node)
        if self.i == start:
            self.get()
        return node

    def parse_scripts(self, base):
        sub = sup = None
        guard = 0
        while True:
            guard += 1
            if guard > 50:
                raise RuntimeError(f"parse_scripts loop i={self.i} ch={self.peek()!r} rest={self.s[self.i:self.i+15]!r}")
            self.skip_space()
            ch = self.peek()
            if ch == "_" and sub is None:
                self.get()
                sub = self.parse_script_arg()
            elif ch == "^" and sup is None:
                self.get()
                sup = self.parse_script_arg()
            else:
                break
        if sub is not None and sup is not None:
            return el("sSubSup", wrap_e(base), el("sub", *sub), el("sup", *sup))
        if sub is not None:
            return el("sSub", wrap_e(base), el("sub", *sub))
        if sup is not None:
            return el("sSup", wrap_e(base), el("sup", *sup))
        return base

    def parse_script_arg(self):
        self.skip_space()
        if self.peek() == "{":
            self.get()
            nodes = self.parse_expr("}")
            if self.peek() == "}":
                self.get()
            return nodes
        return [self.parse_atom()]

    def parse_number(self):
        j = self.i
        while self.i < self.n and self.s[self.i] in "0123456789.":
            self.get()
        return run(self.s[j:self.i])

    def parse_ident(self):
        j = self.i
        while self.peek().isalpha():
            self.get()
        name = self.s[j:self.i]
        if name in UPRIGHT:
            return run(name, sty="p")
        if len(name) > 1:
            return run(name, sty="p")
        return run(name)

    def parse_command(self):
        self.get()  # \
        if self.peek() == ",":
            self.get()
            return run(" ")
        if self.peek() in r"\{}[]|_^%&":
            ch = self.get()
            if ch in "{}":
                return run(ch)
            if ch == " ":
                return run(" ")
            return run(ch)
        j = self.i
        while self.peek().isalpha():
            self.get()
        cmd = self.s[j:self.i]
        return self.apply_command(cmd)

    def apply_command(self, cmd):
        if cmd in GREEK:
            return run(GREEK[cmd])
        if cmd in SYMS:
            return run(SYMS[cmd])
        if cmd in ("left", "right"):
            return self.parse_delim_side(cmd == "left")
        if cmd == "frac":
            num = self.parse_group()
            den = self.parse_group()
            return el("f", el("num", *num), el("den", *den))
        if cmd in ("dfrac", "tfrac"):
            num = self.parse_group()
            den = self.parse_group()
            return el("f", el("num", *num), el("den", *den))
        if cmd == "sqrt":
            return el("rad", el("radPr", el("degHide", val="on")), wrap_e(*self.parse_group()))
        if cmd in ("hat", "widehat"):
            return el("acc", el("accPr", el("chr", val="ˆ")), wrap_e(*self.parse_group()))
        if cmd in ("tilde", "widetilde"):
            return el("acc", el("accPr", el("chr", val="˜")), wrap_e(*self.parse_group()))
        if cmd in ("bar", "overline"):
            return el("bar", wrap_e(*self.parse_group()))
        if cmd == "xrightarrow":
            sup = self.parse_group() if self.peek() == "{" else None
            arrow = run("→")
            if sup:
                return el("sSup", wrap_e(arrow), el("sup", *sup))
            return arrow
        if cmd in ("mathrm", "mathsf", "operatorname", "text"):
            return run(self.read_group_text(), sty="p")
        if cmd == "mathbf":
            return run(self.read_group_text(), sty="b")
        if cmd == "boldsymbol":
            inner = self.parse_group()
            # bold-italic greek typically
            for n in inner:
                rpr = n.find(f"{{{M}}}rPr")
                if rpr is None:
                    rpr = el("rPr")
                    n.insert(0, rpr)
                rpr.append(el("sty", val="bi"))
            return inner[0] if len(inner) == 1 else el("e", *inner)  # fallback
        if cmd == "mathbb":
            txt = self.read_group_text()
            return run(txt, sty="p", scr="double-struck")
        if cmd == "mathcal":
            txt = self.read_group_text()
            return run(txt, sty="p", scr="script")
        if cmd == "sum":
            return self.parse_nary("∑")
        if cmd == "prod":
            return self.parse_nary("∏")
        if cmd == "infty":
            return run("∞")
        if cmd == "cup":
            return run("∪")
        if cmd == "cap":
            return run("∩")
        if cmd == "quad":
            return run("    ")
        if cmd == "qquad":
            return run("        ")
        if cmd == ",":
            return run(" ")
        if cmd == ";":
            return run("  ")
        if cmd == "!":
            return run("")
        if cmd in UPRIGHT or cmd in ("Inv", "Uniform"):
            return run(cmd, sty="p")
        if cmd == "begin":
            env = self.read_group_text()
            return self.parse_env(env)
        if cmd in ("Bigl", "Bigr", "bigl", "bigr", "Big", "bigg", "Bigg"):
            self.skip_space()
            if self.peek() == "\\":
                return self.parse_command()
            if self.peek():
                return run(self.get())
            return run("")
        if cmd == "to":
            return run("→")
        if cmd in ("lVert", "rVert", "Vert"):
            return run("‖")
        if cmd in ("lvert", "rvert", "vert"):
            return run("|")
        if cmd == "backslash":
            return run("\\")
        if cmd == "I":  # indicator sometimes \mathbb{I}
            return run("I")
        # unknown command: upright name
        return run(cmd, sty="p")

    def parse_delim_side(self, is_left):
        self.skip_space()
        ch = self.peek()
        if ch == "\\":
            self.get()
            j = self.i
            while self.peek().isalpha():
                self.get()
            name = self.s[j:self.i]
            mapping = {"lVert": "‖", "rVert": "‖", "Vert": "‖", "lvert": "|", "rvert": "|"}
            delim_ch = mapping.get(name, name[:1] or ".")
        else:
            delim_ch = self.get()
            if delim_ch == ".":
                delim_ch = ""
        if is_left:
            # parse until \right
            inner = []
            while self.i < self.n:
                if self.s.startswith("\\right", self.i):
                    break
                inner.append(self.parse_atom())
            # consume \right ...
            if self.s.startswith("\\right", self.i):
                self.i += 6
                self.skip_space()
                if self.peek() == "\\":
                    self.get()
                    while self.peek().isalpha():
                        self.get()
                elif self.peek():
                    self.get()
            end_ch = {"(": ")", "[": "]", "{": "}"}.get(delim_ch, delim_ch)
            return el(
                "d",
                el("dPr", el("begChr", val=delim_ch or ""), el("endChr", val=end_ch or ""), el("grow", val="1")),
                wrap_e(*inner),
            )
        return run(delim_ch)

    def parse_nary(self, chr_):
        sub = sup = None
        while True:
            self.skip_space()
            if self.peek() == "_" and sub is None:
                self.get()
                sub = self.parse_script_arg()
            elif self.peek() == "^" and sup is None:
                self.get()
                sup = self.parse_script_arg()
            else:
                break
        operand = self.parse_atom()
        return el(
            "nary",
            el("naryPr", el("chr", val=chr_), el("limLoc", val="undOvr"), el("grow", val="1")),
            el("sub", *(sub or [run("")])),
            el("sup", *(sup or [run("")])),
            wrap_e(operand),
        )

    def parse_group(self):
        self.skip_space()
        if self.peek() == "{":
            self.get()
            nodes = self.parse_expr("}")
            if self.peek() == "}":
                self.get()
            return nodes
        return [self.parse_atom()]

    def read_group_text(self):
        nodes_as_text = []
        self.skip_space()
        if self.peek() != "{":
            return self.get()
        self.get()
        depth = 1
        j = self.i
        while self.i < self.n and depth:
            ch = self.get()
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
        return self.s[j:self.i - 1]

    def parse_env(self, env):
        content = []
        end = r"\end{" + env + "}"
        while self.i < self.n and not self.s.startswith(end, self.i):
            if self.peek() == "&":
                self.get()
                content.append(("COL",))
            elif self.s.startswith(r"\\", self.i):
                self.i += 2
                content.append(("ROW",))
            else:
                content.append(self.parse_atom())
        if self.s.startswith(end, self.i):
            self.i += len(end)
        if env in ("bmatrix", "pmatrix", "matrix", "Bmatrix"):
            return self.make_matrix(content, env)
        return wrap_e(*[c for c in content if not isinstance(c, tuple)])

    def make_matrix(self, content, env):
        rows, row = [], []
        cell = []
        def flush_cell():
            nonlocal cell
            row.append(list(cell))
            cell = []
        def flush_row():
            nonlocal row
            if cell:
                flush_cell()
            if row:
                rows.append(row)
                row = []
        for item in content:
            if item == ("COL",):
                flush_cell()
            elif item == ("ROW",):
                flush_row()
            else:
                cell.append(item)
        flush_row()
        if not rows:
            rows = [[[run("")]]]
        n_cols = max(len(r) for r in rows)
        for r in rows:
            while len(r) < n_cols:
                r.append([run("")])
        mcs = el("mcs", el("mc", el("mcPr", el("count", val=str(n_cols)), el("mcJc", val="center"))))
        mrs = [el("mr", *[wrap_e(*cell) for cell in r]) for r in rows]
        mat = el("m", el("mPr", mcs), *mrs)
        br = {"bmatrix": ("[", "]"), "pmatrix": ("(", ")"), "Bmatrix": ("{", "}")}.get(env, ("", ""))
        if br[0]:
            return el(
                "d",
                el("dPr", el("begChr", val=br[0]), el("endChr", val=br[1]), el("grow", val="1")),
                wrap_e(mat),
            )
        return mat

    def _group_or_single(self, nodes):
        if len(nodes) == 1:
            return nodes[0]
        # concatenate into a phantom e
        box = el("e", *nodes)
        return box


def latex_nodes(tex: str):
    p = LatexParser(tex)
    nodes = p.parse_expr()
    return nodes


def omath_xml(tex: str) -> str:
    nodes = latex_nodes(tex)
    om = el("oMath", *nodes)
    return etree.tostring(om, encoding="unicode")
