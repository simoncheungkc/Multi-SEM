"""Manuscript v30 and supplement v28. Does not overwrite v29 or supplement v27."""

from __future__ import annotations

import copy
import json
from pathlib import Path

from lxml import etree

from build_revision_docs import following_table, make_grid_table
from build_v10_docx import make_p, omath, q, set_cell_text
from build_v12_docx import W, blocks_from, load_zip, ptext, store_zip
from build_v26_docs import upright_greek
from build_v29_docs import assert_word_safe, replace_text
from mnreg.latex_omml import run

ROOT = Path(__file__).resolve().parent
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v29.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v30.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v27.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v28.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def load_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def _fill(cell, text):
    paragraph = cell.find(q(W, "p"))
    for child in list(paragraph):
        if etree.QName(child).localname != "pPr":
            paragraph.remove(child)
    built = make_p(blocks_from(text))
    for child in list(built):
        if etree.QName(child).localname != "pPr":
            paragraph.append(child)


def _replace_table(old, rows):
    blank = [[" "] * len(rows[0]) for _ in rows]
    table = make_grid_table(old, blank)
    for source, row in zip(table.findall(q(W, "tr")), rows):
        for cell, value in zip(source.findall(q(W, "tc")), row):
            _fill(cell, value)
    old.addnext(table)
    old.getparent().remove(old)
    return table


def _num(value, digits=3):
    text = f"{float(value):.{digits}f}"
    return "−" + text[1:] if text.startswith("-") else text


def _ci(mean, lo, hi):
    return f"{_num(mean)} ({_num(lo)}, {_num(hi)})"


def _signed(value):
    sign = "−" if float(value) < 0 else "+"
    return sign + _num(abs(float(value)))


def table_4c(body):
    data = load_json("results/unified/table4c.json")
    rows = [[
        "Estimator", "Parameter", "Truth", "Posterior mean", "Bias", "RMSE",
        "95% coverage (MCSE)", r"$\mathbb{E}(L\mid Y)$", r"$P(L>0.95\mid Y)$",
        r"Max $\hat{R}$", "Holdout RMSE",
    ]]
    labels = [
        ("l1", r"$\ell_1$-constrained Multi-SEM", r"$\lambda_1$, Poisson $d=8$", 0),
        ("l1", r"$\ell_1$-constrained Multi-SEM", r"$\lambda_2$, power-law $\gamma=2.2$", 1),
        ("spectral", r"Spectral-stability Multi-SEM ($\rho(B(\lambda))<1$)", r"$\lambda_1$, Poisson $d=8$", 0),
        ("spectral", r"Spectral-stability Multi-SEM ($\rho(B(\lambda))<1$)", r"$\lambda_2$, power-law $\gamma=2.2$", 1),
    ]
    for key, name, parameter, index in labels:
        block = data["estimators"][key]
        rows.append([
            name,
            parameter,
            _signed(0.75 if index == 0 else -0.55),
            _num(block["posterior_mean"][index]),
            _signed(block["bias"][index]),
            _num(block["rmse"][index]),
            f"{_num(block['coverage'][index])} ({_num(block['coverage_mcse'][index])})",
            _num(block["mean_L"]),
            _num(block["p_L_gt_095"]),
            _num(block["rhat_max"]),
            _num(block["holdout_rmse"]),
        ])
    _replace_table(following_table(body, "Table 4c."), rows)
    replace_text(
        body,
        "Each replication fits the",
        "Maximum split-R-hat and holdout RMSE are separate columns. "
        "For the l1 estimator they both round to 1.027 "
        "(maximum split-R-hat 1.02719; holdout RMSE 1.02721). "
        "The spectral estimator has maximum split-R-hat 1.011 and holdout RMSE 1.014. "
        "Each replication fits the",
        minimum=1,
    )


def table_s4(body):
    data = load_json("results/unified/table_s4_prior.json")
    by_label = {row["label"]: row for row in data["rows"]}
    names = [
        ("baseline", r"Baseline: $\xi=1$, $\alpha_0=\gamma_0=0.001$"),
        ("wider", r"Wider Gaussian: $\xi=0.1$, $\alpha_0=\gamma_0=0.001$"),
        ("ridge", r"Strong ridge: $\xi=5$, $\alpha_0=\gamma_0=0.001$"),
        ("informative", r"Informative scale: $\xi=1$, $\alpha_0=2.0$, $\gamma_0=1.0$"),
        ("beta", r"Radial $\mathrm{Beta}(1,2)\times\mathrm{Dirichlet}(1^5)$"),
    ]
    rows = [[
        "Prior (n = 200 synthetic benchmark)",
        r"$\lambda_1$ (true 0.40)",
        r"$\lambda_2$ (true 0.35)",
        r"Max inactive $|\hat{\lambda}_s|$",
        r"$\sigma^2$ (true 1.00)",
        r"$\mathbb{E}(L\mid Y)$",
        r"$P(L>0.95\mid Y)$",
    ]]
    for key, label in names:
        row = by_label[key]
        rows.append([
            label,
            _ci(row["lambda_mean"][0], row["lambda_lo"][0], row["lambda_hi"][0]),
            _ci(row["lambda_mean"][1], row["lambda_lo"][1], row["lambda_hi"][1]),
            _num(row["inactive_max_abs_mean"]),
            _ci(row["sigma2_mean"], row["sigma2_lo"], row["sigma2_hi"]),
            _num(row["mean_L"]),
            _num(row["p_L_gt_095"]),
        ])
    _replace_table(following_table(body, "Table S4. Prior sensitivity"), rows)


def table_13(body):
    table = following_table(body, "Table 13.")
    for row in table.findall(q(W, "tr")):
        cells = row.findall(q(W, "tc"))
        label = "".join(node.text or "" for node in cells[0].iter(q(W, "t")))
        if "safeguard" not in label:
            continue
        set_cell_text(cells[0], "Multi-SEM (screened + non-FDR safeguard, kriging)")
        added = copy.deepcopy(row)
        added_cells = added.findall(q(W, "tc"))
        set_cell_text(added_cells[0], "Multi-SEM (strict BH at q* = 0.05, else OLS)")
        set_cell_text(added_cells[1], "0.06558")
        set_cell_text(added_cells[2], "—")
        row.addprevious(added)
        return
    raise SystemExit("Table 13 safeguard row was not found")


def _append_caption(body, start, sentence):
    for paragraph in body.iter(q(W, "p")):
        if ptext(paragraph).startswith(start):
            nodes = list(paragraph.iter(q(W, "t")))
            nodes[-1].text = (nodes[-1].text or "") + sentence
            return
    raise SystemExit(f"caption missing: {start}")


def _wrap_base(base):
    texts = "".join(node.text or "" for node in base.iter(q(M, "t")))
    if "|" in texts:
        return
    children = list(base)
    if children and etree.QName(children[0]).localname == "d":
        inner = children[0].find(q(M, "e"))
        pieces = [run("|", sty="p")]
        if inner is not None:
            pieces.extend(list(inner))
        pieces.append(run("|", sty="p"))
        base.remove(children[0])
        for index, piece in enumerate(pieces):
            base.insert(index, piece)
        return
    base.insert(0, run("|", sty="p"))
    base.append(run("|", sty="p"))


def absolute_value_bars(body):
    for math in body.iter(q(M, "oMath")):
        text = "".join(node.text or "" for node in math.iter(q(M, "t")))
        if not text.startswith("L"):
            continue
        for nary in math.findall(q(M, "nary")):
            previous = nary.getprevious()
            earlier = previous.getprevious() if previous is not None else None
            letter = "".join(earlier.itertext()) if earlier is not None else ""
            if letter.startswith("L"):
                _wrap_base(nary.find(q(M, "e")))
        if "sec" in text and "|" not in "".join(node.text or "" for node in math.iter(q(M, "t"))):
            for child in list(math):
                bits = "".join(child.itertext())
                if "λ" in bits and ("sec" in bits or "ind" in bits):
                    child.addprevious(run("|", sty="p"))
                    child.addnext(run("|", sty="p"))


def probability_draws(body):
    for node in list(body.iter(q(M, "t"))):
        if not node.text or not node.text.startswith("(<6.3"):
            continue
        math = node
        while math is not None and etree.QName(math).localname != "oMath":
            math = math.getparent()
        paragraph = math
        while paragraph is not None and etree.QName(paragraph).localname != "p":
            paragraph = paragraph.getparent()
        plain = "".join(item.text or "" for item in paragraph.iter(q(W, "t"))).strip()
        if plain:
            run_el = etree.Element(q(W, "r"))
            text = etree.SubElement(run_el, q(W, "t"))
            text.text = "0 / 16,000 draws (0.0000)"
            math.addnext(run_el)
            math.getparent().remove(math)
            continue
        for child in list(paragraph):
            if etree.QName(child).localname != "pPr":
                paragraph.remove(child)
        run_el = etree.SubElement(paragraph, q(W, "r"))
        text = etree.SubElement(run_el, q(W, "t"))
        text.text = "0 / 16,000 draws (0.0000)"


def step4_and_stability(body):
    replace_text(
        body,
        "Table 4a uses 4 chains of 4000 iterations (2000 burn-in). Its worst split-",
        "Table 4a uses four chains of 4,000 iterations (2,000 burn-in), extending any replication with maximum split-",
    )
    replace_text(
        body,
        " is 1.103.",
        " > 1.05 to 8,000 iterations (4,000 burn-in); after extension, the maximum split-R̂ across all 200 replications is 1.049.",
    )
    replace_text(
        body,
        "and then update the posterior ordinate for ",
        "and refresh the residual vector ",
    )
    for node in body.iter(q(W, "t")):
        if node.text and "refresh the residual vector " in node.text:
            math = node.getparent().getnext()
            math.addprevious(omath(r"e = Y - X\beta"))
            math.getparent().remove(math)
            break
    else:
        raise SystemExit("Algorithm 2 step 4 was not found")
    for node in body.iter(q(W, "t")):
        if node.text and "beyond unity, rendering " in node.text:
            node.text = node.text.replace(
                "beyond unity, rendering ",
                "beyond unity, violating the Neumann-series stability condition and rendering ",
            )
            after = node.getparent().getnext().getnext()
            text_node = after.find(".//" + q(W, "t"))
            if text_node is None:
                text_node = next(after.iter(q(W, "t")))
            text_node.text = text_node.text.replace(
                " singular.",
                " singular whenever unity is an eigenvalue of the combined operator.",
                1,
            )
            break
    else:
        raise SystemExit("signed-sum sentence was not found")
    replace_text(
        body,
        "The spectral-stability region has mean radius 0.693 and maximum 0.741",
        "At the true parameter vector, the combined operator has mean spectral radius 0.693 and maximum 0.741",
    )


def supplement_text(body):
    replace_text(
        body,
        "Live downloading is disabled unless that cache file is absent.",
        "If data/sp100_clean_snapshot.csv is missing, the script raises FileNotFoundError. "
        "Live Yahoo Finance prices are fetched only when --download is passed.",
    )
    replace_text(
        body,
        "The covariance-adapted Boston posteriors were produced by python run_boston.py --adaptive. ",
        "Run python run_boston.py --adaptive. Running without --adaptive writes legacy isotropic chains "
        "in results/boston that are excluded from all reported tables and figures. ",
    )
    replace_text(
        body,
        "Legacy isotropic runs in results/boston are excluded from all reported displays. ",
        "",
    )
    note = (
        " The value 0.0576 in main-text Table 12 is the plug-in posterior-mean point prediction, "
        "whereas 0.0578 is the posterior predictive mean integrated over 2,000 MCMC draws and 20 Gaussian replicates."
    )
    _append_caption(body, "Table S4b.", note)
    _append_caption(body, "Table S13b.", note)
    _append_caption(
        body,
        "Table S4. Prior sensitivity",
        " The rows are the covariance-adapted refit of the Table 4 design "
        "(seed 2026; four chains of 6,000 iterations, 3,000 burn-in), including the radial Beta(1, 2) prior.",
    )
    for node in body.iter(q(W, "t")):
        if node.text and "outside-" in node.text and "ball" in ptext(node.getparent().getparent()):
            node.text = node.text.replace("outside-", "outside ")
    for node in body.iter(q(W, "t")):
        if node.text and "table4b.json" in node.text:
            row = node
            while etree.QName(row).localname != "tr":
                row = row.getparent()
            cells = row.findall(q(W, "tc"))
            set_cell_text(cells[1], "run_simulation.py")
            set_cell_text(cells[2], "[202611, rep]")
            set_cell_text(cells[3], "4 x 6000, burn 3000, adapt_covariance=True")
            break
    else:
        raise SystemExit("Table S15 row for Table 4b was not found")


def main():
    main_files, main_root = load_zip(MAIN_IN)
    main_body = main_root.find(q(W, "body"))
    sup_files, sup_root = load_zip(SUP_IN)
    sup_body = sup_root.find(q(W, "body"))

    summary = load_json("results/unified/table4a.json")
    step4_and_stability(main_body)
    table_4c(main_body)
    table_13(main_body)
    probability_draws(main_body)
    absolute_value_bars(main_body)
    _append_caption(
        main_body,
        "Table 4a.",
        f" Three of the 200 replications exceeded a maximum split-R-hat of 1.05 at 4,000 iterations "
        f"and were extended to 8,000 iterations (4,000 burn-in and 4,000 retained). "
        f"The minimum network ESS across all 200 replications is {summary['min_network_ess']:.0f}.",
    )
    upright_greek(main_body)
    assert_word_safe(main_root, "v30")
    store_zip(MAIN_OUT, main_files, main_root)

    table_s4(sup_body)
    supplement_text(sup_body)
    probability_draws(sup_body)
    absolute_value_bars(sup_body)
    upright_greek(sup_body)
    assert_word_safe(sup_root, "supplement v28")
    store_zip(SUP_OUT, sup_files, sup_root)
    print("wrote", MAIN_OUT.name, "and", SUP_OUT.name)


if __name__ == "__main__":
    main()
