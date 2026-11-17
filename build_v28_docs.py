"""Manuscript v28 and supplement v26. Does not overwrite v27 or supplement v25."""

from __future__ import annotations

import json
from pathlib import Path

from lxml import etree

from build_revision_docs import (
    following_table,
    insert_after,
    insert_before,
    make_grid_table,
    replace_paragraph,
)
from build_v10_docx import XML, append_row, make_p, q, set_cell_text
from build_v12_docx import W, blocks_from, load_zip, ptext, require_one, store_zip
from build_v26_docs import replace_exact_math, upright_greek

ROOT = Path(__file__).resolve().parent
V28 = ROOT / "results" / "v28"
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v27.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v28.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v25.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v26.docx"
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_NS = "http://schemas.openxmlformats.org/package/2006/relationships"

TIMING = (
    " Table S8 records dense LU times of 0.082 ms at n = 100 and 1.820 ms at n = 500. "
    "Precision conditioning is an algebraic identity (Rue and Held, 2005; Goulard, Laurent, "
    "and Thomas-Agnan, 2017), and the end-to-end runtime of each MCMC step remains the "
    "training-filter factorization."
)


def load(name):
    path = V28 / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def num(value, digits=3):
    text = f"{float(value):.{digits}f}"
    return ("−" + text[1:]) if text.startswith("-") else text


def fp(value):
    value = float(value)
    if value == 0:
        return "0"
    return f"{value:.3f}" if value >= 0.001 else f"{value:.5f}"


def replace_text(body, old, new, expected=1):
    count = 0
    for node in body.iter(q(W, "t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, new)
            count += 1
    if count != expected:
        raise SystemExit(f"{old!r} replaced in {count} runs, expected {expected}")


def cut_from(body, marker):
    """Delete marker and everything after it in that paragraph, keeping the prefix."""
    for node in body.iter(q(W, "t")):
        if not node.text or marker not in node.text:
            continue
        node.text = node.text[: node.text.index(marker)].rstrip()
        if node.text[:1] == " " or node.text[-1:] == " ":
            node.set(q(XML, "space"), "preserve")
        run = node.getparent()
        paragraph = run.getparent()
        dropping = False
        for child in list(paragraph):
            if dropping:
                paragraph.remove(child)
            elif child is run:
                dropping = True
        return paragraph
    raise SystemExit(f"marker not found: {marker}")


def truncate_from(body, marker, replacement):
    paragraph = cut_from(body, marker)
    run = etree.SubElement(paragraph, q(W, "r"))
    text = etree.SubElement(run, q(W, "t"))
    text.set(q(XML, "space"), "preserve")
    text.text = replacement


def append_run(paragraph, text):
    run = etree.SubElement(paragraph, q(W, "r"))
    node = etree.SubElement(run, q(W, "t"))
    if text[:1] == " " or text[-1:] == " ":
        node.set(q(XML, "space"), "preserve")
    node.text = text


def splice_sentence(body, old, sentence):
    for node in body.iter(q(W, "t")):
        if node.text and old in node.text:
            node.text = node.text.replace(old, "")
            if not node.text.strip():
                node.text = "). "
                node.set(q(XML, "space"), "preserve")
            run = node.getparent()
            paragraph = run.getparent()
            temporary = make_p(blocks_from(sentence), paragraph)
            anchor = run
            for block in list(temporary):
                if block.tag == q(W, "pPr"):
                    continue
                anchor.addnext(block)
                anchor = block
            return
    raise SystemExit(f"sentence host not found: {old}")


def replace_media(files, body, caption_prefix, png_path):
    caption = None
    for paragraph in body.iter(q(W, "p")):
        if ptext(paragraph).startswith(caption_prefix):
            caption = paragraph
            break
    if caption is None:
        raise SystemExit(f"caption missing: {caption_prefix}")
    blip = None
    node = caption.getprevious()
    while node is not None and blip is None:
        for child in node.iter():
            if etree.QName(child).localname == "blip":
                blip = child
                break
        node = node.getprevious()
    if blip is None:
        raise SystemExit(f"no image before {caption_prefix}")
    embed = blip.get(q(R_NS, "embed"))
    rels = etree.fromstring(files["word/_rels/document.xml.rels"])
    target = None
    for rel in rels:
        if rel.get("Id") == embed:
            target = rel.get("Target")
    if not target:
        raise SystemExit(f"relationship {embed} missing")
    media = "word/" + target.replace("\\", "/").lstrip("/")
    files[media] = Path(png_path).read_bytes()
    print("replaced", media)


def assert_word_safe(root, label):
    bad = nested = 0
    for node in root.iter(q(M, "rPr")):
        for child in node:
            if etree.QName(child).localname == "rPr":
                bad += 1
    for node in root.iter(q(M, "e")):
        for child in node:
            if etree.QName(child).localname == "e":
                nested += 1
    if bad or nested:
        raise SystemExit(f"{label} invalid OMML: w:rPr inside m:rPr={bad}, nested m:e={nested}")


def boston_schur_mean():
    panel = V28 / "schur_panel.json"
    if panel.exists():
        data = json.loads(panel.read_text(encoding="utf-8"))
        return data["boston_mean"], len(data["boston_splits"])
    rows = []
    for split in range(10):
        path = V28 / "schur" / f"boston_{split:02d}.json"
        if not path.exists():
            raise SystemExit(f"Boston Schur incomplete at split {split}")
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    fields = ("rmse", "mae", "cover90", "cover95", "width95", "crps")

    def average(key):
        return {field: sum(row[key][field] for row in rows) / len(rows) for field in fields}

    return {"induced": average("induced"), "schur": average("schur")}, len(rows)


def lam_cell(stats, digits=3):
    parts = []
    for mean, low, high in zip(stats["lambda_mean"], stats["lambda_lo"], stats["lambda_hi"]):
        parts.append(f"{num(mean, digits)} ({num(low, digits)}, {num(high, digits)})")
    return "; ".join(parts)


def edit_main(files, root, analyses, rhat, schur_sp, boston_mean):
    body = root.find(q(W, "body"))
    gauss = analyses["sp_priors"]["gaussian"]
    screen = analyses["sp_priors"]["training_screen"]
    bh = analyses["bh_months"]
    rho = analyses["rho"]
    abstract = require_one(
        body,
        "although autocorrelation-robust inference for the monthly loss differential was weaker",
    )
    append_run(
        abstract,
        " Under spatial extrapolation the ranking changes: leave-one-town-out RMSE is 0.1834 "
        "for screened Multi-SEM and 0.1859 for the geographic SEM, while 10-fold spatial-cluster "
        "RMSE is 0.2165 for Multi-SEM and 0.2038 for the town SEM.",
    )
    replace_text(body, "By contrast, the additive specification", "By contrast, the additive higher-order specification")
    florax = None
    for node in body.iter(q(W, "t")):
        if node.text and node.text.startswith("avoids cross-layer interaction products"):
            florax = node
            break
    if florax is None:
        raise SystemExit("Florax sentence missing")
    florax.text = (
        "(Anselin, 1988; Lee and Liu, 2010) avoids cross-layer interaction products and provides "
        "a direct, transparent decomposition of disturbance dependence across network layers while "
        "avoiding the misspecification distortions documented in single-weight specifications ("
    )
    replace_text(
        body,
        "2 chains, 1600 iterations, 800 burn-in",
        "4 chains, 4,000 iterations, 2,000 burn-in",
    )
    caption = require_one(body, "Table 4a.  Monte Carlo recovery")
    append_run(
        caption,
        f" Across the 200 replications the median split-R-hat is {rhat['rhat_median']:.3f} "
        f"and the 95th percentile is {rhat['rhat_p95']:.3f}; the maximum is {rhat['rhat_max']:.3f}.",
    )
    renamed = 0
    for paragraph in body.iter(q(W, "p")):
        if ptext(paragraph) != "Exact-admissible Multi-SEM":
            continue
        nodes = [node for node in paragraph.iter(q(W, "t")) if node.text]
        nodes[0].text = "Spectral-stability Multi-SEM (ρ(B(λ)) < 1)"
        for extra in nodes[1:]:
            extra.text = ""
        renamed += 1
    if renamed != 2:
        raise SystemExit(f"renamed {renamed} admissible labels")
    replace_text(
        body,
        " is 0.693, which keeps the segment from the origin inside ",
        f" is {rho['rho_mean']:.3f} and the maximum is {rho['rho_max']:.3f}, below 1 in "
        f"{rho['rho_lt_1']} of {rho['reps']} replications. This spectral-stability region is a "
        "sufficient subset of the origin-connected component on which the determinant is nonzero; "
        "the mean radius keeps the segment from the origin inside ",
    )
    replace_text(
        body,
        "-constrained sampler and the exact spectral constraint, with four adapted chains of 6,000 iterations and 3,000 burn-in. The ",
        "-constrained sampler and the spectral-stability constraint, with four adapted chains of 6,000 iterations and 3,000 burn-in. The ",
    )
    replace_text(
        body,
        " ball but inside the origin's admissible component: the mean spectral radius is 0.693. The ",
        f" ball. The spectral-stability region has mean radius {rho['rho_mean']:.3f} and maximum "
        f"{rho['rho_max']:.3f}, below 1 in {rho['rho_lt_1']} of {rho['reps']} replications. "
        "The ℓ1 bound guarantees invertibility but truncates parameter inference. The ",
    )
    replace_text(
        body,
        "interval for the first coefficient covers the truth in 10.0% of replications.",
        "interval for the first coefficient has severe undercoverage, covering the truth in 10.0% of replications when the truth lies outside the ℓ1 ball.",
    )
    replace_text(body, "Three checks close the questions left open by the earlier draft. ",
                 "Three additional analyses evaluate the sensitivity of the pipeline to parameter-space truncation, predictive uncertainty quantification, and spatial extrapolation. ")
    replace_paragraph(
        body,
        "Four components underpin the reported analysis.",
        "The four pipeline components are an integrated Bayesian implementation and empirical workflow "
        "for the established additive higher-order error specification "
        r"$u=\sum_{s=1}^{k}\lambda_s A_s u+\varepsilon$, rather than a new norm or conditioning theory. "
        r"The restriction $\Vert\lambda\Vert_1<1$ is the sufficient row-substochastic bound used for estimation. "
        r"Residual Moran's $I$ uses its exact design-conditional moments; when a candidate layer satisfies "
        r"$\kappa>3$ or $\psi>0.20$, its $p$-value comes from a conditional Gaussian Monte Carlo procedure "
        "before Benjamini–Hochberg control. Clique resolvents that reproduce a base layer, identified by "
        "an off-diagonal correlation of at least 0.99, are removed before screening. Holdout imputation "
        "uses the precision form of the conditional Gaussian. That identity is algebraic (Rue and Held, 2005; "
        "Goulard, Laurent, and Thomas-Agnan, 2017), and each MCMC step is governed by the training-filter factorization.",
    )
    truncate_from(body, ", so the holdout factorization is smaller", TIMING)
    truncate_from(body, "giving a ratio of cubic computational costs of", TIMING)
    replace_exact_math(body, "=0.209,\xa0", "=0.206,\xa0", 1)
    replace_exact_math(body, "=0.181", "=0.186", 1)
    n_tail = int(round(gauss["p_L_gt_095"] * 16000))
    if n_tail == 0:
        tail = "No retained draw exceeds 0.95, so the posterior probability is 0, "
    else:
        tail = (
            f"{n_tail} of the 16,000 retained draws exceed 0.95, so the posterior probability is "
            f"{fp(gauss['p_L_gt_095'])}, "
        )
    replace_text(
        body,
        " is 0.430 and its posterior 95th percentile is 0.661. Only four retained draws exceed 0.95, and none reaches the stability boundary at 1.",
        f" is {gauss['mean_L']:.3f} and its posterior 95th percentile is {gauss['L_0.95']:.3f}. "
        f"{tail}and none reaches the stability boundary at 1.",
    )
    for cell in body.iter(q(W, "tc")):
        if any(node.text and "draws not archived" in node.text for node in cell.iter(q(W, "t"))):
            row = cell.getparent()
            cells = row.findall(q(W, "tc"))
            set_cell_text(cells[1], num(gauss["mean_L"], 3))
            set_cell_text(cells[4], f"L0.95={num(gauss['L_0.95'], 3)}; P={fp(gauss['p_L_gt_095'])}")
            break
    else:
        raise SystemExit("Table 11 archive note missing")
    replace_text(
        body,
        "On the ten contiguous blocks the town SEM has RMSE 0.2038",
        "On the ten spatial clusters, nine of which induce a connected geographic subgraph, the town SEM has RMSE 0.2038",
    )
    replace_text(
        body,
        "On the ten contiguous spatial blocks the town SEM has RMSE 0.204",
        "On the ten spatial clusters the town SEM has RMSE 0.204",
    )
    insert_after(
        require_one(body, "Across the 63 monthly cross-sections, the layers selected"),
        [
            f"Of the {bh['months']} months, {bh['bh']['months']} have at least one layer that survives "
            f"Benjamini–Hochberg control at $q^*=0.05$. The other {bh['fallback']['months']} months use the "
            "non-FDR predictive safeguard, which retains layers with $p<0.10$ or, if that set is empty, the "
            f"minimum-$p$ layer. Mean RMSE on the BH months is {bh['bh']['multi_rmse']:.4f} for Multi-SEM and "
            f"{bh['bh']['ols_rmse']:.4f} for OLS. On the fallback months the means are "
            f"{bh['fallback']['multi_rmse']:.4f} and {bh['fallback']['ols_rmse']:.4f}.",
        ],
        require_one(body, "Across the 63 monthly cross-sections, the layers selected"),
    )
    p_text = "; ".join(f"{name} {value:.4f}" for name, value in screen["pvalues"].items())
    table10 = following_table(body, "Table 10. Residual")
    append_row(
        table10,
        [
            "Training-only screen (80 firms)",
            "—",
            "—",
            "—",
            "—",
            "non-FDR fallback",
            "BH empty",
        ],
    )
    table10.addnext(
        make_p(
            blocks_from(
                "The training-only screen on the August 80-firm block does not pass Benjamini–Hochberg "
                f"at $q^*=0.05$ (two-sided training $p$-values: {p_text}). The $p<0.10$ safeguard retains "
                + " and ".join(screen["names"])
                + f", with holdout RMSE {screen['holdout_rmse']:.4f} and MAE {screen['holdout_mae']:.4f}. "
                "This row is the leak-free counterpart of the archived full-sample showcase."
            ),
            require_one(body, "Table 10. Residual"),
        )
    )
    append_row(
        following_table(body, "Table 12."),
        [
            "Multi-SEM (training-only screen)",
            f"{screen['holdout_rmse']:.4f}",
            f"{screen['holdout_mae']:.4f}",
            "—",
        ],
    )
    insert_before(
        require_one(body, "References"),
        [
            "Data and code availability. The replication scripts, requirements.txt, MANIFEST.sha256, "
            "data/boston_corrected.txt, and data/sp100_clean_snapshot.csv are the project files used for "
            "these results. run_sp100.py sets USE_STATIC_CACHE = True and reads the snapshot without calling "
            "Yahoo Finance. The public address https://github.com/simoncheungkc/Multi-SEM did not resolve, "
            "and no Zenodo DOI has been minted.",
        ],
        require_one(body, "References"),
    )
    note = (
        f"On the August 2026 panel the induced-subgraph posterior means are "
        f"{schur_sp['induced']['lambda_mean'][0]:.3f} and {schur_sp['induced']['lambda_mean'][1]:.3f}, "
        f"with 95% coverage {schur_sp['induced']['cover95']:.2f}, mean interval width "
        f"{schur_sp['induced']['width95']:.3f}, and CRPS {schur_sp['induced']['crps']:.3f}. "
        f"The coherent Schur fit has means {schur_sp['schur']['lambda_mean'][0]:.3f} and "
        f"{schur_sp['schur']['lambda_mean'][1]:.3f}, coverage {schur_sp['schur']['cover95']:.2f}, "
        f"width {schur_sp['schur']['width95']:.3f}, and CRPS {schur_sp['schur']['crps']:.3f}. "
        f"Across Boston splits 0–9 the mean induced RMSE is {boston_mean['induced']['rmse']:.3f} "
        f"and the coherent Schur RMSE is {boston_mean['schur']['rmse']:.3f} (Table S13b)."
    )
    insert_after(
        require_one(body, "Three additional analyses evaluate"),
        [note],
        require_one(body, "Three additional analyses evaluate"),
    )
    upright_greek(body)
    assert_word_safe(root, "v28")
    return boston_mean


def prior_rows(analyses):
    boston = analyses["boston_priors"]
    sp = analyses["sp_priors"]
    beta_boston = analyses["boston_beta"]
    rows = [["Application and prior", "Active means (95% CrI)", "E(L|Y)", "L 0.95", "P(L>0.95|Y)", "RMSE", "CRPS"]]
    specs = [
        ("Boston Split 5, truncated Gaussian", boston["gaussian"], 3, 3),
        ("Boston Split 5, Beta(1,2) radial", {**beta_boston, "holdout": beta_boston["holdout"]}, 3, 3),
        ("S&P August 2026, truncated Gaussian", sp["gaussian"], 4, 4),
        ("S&P August 2026, Beta(1,2) radial", sp["beta_l1"], 4, 4),
    ]
    for label, stats, rmse_digits, _crps_digits in specs:
        rows.append([
            label,
            lam_cell(stats),
            num(stats["mean_L"], 3),
            num(stats["L_0.95"], 3),
            fp(stats["p_L_gt_095"]),
            num(stats["holdout"]["rmse"], rmse_digits),
            num(stats["holdout"]["crps"], 3),
        ])
    return rows


def schur_rows(schur_sp, boston_mean):
    def line(label, kind, block, digits):
        means = block.get("lambda_mean") or []
        return [
            label,
            kind,
            "; ".join(num(v, 3) for v in means) if means else "—",
            num(block["cover90"], 3),
            num(block["cover95"], 3),
            num(block["width95"], 3),
            num(block["crps"], 3),
            num(block["rmse"], digits),
        ]

    rows = [[
        "Application", "Likelihood", "Posterior means", "90% coverage", "95% coverage",
        "95% width", "CRPS", "RMSE",
    ]]
    rows.append(line("S&P August 2026", "Induced subgraph", schur_sp["induced"], 4))
    rows.append(line("S&P August 2026", "Coherent Schur", schur_sp["schur"], 4))
    rows.append(line("Boston splits 0–9", "Induced subgraph", boston_mean["induced"], 3))
    rows.append(line("Boston splits 0–9", "Coherent Schur", boston_mean["schur"], 3))
    return rows


def edit_supplement(files, root, analyses, schur_sp, boston_mean):
    body = root.find(q(W, "body"))
    cut_from(body, "The Boston housing chains use this covariance.")
    gauss = analyses["sp_priors"]["gaussian"]
    radial = analyses["sp_priors"]["beta_l1"]
    replace_paragraph(
        body,
        "The August 2026 showcase was refit",
        "The August 2026 showcase was refit on the same 80-firm training block with "
        r"adapt_covariance=True. The truncated-Gaussian prior gives $\lambda_{\mathrm{sec}}="
        f"{gauss['lambda_mean'][0]:.3f}$ and $\\lambda_{{\\mathrm{{ind}}}}={gauss['lambda_mean'][1]:.3f}$, "
        f"with $\\mathbb{{E}}(L\\mid Y)={gauss['mean_L']:.3f}$, $L_{{0.95}}={gauss['L_0.95']:.3f}$, and "
        f"$P(L>0.95\\mid Y)={fp(gauss['p_L_gt_095'])}$. The radial "
        r"$\mathrm{Beta}(1,2)\times\mathrm{Dirichlet}(1,1)$ prior, whose Cartesian log-density is "
        r"$(1-k)\ln L+\ln(1-L)$ on $L<1$, gives means "
        f"{radial['lambda_mean'][0]:.3f} and {radial['lambda_mean'][1]:.3f}, with "
        f"$\\mathbb{{E}}(L\\mid Y)={radial['mean_L']:.3f}$. Table S4b reports both applications.",
    )
    replace_paragraph(
        body,
        "Figure S10 overlays draws",
        "Figure S10 compares the Split 5 posterior with 200,000 rejection draws from the exact "
        r"truncated-Gaussian prior ($\xi=1$, $\Vert\lambda\Vert_1<1$). The left panel shows that prior "
        r"density of $L$, its prior mean, and the analytical shell $3L^{2}$. The radial "
        r"$\mathrm{Beta}(1,2)$ prior implements $(1-k)\ln\Vert\lambda\Vert_1+\ln(1-\Vert\lambda\Vert_1)$ "
        r"on the open $\ell_1$ ball, and both applications were refit with adapt_covariance=True.",
    )
    replace_paragraph(
        body,
        "Figure S10. Prior and posterior",
        "Figure S10. Prior and posterior densities of $L$ on Boston Split 5 ($k=3$). Left: the exact "
        r"truncated-Gaussian prior, its mean, and the shell $3L^{2}$, with the covariance-adapted posterior. "
        r"Right: the $\mathrm{Beta}(1,2)$ radial prior and its posterior.",
    )
    caption = require_one(body, "Figure S10. Prior and posterior")
    table = make_grid_table(following_table(body, "Table S4. Prior sensitivity"), prior_rows(analyses))
    caption.addnext(table)
    caption.addnext(
        make_p(
            blocks_from(
                "Table S4b. Exact truncated-Gaussian and radial Beta(1, 2) priors under "
                "adapt_covariance=True. Credible intervals are central 95% intervals. "
                "Holdout scores use 2,000 thinned draws and 20 Gaussian replicates, except the Boston "
                "truncated-Gaussian row, which uses the archived Split 5 ensemble."
            ),
            caption,
        )
    )
    heading = None
    stop = None
    anchor = None
    for child in list(body):
        if child.tag != q(W, "p"):
            continue
        text = ptext(child).strip()
        if text.startswith("S3.7 "):
            heading = child
        elif text.startswith("S4.8 "):
            stop = child
        elif text.startswith("Section S4."):
            anchor = child
    if None in (heading, stop, anchor):
        raise SystemExit("could not locate S3.7 for the move")
    node = heading
    while node is not stop:
        nxt = node.getnext()
        anchor.addprevious(node)
        node = nxt
    splice_sentence(
        body,
        "), substantially improving effective sample size and parameter recovery.",
        "Covariance adaptation increases minimum effective sample size more than fourfold "
        r"(from $100.3$ to $442.5$) and lowers maximum split-$\hat{R}$ from $1.187$ to $1.040$, "
        "while leaving parameter and holdout RMSE essentially unchanged.",
    )
    replace_exact_math(body, "=0.209+0.181=0.390,", "=0.206+0.186=0.392,", 1)
    replace_text(body, "The ten contiguous blocks are k-means clusters", "The ten spatial clusters are k-means clusters")
    replace_text(body, "On the contiguous blocks the town SEM", "On the spatial clusters the town SEM")
    replace_text(body, "10-fold contiguous spatial cross-validation", "10-fold spatial-cluster cross-validation")
    replace_text(
        body,
        "their sizes are 10, 26, 30, 34, 37, 44, 61, 64, 83, and 117 tracts.",
        "their sizes are 10, 26, 30, 34, 37, 44, 61, 64, 83, and 117 tracts. Nine of the ten induce a "
        "connected subgraph of the geographic adjacency; the cluster of 64 tracts has two components, "
        "the larger containing 53 tracts.",
    )
    bh = analyses["bh_months"]
    insert_after(
        require_one(body, "The active network structure selected by training-only"),
        [
            f"Benjamini–Hochberg at $q^*=0.05$ is active in {bh['bh']['months']} of {bh['months']} months. "
            f"The other {bh['fallback']['months']} months use the non-FDR safeguard. Mean RMSE is "
            f"{bh['bh']['multi_rmse']:.4f} for Multi-SEM and {bh['bh']['ols_rmse']:.4f} for OLS on the BH "
            f"months, and {bh['fallback']['multi_rmse']:.4f} versus {bh['fallback']['ols_rmse']:.4f} on the "
            "fallback months.",
        ],
        require_one(body, "The active network structure selected by training-only"),
    )
    s49 = None
    for child in body:
        if child.tag == q(W, "p") and ptext(child).startswith("S4.9 "):
            s49 = child
            break
    insert_before(
        s49,
        [
            "Table S13b compares the induced-subgraph likelihood with the coherent Schur complement "
            r"$P_T=Q_{tt}-Q_{t*}Q_{**}^{-1}Q_{*t}$ on the August 2026 panel and on Boston splits 0–9. "
            "Scores use 2,000 thinned posterior draws and 20 Gaussian replicates.",
        ],
        s49,
    )
    grid = make_grid_table(following_table(body, "Table S4. Prior sensitivity"), schur_rows(schur_sp, boston_mean))
    s49.addprevious(
        make_p(
            blocks_from(
                "Table S13b. Induced-subgraph and coherent Schur-complement predictive scores. "
                "Boston entries are means over splits 0–9."
            ),
            s49,
        )
    )
    s49.addprevious(grid)
    for cell in body.iter(q(W, "tc")):
        texts = [node.text or "" for node in cell.iter(q(W, "t"))]
        if texts == ["4.350 ", "ms"]:
            set_cell_text(cell, "4.350 ms")
            break
    else:
        raise SystemExit("Table S8 4.350 ms cell not found")
    relabeled = False
    for cell in body.iter(q(W, "tc")):
        joined = "".join(node.text or "" for node in cell.iter(q(W, "t")))
        if joined.strip() == "yfinance, pandas, numpy":
            set_cell_text(
                cell,
                "pandas, numpy; USE_STATIC_CACHE=True reads data/sp100_clean_snapshot.csv and does not call yfinance",
            )
            relabeled = True
    if not relabeled:
        raise SystemExit("Table S7 yfinance cell not found")
    step = require_one(body, "This script reproduces the empirical S&P 100 analysis")
    append_run(
        step,
        " The default USE_STATIC_CACHE = True reads data/sp100_clean_snapshot.csv and does not call "
        "Yahoo Finance. Live downloading is disabled unless that cache file is absent.",
    )
    replace_text(
        body,
        "No live Zenodo DOI is listed because the public repository address did not resolve.",
        "The intended repository is https://github.com/simoncheungkc/Multi-SEM. No live Zenodo DOI is "
        "listed because that address did not resolve.",
    )
    replace_media(files, body, "Figure S10.", ROOT / "results" / "figures" / "figure_s10_prior_posterior.png")
    replace_media(files, body, "Figure S11.", ROOT / "results" / "figures" / "figure_s11_boundary.png")
    upright_greek(body)
    assert_word_safe(root, "supplement v26")


def main():
    analyses = load("analyses.json")
    rhat = json.loads((ROOT / "results" / "unified" / "table4a_rhat.json").read_text(encoding="utf-8"))
    schur_sp = load("schur_sp.json")
    boston_mean, n_splits = boston_schur_mean()
    if n_splits != 10:
        raise SystemExit(f"expected 10 Boston splits, found {n_splits}")
    main_files, main_root = load_zip(MAIN_IN)
    edit_main(main_files, main_root, analyses, rhat, schur_sp, boston_mean)
    store_zip(MAIN_OUT, main_files, main_root)
    sup_files, sup_root = load_zip(SUP_IN)
    edit_supplement(sup_files, sup_root, analyses, schur_sp, boston_mean)
    store_zip(SUP_OUT, sup_files, sup_root)
    print("wrote", MAIN_OUT.name, "and", SUP_OUT.name)


if __name__ == "__main__":
    main()
