"""Build v19 and supplement v17 from the covariance-adapted Boston rerun.

Reads results/boston_adaptive. Does not overwrite v18, supplement v16,
or the isotropic archive in results/boston.
"""

from __future__ import annotations

import json
import math

from lxml import etree

import build_v12_docx as b
from build_v10_docx import set_cell_text

ROOT = b.ROOT
W = b.W
ADAPTIVE = ROOT / "results" / "boston_adaptive"
MAIN_IN = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v18.docx"
SUP_IN = ROOT / "Supplementary_Materials_MultiSEM_v16.docx"
MAIN_OUT = ROOT / "REGRESSION WITH MULTIPLE NETWORKS v19.docx"
SUP_OUT = ROOT / "Supplementary_Materials_MultiSEM_v17.docx"


def signed(value, digits):
    return f"{value:.{digits}f}".replace("-", "−")


def ess(value):
    return f"{int(round(value)):,}"


def sci(value):
    if value <= 0:
        return "0"
    exp = int(math.floor(math.log10(value)))
    mant = value / 10 ** exp
    return f"{mant:.2f} \\times 10^{{{exp}}}"


def cell_text(tc):
    return "".join(t.text or "" for t in tc.iter(b.q(W, "t")))


def replace_once(body, old, new):
    hits = [t for t in body.iter(b.q(W, "t")) if t.text and old in t.text]
    if len(hits) != 1:
        preview = " | ".join((t.text or "")[:60] for t in hits[:3])
        raise SystemExit(f"{old!r} matched {len(hits)}: {preview}")
    hits[0].text = hits[0].text.replace(old, new, 1)


def load_results():
    paths = sorted(ADAPTIVE.glob("split_[0-9][0-9].json"))
    if len(paths) != 50:
        raise SystemExit(f"expected 50 adaptive splits, found {len(paths)}")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    summary = json.loads((ADAPTIVE / "summary.json").read_text(encoding="utf-8"))
    by_model = {row["model"]: row for row in summary["table"]}
    split5 = next(row for row in records if int(row["split"]) == 5)
    lam_rhat = [max(row["multi"]["rhat_lambda"]) for row in records]
    lam_ess = [min(row["multi"]["ess_lambda"]) for row in records]
    beta_ess = [min(row["multi"]["ess_beta"]) for row in records]
    max_l = [row["multi"]["max_L"] for row in records]
    facts = {
        "records": records,
        "summary": summary,
        "by_model": by_model,
        "split5": split5,
        "max_rhat_lambda": max(lam_rhat),
        "min_ess_lambda": min(lam_ess),
        "min_ess_beta": min(beta_ess),
        "rhat_ok": sum(value <= 1.025 for value in lam_rhat),
        "ess_ok": sum(value > 400 for value in lam_ess),
        "ess600": sum(value > 600 for value in lam_ess),
        "max_L": max(max_l),
        "ols": by_model["OLS (no network)"]["mean_rmse"],
        "geo": by_model["Single SEM (geographic 6-NN)"]["mean_rmse"],
        "town": by_model["Single SEM (town boundaries)"]["mean_rmse"],
        "multi": by_model["Multi-SEM (screened, precision kriging)"]["mean_rmse"],
        "wins": by_model["Multi-SEM (screened, precision kriging)"]["wins_vs_ols"],
        "p": summary["wilcoxon"]["pvalue"],
    }
    facts["pct_ols"] = 100.0 * (1.0 - facts["multi"] / facts["ols"])
    facts["pct_geo"] = 100.0 * (1.0 - facts["geo"] / facts["ols"])
    facts["pct_inc"] = 100.0 * (1.0 - facts["multi"] / facts["geo"])
    return facts


def row_key(label, joined):
    if "percentile" in joined or "p(l" in joined:
        return "L"
    if "6-nn" in label:
        return "geo"
    if "resolvent" in label:
        return "resolvent"
    if "municipal" in label:
        return "town"
    if "intercept" in label:
        return "intercept"
    if "rooms" in label:
        return "rm"
    if "lower status" in label:
        return "log_lstat"
    if "crime" in label:
        return "crim"
    if "tax" in label:
        return "tax"
    if "pupil" in label:
        return "ptratio"
    if "distance" in label:
        return "log_dis"
    if "nitric" in label or "oxide" in label:
        return "nox"
    if "zoning" in label:
        return "zn"
    if "acreage" in label or "industry" in label:
        return "indus"
    if "innovation" in label or "variance" in label:
        return "sigma2"
    if label.strip(" ()") == "age":
        return "age"
    return None


def fill_table8(body, split5):
    by_name = {row["parameter"]: row for row in split5["multi"]["posterior"]}
    caption = b.require_one(body, "Table 8. Posterior parameter estimates")
    table = caption.getnext()
    while table is not None and table.tag != b.q(W, "tbl"):
        table = table.getnext()
    seen = set()
    for tr in table.findall(b.q(W, "tr"))[1:]:
        cells = tr.findall(b.q(W, "tc"))
        label = cell_text(cells[0]).lower()
        joined = " ".join(cell_text(tc).lower() for tc in cells)
        key = row_key(label, joined)
        if key is None or key in seen or key not in by_name:
            raise SystemExit(f"unmatched Table 8 row {label!r}")
        seen.add(key)
        row = by_name[key]
        digits = 4 if key == "sigma2" else 3
        set_cell_text(cells[1], signed(row["mean"], digits))
        set_cell_text(cells[2], signed(row["median"], digits))
        set_cell_text(cells[3], signed(row["sd"], digits))
        if key == "L":
            set_cell_text(cells[4], f"95th percentile: {row['q95']:.3f}")
            set_cell_text(cells[5], f"{row['Rhat']:.3f}")
            set_cell_text(cells[6], f"P(L > 0.95) = {split5['multi']['p_L_gt_095']:.3f}")
        else:
            set_cell_text(cells[4], f"({signed(row['q025'], digits)}, {signed(row['q975'], digits)})")
            set_cell_text(cells[5], f"{row['Rhat']:.3f}")
            set_cell_text(cells[6], ess(row["ESS"]))
    if seen != set(by_name):
        raise SystemExit(f"Table 8 missed {set(by_name) - seen}")


def fill_table9(body, by_model):
    aliases = {
        "ols": "OLS (no network)",
        "municipal": "Municipal fixed effects",
        "geographic": "Single SEM (geographic 6-NN)",
        "town": "Single SEM (town boundaries)",
        "no kriging": "Multi-SEM (no kriging)",
        "precision": "Multi-SEM (screened, precision kriging)",
    }
    caption = b.require_one(body, "Table 9. Mean holdout error")
    table = caption.getnext()
    while table is not None and table.tag != b.q(W, "tbl"):
        table = table.getnext()
    for tr in table.findall(b.q(W, "tr"))[1:]:
        cells = tr.findall(b.q(W, "tc"))
        label = cell_text(cells[0]).lower()
        key = next(name for token, name in aliases.items() if token in label)
        row = by_model[key]
        set_cell_text(cells[1], f"{row['mean_rmse']:.4f}")
        set_cell_text(cells[2], f"{row['mean_mae']:.4f}")
        set_cell_text(cells[3], f"{row['mean_corr']:.3f}")
        if row["wins_vs_ols"] is None:
            set_cell_text(cells[4], "Reference")
        else:
            set_cell_text(cells[4], f"{row['wins_vs_ols']} / 50")


def edit_main(body, facts):
    s5 = facts["split5"]["multi"]
    post = {row["parameter"]: row for row in s5["posterior"]}
    names = s5["names"]
    corr = s5["proposal_correlation"]
    geo_res = corr[names.index("geo")][names.index("resolvent")] if "geo" in names and "resolvent" in names else None
    conv = facts["summary"]["convergence"]
    p = sci(facts["p"])
    b.replace_containing(
        body,
        "The update is implemented using a random-walk Metropolis step",
        "To accommodate candidate layers that are nearly collinear, such as geographic distance and its "
        "resolvent, the Metropolis proposal adapts its covariance during burn-in (Haario, Saksman, and "
        "Tamminen, 2001; Roberts and Rosenthal, 2009). After the first 50 burn-in draws, which use an "
        "isotropic Gaussian step so that the chain can move before the covariance is estimated, candidates "
        "are drawn from $\\lambda^* \\sim N(\\lambda^{(t-1)}, \\tau_t^2 \\Sigma_t)$, with "
        "$\\Sigma_t = (2.38^2/k)\\,\\mathrm{Cov}(\\lambda^{(0:t-1)}) + 10^{-5} I_k$. The factor "
        "$(2.38)^2/k$ is the Gelman–Roberts–Gilks scaling (Roberts, Gelman, and Gilks, 1997), and $k$ is "
        "the number of active layers. The scalar $\\tau_t$ starts at 0.08 and is multiplied by 0.85 or 1.15 "
        "every 50 iterations when the acceptance rate in that window is outside $[0.22, 0.38]$. It is "
        "restricted to $[10^{-4}, 1]$. A proposal with "
        "$\\lVert\\lambda^*\\rVert_1 = \\sum_s |\\lambda_s^*| \\geq 1$ is rejected before the determinant "
        "is evaluated. The proposal is symmetric, so it is accepted with probability $\\min\\{1, \\exp(\\Delta \\log \\pi)\\}$, where",
    )
    b.replace_containing(
        body,
        "During burn-in,  is adapted every 50 iterations",
        "After burn-in, both $\\Sigma_t$ and $\\tau_t$ are fixed at the values from the last burn-in "
        "iteration. The transition kernel on the retained draws is therefore time-homogeneous (Roberts and "
        "Rosenthal, 2009). The simulation experiments and the S&P 100 fits were computed with the earlier "
        "isotropic proposal $\\tau^2 I_k$ and have not been rerun. The Boston posteriors in Section 8.1 use "
        "the covariance-adapted kernel, with 8,000 iterations and 4,000 burn-in draws on each of four chains.",
    )
    b.replace_containing(
        body,
        "Step 1. Propose",
        "Step 1. Propose $\\lambda^* \\sim N(\\lambda^{(t-1)}, \\tau_t^2 \\Sigma_t)$. If "
        "$\\lVert\\lambda^*\\rVert_1 = \\sum_{s=1}^{k} |\\lambda_s^*| \\geq 1$, reject the proposal "
        "immediately. $\\Sigma_t = I_k$ for the first 50 burn-in iterations and is the scaled empirical "
        "covariance after that, until burn-in ends. On the retained sample, $\\Sigma_t$ and $\\tau_t$ are fixed.",
    )
    replace_once(
        body,
        "estimated using Algorithm 2.",
        "estimated using Algorithm 2 with the isotropic random-walk proposal.",
    )
    b.replace_containing(
        body,
        "Table 8. Posterior parameter estimates for a representative central Boston split",
        "Table 8. Posterior parameter estimates for a representative central Boston split "
        "(Split 5, $n_t = 405$ training tracts), covariance-adapted MCMC. The radial layer was not "
        "selected. The interval is the central 95 percent of the retained draws.",
    )
    fill_table8(body, facts["split5"])
    b.replace_containing(
        body,
        "On Split 5 the 95th percentile",
        f"On Split 5 the 95th percentile of $L$ is {post['L']['q95']:.3f}, and "
        f"$P(L > 0.95 \\mid Y_t) = {s5['p_L_gt_095']:.3f}$. "
        f"The effective sample sizes of the geographic, resolvent, and town coefficients are "
        f"{post['geo']['ESS']:.0f}, {post['resolvent']['ESS']:.0f}, and {post['town']['ESS']:.0f}. "
        f"Every retained draw satisfies $L < 1$ (the largest draw is {s5['max_L']:.3f}).",
    )
    fill_table9(body, facts["by_model"])
    b.replace_containing(
        body,
        "We evaluate Multi-SEM using two substantive empirical benchmarks.",
        "We evaluate Multi-SEM using two substantive empirical benchmarks. "
        f"In a spatial hedonic study of $n = 506$ Boston census tracts across 92 municipalities, "
        f"Multi-SEM simultaneously captures geographic distance decay, regional multi-hop spillovers, "
        f"and municipal boundaries. Across 50 repeated random holdout splits, the covariance-adapted "
        f"Multi-SEM achieves an average holdout RMSE of {facts['multi']:.4f}, a {facts['pct_ols']:.1f}% "
        f"reduction relative to OLS ({facts['ols']:.4f}; Wilcoxon $p = {p}$) and a {facts['pct_inc']:.1f}% "
        f"reduction relative to the single-layer geographic SEM ({facts['geo']:.4f}), and it outperforms "
        f"OLS on {facts['wins']} of 50 splits. "
        "A 63-month longitudinal panel of S&P 100 equity returns confirms that the pipeline scales "
        "to dynamic, non-geographic networks, outperforming OLS in 44 of 63 months "
        "(sign test $p = 0.0022$; Newey–West $p = 0.072$). Those equity fits used the isotropic proposal.",
    )
    b.replace_containing(
        body,
        "Pace and Gilley (1997) exploit the spatial configuration",
        "Pace and Gilley (1997) exploit the spatial configuration of these tracts to improve hedonic fit. "
        "Table 9 separates this continuous distance channel from the municipal-boundary channel. "
        f"Relative to the OLS mean RMSE of {facts['ols']:.4f}, the geographic six-nearest-neighbour spatial "
        f"error model reduces mean RMSE to {facts['geo']:.4f}, a {facts['pct_geo']:.1f}% improvement. "
        f"Screened Multi-SEM, estimated with the covariance-adapted proposal and retaining the geographic "
        f"resolvent and the town layer on all 50 splits, reduces mean RMSE further to {facts['multi']:.4f}. "
        f"That is a {facts['pct_inc']:.1f}% reduction relative to the geographic SEM and a {facts['pct_ols']:.1f}% "
        f"reduction relative to OLS (Wilcoxon $p = {p}$). Multi-SEM outperforms OLS on {facts['wins']} of 50 splits. "
        f"The town SEM alone has a mean RMSE of {facts['town']:.4f}, so the reduction relative to OLS is not "
        "the marginal contribution of the town layer.",
    )
    ridge = ""
    if geo_res is not None:
        ridge = (
            f" On Split 5 the frozen proposal correlation between the geographic coefficient and the "
            f"resolvent is {geo_res:.2f}, so the adapted steps run along that ridge rather than perpendicular to it."
        )
    b.replace_containing(
        body,
        "MCMC convergence diagnostics reveal a structural distinction",
        "Convergence diagnostics for the covariance-adapted sampler are computed from the original "
        f"8,000-iteration runs. The single-layer geographic fits have maximum split-$\\hat{{R}}$ "
        f"{conv['geo']['max_rhat']:.3f} and minimum $\\lambda$ effective sample size "
        f"{conv['geo']['min_ess_lambda']:.0f}. The town fits have maximum split-$\\hat{{R}}$ "
        f"{conv['town_sem']['max_rhat']:.3f} and minimum $\\lambda$ effective sample size "
        f"{conv['town_sem']['min_ess_lambda']:.0f}. "
        f"Across the 50 screened fits, the largest split-$\\hat{{R}}$ on a network coefficient is "
        f"{facts['max_rhat_lambda']:.3f}, and the smallest $\\lambda$ effective sample size is "
        f"{facts['min_ess_lambda']:.0f}. Every network coefficient has split-$\\hat{{R}} \\le 1.025$ "
        f"on {facts['rhat_ok']} of 50 splits, and every $\\lambda$ effective sample size exceeds 400 on "
        f"{facts['ess_ok']} of 50 splits. The smallest slope effective sample size on a screened fit is "
        f"{facts['min_ess_beta']:.0f}. "
        f"The mean of $E[L \\mid Y_t]$ is {conv['multi']['mean_E_L']:.3f}, and the mean of "
        f"$P(L > 0.95 \\mid Y_t)$ is {conv['multi']['mean_P_L_gt_095']:.3f}. "
        f"On Split 5, $E[L \\mid Y_t] = {post['L']['mean']:.3f}$ and "
        f"$P(L > 0.95 \\mid Y_t) = {s5['p_L_gt_095']:.3f}$. "
        f"The largest retained value of $L$ across all 50 screened fits is {facts['max_L']:.3f}, so every "
        f"retained draw stays inside the open $\\ell_1$ ball."
        + ridge
        + " Table 9 uses these posterior means. The draws were not replaced by a longer chain.",
    )
    ols = facts["split5"]["ols"]
    multi_m = facts["split5"]["multi"]
    b.replace_containing(
        body,
        "Figure 3. Predicted versus actual",
        "Figure 3. Predicted versus actual $\\log(\\mathrm{cmedv})$ on the $n_* = 101$ holdout census tracts "
        f"for Split 5 of the Boston housing study. Left: OLS regression "
        f"($\\mathrm{{RMSE}} = {ols['RMSE']:.4f}$, $\\mathrm{{MAE}} = {ols['MAE']:.4f}$, "
        f"$r = {ols['corr']:.3f}$). Right: Screened Multi-SEM precision-form spatial kriging "
        f"($\\mathrm{{RMSE}} = {multi_m['RMSE']:.4f}$, $\\mathrm{{MAE}} = {multi_m['MAE']:.4f}$, "
        f"$r = {multi_m['corr']:.3f}$). The solid line is the 45-degree identity line ($y = \\hat{{y}}$). "
        "Split 5 is the lower of the two central Multi-SEM holdout RMSEs across the 50 random 80/20 partitions.",
    )
    b.insert_before_snippet(
        body,
        "Harrison, D., and Rubinfeld, D. L. (1978).",
        [
            "Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. "
            "Bernoulli, 7(2), 223–242.",
        ],
    )


def edit_supplement(body):
    b.replace_containing(
        body,
        "The proposal covariance is the identity scaled by",
        "At MCMC iteration $t$, given the current value $\\lambda^{(t-1)}$, propose "
        "$\\lambda^* \\sim N(\\lambda^{(t-1)}, \\tau_t^2 \\Sigma_t)$. For the first 50 burn-in draws, "
        "$\\Sigma_t = I_k$. While burn-in continues after that point, "
        "$\\Sigma_t = (2.38^2/k)\\,\\mathrm{Cov}(\\lambda^{(0:t-1)}) + 10^{-5} I_k$ "
        "(Haario, Saksman, and Tamminen, 2001; Roberts, Gelman, and Gilks, 1997). "
        "The Gaussian proposal does not depend on which of $\\lambda^{(t-1)}$ and $\\lambda^*$ is the "
        "current state, so the proposal densities cancel from the Metropolis–Hastings ratio. "
        "The Boston housing chains use this covariance. The simulation and S&P chains kept $\\Sigma_t = I_k$.",
    )
    b.replace_containing(
        body,
        "Every proposal uses the isotropic covariance",
        "Set $\\tau_0 = 0.08$. For the covariance-adapted kernel used in the Boston application, "
        "$\\Sigma_t = I_k$ during the first 50 burn-in iterations and "
        "$\\Sigma_t = (2.38^2/k)\\,\\mathrm{Cov}(\\lambda^{(0:t-1)}) + 10^{-5} I_k$ after that, "
        "until burn-in ends. The isotropic kernel used for the simulation and S&P fits keeps $\\Sigma_t = I_k$.",
    )
    b.replace_containing(
        body,
        "During burn-in, and only when the iteration count is a multiple of 50",
        "During burn-in, and only when 50 proposals have accumulated in the current window, update "
        "$\\tau_t$ from that window's acceptance rate, using the same 0.85 and 1.15 factors and the "
        "target 0.30. Restrict $\\tau_t$ to $[10^{-4}, 1]$. At the end of burn-in, freeze both "
        "$\\Sigma_t$ and $\\tau_t$.",
    )
    b.replace_containing(
        body,
        "Once burn-in ends, the adaptation increment is identically zero",
        "Once burn-in ends, $\\Sigma_t$ and $\\tau_t$ stay fixed. The kernel on the retained draws is "
        "time-homogeneous (Roberts and Rosenthal, 2009). The function that implements this choice is "
        "gibbs_multi_sem in mnreg/model.py. The Boston run sets adapt_covariance to true. The default, "
        "used for the stored simulation and S&P draws, leaves that switch off and proposes with $\\tau^2 I_k$.",
    )
    node = b.require_one(body, "plot_boston_split5.py")
    node.addnext(b.make_p(b.blocks_from(
        "The covariance-adapted Boston posteriors were produced by python run_boston.py --adaptive. "
        "That option calls gibbs_multi_sem in mnreg/model.py with adapt_covariance set, using 8,000 "
        "iterations, 4,000 burn-in draws, and four chains, and it writes results/boston_adaptive. "
        "The earlier isotropic fits remain in results/boston. The adaptation follows Haario, Saksman, "
        "and Tamminen (2001), and the learned covariance is frozen after burn-in (Roberts and Rosenthal, 2009)."
    ), node))
    b.insert_before_snippet(
        body,
        "Rue, H., & Held, L. (2005).",
        [
            "Haario, H., Saksman, E., & Tamminen, J. (2001). An adaptive Metropolis algorithm. "
            "Bernoulli, 7(2), 223–242.",
            "Roberts, G. O., Gelman, A., & Gilks, W. R. (1997). Weak convergence and optimal scaling of "
            "random walk Metropolis algorithms. The Annals of Applied Probability, 7(1), 110–120.",
            "Roberts, G. O., & Rosenthal, J. S. (2009). Examples of adaptive MCMC. Journal of "
            "Computational and Graphical Statistics, 18(2), 349–367.",
        ],
    )


def refresh_figure(files, root):
    png = ADAPTIVE / "boston_split5_pred_vs_actual.png"
    if not png.exists():
        raise SystemExit(f"missing {png}")
    body = root.find(b.q(W, "body"))
    caption = b.require_one(body, "Figure 3. Predicted versus actual")
    drawing = caption.getprevious()
    office = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    rid = None
    for blip in drawing.iter():
        if blip.tag.endswith("blip"):
            rid = blip.get(f"{office}embed")
    if not rid:
        raise SystemExit("Figure 3 drawing has no image")
    rels = etree.fromstring(files["word/_rels/document.xml.rels"])
    target = next(rel.get("Target") for rel in rels if rel.get("Id") == rid)
    files["word/" + target.replace("\\", "/").lstrip("/")] = png.read_bytes()


def main():
    facts = load_results()
    files, root = b.load_zip(MAIN_IN)
    edit_main(root.find(b.q(W, "body")), facts)
    refresh_figure(files, root)
    print("main words", b.word_count_root(root, "References"))
    b.store_zip(MAIN_OUT, files, root)
    files, root = b.load_zip(SUP_IN)
    edit_supplement(root.find(b.q(W, "body")))
    print("supp words", b.word_count_root(root, "References"))
    b.store_zip(SUP_OUT, files, root)
    print(facts["pct_ols"], facts["pct_inc"], facts["min_ess_lambda"], facts["max_rhat_lambda"], facts["max_L"])
    print(MAIN_OUT.name)
    print(SUP_OUT.name)


if __name__ == "__main__":
    main()
