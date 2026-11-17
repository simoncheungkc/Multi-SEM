"""Point-by-point letter for the results that actually exist in v26 / supplement v24."""

from pathlib import Path

from build_revision_docs import replace_paragraph
from build_v10_docx import q
from build_v12_docx import W, load_zip, store_zip

ROOT = Path(__file__).resolve().parent


def main():
    files, root = load_zip(ROOT / "Response_to_Reviewers_2.docx")
    body = root.find(q(W, "body"))
    replace_paragraph(
        body,
        "The reported RMSE comparison does not calibrate predictive intervals.",
        "Section 7.5 states that the induced-subgraph criterion is a pseudo-posterior: "
        "re-row-standardizing the training block changes the operator scale. When the full "
        "topology is observed, the Schur precision is the coherent training likelihood. The "
        "induced fit is retained as a fast surrogate when the target is point imputation. "
        "Table S13 in Supplement S4.8, and the cross-reference after Table 9 in manuscript v27, "
        "now report Split 5 posterior predictive RMSE and MAE, empirical 90% and 95% interval "
        "coverage, mean 95% interval width, and mean CRPS from 2,000 thinned draws and 20 "
        "conditional Gaussian replicates (40,000 draws). Schur Multi-SEM has RMSE 0.137, "
        "95% coverage 0.970, mean width 0.510, and mean CRPS 0.069. Induced Multi-SEM has "
        "RMSE 0.135 and CRPS 0.070. Bayesian OLS has RMSE 0.191 and CRPS 0.105. The geographic "
        "row is the induced-training geographic SEM with full-graph conditional replicates "
        "(RMSE 0.142, CRPS 0.074); it is not a Schur-likelihood geographic fit.",
    )
    replace_paragraph(
        body,
        "The RMSE comparison is not a calibration of predictive intervals.",
        "Section 7.5 acknowledges the change of scale and discusses data augmentation. The "
        "induced subgraph is retained because the target in the reported point-RMSE tables is "
        "point imputation. Split 5 interval calibration is reported separately in Table S13 "
        "(Supplement S4.8): Schur Multi-SEM 95% coverage is 0.970 and mean CRPS is 0.069, "
        "against 0.941 and 0.105 for Bayesian OLS.",
    )
    replace_paragraph(
        body,
        "No design outside the l1 ball was run.",
        "Table 4b reports 200 replications at n = 200 with lambda = (0.50, -0.30, 0, 0, 0), signed sum 0.20 and l1 norm 0.80. Both active signs are recovered. The Poisson degree-8 coefficient has bias -0.106, RMSE 0.142, and 91.0% coverage. The power-law coefficient has bias +0.066, RMSE 0.100, and 91.5% coverage. The three inactive coefficients stay near zero, with coverage 97.0% to 97.5%. The largest R-hat in the ensemble is 1.042. The coefficients are shrunk toward zero; the table does not show unbiased recovery. Table 4c in Section 7.3.2 is the design outside the l1 ball: lambda = (0.75, -0.55), l1 norm 1.30, mean spectral radius 0.693. The l1-constrained posterior means are 0.507 and -0.442, with coverage 0.100 and 0.685. The spectral-admissible estimator recovers 0.725 and -0.535, with coverage 0.960 and 0.955. Its maximum split-R is 1.011. The l1-constrained run reaches 1.027. Figure S11 shows replication 1.",
    )
    replace_paragraph(
        body,
        "Leave-one-town-out cross-validation was not added.",
        "Table S10 reruns the 50 splits with the Charles River dummy and, separately, drops the 16 ceiling tracts (n = 490, training size 392). With the dummy, mean RMSE is 0.1904 for OLS, 0.1410 for the geographic SEM, and 0.1378 for Multi-SEM: 27.6% below OLS (t_49 = 7.07, p = 5.11e-9, 50 of 50) and 2.3% below the geographic SEM (t_49 = 1.29, p = 0.204, 38 of 50). Without the ceiling tracts, the means are 0.1806, 0.1297, and 0.1289: 28.7% below OLS (t_49 = 6.82, p = 1.26e-8, 50 of 50) and 0.6% below the geographic SEM (t_49 = 0.49, p = 0.623, 35 of 50). The geographic increment does not remain under the censoring check. Section 8.1 gives the no-kriging explanation. Table 9 is within-region interpolation on random 80/20 splits. Universal kriging (Matern 3/2, REML) is appended there: mean RMSE 0.1473, mean MAE 0.1091, correlation 0.931, and a lower RMSE than OLS on 50 of 50 splits. Table S14 is out-of-domain extrapolation, with one prediction per tract. Leave-one-town-out RMSE is 0.2075 for OLS, 0.2414 for municipal fixed effects, 0.1969 for universal kriging, 0.2146 for the town SEM, 0.1859 for the geographic SEM, and 0.1834 for screened Multi-SEM. The corresponding MAE values are 0.1506, 0.1714, 0.1399, 0.1503, 0.1269, and 0.1255. On the ten contiguous blocks, RMSE is 0.2161, 0.2245, 0.2171, 0.2038, 0.2115, and 0.2165 in that same order. The town SEM is the lowest-error block specification. An unseen municipality has no town-level edge and uses the reference intercept, which is why fixed effects and the town SEM lose their interpolation advantage under leave-one-town-out. Section 5.3 and Section 8.1 state that Tables 8 and 9 use the covariance-adapted kernel.",
    )
    replace_paragraph(
        body,
        "Section 5.3 still states that the simulation and S&P fits used the isotropic proposal",
        "Section 5.3 of manuscript v27 states that the reported simulation and empirical fits use the "
        "covariance-adapted Metropolis-within-Gibbs sampler (adapt_covariance=True). The Table 4 "
        "run, four chains of 6,000 iterations with 3,000 burn-in and seed 2026, has maximum "
        "split-R 1.012. Table 4a was lengthened to four chains of 4,000 iterations with 2,000 "
        "burn-in. Its worst split-R is 1.102, so the uniform 1.025 bound is still missed by the "
        "worst replication. Active-coefficient 95% coverage is 89.5% (MCSE 0.022) and 92.5% "
        "(MCSE 0.019). Table 5 was recomputed with the same adapted sampler: oracle RMSE 0.998 and "
        "screened RMSE 1.033, against OLS 1.089. The 63-month S&P panel was recomputed with four "
        "adapted chains of 8,000 iterations; its largest split-R is 1.013. The August posterior means "
        "are 0.206 for sector and 0.186 for industry. Table S12 shows why adaptation matters when "
        "the layer correlation is 0.979: minimum lambda ESS rises from 100 to 443 and the worst "
        "split-R falls from 1.187 to 1.040. That adapted collinear run still exceeds 1.025. "
        "run_sp100.py sets USE_STATIC_CACHE = True "
        "and reads data/sp100_clean_snapshot.csv: 100 equities and 2,187 adjusted closes. That "
        "path does not call Yahoo Finance. The public address https://github.com/simoncheungkc/Multi-SEM "
        "did not resolve, and no live Zenodo DOI has been minted. The numerical results are "
        "reproduced from the project files. Table S11 reports Moran's I on the filtered "
        "innovations. On the geographic layer the mean OLS statistic is 0.442 and the mean "
        "filtered statistic is -0.021. On Split 5 the filtered statistic is -0.012 (p = 0.755). "
        "The share of splits with p > 0.30 is 0.80, 0.94, 0.92, and 0.96 on the four candidate "
        "matrices. The filtered geographic statistics range from -0.077 to 0.007.",
    )
    out = ROOT / "Response_to_Reviewers_3.docx"
    store_zip(out, files, root)
    print("wrote", out.name)


if __name__ == "__main__":
    main()
