"""A1. Is the AA AUROC advantage carried by the positives or by the comparison pool?

Outputs (results/):
  a1_score_by_class.tsv       median |score| (gene-bootstrap 95% CI) per model x group x class x threshold
  a1_pos_vs_low_ratio.tsv     ratio of medians positives/low-PIP with CI
  a1_group_contrasts.tsv      AA vs NHW, AA vs CH, CH vs NHW among positives: Mann-Whitney and
                              gene-clustered OLS of log10|score| on group + TSS-bin + log distance
  a1_positive_characteristics.tsv  |z|, PIP, distance, promoter share, MAF (where available) per group
"""
import sys, numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
import statsmodels.formula.api as smf
from fu_common import load_pairs, split_classes, GROUPS, MODELS, TSS_BINS, ci

pairs_path, out_dir = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(42)
B = 1000
df = load_pairs(pairs_path)

def gene_boot_median(d, col, B=B):
    if len(d) == 0:
        return np.nan, (np.nan, np.nan)
    genes = d["gene_id"].unique(); gi = d.groupby("gene_id").indices
    vals = np.abs(d[col].values); out = []
    for _ in range(B):
        pick = rng.choice(genes, size=len(genes), replace=True)
        idx = np.concatenate([gi[g] for g in pick])
        out.append(np.median(vals[idx]))
    return np.median(vals), ci(out)

rows, ratios, contrasts, chars = [], [], [], []
for model, col in MODELS.items():
    for t in (0.9, 0.5):
        for g in GROUPS:
            d = df[df["ancestry"] == g]
            pos, low, mid = split_classes(d, col, t)
            med = {}
            for cls, sub in (("positive", pos), ("low_pip", low), ("intermediate_pip", mid)):
                m, (lo, hi) = gene_boot_median(sub, col)
                med[cls] = m
                rows.append(dict(model=model, pip_threshold=t, group=g, variant_class=cls, n=len(sub),
                                 n_genes=sub["gene_id"].nunique(), median_abs_score=m, ci_low=lo, ci_high=hi,
                                 q25=np.percentile(np.abs(sub[col]), 25) if len(sub) else np.nan, q75=np.percentile(np.abs(sub[col]), 75) if len(sub) else np.nan))
            # ratio of medians with joint gene bootstrap
            if len(pos) == 0 or len(low) == 0:
                continue
            gp, gl = pos.groupby("gene_id").indices, low.groupby("gene_id").indices
            vp, vl = np.abs(pos[col].values), np.abs(low[col].values)
            rs = []
            for _ in range(B):
                ip = np.concatenate([gp[x] for x in rng.choice(list(gp), size=len(gp), replace=True)])
                il = np.concatenate([gl[x] for x in rng.choice(list(gl), size=len(gl), replace=True)])
                rs.append(np.median(vp[ip]) / np.median(vl[il]))
            lo, hi = ci(rs)
            ratios.append(dict(model=model, pip_threshold=t, group=g, ratio_pos_over_low=med["positive"] / med["low_pip"],
                               ci_low=lo, ci_high=hi))
        # group contrasts among positives
        pos_all = pd.concat([split_classes(df[df["ancestry"] == g], col, t)[0].assign(group=g) for g in GROUPS])
        pos_all["log_abs_score"] = np.log10(np.abs(pos_all[col]) + 1e-6)
        pos_all["tss_bin"] = pd.Categorical(pos_all["tss_distance_bin"], categories=TSS_BINS)
        for a, b in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
            xa, xb = np.abs(pos_all.loc[pos_all.group == a, col]), np.abs(pos_all.loc[pos_all.group == b, col])
            if len(xa) < 3 or len(xb) < 3:
                continue
            u, p_mw = mannwhitneyu(xa, xb, alternative="two-sided")
            sub = pos_all[pos_all.group.isin([a, b])].copy(); sub["is_a"] = (sub.group == a).astype(int)
            fit = smf.ols("log_abs_score ~ is_a + C(tss_bin) + log_dist", data=sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["gene_id"])[0]})
            coef = fit.params["is_a"]; se = fit.bse["is_a"]
            contrasts.append(dict(model=model, pip_threshold=t, contrast=f"{a}_vs_{b}", n_a=len(xa), n_b=len(xb),
                                  median_a=np.median(xa), median_b=np.median(xb), mannwhitney_p=p_mw,
                                  ols_log10_ratio=coef, ols_ci_low=coef - 1.96 * se, ols_ci_high=coef + 1.96 * se,
                                  ols_fold=10 ** coef, ols_p=fit.pvalues["is_a"]))
        # characteristics of positives
        for g in GROUPS:
            p = pos_all[pos_all.group == g]
            rho = spearmanr(np.abs(p[col]), p["abs_z"]).correlation if len(p) > 5 else np.nan
            chars.append(dict(model=model, pip_threshold=t, group=g, n=len(p), median_abs_z=p["abs_z"].median(),
                              share_abs_z_gt8=(p["abs_z"] > 8).mean(), median_pip=p["pip"].median(),
                              median_abs_dist=p["distance_to_tss"].abs().median(), share_0_3kb=(p["tss_distance_bin"] == "0-3kb").mean(),
                              median_maf=p["maf"].median(), spearman_abs_score_vs_abs_z=rho))
pd.DataFrame(rows).to_csv(f"{out_dir}/a1_score_by_class.tsv", sep="\t", index=False)
pd.DataFrame(ratios).to_csv(f"{out_dir}/a1_pos_vs_low_ratio.tsv", sep="\t", index=False)
pd.DataFrame(contrasts).to_csv(f"{out_dir}/a1_group_contrasts.tsv", sep="\t", index=False)
pd.DataFrame(chars).to_csv(f"{out_dir}/a1_positive_characteristics.tsv", sep="\t", index=False)
print("A1 done")
