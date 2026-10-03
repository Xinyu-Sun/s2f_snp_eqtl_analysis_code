"""A9. Does accessible-chromatin context account for the larger S2F scores of AA high-PIP variants?
Outputs: a9_accessible_summary.tsv (share accessible; median |score| inside/outside),
         a9_accessible_regression.tsv (group contrast in log10|score| adjusted for accessibility, and for accessibility + LD proxies),
         a9_accessible_auroc.tsv (distance-matched AUROC for positives inside vs outside accessible chromatin)."""
import sys, numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import fisher_exact
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, TSS_BINS, ci
pairs_path, filer_path, ld_path, out_dir = sys.argv[1:5]
df = load_pairs(pairs_path)
fl = pd.read_csv(filer_path, sep="\t", usecols=["chromosome", "position", "ref_allele", "alt_allele", "filer_binary_Accessible_chromatin"]).rename(columns={"filer_binary_Accessible_chromatin": "acc"})
df = df.merge(fl.drop_duplicates(["chromosome", "position", "ref_allele", "alt_allele"]), on=["chromosome", "position", "ref_allele", "alt_allele"], how="left")
df["plink_id"] = df["chromosome"].astype(str) + ":" + df["position"].astype(str) + "_" + df["ref_allele"] + "_" + df["alt_allele"]
ld = pd.read_csv(ld_path, sep="\t", usecols=["ancestry", "plink_id", "n80_200kb"]).drop_duplicates(["ancestry", "plink_id"])
df = df.merge(ld, on=["ancestry", "plink_id"], how="left")
summ, reg, au = [], [], []
for t in (0.9, 0.5):
    for model, col in MODELS.items():
        P = {g: split_classes(df[df["ancestry"] == g], col, t)[0] for g in GROUPS}
        L = {g: split_classes(df[df["ancestry"] == g], col, t)[1] for g in GROUPS}
        for g in GROUPS:
            p = P[g]; a = p[p["acc"] == 1]; o = p[p["acc"] == 0]
            summ.append(dict(pip_threshold=t, model=model, group=g, n_pos=len(p), n_with_call=int(p["acc"].notna().sum()), share_accessible=(p["acc"] == 1).mean(),
                             median_abs_score_accessible=np.abs(a[col]).median(), median_abs_score_not_accessible=np.abs(o[col]).median(), n_accessible=len(a), n_not_accessible=len(o),
                             low_pip_with_call=int(L[g]["acc"].notna().sum()), low_pip_share_accessible_among_called=(L[g]["acc"] == 1).sum() / max(1, L[g]["acc"].notna().sum())))
            for lab, sub in (("accessible", a), ("not_accessible", o)):
                if len(sub) < 5: au.append(dict(pip_threshold=t, model=model, group=g, stratum=lab, n_pos=len(sub), note="too few")); continue
                rng = np.random.default_rng(21); e = distance_bins(sub, L[g]); x = [matched_auroc(sub, L[g], col, rng, e)[0] for _ in range(300)]
                au.append(dict(pip_threshold=t, model=model, group=g, stratum=lab, n_pos=len(sub), auroc_mean=np.nanmean(x), ci_low=ci(x)[0], ci_high=ci(x)[1]))
        allp = pd.concat([P[g].assign(group=g) for g in GROUPS]).dropna(subset=["acc"]).copy()
        allp["log_abs_score"] = np.log10(np.abs(allp[col]) + 1e-6); allp["tss_bin"] = pd.Categorical(allp["tss_distance_bin"], categories=TSS_BINS)
        allp["log1p_n80"] = np.log1p(allp["n80_200kb"])  # NaN for positives without an LD count (not in the a5 output)
        for a_, b_ in (("AA", "NHW"), ("AA", "CH")):
            sub = allp[allp.group.isin([a_, b_])].copy(); sub["is_a"] = (sub.group == a_).astype(int); cl = {"groups": pd.factorize(sub["gene_id"])[0]}
            f0 = smf.ols("log_abs_score ~ is_a + C(tss_bin) + log_dist", data=sub).fit(cov_type="cluster", cov_kwds=cl)
            f1 = smf.ols("log_abs_score ~ is_a + acc + C(tss_bin) + log_dist", data=sub).fit(cov_type="cluster", cov_kwds=cl)
            # LD-adjusted model only on positives with an LD count; its unadjusted reference is refitted on the same rows
            sub_ld = sub.dropna(subset=["log1p_n80"]).copy(); cl_ld = {"groups": pd.factorize(sub_ld["gene_id"])[0]}
            f0_ld = smf.ols("log_abs_score ~ is_a + C(tss_bin) + log_dist", data=sub_ld).fit(cov_type="cluster", cov_kwds=cl_ld)
            f2 = smf.ols("log_abs_score ~ is_a + acc + log1p_n80 + C(tss_bin) + log_dist", data=sub_ld).fit(cov_type="cluster", cov_kwds=cl_ld)
            tab = pd.crosstab(sub["is_a"], sub["acc"]); odds, pf = fisher_exact(tab.values) if tab.shape == (2, 2) else (np.nan, np.nan)
            def fold(f): c = f.params["is_a"]; s = f.bse["is_a"]; return 10 ** c, 10 ** (c - 1.96 * s), 10 ** (c + 1.96 * s), f.pvalues["is_a"]
            r = dict(pip_threshold=t, model=model, contrast=f"{a_}_vs_{b_}", n=len(sub), n_adj_acc_ld=len(sub_ld), share_acc_a=sub.loc[sub.is_a == 1, "acc"].mean(), share_acc_b=sub.loc[sub.is_a == 0, "acc"].mean(), fisher_p_accessible=pf)
            for nm, f in (("unadj", f0), ("adj_acc", f1), ("adj_acc_ld", f2)):
                fo, lo, hi, p = fold(f); r.update({f"fold_{nm}": fo, f"fold_{nm}_lo": lo, f"fold_{nm}_hi": hi, f"p_{nm}": p})
            r["acc_fold"] = 10 ** f1.params["acc"]; r["acc_p"] = f1.pvalues["acc"]
            r["share_explained_acc"] = 1 - f1.params["is_a"] / f0.params["is_a"]; r["share_explained_acc_ld"] = 1 - f2.params["is_a"] / f0_ld.params["is_a"]
            reg.append(r)
pd.DataFrame(summ).to_csv(f"{out_dir}/a9_accessible_summary.tsv", sep="\t", index=False)
pd.DataFrame(reg).to_csv(f"{out_dir}/a9_accessible_regression.tsv", sep="\t", index=False)
pd.DataFrame(au).to_csv(f"{out_dir}/a9_accessible_auroc.tsv", sep="\t", index=False); print("A9 done")
