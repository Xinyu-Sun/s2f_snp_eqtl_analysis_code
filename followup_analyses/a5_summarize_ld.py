"""A5. LD-partner counts per positive (within-group PLINK r2) and their relation to score magnitude and the group gap."""
import sys, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
import statsmodels.formula.api as smf
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, TSS_BINS, ci
pairs_path, ld_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
freq_dir = sys.argv[4] if len(sys.argv) > 4 else None
df = load_pairs(pairs_path)
df["plink_id"] = df["chromosome"].astype(str) + ":" + df["position"].astype(str) + "_" + df["ref_allele"] + "_" + df["alt_allele"]
recs = []
ld_run = {}  # variants PLINK computed LD for (each has a self row SNP_A == SNP_B)
for g in GROUPS:
    parts = []
    for f in glob.glob(f"{ld_dir}/{g}.chr*.ld"):
        try: parts.append(pd.read_csv(f, sep=r"\s+"))
        except Exception: pass
    if not parts: continue
    ld = pd.concat(parts); ld_run[g] = set(ld["SNP_A"]); ld = ld[ld["SNP_A"] != ld["SNP_B"]]
    ld["dist"] = (ld["BP_A"] - ld["BP_B"]).abs()
    for w, lab in ((200000, "200kb"), (1000000, "1mb")):
        sub = ld[ld["dist"] <= w]
        agg = sub.groupby("SNP_A").agg(n80=("R2", lambda r: (r >= 0.8).sum()), n50=("R2", lambda r: (r >= 0.5).sum()), n20=("R2", lambda r: (r >= 0.2).sum()), max_r2=("R2", "max"))
        agg.columns = [f"{c}_{lab}" for c in agg.columns]; agg = agg.reset_index().rename(columns={"SNP_A": "plink_id"})
        recs.append((g, lab, agg))
out = {}
for g, lab, agg in recs:
    out.setdefault(g, None); out[g] = agg if out[g] is None else out[g].merge(agg, on="plink_id", how="outer")
ldtab = pd.concat([v.assign(ancestry=g) for g, v in out.items()]) if out else pd.DataFrame()
# positives scored by either model (the earlier version kept Borzoi-scored positives only, which made the
# AlphaGenome regressions run on the Borzoi subset)
_d = df.dropna(subset=["log_dist", "pip"])
_d = _d[(_d["pip"] >= 0.5) & (_d["borzoi_score"].notna() | _d["alphagenome_score"].notna())]
pos_all = pd.concat([_d[_d["ancestry"] == g].assign(group=g) for g in GROUPS])
pos_all = pos_all.merge(ldtab, on=["ancestry", "plink_id"], how="left")
for c in [c for c in pos_all.columns if c.startswith(("n80", "n50", "n20"))]:
    pos_all[c] = pos_all[c].fillna(0)  # positive present in LD list but no partner above r2 0.1 -> zero partners
pos_all["in_ld_output"] = pos_all["plink_id"].isin(ldtab["plink_id"]) if len(ldtab) else False
if freq_dir:
    inbim = {}
    for g in GROUPS:
        ids = set()
        for f in glob.glob(f"{freq_dir}/{g}.chr*.frq"):
            ids |= set(pd.read_csv(f, sep=r"\s+")["SNP"])
        inbim[g] = ids
    pos_all["in_bim"] = [pid in inbim.get(g, set()) for g, pid in zip(pos_all["ancestry"], pos_all["plink_id"])]
    pos_all = pos_all[pos_all["in_bim"]].copy()  # partner counts are only defined for variants present in the genotype file
# positives without an LD computation (not in the LD list or not in the genotype file) have no count; leave them out
# rather than counting them as having zero partners
pos_all["in_ld_run"] = [pid in ld_run.get(g, set()) for g, pid in zip(pos_all["ancestry"], pos_all["plink_id"])]
pos_all = pos_all[pos_all["in_ld_run"]].copy()
pos_all.to_csv(f"{out_dir}/a5_positive_ld_partners.tsv", sep="\t", index=False)
summ, strat = [], []
for t in (0.9, 0.5):
    p = pos_all[pos_all["pip"] >= t]
    for g in GROUPS:
        d = p[p["group"] == g]
        summ.append(dict(pip_threshold=t, group=g, n_in_bim=len(d), n_borzoi=int(d["borzoi_score"].notna().sum()), n_alphagenome=int(d["alphagenome_score"].notna().sum()), n_with_any_partner_r2_0p1=int(d["in_ld_output"].sum()),
                         median_n80_200kb=d["n80_200kb"].median(), mean_n80_200kb=d["n80_200kb"].mean(), share_zero_partners_r2_0p8_200kb=(d["n80_200kb"] == 0).mean(),
                         median_n50_200kb=d["n50_200kb"].median(), median_n20_200kb=d["n20_200kb"].median(), median_max_r2_200kb=d["max_r2_200kb"].median(),
                         **{f"spearman_abs_{m}_vs_n80": spearmanr(np.abs(d[c]), d["n80_200kb"], nan_policy="omit").correlation for m, c in MODELS.items()}))
    for model, col in MODELS.items():
        dd = p.dropna(subset=[col]).copy(); dd["log_abs_score"] = np.log10(np.abs(dd[col]) + 1e-6); dd["log1p_n80"] = np.log1p(dd["n80_200kb"])
        dd["tss_bin"] = pd.Categorical(dd["tss_distance_bin"], categories=TSS_BINS)
        for a, b in (("AA", "NHW"), ("AA", "CH")):
            sub = dd[dd.group.isin([a, b])].copy(); sub["is_a"] = (sub.group == a).astype(int)
            f0 = smf.ols("log_abs_score ~ is_a + C(tss_bin) + log_dist", data=sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["gene_id"])[0]})
            f1 = smf.ols("log_abs_score ~ is_a + log1p_n80 + C(tss_bin) + log_dist", data=sub).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(sub["gene_id"])[0]})
            strat.append(dict(pip_threshold=t, model=model, contrast=f"{a}_vs_{b}", n=len(sub), group_coef_unadjusted=f0.params["is_a"], p_unadjusted=f0.pvalues["is_a"],
                              group_coef_adj_n80=f1.params["is_a"], p_adj_n80=f1.pvalues["is_a"], n80_coef=f1.params["log1p_n80"], n80_p=f1.pvalues["log1p_n80"],
                              share_group_coef_explained=1 - f1.params["is_a"] / f0.params["is_a"] if f0.params["is_a"] else np.nan))
        # AUROC by partner stratum, each group vs its own low-PIP pool
        for g in GROUPS:
            pos, low, _ = split_classes(df[df["ancestry"] == g], col, t)
            pos = pos.merge(pos_all[["ancestry", "plink_id", "n80_200kb"]].drop_duplicates(), on=["ancestry", "plink_id"], how="left")
            # positives without an LD count (not in the genotype file / LD output) are left out of both strata
            for lab, m in (("zero_partners_r2_0.8", pos["n80_200kb"] == 0), ("one_or_more_partners_r2_0.8", pos["n80_200kb"] >= 1)):
                sub = pos[m]
                if len(sub) < 5: strat.append(dict(pip_threshold=t, model=model, group=g, stratum=lab, n_pos=len(sub), note="too few")); continue
                rng = np.random.default_rng(9); e = distance_bins(sub, low); a = [matched_auroc(sub, low, col, rng, e)[0] for _ in range(300)]
                strat.append(dict(pip_threshold=t, model=model, group=g, stratum=lab, n_pos=len(sub), auroc_mean=np.nanmean(a), ci_low=ci(a)[0], ci_high=ci(a)[1]))
pd.DataFrame(summ).to_csv(f"{out_dir}/a5_ld_partner_summary.tsv", sep="\t", index=False)
pd.DataFrame(strat).to_csv(f"{out_dir}/a5_ld_partner_models_and_strata.tsv", sep="\t", index=False); print("A5 done")
