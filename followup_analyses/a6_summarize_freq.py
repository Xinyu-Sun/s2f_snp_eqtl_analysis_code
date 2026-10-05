"""A6. Frequency class of each group's positives in the other two cohorts (PLINK --freq with --keep-allele-order: the MAF
column is the ALT frequency and NCHROBS the number of observed allele copies), and score magnitude / AUROC by class.
A variant counts as absent from a cohort when it is not in that cohort's genotype file or its minor allele count there
(minor allele frequency x NCHROBS) is below 10."""
import sys, glob, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, ci
pairs_path, freq_dir, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
df = load_pairs(pairs_path)
df["plink_id"] = df["chromosome"].astype(str) + ":" + df["position"].astype(str) + "_" + df["ref_allele"] + "_" + df["alt_allele"]
fr = {}
for g in GROUPS:
    parts = [pd.read_csv(f, sep=r"\s+") for f in glob.glob(f"{freq_dir}/{g}.chr*.frq")]
    if parts:
        t = pd.concat(parts); t["maf_calc"] = np.minimum(t["MAF"], 1 - t["MAF"])
        t["mac"] = np.rint(t["maf_calc"] * t["NCHROBS"])
        t.loc[t["mac"] < 10, "maf_calc"] = np.nan  # MAC < 10 in this cohort: treated as absent
        fr[g] = t.set_index("SNP")["maf_calc"]
# positives scored by either model (as in a5); each model's median and AUROC below use that model's own scored positives
_d = df.dropna(subset=["log_dist", "pip"])
_d = _d[(_d["pip"] >= 0.5) & (_d["borzoi_score"].notna() | _d["alphagenome_score"].notna())]
pos_all = pd.concat([_d[_d["ancestry"] == g].assign(group=g) for g in GROUPS]).drop_duplicates(["ancestry", "plink_id", "gene_id"])
for g in GROUPS:
    pos_all[f"maf_{g}"] = pos_all["plink_id"].map(fr.get(g, pd.Series(dtype=float)))
def classify(r):
    others = [o for o in GROUPS if o != r["group"]]
    vals = [r[f"maf_{o}"] for o in others]
    if any(pd.isna(v) for v in vals): return "absent_or_MAC<10_in_another_group"
    if all(v >= 0.05 for v in vals): return "common_in_all_groups"
    return "low_frequency_in_another_group"
pos_all["freq_class"] = pos_all.apply(classify, axis=1)
pos_all.to_csv(f"{out_dir}/a6_positive_freq.tsv", sep="\t", index=False)
rows = []
for t in (0.9, 0.5):
    for g in GROUPS:
        for cls, sub in pos_all[(pos_all["pip"] >= t) & (pos_all["group"] == g)].groupby("freq_class"):
            r = dict(pip_threshold=t, group=g, freq_class=cls, n=len(sub), share=len(sub) / max(1, (pos_all[(pos_all["pip"] >= t) & (pos_all["group"] == g)]).shape[0]))
            for model, col in MODELS.items():
                msub = sub[sub[col].notna()]  # this model's own scored positives in the class
                r[f"n_{model}"] = len(msub); r[f"median_abs_{model}"] = np.abs(msub[col]).median()
                pos, low, _ = split_classes(df[df["ancestry"] == g], col, t)
                keys = set(msub["plink_id"]); p2 = pos[pos["plink_id"].isin(keys)]
                if len(p2) >= 5:
                    rng = np.random.default_rng(13); e = distance_bins(p2, low); a = [matched_auroc(p2, low, col, rng, e)[0] for _ in range(300)]
                    r[f"auroc_{model}"] = np.nanmean(a); r[f"auroc_{model}_ci_low"], r[f"auroc_{model}_ci_high"] = ci(a)
                else:
                    r[f"auroc_{model}"] = np.nan
            rows.append(r)
pd.DataFrame(rows).to_csv(f"{out_dir}/a6_freq_class_summary.tsv", sep="\t", index=False); print("A6 done")
