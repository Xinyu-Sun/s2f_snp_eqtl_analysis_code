"""B7. (a) Is the AA AUROC unusual for a positive set of AA's size? Random subsets of the CH and NHW positives of the same
size as the AA set (without replacement), each evaluated against that group's comparison pool as the mean of 20
comparison draws (the AA reference is the mean of 200 draws).
(b) PIP-weighted AUROC: positives weighted by their PIP, to propagate label uncertainty.

Outputs: b7_random_subsets.tsv, b7_pip_weighted_auroc.tsv, b7_pip_weighted_differences.tsv
"""
import sys, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, ci, group_rng, bootstrap_p
pairs_path, out_dir = sys.argv[1], sys.argv[2]
df = load_pairs(pairs_path); B_SUB, B_REF, B_JOINT, DRAWS_PER_SUBSET = 1000, 200, 1000, 20
sub_rows, w_rows, w_diff = [], [], []
for model, col in MODELS.items():
    for comp_name in ("low_pip", "intermediate_pip"):
        S = {g: split_classes(df[df["ancestry"] == g], col, 0.9) for g in GROUPS}
        C = {g: (S[g][1] if comp_name == "low_pip" else S[g][2]) for g in GROUPS}
        rng = group_rng(42, "AA"); pa = S["AA"][0]; n_aa = len(pa); e = distance_bins(pa, C["AA"])
        aa_ref = np.nanmean([matched_auroc(pa, C["AA"], col, rng, e)[0] for _ in range(B_REF)])
        for g in ("CH", "NHW"):
            p = S[g][0]
            if len(p) <= n_aa: continue
            rng = group_rng(5, g); vals = []
            for _ in range(B_SUB):
                s = p.iloc[rng.choice(len(p), size=n_aa, replace=False)]
                e_s = distance_bins(s, C[g])
                vals.append(np.nanmean([matched_auroc(s, C[g], col, rng, e_s)[0] for _ in range(DRAWS_PER_SUBSET)]))
            vals = np.array(vals); vals = vals[np.isfinite(vals)]
            sub_rows.append(dict(model=model, comparison=comp_name, group=g, n_pos_group=len(p), subset_size=n_aa, n_subsets=len(vals), draws_per_subset=DRAWS_PER_SUBSET, aa_auroc=aa_ref,
                                 subset_auroc_mean=vals.mean(), subset_p2_5=np.percentile(vals, 2.5), subset_p97_5=np.percentile(vals, 97.5),
                                 share_of_subsets_at_or_above_aa=(vals >= aa_ref).mean()))
        store = {}
        for g in GROUPS:
            p = S[g][0]; e = distance_bins(p, C[g]); rng = group_rng(42, g)
            w = np.array([matched_auroc(p, C[g], col, rng, e, resample_pos_genes=True, pos_weight_col="pip")[0] for _ in range(B_JOINT)])
            rng = group_rng(42, g)
            u = np.array([matched_auroc(p, C[g], col, rng, e, resample_pos_genes=True)[0] for _ in range(B_JOINT)])
            store[g] = w
            w_rows.append(dict(model=model, comparison=comp_name, group=g, n_pos=len(p), median_pip=p["pip"].median(), auroc_unweighted=np.nanmean(u),
                               auroc_pip_weighted=np.nanmean(w), weighted_ci_low=ci(w)[0], weighted_ci_high=ci(w)[1]))
        for a_, b_ in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
            d = store[a_] - store[b_]; d = d[np.isfinite(d)]
            w_diff.append(dict(model=model, comparison=comp_name, contrast=f"{a_}_minus_{b_}", diff_mean=d.mean(), ci_low=ci(d)[0], ci_high=ci(d)[1],
                               bootstrap_two_sided_p=bootstrap_p(d), n_replicates=len(d)))
pd.DataFrame(sub_rows).to_csv(f"{out_dir}/b7_random_subsets.tsv", sep="\t", index=False)
pd.DataFrame(w_rows).to_csv(f"{out_dir}/b7_pip_weighted_auroc.tsv", sep="\t", index=False)
pd.DataFrame(w_diff).to_csv(f"{out_dir}/b7_pip_weighted_differences.tsv", sep="\t", index=False); print("B7 done")
