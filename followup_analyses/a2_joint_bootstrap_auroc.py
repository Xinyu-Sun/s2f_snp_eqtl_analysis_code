"""A2. Joint uncertainty for the group contrast in distance-matched AUROC, and a cross-pool 2x2 decomposition
(positives from one group against the comparison pool of another).

Outputs (results/):
  a2_auroc_bootstrap.tsv       comparison-only (manuscript design) and joint (positive-gene + comparison) bootstrap
  a2_group_differences.tsv     AA-NHW, AA-CH, CH-NHW differences under the joint bootstrap, percentile CI, bootstrap p
  a2_cross_pool_auroc.tsv      positives(group i) vs comparison pool(group j), distance-matched
The primary contrast (PIP >= 0.9 against low-PIP comparison variants) uses 10,000 joint replicates, the others 1,000.
Each group has its own random stream.
"""
import sys, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, ci, group_rng, bootstrap_p

pairs_path, out_dir = sys.argv[1], sys.argv[2]
B_COMP, B_JOINT, B_JOINT_PRIMARY, B_CROSS = 200, 1000, 10000, 200
df = load_pairs(pairs_path)

boot_rows, diff_rows, cross_rows = [], [], []
store = {}
for model, col in MODELS.items():
    for t in (0.9, 0.5):
        for comp_name in ("low_pip", "intermediate_pip"):
            b_joint = B_JOINT_PRIMARY if (t == 0.9 and comp_name == "low_pip") else B_JOINT
            for g in GROUPS:
                rng = group_rng(42, g)
                pos, low, mid = split_classes(df[df["ancestry"] == g], col, t)
                comp = low if comp_name == "low_pip" else mid
                edges = distance_bins(pos, comp)
                a_comp = [matched_auroc(pos, comp, col, rng, edges)[0] for _ in range(B_COMP)]
                a_joint = [matched_auroc(pos, comp, col, rng, edges, resample_pos_genes=True)[0] for _ in range(b_joint)]
                store[(model, t, comp_name, g)] = np.array(a_joint)
                boot_rows.append(dict(model=model, pip_threshold=t, comparison=comp_name, group=g, n_pos=len(pos), n_pos_genes=pos["gene_id"].nunique(),
                                      n_comp=len(comp), auroc_comp_only_mean=np.nanmean(a_comp), comp_only_ci_low=ci(a_comp)[0], comp_only_ci_high=ci(a_comp)[1],
                                      auroc_joint_mean=np.nanmean(a_joint), joint_ci_low=ci(a_joint)[0], joint_ci_high=ci(a_joint)[1],
                                      n_comp_only_replicates=B_COMP, n_joint_replicates=b_joint))
            for a, b in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
                d = store[(model, t, comp_name, a)] - store[(model, t, comp_name, b)]
                diff_rows.append(dict(model=model, pip_threshold=t, comparison=comp_name, contrast=f"{a}_minus_{b}", diff_mean=np.nanmean(d),
                                      ci_low=ci(d)[0], ci_high=ci(d)[1], bootstrap_two_sided_p=bootstrap_p(d), n_replicates=int(np.isfinite(d).sum())))
            # cross-pool 2x2 at the primary threshold
            if t == 0.9:
                for gi in GROUPS:
                    pos_i = split_classes(df[df["ancestry"] == gi], col, t)[0]
                    for gj in GROUPS:
                        rng = np.random.default_rng(11)
                        _, low_j, mid_j = split_classes(df[df["ancestry"] == gj], col, t)
                        comp_j = low_j if comp_name == "low_pip" else mid_j
                        e = distance_bins(pos_i, comp_j)
                        a = [matched_auroc(pos_i, comp_j, col, rng, e)[0] for _ in range(B_CROSS)]
                        cross_rows.append(dict(model=model, pip_threshold=t, comparison=comp_name, positives_from=gi, comparison_pool_from=gj,
                                               n_pos=len(pos_i), n_comp=len(comp_j), auroc_mean=np.nanmean(a), ci_low=ci(a)[0], ci_high=ci(a)[1]))
pd.DataFrame(boot_rows).to_csv(f"{out_dir}/a2_auroc_bootstrap.tsv", sep="\t", index=False)
pd.DataFrame(diff_rows).to_csv(f"{out_dir}/a2_group_differences.tsv", sep="\t", index=False)
pd.DataFrame(cross_rows).to_csv(f"{out_dir}/a2_cross_pool_auroc.tsv", sep="\t", index=False)
print("A2 done")
