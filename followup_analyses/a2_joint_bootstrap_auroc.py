"""A2. Joint uncertainty for the group contrast in distance-matched AUROC, a permutation test,
and a cross-pool 2x2 decomposition (positives from one group against the comparison pool of another).

Outputs (results/):
  a2_auroc_bootstrap.tsv       comparison-only (manuscript design) and joint (positive-gene + comparison) bootstrap
  a2_group_differences.tsv     AA-NHW, AA-CH, CH-NHW differences under the joint bootstrap, percentile CI, bootstrap p
  a2_permutation_tests.tsv     permutation of positive group labels within TSS bins, p for the observed difference
  a2_cross_pool_auroc.tsv      positives(group i) vs comparison pool(group j), distance-matched
"""
import sys, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, TSS_BINS, ci

pairs_path, out_dir = sys.argv[1], sys.argv[2]
B_COMP, B_JOINT, B_PERM, B_CROSS = 200, 1000, 1000, 200
df = load_pairs(pairs_path)

boot_rows, diff_rows, perm_rows, cross_rows = [], [], [], []
store = {}
for model, col in MODELS.items():
    for t in (0.9, 0.5):
        for comp_name in ("low_pip", "intermediate_pip"):
            for g in GROUPS:
                rng = np.random.default_rng(42)
                pos, low, mid = split_classes(df[df["ancestry"] == g], col, t)
                comp = low if comp_name == "low_pip" else mid
                edges = distance_bins(pos, comp)
                a_comp = [matched_auroc(pos, comp, col, rng, edges)[0] for _ in range(B_COMP)]
                a_joint = [matched_auroc(pos, comp, col, rng, edges, resample_pos_genes=True)[0] for _ in range(B_JOINT)]
                store[(model, t, comp_name, g)] = np.array(a_joint)
                boot_rows.append(dict(model=model, pip_threshold=t, comparison=comp_name, group=g, n_pos=len(pos), n_pos_genes=pos["gene_id"].nunique(),
                                      n_comp=len(comp), auroc_comp_only_mean=np.nanmean(a_comp), comp_only_ci_low=ci(a_comp)[0], comp_only_ci_high=ci(a_comp)[1],
                                      auroc_joint_mean=np.nanmean(a_joint), joint_ci_low=ci(a_joint)[0], joint_ci_high=ci(a_joint)[1]))
            for a, b in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
                d = store[(model, t, comp_name, a)] - store[(model, t, comp_name, b)]
                p = 2 * min((d <= 0).mean(), (d >= 0).mean())
                diff_rows.append(dict(model=model, pip_threshold=t, comparison=comp_name, contrast=f"{a}_minus_{b}", diff_mean=np.nanmean(d),
                                      ci_low=ci(d)[0], ci_high=ci(d)[1], bootstrap_two_sided_p=min(1.0, p)))
            # permutation of positive labels within TSS bins (comparison pools stay group-specific)
            if t == 0.9:
                for a, b in (("AA", "NHW"), ("AA", "CH")):
                    rng = np.random.default_rng(7)
                    pa, la, ma = split_classes(df[df["ancestry"] == a], col, t); pb, lb, mb = split_classes(df[df["ancestry"] == b], col, t)
                    ca = la if comp_name == "low_pip" else ma; cb = lb if comp_name == "low_pip" else mb
                    ea, eb = distance_bins(pa, ca), distance_bins(pb, cb)
                    obs = np.mean([matched_auroc(pa, ca, col, rng, ea)[0] - matched_auroc(pb, cb, col, rng, eb)[0] for _ in range(50)])
                    pooled = pd.concat([pa.assign(_g=a), pb.assign(_g=b)])
                    perm = []
                    for _ in range(B_PERM):
                        lab = pooled["_g"].values.copy()
                        for tb in TSS_BINS:
                            idx = np.where(pooled["tss_distance_bin"].values == tb)[0]
                            lab[idx] = rng.permutation(lab[idx])
                        qa, qb = pooled[lab == a], pooled[lab == b]
                        perm.append(matched_auroc(qa, ca, col, rng, ea)[0] - matched_auroc(qb, cb, col, rng, eb)[0])
                    perm = np.array(perm)
                    perm_rows.append(dict(model=model, pip_threshold=t, comparison=comp_name, contrast=f"{a}_minus_{b}", observed_diff=obs,
                                          perm_mean=np.nanmean(perm), perm_sd=np.nanstd(perm), perm_p=(np.abs(perm) >= abs(obs)).mean(), n_perm=B_PERM))
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
pd.DataFrame(perm_rows).to_csv(f"{out_dir}/a2_permutation_tests.tsv", sep="\t", index=False)
pd.DataFrame(cross_rows).to_csv(f"{out_dir}/a2_cross_pool_auroc.tsv", sep="\t", index=False)
print("A2 done")
