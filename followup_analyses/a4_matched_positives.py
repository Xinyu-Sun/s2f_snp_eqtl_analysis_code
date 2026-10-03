"""A4. Does the AA advantage survive matching the positives across groups on |z|, TSS bin and MAF bin?
1:1 nearest-neighbour matching (without replacement) of each AA PIP>=0.9 positive to a positive from the other group
with exact TSS bin and MAF bin and the closest |z| (caliper 1.5); then distance-matched AUROC of each matched subset
against its own group's low-PIP pool (comparison-only bootstrap, 300 iterations)."""
import sys, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, ci
pairs_path, maf_path, out_dir = sys.argv[1], sys.argv[2], sys.argv[3]
df = load_pairs(pairs_path)
df["maf_use"] = df["maf"]  # filled for all groups by fu_common when FU_MAF is set
df["maf_bin"] = pd.cut(df["maf_use"], [0, 0.05, 0.2, 0.5], labels=["lt0.05", "0.05-0.2", "ge0.2"], include_lowest=True)
rows, pairs_out = [], []
for model, col in MODELS.items():
    t = 0.9
    pa, la, _ = split_classes(df[df["ancestry"] == "AA"], col, t)
    for other in ("NHW", "CH"):
        rng = np.random.default_rng(3)
        pb, lb, _ = split_classes(df[df["ancestry"] == other], col, t)
        pb = pb.copy(); pb["_used"] = False
        keep_a, keep_b = [], []
        for ia, ra in pa.sample(frac=1, random_state=1).iterrows():
            cand = pb[(~pb["_used"]) & (pb["tss_distance_bin"] == ra["tss_distance_bin"]) & (pb["maf_bin"].astype(str) == str(ra["maf_bin"]))]
            if cand.empty: continue
            dz = (cand["abs_z"] - ra["abs_z"]).abs()
            j = dz.idxmin()
            if dz.loc[j] > 1.5: continue
            pb.loc[j, "_used"] = True; keep_a.append(ia); keep_b.append(j)
            pairs_out.append(dict(model=model, other=other, aa_variant=ra["variant_key"], aa_gene=ra["gene_id"], aa_abs_z=ra["abs_z"],
                                  other_variant=pb.loc[j, "variant_key"], other_gene=pb.loc[j, "gene_id"], other_abs_z=pb.loc[j, "abs_z"],
                                  tss_bin=ra["tss_distance_bin"], maf_bin=str(ra["maf_bin"])))
        ma, mb = pa.loc[keep_a], pb.loc[keep_b]
        ea, eb = distance_bins(ma, la), distance_bins(mb, lb)
        aa = [matched_auroc(ma, la, col, rng, ea)[0] for _ in range(300)]
        ab = [matched_auroc(mb, lb, col, rng, eb)[0] for _ in range(300)]
        rows.append(dict(model=model, contrast=f"AA_vs_{other}", n_matched_pairs=len(ma), n_aa_total=len(pa), n_other_total=len(pb),
                         median_abs_z_aa=ma["abs_z"].median(), median_abs_z_other=mb["abs_z"].median(),
                         median_abs_score_aa=np.abs(ma[col]).median(), median_abs_score_other=np.abs(mb[col]).median(),
                         auroc_aa_matched=np.nanmean(aa), aa_ci_low=ci(aa)[0], aa_ci_high=ci(aa)[1],
                         auroc_other_matched=np.nanmean(ab), other_ci_low=ci(ab)[0], other_ci_high=ci(ab)[1],
                         diff_mean=np.nanmean(np.array(aa) - np.array(ab))))
pd.DataFrame(rows).to_csv(f"{out_dir}/a4_matched_positives.tsv", sep="\t", index=False)
pd.DataFrame(pairs_out).to_csv(f"{out_dir}/a4_matched_pairs.tsv", sep="\t", index=False); print("A4 done")
