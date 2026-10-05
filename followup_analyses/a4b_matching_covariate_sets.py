"""A4b. Which matching covariate drives the matched-positive result? Repeat the 1:1 matching of AA PIP>=0.9 positives
to NHW/CH positives under different covariate sets and recompute distance-matched AUROC for both matched subsets."""
import sys, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, MODELS, ci
pairs_path, out_dir = sys.argv[1], sys.argv[2]
df = load_pairs(pairs_path)
df["maf_bin"] = pd.cut(df["maf"], [0, 0.05, 0.2, 0.5 + 1e-9], labels=["lt0.05", "0.05-0.2", "ge0.2"], right=False).astype(str)  # MAF < 0.05, 0.05-0.2, >= 0.2
SETS = {"tss_only": (True, False, False), "tss_maf": (True, True, False), "tss_z": (True, False, True), "z_only": (False, False, True), "tss_maf_z": (True, True, True)}
rows = []
for model, col in MODELS.items():
    pa, la, _ = split_classes(df[df["ancestry"] == "AA"], col, 0.9)
    for other in ("NHW", "CH"):
        pb0, lb, _ = split_classes(df[df["ancestry"] == other], col, 0.9)
        for name, (use_tss, use_maf, use_z) in SETS.items():
            rng = np.random.default_rng(3); pb = pb0.copy(); pb["_used"] = False; ka, kb = [], []
            for ia, ra in pa.sample(frac=1, random_state=1).iterrows():
                cand = pb[~pb["_used"]]
                if use_tss: cand = cand[cand["tss_distance_bin"] == ra["tss_distance_bin"]]
                if use_maf: cand = cand[cand["maf_bin"] == ra["maf_bin"]]
                if cand.empty: continue
                if use_z:
                    dz = (cand["abs_z"] - ra["abs_z"]).abs(); j = dz.idxmin()
                    if dz.loc[j] > 1.5: continue
                else:
                    j = cand.sample(1, random_state=int(rng.integers(1e9))).index[0]
                pb.loc[j, "_used"] = True; ka.append(ia); kb.append(j)
            ma, mb = pa.loc[ka], pb.loc[kb]
            ea, eb = distance_bins(ma, la), distance_bins(mb, lb)
            aa = [matched_auroc(ma, la, col, rng, ea)[0] for _ in range(300)]; ab = [matched_auroc(mb, lb, col, rng, eb)[0] for _ in range(300)]
            rows.append(dict(model=model, contrast=f"AA_vs_{other}", covariates=name, n_pairs=len(ma), median_abs_z_aa=ma["abs_z"].median(), median_abs_z_other=mb["abs_z"].median(),
                             share_0_3kb_other=(mb["tss_distance_bin"] == "0-3kb").mean(), share_maf_lt05_other=(mb["maf_bin"] == "lt0.05").mean(),
                             median_abs_score_aa=np.abs(ma[col]).median(), median_abs_score_other=np.abs(mb[col]).median(),
                             auroc_aa=np.nanmean(aa), auroc_other=np.nanmean(ab), other_ci_low=ci(ab)[0], other_ci_high=ci(ab)[1], diff=np.nanmean(np.array(aa) - np.array(ab))))
pd.DataFrame(rows).to_csv(f"{out_dir}/a4b_matching_covariate_sets.tsv", sep="\t", index=False); print("A4b done")
