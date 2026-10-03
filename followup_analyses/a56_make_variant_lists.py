"""Write per-group, per-chromosome PLINK variant-ID lists for the LD (A5) and frequency (A6) steps.
IDs follow the MAGENTA bim convention chr{chrom}:{pos}_{REF}_{ALT}. Positives = PIP >= 0.5 with any model score
(PIP >= 0.9 flagged downstream). All positives from all groups go into the frequency lists for every cohort."""
import sys, os, pandas as pd
from fu_common import load_pairs, GROUPS
pairs_path, out_dir = sys.argv[1], sys.argv[2]
df = load_pairs(pairs_path)
pos = df[(df["pip"] >= 0.5) & (df["borzoi_score"].notna() | df["alphagenome_score"].notna())].copy()
pos["plink_id"] = pos["chromosome"].astype(str) + ":" + pos["position"].astype(str) + "_" + pos["ref_allele"] + "_" + pos["alt_allele"]
pos["chrom_num"] = pos["chromosome"].str.replace("chr", "", regex=False)
pos[["ancestry", "plink_id", "chromosome", "position", "ref_allele", "alt_allele", "gene_id", "pip", "CS", "tss_distance_bin", "distance_to_tss"]].drop_duplicates().to_csv(f"{out_dir}/positives_pip_ge_0.5.tsv", sep="\t", index=False)
os.makedirs(f"{out_dir}/ld_lists", exist_ok=True); os.makedirs(f"{out_dir}/freq_lists", exist_ok=True)
for g in GROUPS:
    for c, sub in pos[pos["ancestry"] == g].groupby("chrom_num"):
        sub["plink_id"].drop_duplicates().to_csv(f"{out_dir}/ld_lists/{g}.chr{c}.txt", index=False, header=False)
for c, sub in pos.groupby("chrom_num"):
    sub["plink_id"].drop_duplicates().to_csv(f"{out_dir}/freq_lists/all.chr{c}.txt", index=False, header=False)
print("lists written:", pos.groupby("ancestry")["plink_id"].nunique().to_dict())
