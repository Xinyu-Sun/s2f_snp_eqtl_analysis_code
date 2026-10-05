"""Table S3. External replication: are each group's significant cis-eQTLs also significant, with the same direction, in
eQTLGen whole blood (31,684 mostly European-ancestry samples)?

Significant pairs: p < 2.02e-5 and within-gene q < 0.05 in the group's nominal association table. For each set (lead eQTL of
each eGene, split by whether the eGene is significant in another group, and all significant pairs) the table reports the
pairs tested, the pairs whose gene has at least one FDR-significant eQTLGen cis-eQTL (present in eQTLGen), the pairs whose
variant-gene pair is FDR-significant in eQTLGen, and direction agreement among significant pairs at non-palindromic
single-nucleotide variants whose alleles match. MAGENTA beta refers to the ALT allele (last allele in chr:pos_REF_ALT);
eQTLGen Z refers to AssessedAllele.

Usage: r1_eqtlgen_replication.py <out.tsv>   (FU_NOMINAL_DIR and FU_EQTLGEN set the inputs)
"""
import os, sys
import numpy as np, pandas as pd

# per-group nominal association tables ({group}_regular/new_nominal.parquet) of the eqtl_four_population_20260922 release
D = os.environ.get("FU_NOMINAL_DIR", "/path/to/eqtl_four_population_20260922/comparison_20260929")
# eQTLGen cis-eQTLs at FDR < 0.05, GRCh38 coordinates
E = os.environ.get("FU_EQTLGEN", "/path/to/blood_cis_eqtls_b38_fdr_sig.txt.gz")
out_path = sys.argv[1]

M = {}
for g in ["AA", "CH", "NHW"]:
    d = pd.read_parquet(f"{D}/{g}_regular/new_nominal.parquet")
    d = d[(d.pvalue < 2.02e-5) & (d.qvalue < 0.05)].copy()
    d["gene"] = d.gene_id.str.split(".").str[0]
    x = d.variant_id.str.extract(r"^(chr[0-9]+):([0-9]+)_([ACGT]+)_([ACGT]+)$")
    d["chr"], d["pos"], d["ref"], d["alt"] = x[0], x[1], x[2], x[3]
    d = d.dropna(subset=["chr"])
    d["key"] = d.chr + ":" + d.pos + "|" + d.gene
    M[g] = d
keys = set().union(*[set(d.key) for d in M.values()])
hits, eqtlgen_genes = [], set()
for chunk in pd.read_csv(E, sep=" ", usecols=["chrSNPChr", "SNPPos_b38", "AssessedAllele", "OtherAllele", "Zscore", "Gene"], chunksize=2_000_000):
    eqtlgen_genes |= set(chunk.Gene.unique())
    chunk["key"] = chunk.chrSNPChr + ":" + chunk.SNPPos_b38.astype(str) + "|" + chunk.Gene
    hits.append(chunk[chunk.key.isin(keys)])
H = pd.concat(hits).drop_duplicates("key").set_index("key")
egenes = {g: set(M[g].gene) for g in M}

rows = []
def summarize(g, d, label):
    present = d[d.gene.isin(eqtlgen_genes)]
    j = present.join(H[["AssessedAllele", "OtherAllele", "Zscore"]], on="key", how="left")
    f = j.dropna(subset=["Zscore"])
    same = (f.alt == f.AssessedAllele) & (f.ref == f.OtherAllele)
    flip = (f.alt == f.OtherAllele) & (f.ref == f.AssessedAllele)
    snv = f.ref.str.len().eq(1) & f.alt.str.len().eq(1) & ~(f.ref + f.alt).isin(["AT", "TA", "CG", "GC"])
    u = f[(same | flip) & snv]
    eqtlgen_sign = np.where(same[u.index], 1, -1) * np.sign(u.Zscore)
    n_same = int((np.sign(u.beta) == eqtlgen_sign).sum())
    rows.append(dict(group=g, set=label, pairs_tested=len(d), present_in_eqtlgen=len(present),
                     share_present_in_eqtlgen=len(present) / len(d) if len(d) else np.nan,
                     significant_in_eqtlgen=len(f), share_significant_in_eqtlgen=len(f) / len(present) if len(present) else np.nan,
                     direction_assessable=len(u), n_same_direction=n_same,
                     share_same_direction=n_same / len(u) if len(u) else np.nan))

for g, d in M.items():
    lead = d.sort_values("pvalue").drop_duplicates("gene")
    others = set().union(*[egenes[o] for o in M if o != g])
    summarize(g, lead, "lead variant of each eGene")
    summarize(g, lead[~lead.gene.isin(others)], "lead variant, eGene in this group only")
    summarize(g, lead[lead.gene.isin(others)], "lead variant, eGene also found in another group")
    summarize(g, d, "all significant pairs")
pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)
print("eQTLGen genes with an FDR-significant cis-eQTL:", len(eqtlgen_genes), "| matched pairs:", len(H))
