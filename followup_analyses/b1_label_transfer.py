"""B1. Cross-group label transfer: where do each group's high-PIP variants sit in the other groups' fine-mapping?

For every high-PIP (PIP >= 0.9, in a credible set) variant-gene pair of group A, classify its status in group B:
  variant_absent          the gene has a credible set in B but the variant was not tested there (allele-count filter)
  gene_not_egene          the gene is not a significant eGene in B
  egene_no_credible_set   eGene in B, but fine-mapping produced no credible set
  pip_ge_0.9 / pip_0.5_0.9 / pip_0.01_0.5 / pip_lt_0.01   PIP of the same pair in B (credible-set membership reported separately)
Then: distance-matched AUROC of A's positives split by whether the pair is also high-PIP in at least one other group.

Inputs: scored pairs table; AA/NHW per-gene SuSiE outputs; the CH compact PIP table (PIP >= 0.001 or in a credible set) and
the list of CH fine-mapped genes; the CH genotype .bim (variants tested in CH); per-group significant-pair tables for eGene membership.
Usage: b1_label_transfer.py <pairs.tsv.gz> <ch_compact.tsv.gz> <ch_gene_list> <out_dir>
"""
import os, sys, glob, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, ci
pairs_path, ch_compact, ch_genes_path, out_dir = sys.argv[1:5]
S = '/mnt/vstor/Data14/metabrain_lasso/magenta_susie/eGenes'
# per-group nominal association tables ({group}_regular/new_nominal.parquet) of the eqtl_four_population_20260922 release
NOM = os.environ.get('FU_NOMINAL_DIR', '/path/to/eqtl_four_population_20260922/comparison_20260929')
# CH eQTL genotype file (the 209 CH participants, MAC >= 10): a variant is tested in CH only if it is in this .bim
CH_BIM = '/mnt/vstor/Data14/metabrain_lasso/rna_seq_norm/MAGENTA_Hispanic/geno/imp_fil/geno_final/ihg_ad/plink_by_chr/MAGENTA_HISP_TOPMed_imputed_b38_chr_1_22_filter_imp_r2_0.8_and_genotyped_var.CH_MAC10.plink_qc'
df = load_pairs(pairs_path)
df['gene'] = df['gene_id'].str.split('.').str[0]
df['vid'] = 'chr' + df['chromosome'].astype(str).str.replace('chr', '') + ':' + df['position'].astype(str) + '_' + df['ref_allele'] + '_' + df['alt_allele']
df['key'] = df['vid'] + '|' + df['gene']
# ---- high-PIP sets (PIP >= 0.9, in a credible set) ----
hp = {}
for g in ('AA', 'NHW'):
    agg = pd.read_csv(f'{S}/MAGENTA_eQTLs_susie_pip_w_cs_{g}.txt', sep='\t', usecols=['molecular_trait_id', 'variant_id', 'pip', 'CS'])
    agg = agg[(agg.pip >= 0.9) & (agg.CS > 0)].drop_duplicates(['molecular_trait_id', 'variant_id'])
    hp[g] = set(agg.variant_id + '|' + agg.molecular_trait_id)
chc = pd.read_csv(ch_compact, sep='\t', usecols=['molecular_trait_id', 'variant_id', 'pip', 'CS']).drop_duplicates(['molecular_trait_id', 'variant_id'])
chc['key'] = chc.variant_id + '|' + chc.molecular_trait_id
hp['CH'] = set(chc.loc[(chc.pip >= 0.9) & (chc.CS.fillna(0) > 0), 'key'])
ch_genes = set(pd.read_csv(ch_genes_path, sep='\t').iloc[:, 0].astype(str))
print({g: len(v) for g, v in hp.items()}, 'CH fine-mapped genes', len(ch_genes), flush=True)
# ---- eGene sets ----
egenes = {}
for g in GROUPS:
    d = pd.read_parquet(f'{NOM}/{g}_regular/new_nominal.parquet', columns=['gene_id', 'pvalue', 'qvalue'])
    egenes[g] = set(d.loc[(d.pvalue < 2.02e-5) & (d.qvalue < 0.05), 'gene_id'].str.split('.').str[0])
# ---- lookup of PIPs for all high-PIP keys in every group ----
allkeys = set().union(*hp.values())
lookup = {}  # (group, key) -> (pip, CS)
cs_genes = {}
for g in ('AA', 'NHW'):
    files = glob.glob(f'{S}/{g}/*/*_susie_pip_w_cs_corrected_n.txt.gz')
    cs_genes[g] = set()
    for f in files:
        d = pd.read_csv(f, sep='\t', usecols=['molecular_trait_id', 'variant_id', 'pip', 'CS'])
        gene = d.molecular_trait_id.iloc[0]; cs_genes[g].add(gene)
        k = d.variant_id + '|' + d.molecular_trait_id
        m = k.isin(allkeys)
        for kk, p, c in zip(k[m], d.pip[m], d.CS[m]): lookup[(g, kk)] = (float(p), int(c))
    print(g, 'genes with a credible-set file', len(cs_genes[g]), flush=True)
cs_genes['CH'] = set(chc.loc[chc.CS.fillna(0) > 0, 'molecular_trait_id'])
for kk, p, c in zip(chc.key, chc.pip, chc.CS.fillna(0)):
    if kk in allkeys: lookup[('CH', kk)] = (float(p), int(c))
need = {k.split('|')[0] for k in allkeys}; ch_tested = set()
for c in range(1, 23):
    ids = pd.read_csv(f'{CH_BIM}.{c}.bim', sep='\t', header=None, usecols=[1], dtype=str)[1]
    ch_tested |= set(ids[ids.isin(need)])
print('high-PIP variants present in the CH genotype file', len(ch_tested), 'of', len(need), flush=True)
def status(g, key):
    vid, gene = key.split('|')
    if (g, key) in lookup:
        p, c = lookup[(g, key)]
        cat = 'pip_ge_0.9' if p >= 0.9 else 'pip_0.5_0.9' if p >= 0.5 else 'pip_0.01_0.5' if p >= 0.01 else 'pip_lt_0.01'
        return cat, p, c
    # fine-mapped in CH and the variant was tested there; absent from the compact table means PIP < 0.001
    if g == 'CH' and gene in ch_genes and vid in ch_tested: return 'pip_lt_0.01', 0.0, 0
    if gene not in egenes[g]: return 'gene_not_egene', np.nan, 0
    if gene not in cs_genes[g]: return 'egene_no_credible_set', np.nan, 0
    return 'variant_absent', np.nan, 0
rows, detail = [], []
for a in GROUPS:
    for key in sorted(hp[a]):
        rec = dict(group=a, key=key)
        for b in GROUPS:
            if b == a: continue
            cat, p, c = status(b, key); rec[f'status_{b}'] = cat; rec[f'pip_{b}'] = p; rec[f'cs_{b}'] = c
        detail.append(rec)
D = pd.DataFrame(detail); D.to_csv(f'{out_dir}/b1_label_transfer_detail.tsv', sep='\t', index=False)
for a in GROUPS:
    for b in GROUPS:
        if b == a: continue
        d = D[D.group == a]; vc = d[f'status_{b}'].value_counts()
        rows.append(dict(positives_from=a, status_in=b, n_positives=len(d), **{k: int(vc.get(k, 0)) for k in ['pip_ge_0.9', 'pip_0.5_0.9', 'pip_0.01_0.5', 'pip_lt_0.01', 'variant_absent', 'egene_no_credible_set', 'gene_not_egene']},
                         in_credible_set=int((d[f'cs_{b}'] > 0).sum())))
pd.DataFrame(rows).to_csv(f'{out_dir}/b1_label_transfer_summary.tsv', sep='\t', index=False)
# ---- AUROC by sharing (positives present in the scored table) ----
au = []
shared = {a: {k for k in hp[a] if any(status(b, k)[0] == 'pip_ge_0.9' for b in GROUPS if b != a)} for a in GROUPS}
for model, col in MODELS.items():
    for a in GROUPS:
        pos, low, _ = split_classes(df[df.ancestry == a], col, 0.9)
        pos = pos[pos.key.isin(hp[a])]
        for lab, sub in (('shared_high_pip_in_another_group', pos[pos.key.isin(shared[a])]), ('high_pip_in_this_group_only', pos[~pos.key.isin(shared[a])]), ('all', pos)):
            if len(sub) < 5: au.append(dict(model=model, group=a, subset=lab, n_pos=len(sub), note='too few')); continue
            rng = np.random.default_rng(42); e = distance_bins(sub, low)
            x = [matched_auroc(sub, low, col, rng, e, resample_pos_genes=True)[0] for _ in range(1000)]
            au.append(dict(model=model, group=a, subset=lab, n_pos=len(sub), n_genes=sub.gene_id.nunique(), auroc_mean=np.nanmean(x), ci_low=ci(x)[0], ci_high=ci(x)[1]))
pd.DataFrame(au).to_csv(f'{out_dir}/b1_auroc_by_sharing.tsv', sep='\t', index=False); print('B1 done')
