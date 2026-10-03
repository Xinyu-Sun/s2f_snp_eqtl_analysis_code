"""Standard external replication: are each group's significant cis-eQTLs also significant, with the same direction,
in eQTLGen whole blood (31,684 mostly European-ancestry samples)? Uses the released per-group results as they are.
MAGENTA slope refers to the ALT allele (last allele in chr:pos_REF_ALT); eQTLGen Z refers to AssessedAllele."""
import os
import pandas as pd, numpy as np
# per-group nominal association tables ({group}_regular/new_nominal.parquet) of the eqtl_four_population_20260922 release
D=os.environ.get('FU_NOMINAL_DIR','/path/to/eqtl_four_population_20260922/comparison_20260929')
E='/mnt/vstor/Data14/metabrain_lasso/eQTLGen/magenta_ad_specific_eqtlgen_comp/blood_cis_eqtls_b38_fdr_sig.txt.gz'
W=os.environ.get('FU_REPLICATION_DIR','/path/to/followup_project/external_replication')
M={}
for g in ['AA','CH','NHW']:
    d=pd.read_parquet(f'{D}/{g}_regular/new_nominal.parquet'); d=d[(d.pvalue<2.02e-5)&(d.qvalue<0.05)].copy()
    d['gene']=d.gene_id.str.split('.').str[0]
    x=d.variant_id.str.extract(r'^(chr[0-9]+):([0-9]+)_([ACGT]+)_([ACGT]+)$'); d['chr']=x[0]; d['pos']=x[1]; d['ref']=x[2]; d['alt']=x[3]
    d=d.dropna(subset=['chr']); d['key']=d.chr+':'+d.pos+'|'+d.gene; M[g]=d
keys=set().union(*[set(d.key) for d in M.values()])
hits=[]; eg=set()
for ch in pd.read_csv(E,sep=' ',usecols=['chrSNPChr','SNPPos_b38','AssessedAllele','OtherAllele','Zscore','Gene'],chunksize=2_000_000):
    eg|=set(ch.Gene.unique()); ch['key']=ch.chrSNPChr+':'+ch.SNPPos_b38.astype(str)+'|'+ch.Gene
    hits.append(ch[ch.key.isin(keys)])
H=pd.concat(hits).drop_duplicates('key').set_index('key')
print('eQTLGen genes with any FDR-significant cis-eQTL:',len(eg),'| matched pairs:',len(H))
egenes={g:set(M[g].gene) for g in M}
rows=[]
def summarize(g,d,label):
    d=d[d.gene.isin(eg)]
    j=d.join(H[['AssessedAllele','OtherAllele','Zscore']],on='key',how='left')
    f=j.dropna(subset=['Zscore'])
    same=(f.alt==f.AssessedAllele)&(f.ref==f.OtherAllele); flip=(f.alt==f.OtherAllele)&(f.ref==f.AssessedAllele)
    snv=f.ref.str.len().eq(1)&f.alt.str.len().eq(1)&~(f.ref+f.alt).isin(['AT','TA','CG','GC'])
    u=f[(same|flip)&snv]; s=np.where(same[u.index],1,-1)*np.sign(u.Zscore)
    conc=(np.sign(u.beta)==s).mean() if len(u) else np.nan
    rows.append(dict(group=g,set=label,pairs_with_gene_in_eqtlgen=len(d),also_significant_in_eqtlgen=len(f),
        share_significant_in_eqtlgen=len(f)/len(d) if len(d) else np.nan,pairs_for_direction=len(u),same_direction=conc))
for g,d in M.items():
    lead=d.sort_values('pvalue').drop_duplicates('gene')
    summarize(g,lead,'lead variant of each eGene')
    others=set().union(*[egenes[o] for o in M if o!=g])
    summarize(g,lead[~lead.gene.isin(others)],'lead variant, eGene in this group only')
    summarize(g,lead[lead.gene.isin(others)],'lead variant, eGene also found in another group')
    summarize(g,d,'all significant pairs')
out=pd.DataFrame(rows); out.to_csv(W+'/results/eqtlgen_replication.tsv',sep='\t',index=False)
pd.set_option('display.width',250); print(out.round(3).to_string(index=False))
print({g:len(egenes[g]) for g in egenes})
