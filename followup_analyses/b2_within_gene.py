"""B2. Within-gene evaluation. The score is a per-gene log fold change, so its scale can differ between genes.
For each high-PIP variant, the percentile of its |score| among low-PIP comparison variants of the SAME gene
(ties count one half). The mean percentile equals a within-gene AUROC averaged over positives.
Two versions: any same-gene comparison variant (at least 5), and same gene plus same TSS-distance bin (at least 3).
Uncertainty: bootstrap over positive genes (B = 2000).

Outputs: b2_within_gene.tsv, b2_within_gene_differences.tsv
"""
import sys, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, GROUPS, MODELS, ci
pairs_path, out_dir = sys.argv[1], sys.argv[2]
df = load_pairs(pairs_path); B = 2000
rows, diffs = [], []
for model, col in MODELS.items():
    for version, min_n in (("same_gene", 5), ("same_gene_same_tss_bin", 3)):
        store = {}
        for g in GROUPS:
            pos, low, _ = split_classes(df[df["ancestry"] == g], col, 0.9)
            keys = ["gene_id"] if version == "same_gene" else ["gene_id", "tss_distance_bin"]
            grp = {(k if isinstance(k, tuple) else (k,)): np.abs(v[col].values) for k, v in low.groupby(keys)}
            rec = []
            for r in pos.itertuples(index=False):
                k = (getattr(r, "gene_id"),) if version == "same_gene" else (getattr(r, "gene_id"), getattr(r, "tss_distance_bin"))
                c = grp.get(k)
                if c is None or len(c) < min_n: continue
                s = abs(getattr(r, col)); rec.append((r.gene_id, ((c < s).sum() + 0.5 * (c == s).sum()) / len(c), len(c)))
            w = pd.DataFrame(rec, columns=["gene_id", "pct", "n_comp"])
            if len(w) < 5:
                rows.append(dict(model=model, version=version, group=g, n_pos=len(pos), n_pos_evaluated=len(w), note="too few")); continue
            rng = np.random.default_rng(42); gi = w.groupby("gene_id").indices; genes = list(gi)
            bs = np.array([w["pct"].values[np.concatenate([gi[x] for x in rng.choice(genes, size=len(genes), replace=True)])].mean() for _ in range(B)])
            store[g] = bs
            rows.append(dict(model=model, version=version, group=g, n_pos=len(pos), n_pos_evaluated=len(w), share_evaluated=len(w) / len(pos), n_genes=w.gene_id.nunique(),
                             median_same_gene_comparison_variants=w.n_comp.median(), mean_percentile=w.pct.mean(), ci_low=ci(bs)[0], ci_high=ci(bs)[1],
                             share_above_all_same_gene_variants=(w.pct >= 1 - 1e-12).mean()))
        for a_, b_ in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
            if a_ in store and b_ in store:
                d = store[a_] - store[b_]
                diffs.append(dict(model=model, version=version, contrast=f"{a_}_minus_{b_}", diff_mean=d.mean(), ci_low=ci(d)[0], ci_high=ci(d)[1],
                                  bootstrap_two_sided_p=min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))))
pd.DataFrame(rows).to_csv(f"{out_dir}/b2_within_gene.tsv", sep="\t", index=False)
pd.DataFrame(diffs).to_csv(f"{out_dir}/b2_within_gene_differences.tsv", sep="\t", index=False); print("B2 done")
