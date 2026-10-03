"""A7. Distance-matched AUROC restricted to genes represented in the fine-mapping pools of all three groups."""
import sys, numpy as np, pandas as pd
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, ci
pairs_path, out_dir = sys.argv[1], sys.argv[2]
df = load_pairs(pairs_path)
shared = set.intersection(*[set(df.loc[df["ancestry"] == g, "gene_id"]) for g in GROUPS])
rows = []
for model, col in MODELS.items():
    for t in (0.9, 0.5):
        for g in GROUPS:
            rng = np.random.default_rng(5)
            d = df[(df["ancestry"] == g) & (df["gene_id"].isin(shared))]
            pos, low, _ = split_classes(d, col, t)
            if len(pos) < 5: rows.append(dict(model=model, pip_threshold=t, group=g, n_pos=len(pos), note="too few")); continue
            e = distance_bins(pos, low); a = [matched_auroc(pos, low, col, rng, e)[0] for _ in range(300)]
            rows.append(dict(model=model, pip_threshold=t, group=g, n_shared_genes=len(shared), n_pos=len(pos), n_pos_genes=pos["gene_id"].nunique(),
                             n_low=len(low), auroc_mean=np.nanmean(a), ci_low=ci(a)[0], ci_high=ci(a)[1]))
pd.DataFrame(rows).to_csv(f"{out_dir}/a7_shared_gene_auroc.tsv", sep="\t", index=False); print("A7 done")
