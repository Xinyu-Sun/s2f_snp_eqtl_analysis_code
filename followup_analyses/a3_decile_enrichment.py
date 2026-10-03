"""A3. Precision-style view: odds of being a high-PIP positive in the top versus bottom decile of |score|,
within the benchmark pool (positives + low-PIP comparison variants), per model x group x threshold,
with a gene-cluster bootstrap CI and a TSS-bin-stratified (Mantel-Haenszel) estimate."""
import sys, numpy as np, pandas as pd
from scipy.stats.contingency import odds_ratio
from fu_common import load_pairs, split_classes, GROUPS, MODELS, TSS_BINS, ci
pairs_path, out_dir = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(42); B = 1000
df = load_pairs(pairs_path)

def or_top_bottom(d, col):
    s = np.abs(d[col].values); y = d["is_pos"].values
    q = np.quantile(s, [0.1, 0.9])
    top, bot = y[s >= q[1]], y[s <= q[0]]
    a, b = top.sum() + 0.5, (len(top) - top.sum()) + 0.5
    c, dd = bot.sum() + 0.5, (len(bot) - bot.sum()) + 0.5
    return (a / b) / (c / dd), int(top.sum()), int(len(top)), int(bot.sum()), int(len(bot))

def mh_or(d, col):
    num = den = 0.0
    for tb in TSS_BINS:
        sub = d[d["tss_distance_bin"] == tb]
        if len(sub) < 20: continue
        s = np.abs(sub[col].values); y = sub["is_pos"].values
        q = np.quantile(s, [0.1, 0.9]); top, bot = y[s >= q[1]], y[s <= q[0]]
        n = len(top) + len(bot)
        if n == 0: continue
        a, b, c, dd = top.sum(), len(top) - top.sum(), bot.sum(), len(bot) - bot.sum()
        num += a * dd / n; den += b * c / n
    return num / den if den > 0 else np.nan

rows = []
for model, col in MODELS.items():
    for t in (0.9, 0.5):
        for g in GROUPS:
            pos, low, _ = split_classes(df[df["ancestry"] == g], col, t)
            d = pd.concat([pos.assign(is_pos=1), low.assign(is_pos=0)]).reset_index(drop=True)
            o, tp, tn, bp, bn = or_top_bottom(d, col)
            gi = d.groupby("gene_id").indices; genes = list(gi)
            boots = []
            for _ in range(B):
                idx = np.concatenate([gi[x] for x in rng.choice(genes, size=len(genes), replace=True)])
                boots.append(or_top_bottom(d.iloc[idx], col)[0])
            lo, hi = ci(boots)
            # exact conditional odds ratio (finite lower bound even when a cell is zero) and a zero-cell-free summary:
            # the share of positives that fall in the top score decile, with a gene-cluster bootstrap interval
            ex = odds_ratio(np.array([[tp, tn - tp], [bp, bn - bp]]), kind="conditional"); exci = ex.confidence_interval(0.95)
            def share_top(dd):
                s_ = np.abs(dd[col].values); y_ = dd["is_pos"].values
                return y_[s_ >= np.quantile(s_, 0.9)].sum() / max(1, y_.sum())
            sb = []
            for _ in range(B):
                idx = np.concatenate([gi[x] for x in rng.choice(genes, size=len(genes), replace=True)])
                sb.append(share_top(d.iloc[idx]))
            rows.append(dict(model=model, pip_threshold=t, group=g, n_pos=len(pos), n_low=len(low), or_top_vs_bottom_decile=o,
                             ci_low=lo, ci_high=hi, pos_in_top=tp, n_top=tn, pos_in_bottom=bp, n_bottom=bn, mh_or_tss_stratified=mh_or(d, col),
                             exact_or=ex.statistic, exact_ci_low=exci.low, exact_ci_high=exci.high,
                             share_pos_in_top_decile=share_top(d), share_top_ci_low=ci(sb)[0], share_top_ci_high=ci(sb)[1]))
pd.DataFrame(rows).to_csv(f"{out_dir}/a3_decile_enrichment.tsv", sep="\t", index=False); print("A3 done")
