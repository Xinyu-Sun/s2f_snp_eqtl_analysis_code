"""B4. How much of the separation, and of the group difference, is available without a sequence model?

Per model and group (PIP >= 0.9 versus low-PIP comparison variants, variants with a FILER annotation call only):
  distance-matched AUROC of (1) |score|, (2) accessible-chromatin overlap alone (0/1), (3) proximity to the TSS alone,
  (4) a logistic model with accessibility + distance spline, (5) the same model plus the score.
(4) and (5) use predictions from 5-fold cross-validation with genes kept together in a fold.
Uncertainty: joint bootstrap (positive genes resampled, comparison variants redrawn), B = 500, same draws for all predictors,
so the gain (5) - (4) and the group differences carry positive-set uncertainty.

Outputs: b4_baseline_auroc.tsv, b4_baseline_group_differences.tsv
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
import patsy
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, ci

pairs_path, filer_path, out_dir = sys.argv[1:4]
B = 500
df = load_pairs(pairs_path)
fl = pd.read_csv(filer_path, sep="\t", usecols=["chromosome", "position", "ref_allele", "alt_allele", "filer_binary_Accessible_chromatin"]).rename(columns={"filer_binary_Accessible_chromatin": "acc"})
df = df.merge(fl.drop_duplicates(["chromosome", "position", "ref_allele", "alt_allele"]), on=["chromosome", "position", "ref_allele", "alt_allele"], how="left")

def cv_pred(d, formula, k=5, seed=3):
    rng = np.random.default_rng(seed); genes = d["gene_id"].unique(); fold = dict(zip(genes, rng.integers(0, k, len(genes))))
    f = d["gene_id"].map(fold).values; out = np.full(len(d), np.nan)
    X = patsy.dmatrix(formula, d, return_type="dataframe")
    for i in range(k):
        tr, te = f != i, f == i
        if d["y"].values[tr].sum() < 5 or te.sum() == 0: continue
        try:
            m = sm.GLM(d["y"].values[tr], X[tr], family=sm.families.Binomial()).fit(maxiter=200)
            if m.converged and np.all(np.abs(m.params) < 50):
                out[te] = m.predict(X[te])
        except Exception: pass
    return out

PRED = ["score", "accessible_only", "tss_proximity_only", "annotation_model", "annotation_plus_score"]
rows, diffs = [], []
for model, col in MODELS.items():
    store = {}
    for g in GROUPS:
        pos, low, _ = split_classes(df[df["ancestry"] == g], col, 0.9)
        n_pos_all, n_low_all = len(pos), len(low)
        d = pd.concat([pos.assign(y=1), low.assign(y=0)]).dropna(subset=["acc"]).copy()
        d["x"] = np.log10(np.abs(d[col]) + 1e-6)
        d["score"] = np.abs(d[col]); d["accessible_only"] = d["acc"].astype(float); d["tss_proximity_only"] = d["log_dist"].max() - d["log_dist"]
        d["annotation_model"] = cv_pred(d, "acc + cr(log_dist, df=4, constraints='center')"); d["annotation_plus_score"] = cv_pred(d, "acc + cr(log_dist, df=4, constraints='center') + x")
        n_before = len(d); d = d.dropna(subset=["annotation_model", "annotation_plus_score"])
        if len(d) < n_before: print(model, g, "rows without a cross-validated prediction:", n_before - len(d), flush=True)
        p, c = d[d.y == 1], d[d.y == 0]; edges = distance_bins(p, c)
        rng = np.random.default_rng(42); res = {k: [] for k in PRED}
        for _ in range(B):
            # one positive-gene resample and one comparison draw shared by all predictors
            genes = p["gene_id"].unique(); cnt = pd.Series(rng.choice(genes, size=len(genes), replace=True)).value_counts()
            pb = p.loc[p["gene_id"].isin(cnt.index)]; pb = pb.loc[pb.index.repeat(pb["gene_id"].map(cnt).astype(int))]
            seed = int(rng.integers(0, 2**31 - 1))
            for k in PRED:
                res[k].append(matched_auroc(pb, c, k, np.random.default_rng(seed), edges)[0])
        for k in PRED:
            a = np.array(res[k]); store[(g, k)] = a
            rows.append(dict(model=model, group=g, predictor=k, n_pos_with_annotation=len(p), n_pos_all=n_pos_all, n_comp_with_annotation=len(c), n_comp_all=n_low_all,
                             share_pos_accessible=p["acc"].mean(), share_comp_accessible=c["acc"].mean(), auroc_mean=np.nanmean(a), ci_low=ci(a)[0], ci_high=ci(a)[1]))
        gain = store[(g, "annotation_plus_score")] - store[(g, "annotation_model")]; store[(g, "gain_from_score")] = gain
        rows.append(dict(model=model, group=g, predictor="gain_from_score", n_pos_with_annotation=len(p), n_pos_all=n_pos_all, n_comp_with_annotation=len(c), n_comp_all=n_low_all,
                         auroc_mean=np.nanmean(gain), ci_low=ci(gain)[0], ci_high=ci(gain)[1]))
    for a_, b_ in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
        for k in PRED + ["gain_from_score"]:
            dd = store[(a_, k)] - store[(b_, k)]; dd = dd[np.isfinite(dd)]
            pv = min(1.0, 2 * min((dd <= 0).mean(), (dd >= 0).mean()))
            diffs.append(dict(model=model, contrast=f"{a_}_minus_{b_}", predictor=k, diff_mean=dd.mean(), ci_low=ci(dd)[0], ci_high=ci(dd)[1], bootstrap_two_sided_p=pv))
pd.DataFrame(rows).to_csv(f"{out_dir}/b4_baseline_auroc.tsv", sep="\t", index=False)
pd.DataFrame(diffs).to_csv(f"{out_dir}/b4_baseline_group_differences.tsv", sep="\t", index=False); print("B4 done")
