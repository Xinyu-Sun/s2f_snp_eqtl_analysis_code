"""Do high-PIP variants that are also high-PIP in another group separate better than group-specific ones after matching?

PIP >= 0.9 positives (all in a credible set) of each group against the group's own low-PIP pool (PIP < 0.01, no credible set).
Sharing status comes from results_rev4/b1_label_transfer_detail.tsv (one row per high-PIP pair, status in each other group).

Definitions of the split (all within one group):
  A  shared    = PIP >= 0.9 in >= 1 other group          vs  not_shared = all other positives (the b1 split)
  B  shared_cs = in a credible set in >= 1 other group    vs  specific   = neither (no credible set, no PIP >= 0.9 elsewhere)
  C  shared    = PIP >= 0.9 in >= 1 other group          vs  specific   = neither (positives in a credible set elsewhere at
                                                              PIP < 0.9 are left out)
Every PIP >= 0.9 pair elsewhere is also in a credible set there, so shared is a subset of shared_cs.

Steps (per group x model x definition):
  s1  characteristics of the two sets (all PIP >= 0.9 positives in the table, score-agnostic)
  s2  unmatched distance-matched AUROC (300 comparison-only draws per subset) and a joint bootstrap of the difference
      (1,000 replicates; positive genes resampled with replacement from the union of both subsets, comparison variants redrawn;
      the same gene draw is used for both subsets); exact b1 re-run for definition A as a check
  s3  one-to-one matching without replacement on TSS-distance bin and MAF bin (exact) and the closest |z| within a caliper
      of 1.5 (the a4 settings; source order shuffled with random_state=1), in both directions; joint bootstrap of the two
      matched subsets (1,000 replicates; clusters = matched pairs joined when they share a gene, resampled with replacement;
      comparison variants redrawn); difference shared minus specific with percentile interval and bootstrap p;
      matching-order sensitivity: 20 further source orders, 100 comparison-only draws each
  s4  OLS of log10(|score| + 1e-6) on the shared indicator + TSS-distance bin + log distance + MAF + |z| (gene-clustered SE),
      then with accessible-chromatin overlap (FILER) added; crude model for reference
  s5  pooled across groups: matched AUROC difference weighted by matched pairs, with a bootstrap whose clusters (genes)
      span groups; pooled regression with group-specific covariates; shared x group interaction test

Usage (from followup_analyses/scripts with FU_GENE_TSS and FU_MAF set):
  b8_shared_matched.py <pairs.tsv.gz> <b1_label_transfer_detail.tsv> <variant_filer_category_counts.tsv.gz> <out_dir>
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, fisher_exact, chi2
import statsmodels.formula.api as smf
from fu_common import load_pairs, split_classes, matched_auroc, distance_bins, GROUPS, MODELS, TSS_BINS, ci, bootstrap_p

pairs_path, detail_path, filer_path, out_dir = sys.argv[1:5]
B_COMP, B_JOINT, N_ORDERS, B_ORDER = 300, 1000, 20, 100
CALIPER = 1.5
DEFS = {
    "A_shared_vs_not_shared": ("shared", "not_shared"),
    "B_shared_cs_vs_specific": ("shared_cs", "specific"),
    "C_shared_vs_specific": ("shared", "specific"),
}

df = load_pairs(pairs_path)
df["key"] = ("chr" + df["chromosome"].astype(str).str.replace("chr", "") + ":" + df["position"].astype(str) + "_"
             + df["ref_allele"] + "_" + df["alt_allele"] + "|" + df["gene_id"].str.split(".").str[0])
df["maf_bin"] = pd.cut(df["maf"], [0, 0.05, 0.2, 0.5 + 1e-9], labels=["lt0.05", "0.05-0.2", "ge0.2"], right=False)
df["tss_bin"] = pd.Categorical(df["tss_distance_bin"], categories=TSS_BINS)
filer = pd.read_csv(filer_path, sep="\t", usecols=["chromosome", "position", "ref_allele", "alt_allele", "filer_binary_Accessible_chromatin"])
filer = filer.rename(columns={"filer_binary_Accessible_chromatin": "accessible"}).drop_duplicates(["chromosome", "position", "ref_allele", "alt_allele"])
df["chromosome"] = df["chromosome"].astype(str)
df = df.merge(filer, on=["chromosome", "position", "ref_allele", "alt_allele"], how="left")
df.index = pd.RangeIndex(len(df))

# sharing flags per high-PIP pair
det = pd.read_csv(detail_path, sep="\t")
flag = {}
for g in GROUPS:
    d = det[det.group == g]
    others = [o for o in GROUPS if o != g]
    sh = d[[f"status_{o}" for o in others]].eq("pip_ge_0.9").any(axis=1)
    cs = (d[[f"cs_{o}" for o in others]].fillna(0) > 0).any(axis=1) | sh
    for k, a, b in zip(d.key, sh, cs):
        flag[(g, k)] = (bool(a), bool(b))
pos_all = df[df.pip >= 0.9]
missing = [(g, k) for g, k in zip(pos_all.ancestry, pos_all.key) if (g, k) not in flag]
assert not missing, f"positives without sharing status: {missing[:5]}"
df["is_shared"] = [flag.get((g, k), (np.nan, np.nan))[0] for g, k in zip(df.ancestry, df.key)]
df["is_shared_cs"] = [flag.get((g, k), (np.nan, np.nan))[1] for g, k in zip(df.ancestry, df.key)]


def split(pos, defn):
    """Return (shared subset, comparator subset) for a definition."""
    if defn.startswith("A"):
        return pos[pos.is_shared == True], pos[pos.is_shared == False]
    if defn.startswith("B"):
        return pos[pos.is_shared_cs == True], pos[pos.is_shared_cs == False]
    return pos[pos.is_shared == True], pos[pos.is_shared_cs == False]


def resample(sub, counts):
    s = sub[sub["cluster"].isin(counts.index)]
    return s.loc[s.index.repeat(s["cluster"].map(counts).astype(int))]


def joint_boot(subsets, edges, comp, col, rng, B):
    """Cluster bootstrap: clusters drawn once per replicate and applied to every subset; comparison variants redrawn."""
    clusters = np.unique(np.concatenate([s["cluster"].values for s in subsets]))
    out = np.full((B, len(subsets)), np.nan)
    for b in range(B):
        counts = pd.Series(rng.choice(clusters, size=len(clusters), replace=True)).value_counts()
        for k, (s, e) in enumerate(zip(subsets, edges)):
            out[b, k] = matched_auroc(resample(s, counts), comp, col, rng, e)[0]
    return out


def greedy_match(src, tgt, seed=1):
    """a4 matching: each source positive, in shuffled order, takes the unused target with the same TSS bin and MAF bin and
    the closest |z|, if within the caliper."""
    tgt = tgt.copy()
    tgt["_used"] = False
    out = []
    for i, r in src.sample(frac=1, random_state=seed).iterrows():
        cand = tgt[(~tgt["_used"]) & (tgt["tss_distance_bin"] == r["tss_distance_bin"]) & (tgt["maf_bin"].astype(str) == str(r["maf_bin"]))]
        if cand.empty:
            continue
        dz = (cand["abs_z"] - r["abs_z"]).abs()
        j = dz.idxmin()
        if dz.loc[j] > CALIPER:
            continue
        tgt.loc[j, "_used"] = True
        out.append((i, j))
    return out


def pair_clusters(pairs, data, prefix=""):
    """Union-find over matched pairs: pairs whose positives share a gene form one cluster. Returns {row index: cluster id}."""
    parent = list(range(len(pairs)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    first = {}
    for p, (a, b) in enumerate(pairs):
        for idx in (a, b):
            gname = data.loc[idx, "gene_id"]
            if gname in first:
                ra, rb = find(p), find(first[gname])
                if ra != rb:
                    parent[ra] = rb
            else:
                first[gname] = p
    return {idx: f"{prefix}{find(p)}" for p, (a, b) in enumerate(pairs) for idx in (a, b)}


def describe(sub):
    return dict(n=len(sub), median_abs_tss_distance_bp=sub["distance_to_tss"].abs().median(), share_0_3kb=(sub["tss_distance_bin"] == "0-3kb").mean(),
                share_maf_ge_005=(sub["maf"] >= 0.05).mean(), median_maf=sub["maf"].median(), median_abs_z=sub["abs_z"].median(),
                share_accessible=sub["accessible"].mean(), n_accessible_known=int(sub["accessible"].notna().sum()))


# ---------------- s1: characteristics ----------------
s1 = []
for g in GROUPS:
    pos = df[(df.ancestry == g) & (df.pip >= 0.9)]
    for defn, (ln, rn) in DEFS.items():
        a, b = split(pos, defn)
        da, db = describe(a), describe(b)
        row = dict(group=g, definition=defn, shared_label=ln, comparator_label=rn)
        row.update({f"{k}_shared": v for k, v in da.items()})
        row.update({f"{k}_comparator": v for k, v in db.items()})
        row["p_distance_mannwhitney"] = mannwhitneyu(a["distance_to_tss"].abs(), b["distance_to_tss"].abs()).pvalue
        row["p_abs_z_mannwhitney"] = mannwhitneyu(a["abs_z"], b["abs_z"]).pvalue
        row["p_maf_mannwhitney"] = mannwhitneyu(a["maf"], b["maf"]).pvalue
        row["p_maf_ge_005_fisher"] = fisher_exact([[(a.maf >= 0.05).sum(), (a.maf < 0.05).sum()], [(b.maf >= 0.05).sum(), (b.maf < 0.05).sum()]])[1]
        aa, bb = a["accessible"].dropna(), b["accessible"].dropna()
        row["p_accessible_fisher"] = fisher_exact([[(aa == 1).sum(), (aa == 0).sum()], [(bb == 1).sum(), (bb == 0).sum()]])[1]
        s1.append(row)
pd.DataFrame(s1).to_csv(f"{out_dir}/b8_describe.tsv", sep="\t", index=False)
print("s1 done", flush=True)

# ---------------- s2: unmatched AUROC ----------------
s2 = []
for mi, (model, col) in enumerate(MODELS.items()):
    for gi, g in enumerate(GROUPS):
        pos, low, _ = split_classes(df[df.ancestry == g], col, 0.9)
        for di, defn in enumerate(DEFS):
            a, b = split(pos, defn)
            a, b = a.assign(cluster=a["gene_id"]), b.assign(cluster=b["gene_id"])
            ea, eb = distance_bins(a, low), distance_bins(b, low)
            rng = np.random.default_rng([7, gi, mi, di])
            ca = [matched_auroc(a, low, col, rng, ea)[0] for _ in range(B_COMP)]
            cb = [matched_auroc(b, low, col, rng, eb)[0] for _ in range(B_COMP)]
            jb = joint_boot([a, b], [ea, eb], low, col, rng, B_JOINT)
            d = jb[:, 0] - jb[:, 1]
            row = dict(model=model, group=g, definition=defn, n_shared=len(a), n_comparator=len(b), n_low_pool=len(low),
                       auroc_shared_comp_only=np.nanmean(ca), shared_comp_ci_low=ci(ca)[0], shared_comp_ci_high=ci(ca)[1],
                       auroc_comparator_comp_only=np.nanmean(cb), comparator_comp_ci_low=ci(cb)[0], comparator_comp_ci_high=ci(cb)[1],
                       auroc_shared_joint=np.nanmean(jb[:, 0]), auroc_comparator_joint=np.nanmean(jb[:, 1]),
                       diff_joint_mean=np.nanmean(d), diff_ci_low=ci(d)[0], diff_ci_high=ci(d)[1], diff_bootstrap_p=bootstrap_p(d))
            if defn.startswith("A"):  # exact b1 re-run (default_rng(42), 1,000 positive-gene resamples within each subset)
                for lab, sub in (("shared", a), ("comparator", b)):
                    r = np.random.default_rng(42)
                    e = distance_bins(sub, low)
                    x = [matched_auroc(sub, low, col, r, e, resample_pos_genes=True)[0] for _ in range(1000)]
                    row[f"b1_rerun_auroc_{lab}"] = np.nanmean(x)
            s2.append(row)
        print("s2", model, g, flush=True)
pd.DataFrame(s2).to_csv(f"{out_dir}/b8_unmatched_auroc.tsv", sep="\t", index=False)

# ---------------- s3: matched AUROC ----------------
s3, s3_pairs, s3_order, store = [], [], [], {}
for mi, (model, col) in enumerate(MODELS.items()):
    for gi, g in enumerate(GROUPS):
        pos, low, _ = split_classes(df[df.ancestry == g], col, 0.9)
        for di, defn in enumerate(DEFS):
            sh, sp = split(pos, defn)
            for ri, direction in enumerate(("specific_to_shared", "shared_to_specific")):
                src, tgt = (sp, sh) if direction == "specific_to_shared" else (sh, sp)
                raw = greedy_match(src, tgt, seed=1)
                pairs = [(j, i) if direction == "specific_to_shared" else (i, j) for i, j in raw]  # (shared idx, specific idx)
                row = dict(model=model, group=g, definition=defn, direction=direction, n_shared=len(sh), n_comparator=len(sp),
                           n_source=len(src), n_matched_pairs=len(pairs), share_source_matched=len(pairs) / len(src) if len(src) else np.nan)
                if len(pairs) < 5:
                    row["note"] = "fewer than 5 matched pairs"
                    s3.append(row)
                    continue
                cl = pair_clusters(pairs, pos, prefix=f"{g}_")
                ms = pos.loc[[p[0] for p in pairs]].copy()
                mc = pos.loc[[p[1] for p in pairs]].copy()
                ms["cluster"], mc["cluster"] = ms.index.map(cl), mc.index.map(cl)
                es, ec = distance_bins(ms, low), distance_bins(mc, low)
                rng = np.random.default_rng([11, gi, mi, di, ri])
                jb = joint_boot([ms, mc], [es, ec], low, col, rng, B_JOINT)
                d = jb[:, 0] - jb[:, 1]
                store[(model, defn, direction, g)] = dict(ms=ms, mc=mc, es=es, ec=ec, low=low, d=d, n=len(pairs))
                for lab, m in (("shared", ms), ("comparator", mc)):
                    row[f"median_abs_z_{lab}"] = m["abs_z"].median()
                    row[f"median_abs_tss_distance_{lab}"] = m["distance_to_tss"].abs().median()
                    row[f"share_maf_ge_005_{lab}"] = (m["maf"] >= 0.05).mean()
                    row[f"share_accessible_{lab}"] = m["accessible"].mean()
                    row[f"median_abs_score_{lab}"] = np.abs(m[col]).median()
                row["n_clusters"] = len(set(cl.values()))
                row.update(auroc_shared=np.nanmean(jb[:, 0]), shared_ci_low=ci(jb[:, 0])[0], shared_ci_high=ci(jb[:, 0])[1],
                           auroc_comparator=np.nanmean(jb[:, 1]), comparator_ci_low=ci(jb[:, 1])[0], comparator_ci_high=ci(jb[:, 1])[1],
                           diff_mean=np.nanmean(d), diff_ci_low=ci(d)[0], diff_ci_high=ci(d)[1], diff_bootstrap_p=bootstrap_p(d))
                # matching-order sensitivity: point difference (comparison-only draws) over further source orders
                diffs, npairs = [], []
                ro = np.random.default_rng([13, gi, mi, di, ri])
                for seed in range(2, 2 + N_ORDERS):
                    rr = greedy_match(src, tgt, seed=seed)
                    pp = [(j, i) if direction == "specific_to_shared" else (i, j) for i, j in rr]
                    if len(pp) < 5:
                        continue
                    a_, b_ = pos.loc[[p[0] for p in pp]], pos.loc[[p[1] for p in pp]]
                    ea_, eb_ = distance_bins(a_, low), distance_bins(b_, low)
                    xa = np.nanmean([matched_auroc(a_, low, col, ro, ea_)[0] for _ in range(B_ORDER)])
                    xb = np.nanmean([matched_auroc(b_, low, col, ro, eb_)[0] for _ in range(B_ORDER)])
                    diffs.append(xa - xb); npairs.append(len(pp))
                    s3_order.append(dict(model=model, group=g, definition=defn, direction=direction, order_seed=seed, n_matched_pairs=len(pp),
                                         auroc_shared=xa, auroc_comparator=xb, diff=xa - xb))
                row.update(order_diff_min=np.min(diffs), order_diff_median=np.median(diffs), order_diff_max=np.max(diffs),
                           order_n_pairs_min=int(np.min(npairs)), order_n_pairs_max=int(np.max(npairs)))
                s3.append(row)
                for (i, j) in pairs:
                    s3_pairs.append(dict(model=model, group=g, definition=defn, direction=direction,
                                         shared_key=pos.loc[i, "key"], shared_abs_z=pos.loc[i, "abs_z"], shared_score=pos.loc[i, col],
                                         comparator_key=pos.loc[j, "key"], comparator_abs_z=pos.loc[j, "abs_z"], comparator_score=pos.loc[j, col],
                                         tss_bin=pos.loc[i, "tss_distance_bin"], maf_bin=str(pos.loc[i, "maf_bin"]), cluster=cl[i]))
        print("s3", model, g, flush=True)
pd.DataFrame(s3).to_csv(f"{out_dir}/b8_matched_auroc.tsv", sep="\t", index=False)
pd.DataFrame(s3_pairs).to_csv(f"{out_dir}/b8_matched_pairs.tsv", sep="\t", index=False)
pd.DataFrame(s3_order).to_csv(f"{out_dir}/b8_matching_order_sensitivity.tsv", sep="\t", index=False)

# ---------------- s4: regression ----------------
s4 = []
FORMS = {"crude": "y ~ x",
         "covariates": "y ~ x + C(tss_bin) + log_dist + maf + abs_z",
         "covariates_plus_accessible": "y ~ x + C(tss_bin) + log_dist + maf + abs_z + accessible"}
for model, col in MODELS.items():
    for g in GROUPS:
        pos = split_classes(df[df.ancestry == g], col, 0.9)[0]
        for defn in DEFS:
            a, b = split(pos, defn)
            sub = pd.concat([a.assign(x=1), b.assign(x=0)])
            sub["y"] = np.log10(np.abs(sub[col]) + 1e-6)
            for fname, f in FORMS.items():
                dd = sub.dropna(subset=["accessible"]) if "accessible" in f else sub
                fit = smf.ols(f, data=dd).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(dd["gene_id"])[0]})
                c, se = fit.params["x"], fit.bse["x"]
                s4.append(dict(model=model, group=g, definition=defn, formula=fname, n=len(dd), n_shared=int(dd.x.sum()), n_genes=dd.gene_id.nunique(),
                               log10_coef=c, fold=10 ** c, fold_ci_low=10 ** (c - 1.96 * se), fold_ci_high=10 ** (c + 1.96 * se), p=fit.pvalues["x"],
                               accessible_fold=10 ** fit.params["accessible"] if "accessible" in f else np.nan,
                               accessible_p=fit.pvalues["accessible"] if "accessible" in f else np.nan))
pd.DataFrame(s4).to_csv(f"{out_dir}/b8_regression.tsv", sep="\t", index=False)
print("s4 done", flush=True)

# ---------------- s5: pooled across groups ----------------
s5 = []
for mi, (model, col) in enumerate(MODELS.items()):
    for di, defn in enumerate(DEFS):
        for ri, direction in enumerate(("specific_to_shared", "shared_to_specific")):
            keys = [(model, defn, direction, g) for g in GROUPS]
            if not all(k in store for k in keys):
                continue
            # (i) within-group bootstraps combined, weights = matched pairs (treats the groups as independent)
            w = np.array([store[k]["n"] for k in keys], dtype=float)
            dmat = np.vstack([store[k]["d"] for k in keys])
            pooled_indep = (w[:, None] * dmat).sum(0) / w.sum()
            # heterogeneity: Cochran's Q with bootstrap variances
            m_ = np.nanmean(dmat, 1); v_ = np.nanvar(dmat, 1, ddof=1)
            ivw = (m_ / v_).sum() / (1 / v_).sum()
            Q = ((m_ - ivw) ** 2 / v_).sum()
            # (ii) one bootstrap over gene clusters that span groups (a variant shared by two groups is a positive in both)
            sets = []
            for k in keys:
                s = store[k]
                ms, mc = s["ms"].copy(), s["mc"].copy()
                sets.append((ms, mc, s["es"], s["ec"], s["low"]))
            allpos = pd.concat([pd.concat([x[0], x[1]]) for x in sets])
            # union-find over genes across groups: positives of the same gene, and the two members of a matched pair, share a cluster
            parent = {}

            def find(x):
                parent.setdefault(x, x)
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for x in sets:
                for (_, r1), (_, r2) in zip(x[0].iterrows(), x[1].iterrows()):
                    a1, a2 = find("g:" + r1["gene_id"]), find("g:" + r2["gene_id"])
                    if a1 != a2:
                        parent[a1] = a2
            for x in sets:
                x[0]["cluster"] = [find("g:" + v) for v in x[0]["gene_id"]]
                x[1]["cluster"] = [find("g:" + v) for v in x[1]["gene_id"]]
            clusters = np.unique(np.concatenate([np.concatenate([x[0]["cluster"].values, x[1]["cluster"].values]) for x in sets]))
            rng = np.random.default_rng([17, mi, di, ri])
            pooled_joint = np.full(B_JOINT, np.nan)
            for b in range(B_JOINT):
                counts = pd.Series(rng.choice(clusters, size=len(clusters), replace=True)).value_counts()
                num, den = 0.0, 0.0
                for (ms, mc, es, ec, low), wk in zip(sets, w):
                    xa = matched_auroc(resample(ms, counts), low, col, rng, es)[0]
                    xb = matched_auroc(resample(mc, counts), low, col, rng, ec)[0]
                    if np.isfinite(xa) and np.isfinite(xb):
                        num += wk * (xa - xb); den += wk
                pooled_joint[b] = num / den if den > 0 else np.nan
            s5.append(dict(analysis="matched_auroc_difference", model=model, definition=defn, direction=direction,
                           n_matched_pairs_total=int(w.sum()), group_diffs=";".join(f"{g}={v:.3f}" for g, v in zip(GROUPS, m_)),
                           cochran_Q=Q, heterogeneity_p=chi2.sf(Q, len(keys) - 1),
                           pooled_diff_indep=np.nanmean(pooled_indep), indep_ci_low=ci(pooled_indep)[0], indep_ci_high=ci(pooled_indep)[1],
                           indep_p=bootstrap_p(pooled_indep),
                           pooled_diff_cross_group_clusters=np.nanmean(pooled_joint), cross_ci_low=ci(pooled_joint)[0],
                           cross_ci_high=ci(pooled_joint)[1], cross_p=bootstrap_p(pooled_joint)))
    # pooled regression: group-specific covariates, gene-clustered SE; interaction test for heterogeneity
    for defn in DEFS:
        parts = []
        for g in GROUPS:
            pos = split_classes(df[df.ancestry == g], col, 0.9)[0]
            a, b = split(pos, defn)
            parts.append(pd.concat([a.assign(x=1), b.assign(x=0)]).assign(group=g))
        sub = pd.concat(parts)
        sub["y"] = np.log10(np.abs(sub[col]) + 1e-6)
        base = "C(group) + C(group):C(tss_bin) + C(group):log_dist + C(group):maf + C(group):abs_z"
        for fname, extra in (("covariates", ""), ("covariates_plus_accessible", " + C(group):accessible")):
            dd = sub.dropna(subset=["accessible"]) if extra else sub
            cl = pd.factorize(dd["gene_id"])[0]
            fit = smf.ols(f"y ~ x + {base}{extra}", data=dd).fit(cov_type="cluster", cov_kwds={"groups": cl})
            fit_i = smf.ols(f"y ~ x * C(group) + {base}{extra}", data=dd).fit(cov_type="cluster", cov_kwds={"groups": cl})
            inter = [p for p in fit_i.params.index if p.startswith("x:")]
            R = np.zeros((len(inter), len(fit_i.params)))
            for r_, p_ in enumerate(inter):
                R[r_, list(fit_i.params.index).index(p_)] = 1.0
            wt = fit_i.wald_test(R, scalar=True)
            c, se = fit.params["x"], fit.bse["x"]
            s5.append(dict(analysis=f"regression_{fname}", model=model, definition=defn, n=len(dd), n_genes=dd.gene_id.nunique(),
                           pooled_fold=10 ** c, fold_ci_low=10 ** (c - 1.96 * se), fold_ci_high=10 ** (c + 1.96 * se), pooled_p=fit.pvalues["x"],
                           interaction_wald_p=float(wt.pvalue)))
pd.DataFrame(s5).to_csv(f"{out_dir}/b8_pooled.tsv", sep="\t", index=False)
print("s5 done", flush=True)
