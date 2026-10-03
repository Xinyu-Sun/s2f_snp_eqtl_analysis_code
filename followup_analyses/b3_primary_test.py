"""B3. One pre-specified test of the group difference.

Model, fitted within each group (equivalent to a pooled logistic model in which every term interacts with group):
    high-PIP status ~ x + natural cubic spline(log10 TSS distance, 4 df, centred) + log10(MAF)
x = log10(|score| + 1e-6), standardised with the mean and SD of the pooled benchmark set of all three groups, so the
slope (log-odds per SD of score magnitude) is on the same scale in every group. Standard errors are cluster-robust by gene.
Group contrasts: difference of slopes, SE = sqrt(se_a^2 + se_b^2) (independent samples), two-sided Wald p.
Primary: PIP >= 0.9 versus low-PIP comparison variants, per model. Other thresholds and comparison sets are sensitivity rows.
Also: leave-one-gene-out refits for the AA slope.

Outputs: b3_group_slopes.tsv, b3_slope_contrasts.tsv, b3_leave_one_gene_out_AA.tsv
"""
import sys, numpy as np, pandas as pd
import statsmodels.api as sm
import patsy
from scipy import stats
from fu_common import load_pairs, split_classes, GROUPS, MODELS

pairs_path, out_dir = sys.argv[1], sys.argv[2]
df = load_pairs(pairs_path)

def build(col, t, comp_name):
    parts = []
    for g in GROUPS:
        pos, low, mid = split_classes(df[df["ancestry"] == g], col, t)
        comp = low if comp_name == "low_pip" else mid
        parts.append(pd.concat([pos.assign(y=1), comp.assign(y=0)]).assign(group=g))
    d = pd.concat(parts).dropna(subset=["maf", "log_dist"]).copy()
    d = d[d["maf"] > 0]
    lx = np.log10(np.abs(d[col]) + 1e-6)
    d["x"] = (lx - lx.mean()) / lx.std()
    d["log_maf"] = np.log10(d["maf"])
    return d

def fit(d, formula="x + cr(log_dist, df=4, constraints='center') + log_maf"):
    """Logistic fit; standard error of the score slope from a cluster (gene) sandwich computed directly."""
    X = patsy.dmatrix(formula, d, return_type="dataframe")
    y = d["y"].values.astype(float)
    m = sm.GLM(y, X, family=sm.families.Binomial()).fit(maxiter=200)
    if (not m.converged) or abs(m.params["x"]) > 20:
        raise RuntimeError("logistic fit did not converge")
    Xv = X.values; p = m.predict(X).values if hasattr(m.predict(X), "values") else m.predict(X)
    bread = np.linalg.inv((Xv * (p * (1 - p))[:, None]).T @ Xv)
    sc = Xv * (y - p)[:, None]
    cl = pd.factorize(d["gene_id"])[0]; G = cl.max() + 1
    S = np.zeros((G, Xv.shape[1])); np.add.at(S, cl, sc)
    V = bread @ (S.T @ S) @ bread * (G / (G - 1))
    j = list(X.columns).index("x")
    return m.params["x"], float(np.sqrt(V[j, j]))


slopes, contrasts, loo = [], [], []
for model, col in MODELS.items():
    for t in (0.9, 0.5):
        for comp_name in ("low_pip", "intermediate_pip"):
            d = build(col, t, comp_name); est = {}
            for g in GROUPS:
                dg = d[d.group == g]
                try: b, se = fit(dg)
                except Exception as e: print("fit failed:", model, t, comp_name, g, repr(e)[:120], flush=True); b, se = np.nan, np.nan
                est[g] = (b, se)
                slopes.append(dict(model=model, pip_threshold=t, comparison=comp_name, primary=(t == 0.9 and comp_name == "low_pip"), group=g,
                                   n_pos=int(dg.y.sum()), n_comp=int((dg.y == 0).sum()), n_genes=dg.gene_id.nunique(),
                                   log_odds_per_sd=b, se=se, odds_ratio_per_sd=np.exp(b), or_ci_low=np.exp(b - 1.96 * se), or_ci_high=np.exp(b + 1.96 * se)))
            for a, b_ in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
                diff = est[a][0] - est[b_][0]; se = np.sqrt(est[a][1] ** 2 + est[b_][1] ** 2); z = diff / se
                contrasts.append(dict(model=model, pip_threshold=t, comparison=comp_name, primary=(t == 0.9 and comp_name == "low_pip"), contrast=f"{a}_minus_{b_}",
                                      slope_difference=diff, se=se, ci_low=diff - 1.96 * se, ci_high=diff + 1.96 * se, z=z, p=2 * stats.norm.sf(abs(z)),
                                      ratio_of_odds_ratios=np.exp(diff)))
            if t == 0.9 and comp_name == "low_pip":
                dg = d[d.group == "AA"]; full = est["AA"][0]
                for gene in dg.loc[dg.y == 1, "gene_id"].unique():
                    try: b, se = fit(dg[dg.gene_id != gene])
                    except Exception: b, se = np.nan, np.nan
                    loo.append(dict(model=model, gene_left_out=gene, aa_slope_full=full, aa_slope_without_gene=b, se=se,
                                    nhw_slope=est["NHW"][0], ch_slope=est["CH"][0]))
pd.DataFrame(slopes).to_csv(f"{out_dir}/b3_group_slopes.tsv", sep="\t", index=False)
pd.DataFrame(contrasts).to_csv(f"{out_dir}/b3_slope_contrasts.tsv", sep="\t", index=False)
pd.DataFrame(loo).to_csv(f"{out_dir}/b3_leave_one_gene_out_AA.tsv", sep="\t", index=False); print("B3 done")
