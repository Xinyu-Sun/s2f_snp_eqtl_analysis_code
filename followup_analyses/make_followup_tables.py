"""Regenerate the follow-up supplementary tables (tables/aa_advantage_*.tex and the review-analysis tables) from a results
directory. Captions and labels of existing table files are kept; only the tabular body is rebuilt.
Usage: python make_followup_tables.py <results_dir> <tables_dir>
"""
import re, sys
from pathlib import Path
import numpy as np, pandas as pd

R = Path(sys.argv[1]); T = Path(sys.argv[2])
GROUPS = ["AA", "CH", "NHW"]; MODELS = [("borzoi", "Borzoi"), ("alphagenome", "AlphaGenome")]; ML = dict(MODELS)
PIP = {0.9: "$\\geq 0.9$", 0.5: "$\\geq 0.5$"}; COMP = {"low_pip": "low-PIP", "intermediate_pip": "intermediate-PIP"}
CON = {"AA_minus_NHW": "AA $-$ NHW", "AA_minus_CH": "AA $-$ CH", "CH_minus_NHW": "CH $-$ NHW",
       "AA_vs_NHW": "AA vs NHW", "AA_vs_CH": "AA vs CH", "CH_vs_NHW": "CH vs NHW"}

def rd(name): return pd.read_csv(R / name, sep="\t")
def f3(x): return "nan" if pd.isna(x) else f"{x:.3f}"
def f4(x): return "nan" if pd.isna(x) else f"{x:.4f}"
def ci3(m, lo, hi): return f"{m:.3f} ({lo:.3f}--{hi:.3f})"
def fp(p): return "nan" if pd.isna(p) else f"{p:.2g}"
def pct(x): return f"{100*x:.0f}\\%"
def bp(p): return "nan" if pd.isna(p) else ("$< 0.001$" if p < 0.0005 else f"{p:.3f}")  # a bootstrap p that would print as 0.000 prints as < 0.001

def existing(name):
    p = T / name
    if not p.exists(): return None, None, None
    t = p.read_text()
    cap = re.search(r"\\caption\{(.*)\}\n\\label", t, flags=re.S); lab = re.search(r"\\label\{([^}]+)\}", t)
    pre = t[: t.index("\\begin{tabular}")]; post = t[t.index("\\end{tabular}"):]
    return pre, post, (cap.group(1) if cap else None)

def write(name, header_lines, body_lines, default_caption, default_label, colspec, resize=True, small="\\scriptsize", extra=""):
    pre, post, _ = existing(name)
    if pre is None:
        pre = ("\\begin{table}[p]\n\\centering\n" + small + "\n" + extra + "\\caption{" + default_caption + "}\n\\label{" + default_label + "}\n"
               + ("\\resizebox{\\textwidth}{!}{%\n" if resize else ""))
        post = "\\end{tabular}" + ("%\n}" if resize else "") + "\n\\end{table}\n"
    body = "\\begin{tabular}{" + colspec + "}\n\\toprule\n" + "\n".join(header_lines) + "\n\\midrule\n" + "\n".join(body_lines) + "\n\\bottomrule\n"
    (T / name).write_text(pre + body + post); print("wrote", name, len(body_lines), "rows")

# ---------- score anatomy ----------
sc = rd("a1_score_by_class.tsv"); ra = rd("a1_pos_vs_low_ratio.tsv"); gc = rd("a1_group_contrasts.tsv"); ch = rd("a1_positive_characteristics.tsv")
rows = []
for t in (0.9, 0.5):
    for m, mn in MODELS:
        for g in GROUPS:
            s = sc[(sc.model == m) & (sc.pip_threshold == t) & (sc.group == g)].set_index("variant_class")
            r = ra[(ra.model == m) & (ra.pip_threshold == t) & (ra.group == g)].iloc[0]
            p, l, i = s.loc["positive"], s.loc["low_pip"], s.loc["intermediate_pip"]
            rows.append(f"{PIP[t]} & {mn} & {g} & {f4(p.median_abs_score)} ({f4(p.ci_low)}--{f4(p.ci_high)}); {int(p.n)} & {f4(l.median_abs_score)} ({f4(l.ci_low)}--{f4(l.ci_high)}); {int(l.n)} & {f4(i.median_abs_score)}; {int(i.n)} & {r.ratio_pos_over_low:.1f} ({r.ci_low:.1f}--{r.ci_high:.1f}) \\\\")
    if t == 0.9: rows.append("\\midrule")
rows += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textbf{Group contrasts among high-PIP variants}} \\\\",
         "\\textbf{PIP} & \\textbf{Model} & \\textbf{Contrast} & \\textbf{Fold difference (95\\% CI)} & \\textbf{Regression $p$} & \\textbf{Mann--Whitney $p$} & \\textbf{$n$} \\\\", "\\midrule"]
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for _, r in gc[(gc.model == m) & (gc.pip_threshold == t)].iterrows():
            rows.append(f"{PIP[t]} & {mn} & {CON.get(r.contrast, r.contrast)} & {r.ols_fold:.2f} ({10**r.ols_ci_low:.2f}--{10**r.ols_ci_high:.2f}) & {fp(r.ols_p)} & {fp(r.mannwhitney_p)} & {int(r.n_a)} vs {int(r.n_b)} \\\\")
rows += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textbf{Characteristics of high-PIP variants}} \\\\",
         "\\textbf{PIP} & \\textbf{Model} & \\textbf{Group} & \\textbf{Median $|z|$ (share $|z| > 8$)} & \\textbf{Median PIP} & \\textbf{Share 0--3~kb; median MAF} & \\textbf{Spearman $|$score$|$ vs $|z|$} \\\\", "\\midrule"]
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for g in GROUPS:
            x = ch[(ch.model == m) & (ch.pip_threshold == t) & (ch.group == g)]
            if x.empty: continue
            x = x.iloc[0]
            rows.append(f"{PIP[t]} & {mn} & {g} & {x.median_abs_z:.1f} ({pct(x.share_abs_z_gt8)}) & {x.median_pip:.3f} & {pct(x.share_0_3kb)}; {x.median_maf:.2f} & {x.spearman_abs_score_vs_abs_z:.2f} \\\\")
write("aa_advantage_score_anatomy.tex", ["\\textbf{PIP} & \\textbf{Model} & \\textbf{Group} & \\textbf{High-PIP median (95\\% CI); $n$} & \\textbf{Low-PIP median (95\\% CI); $n$} & \\textbf{Intermediate-PIP median; $n$} & \\textbf{Ratio high/low (95\\% CI)} \\\\"],
      rows, "Score magnitude of high-PIP variants and comparison variants by group.", "stab:score_anatomy", "@{}lllrrrr@{}")

# ---------- joint bootstrap ----------
ab = rd("a2_auroc_bootstrap.tsv"); gd = rd("a2_group_differences.tsv"); cp = rd("a2_cross_pool_auroc.tsv")
rows = []
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for c in ("low_pip", "intermediate_pip"):
            for g in GROUPS:
                r = ab[(ab.model == m) & (ab.pip_threshold == t) & (ab.comparison == c) & (ab.group == g)].iloc[0]
                rows.append(f"{PIP[t]} & {mn} & {COMP[c]} & {g} & {int(r.n_pos)} ({int(r.n_pos_genes)}) & {ci3(r.auroc_comp_only_mean, r.comp_only_ci_low, r.comp_only_ci_high)} & {ci3(r.auroc_joint_mean, r.joint_ci_low, r.joint_ci_high)} \\\\")
rows += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textbf{Group differences under joint resampling}} \\\\",
         "\\textbf{PIP} & \\textbf{Model} & \\textbf{Comparison} & \\textbf{Contrast} & \\textbf{Difference} & \\textbf{95\\% CI} & \\textbf{Bootstrap $p$} \\\\", "\\midrule"]
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for c in ("low_pip", "intermediate_pip"):
            for _, r in gd[(gd.model == m) & (gd.pip_threshold == t) & (gd.comparison == c)].iterrows():
                rows.append(f"{PIP[t]} & {mn} & {COMP[c]} & {CON[r.contrast]} & {r.diff_mean:.3f} & {r.ci_low:.3f} to {r.ci_high:.3f} & {bp(r.bootstrap_two_sided_p)} \\\\")
rows += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textbf{Cross-pool evaluation (PIP $\\geq 0.9$ positives; AUROC, 95\\% CI)}} \\\\",
         "\\textbf{Model} & \\textbf{Comparison} & \\textbf{Positives from} & \\textbf{vs AA pool} & \\textbf{vs CH pool} & \\textbf{vs NHW pool} & \\\\", "\\midrule"]
for m, mn in MODELS:
    for c in ("low_pip", "intermediate_pip"):
        for g in GROUPS:
            cells = []
            for gj in GROUPS:
                r = cp[(cp.model == m) & (cp.comparison == c) & (cp.positives_from == g) & (cp.comparison_pool_from == gj)].iloc[0]
                cells.append(ci3(r.auroc_mean, r.ci_low, r.ci_high))
            rows.append(f"{mn} & {COMP[c]} & {g} & " + " & ".join(cells) + " &  \\\\")
write("aa_advantage_joint_bootstrap.tex", ["\\textbf{PIP} & \\textbf{Model} & \\textbf{Comparison} & \\textbf{Group} & \\textbf{$n$ positives (genes)} & \\textbf{AUROC, comparison-only (95\\% CI)} & \\textbf{AUROC, joint (95\\% CI)} \\\\"],
      rows, "Distance-matched AUROC with joint resampling of positives and comparison variants, group differences, and cross-pool evaluation.", "stab:joint_bootstrap", "@{}lllllrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")

# ---------- shared genes ----------
sg = rd("a7_shared_gene_auroc.tsv"); rows = []
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for g in GROUPS:
            x = sg[(sg.model == m) & (sg.pip_threshold == t) & (sg.group == g)]
            if x.empty: continue
            r = x.iloc[0]; rows.append(f"{PIP[t]} & {mn} & {g} & {int(r.n_pos)} & {int(r.n_low)} & {ci3(r.auroc_mean, r.ci_low, r.ci_high)} \\\\")
n_shared = int(sg.n_shared_genes.iloc[0])
pre, post, cap = existing("aa_advantage_shared_genes.tex")
if cap:
    t = (T / "aa_advantage_shared_genes.tex").read_text(); t2 = re.sub(r"\}\s*\d{2,4} genes;", "} " + f"{n_shared} genes;", t, count=1)
    (T / "aa_advantage_shared_genes.tex").write_text(t2)
write("aa_advantage_shared_genes.tex", ["\\textbf{PIP} & \\textbf{Model} & \\textbf{Group} & \\textbf{$n$ positives} & \\textbf{$n$ comparison} & \\textbf{AUROC (95\\% CI)} \\\\"], rows,
      f"Distance-matched AUROC restricted to genes represented in the fine-mapping pools of all three groups. {n_shared} genes.", "stab:shared_genes", "@{}lllrrr@{}", resize=False)

# ---------- decile enrichment ----------
de = rd("a3_decile_enrichment.tsv"); rows = []
def orfmt(r):
    if np.isinf(r.exact_or) or pd.isna(r.exact_or): return f"$>${r.exact_ci_low:.1f} (no high-PIP variant in the bottom decile)"
    hi = "$\\infty$" if np.isinf(r.exact_ci_high) else f"{r.exact_ci_high:.1f}"
    return f"{r.exact_or:.1f} ({r.exact_ci_low:.1f}--{hi})"
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for g in GROUPS:
            r = de[(de.model == m) & (de.pip_threshold == t) & (de.group == g)].iloc[0]
            rows.append(f"{PIP[t]} & {mn} & {g} & {int(r.pos_in_top)} / {int(r.pos_in_bottom)} of {int(r.n_top)} & {orfmt(r)} & {r.mh_or_tss_stratified:.1f} & {100*r.share_pos_in_top_decile:.0f}\\% ({100*r.share_top_ci_low:.0f}--{100*r.share_top_ci_high:.0f}) \\\\")
write("aa_advantage_decile_enrichment.tex", ["\\textbf{PIP} & \\textbf{Model} & \\textbf{Group} & \\textbf{High-PIP in top / bottom decile} & \\textbf{Exact odds ratio (95\\% CI)} & \\textbf{MH odds ratio} & \\textbf{Share of high-PIP in top decile (95\\% CI)} \\\\"],
      rows, "Enrichment of high-PIP variants in the top versus bottom decile of score magnitude.", "stab:decile_enrichment", "@{}lllrrrr@{}", resize=True)

# ---------- matched positives ----------
mp = rd("a4_matched_positives.tsv"); mb = rd("a4b_matching_covariate_sets.tsv"); rows = []
LAB = {"tss_only": "TSS bin", "tss_maf": "TSS bin + MAF bin", "tss_z": "TSS bin + $|z|$", "z_only": "$|z|$ only", "tss_maf_z": "TSS bin + MAF bin + $|z|$"}
for m, mn in MODELS:
    for c in ("AA_vs_NHW", "AA_vs_CH"):
        for _, b in mb[(mb.model == m) & (mb.contrast == c)].iterrows():
            rows.append(f"{mn} & {CON[c]} & {LAB.get(b.covariates, b.covariates)} & {int(b.n_pairs)} & {b.median_abs_z_aa:.1f} / {b.median_abs_z_other:.1f} & {f4(b.median_abs_score_aa)} / {f4(b.median_abs_score_other)} & {b.auroc_aa:.3f} & {ci3(b.auroc_other, b.other_ci_low, b.other_ci_high)} & {b['diff']:+.3f} \\\\")
write("aa_advantage_matched_positives.tex", ["\\textbf{Model} & \\textbf{Contrast} & \\textbf{Matching covariates} & \\textbf{Pairs} & \\textbf{Median $|z|$ AA / other} & \\textbf{Median $|$score$|$ AA / other} & \\textbf{AUROC AA} & \\textbf{AUROC other (95\\% CI)} & \\textbf{Difference} \\\\"],
      rows, "Distance-matched AUROC after one-to-one matching of African American high-PIP variants to Non-Hispanic White or Caribbean Hispanic high-PIP variants.", "stab:matched_positives", "@{}lllrrrrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")

# ---------- LD proxies ----------
ls = rd("a5_ld_partner_summary.tsv"); lm = rd("a5_ld_partner_models_and_strata.tsv"); rows = []
for t in (0.9, 0.5):
    for g in GROUPS:
        r = ls[(ls.pip_threshold == t) & (ls.group == g)].iloc[0]; cells = []
        for st in ("zero_partners_r2_0.8", "one_or_more_partners_r2_0.8"):
            for m, mn in MODELS:
                x = lm[(lm.pip_threshold == t) & (lm.model == m) & (lm.group == g) & (lm.stratum == st)]
                if x.empty or pd.isna(x.iloc[0].get("auroc_mean", np.nan)): cells.append(f"-- ({int(x.iloc[0].n_pos) if len(x) else 0})"); continue
                x = x.iloc[0]; cells.append(f"{ci3(x.auroc_mean, x.ci_low, x.ci_high)[:-1]}; {int(x.n_pos)})")
        rows.append(f"{PIP[t]} & {g} & {int(r.n_in_bim)} & {100*r.share_zero_partners_r2_0p8_200kb:.1f} & {r.mean_n80_200kb:.2f} & {r.spearman_abs_borzoi_vs_n80:.2f} & {r.spearman_abs_alphagenome_vs_n80:.2f} & " + " & ".join(cells) + " \\\\")
    if t == 0.9: rows.append("\\midrule")
write("aa_advantage_ld_proxies.tex", ["\\textbf{PIP} & \\textbf{Group} & \\textbf{$n$} & \\textbf{No LD proxy (\\%)} & \\textbf{Mean LD proxies} & \\textbf{$\\rho$ Borzoi} & \\textbf{$\\rho$ AlphaGenome} & \\multicolumn{2}{c}{\\textbf{AUROC, no LD proxy}} & \\multicolumn{2}{c}{\\textbf{AUROC, $\\geq$1 LD proxy}} \\\\",
      "\\cmidrule(lr){8-9}\\cmidrule(lr){10-11}", " & & & & & & & \\textbf{Borzoi ($n$)} & \\textbf{AlphaGenome ($n$)} & \\textbf{Borzoi ($n$)} & \\textbf{AlphaGenome ($n$)} \\\\"],
      rows, "Variants in strong LD with high-PIP variants and distance-matched AUROC by LD stratum.", "stab:ld_proxies", "@{}llrrrrrrrrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")
rows = []
for t in (0.9, 0.5):
    for m, mn in MODELS:
        for c in ("AA_vs_NHW", "AA_vs_CH"):
            x = lm[(lm.pip_threshold == t) & (lm.model == m) & (lm.contrast == c)].iloc[0]
            rows.append(f"{PIP[t]} & {mn} & {CON[c]} & {10**x.group_coef_unadjusted:.2f} & {fp(x.p_unadjusted)} & {10**x.group_coef_adj_n80:.2f} & {fp(x.p_adj_n80)} & {x.n80_coef:.3f} ({fp(x.n80_p)}) & {pct(x.share_group_coef_explained)} \\\\")
write("aa_advantage_ld_proxy_regression.tex", ["\\textbf{PIP} & \\textbf{Model} & \\textbf{Contrast} & \\textbf{Fold (unadj.)} & \\textbf{$p$} & \\textbf{Fold (adj.)} & \\textbf{$p$} & \\textbf{LD-proxy coef.\\ ($p$)} & \\textbf{Explained} \\\\"],
      rows, "Group difference in score magnitude before and after adjustment for the number of LD proxies.", "stab:ld_proxy_regression", "@{}lllrrrrrr@{}", resize=True)

# ---------- frequency classes ----------
fq = rd("a6_freq_class_summary.tsv"); rows = []
FL = {"common_in_all_groups": "Common (MAF $\\geq$ 0.05) in both other cohorts", "low_frequency_in_another_group": "MAF $<$ 0.05 in another cohort", "absent_or_MAC<10_in_another_group": "Absent or MAC $<$ 10 in another cohort"}
for t in (0.9, 0.5):
    for g in GROUPS:
        sub = fq[(fq.pip_threshold == t) & (fq.group == g)]
        for _, r in sub.iterrows():
            def au(a, lo, hi): return "--" if pd.isna(a) else ci3(a, lo, hi)
            rows.append(f"{PIP[t]} & {g} & {FL.get(r.freq_class, r.freq_class)} & {int(r.n)} ({100*r.share:.0f}) & {f4(r.median_abs_borzoi)} & {f4(r.median_abs_alphagenome)} & {au(r.auroc_borzoi, r.auroc_borzoi_ci_low, r.auroc_borzoi_ci_high)} & {au(r.auroc_alphagenome, r.auroc_alphagenome_ci_low, r.auroc_alphagenome_ci_high)} \\\\")
    if t == 0.9: rows.append("\\midrule")
write("aa_advantage_freq_class.tex", ["\\textbf{PIP} & \\textbf{Group} & \\textbf{Class in other cohorts} & \\textbf{$n$ (\\%)} & \\textbf{Median $|$Borzoi$|$} & \\textbf{Median $|$AlphaGenome$|$} & \\textbf{AUROC Borzoi} & \\textbf{AUROC AlphaGenome} \\\\"],
      rows, "High-PIP variants classified by allele frequency in the other two cohorts.", "stab:freq_class", "@{}lllrrrrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")

# ---------- accessible chromatin ----------
asu = rd("a9_accessible_summary.tsv"); are = rd("a9_accessible_regression.tsv"); aau = rd("a9_accessible_auroc.tsv"); rows = []
for t in (0.9, 0.5):
    for m, mn in MODELS:
        for g in GROUPS:
            r = asu[(asu.pip_threshold == t) & (asu.model == m) & (asu.group == g)].iloc[0]
            def au(st):
                x = aau[(aau.pip_threshold == t) & (aau.model == m) & (aau.group == g) & (aau.stratum == st)].iloc[0]
                return "--" if pd.isna(x.get("auroc_mean", np.nan)) else ci3(x.auroc_mean, x.ci_low, x.ci_high)
            rows.append(f"{PIP[t]} & {mn} & {g} & {int(r.n_accessible)} of {int(r.n_with_call)} ({100*r.share_accessible:.0f}) & {f4(r.median_abs_score_accessible)} & {f4(r.median_abs_score_not_accessible)} & {au('accessible')} & {au('not_accessible')} \\\\")
    if t == 0.9: rows.append("\\midrule")
rows += ["\\midrule", "\\textbf{PIP} & \\textbf{Model} & \\textbf{Contrast} & \\textbf{Fold, unadjusted; $p$} & \\textbf{Fold, adj.\\ accessibility; $p$} & \\textbf{Fold, adj.\\ accessibility + LD; $p$} & \\textbf{Accessibility fold; $p$} & \\textbf{Explained (acc.; acc.+LD)} \\\\", "\\midrule"]
for t in (0.9, 0.5):
    for m, mn in MODELS:
        for c in ("AA_vs_NHW", "AA_vs_CH"):
            r = are[(are.pip_threshold == t) & (are.model == m) & (are.contrast == c)].iloc[0]
            rows.append(f"{PIP[t]} & {mn} & {CON[c]} & {r.fold_unadj:.2f} ({r.fold_unadj_lo:.2f}--{r.fold_unadj_hi:.2f}); {fp(r.p_unadj)} & {r.fold_adj_acc:.2f} ({r.fold_adj_acc_lo:.2f}--{r.fold_adj_acc_hi:.2f}); {fp(r.p_adj_acc)} & {r.fold_adj_acc_ld:.2f} ({r.fold_adj_acc_ld_lo:.2f}--{r.fold_adj_acc_ld_hi:.2f}); {fp(r.p_adj_acc_ld)} & {r.acc_fold:.1f}; {fp(r.acc_p)} & {pct(r.share_explained_acc)}; {pct(r.share_explained_acc_ld)} \\\\")
write("aa_advantage_accessible_chromatin.tex", ["\\textbf{PIP} & \\textbf{Model} & \\textbf{Group} & \\textbf{Accessible, $n$ (\\%)} & \\textbf{Median $|$score$|$ inside} & \\textbf{Median $|$score$|$ outside} & \\textbf{AUROC inside (95\\% CI)} & \\textbf{AUROC outside (95\\% CI)} \\\\"],
      rows, "Accessible-chromatin context of high-PIP variants, score magnitude, and distance-matched AUROC.", "stab:accessible", "@{}lllrrrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")

# ---------- NEW: primary logistic test (b3) ----------
b3 = rd("b3_group_slopes.tsv"); b3c = rd("b3_slope_contrasts.tsv"); loo = rd("b3_leave_one_gene_out_AA.tsv"); rows = []
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for c in ("low_pip", "intermediate_pip"):
            for g in GROUPS:
                r = b3[(b3.model == m) & (b3.pip_threshold == t) & (b3.comparison == c) & (b3.group == g)].iloc[0]
                rows.append(f"{PIP[t]} & {mn} & {COMP[c]} & {g} & {int(r.n_pos)} / {int(r.n_comp)} & {r.log_odds_per_sd:.2f} ({r.se:.2f}) & {r.odds_ratio_per_sd:.2f} ({r.or_ci_low:.2f}--{r.or_ci_high:.2f}) \\\\")
rows += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textbf{Differences between groups (slope of the first group minus the second)}} \\\\",
         "\\textbf{PIP} & \\textbf{Model} & \\textbf{Comparison} & \\textbf{Contrast} & \\textbf{Difference (SE)} & \\textbf{95\\% CI} & \\textbf{$p$} \\\\", "\\midrule"]
for m, mn in MODELS:
    for t in (0.9, 0.5):
        for c in ("low_pip", "intermediate_pip"):
            for _, r in b3c[(b3c.model == m) & (b3c.pip_threshold == t) & (b3c.comparison == c)].iterrows():
                rows.append(f"{PIP[t]} & {mn} & {COMP[c]} & {CON[r.contrast]} & {r.slope_difference:.2f} ({r.se:.2f}) & {r.ci_low:.2f} to {r.ci_high:.2f} & {fp(r.p)} \\\\")
rows += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textbf{Leave-one-gene-out refits of the AA slope (PIP $\\geq 0.9$, low-PIP comparison)}} \\\\",
         "\\textbf{Model} & \\textbf{Genes left out} & \\textbf{Full AA slope} & \\textbf{Range without one gene} & \\textbf{NHW slope} & \\textbf{CH slope} & \\\\", "\\midrule"]
for m, mn in MODELS:
    x = loo[loo.model == m]
    rows.append(f"{mn} & {len(x)} & {x.aa_slope_full.iloc[0]:.2f} & {x.aa_slope_without_gene.min():.2f} to {x.aa_slope_without_gene.max():.2f} & {x.nhw_slope.iloc[0]:.2f} & {x.ch_slope.iloc[0]:.2f} & \\\\")
write("review_logistic_slopes.tex", ["\\textbf{PIP} & \\textbf{Model} & \\textbf{Comparison} & \\textbf{Group} & \\textbf{$n$ high-PIP / comparison} & \\textbf{Log-odds per SD (SE)} & \\textbf{Odds ratio per SD (95\\% CI)} \\\\"], rows,
      "Logistic regression of high-PIP status on score magnitude within each group. The predictor is $\\log_{10}|$score$|$ standardised on the pooled benchmark set of all three groups; the model adjusts for a natural cubic spline of $\\log_{10}$ TSS distance (4 df) and $\\log_{10}$ MAF, with standard errors clustered by gene. Group differences use the two groups' standard errors; the primary comparison is PIP $\\geq 0.9$ against low-PIP comparison variants.",
      "stab:logistic_slopes", "@{}lllllrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")

# ---------- NEW: baselines (b4) ----------
b4 = rd("b4_baseline_auroc.tsv"); b4d = rd("b4_baseline_group_differences.tsv"); rows = []
PL = [("score", "Model score"), ("accessible_only", "Accessible-chromatin overlap alone"), ("tss_proximity_only", "Proximity to the TSS alone"), ("annotation_model", "Accessibility + distance model"), ("annotation_plus_score", "Accessibility + distance model + score"), ("gain_from_score", "Gain from adding the score")]
for m, mn in MODELS:
    for k, kl in PL:
        cells = []
        for g in GROUPS:
            r = b4[(b4.model == m) & (b4.group == g) & (b4.predictor == k)].iloc[0]; cells.append(ci3(r.auroc_mean, r.ci_low, r.ci_high))
        rows.append(f"{mn} & {kl} & " + " & ".join(cells) + " \\\\")
    if m == "borzoi": rows.append("\\midrule")
rows += ["\\midrule", "\\multicolumn{5}{@{}l}{\\textbf{Group differences (difference, 95\\% CI, bootstrap $p$)}} \\\\",
         "\\textbf{Model} & \\textbf{Predictor} & \\textbf{AA $-$ NHW} & \\textbf{AA $-$ CH} & \\textbf{CH $-$ NHW} \\\\", "\\midrule"]
for m, mn in MODELS:
    for k, kl in PL:
        if k == "tss_proximity_only": continue
        cells = []
        for c in ("AA_minus_NHW", "AA_minus_CH", "CH_minus_NHW"):
            r = b4d[(b4d.model == m) & (b4d.predictor == k) & (b4d.contrast == c)].iloc[0]; cells.append(f"{r.diff_mean:.3f} ({r.ci_low:.3f} to {r.ci_high:.3f}; {bp(r.bootstrap_two_sided_p)})")
        rows.append(f"{mn} & {kl} & " + " & ".join(cells) + " \\\\")
n_note = b4[b4.predictor == "score"][["group", "n_pos_with_annotation", "n_comp_with_annotation"]].drop_duplicates("group")
write("review_annotation_baselines.tex", ["\\textbf{Model} & \\textbf{Predictor} & \\textbf{AA} & \\textbf{CH} & \\textbf{NHW} \\\\"], rows,
      "Distance-matched AUROC of annotation-only predictors compared with the model score, PIP $\\geq 0.9$ against low-PIP comparison variants, restricted to variants with a FILER annotation call. The accessibility + distance model is a logistic regression with predictions from five-fold cross-validation that keeps all variants of a gene in one fold. Intervals and group differences come from 500 joint resamples of positive genes and comparison variants, shared across predictors.",
      "stab:annotation_baselines", "@{}llrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")

# ---------- NEW: within gene (b2), random subsets and PIP weighting (b7) ----------
b2 = rd("b2_within_gene.tsv"); b2d = rd("b2_within_gene_differences.tsv"); rows = []
VL = {"same_gene": "Same gene", "same_gene_same_tss_bin": "Same gene and TSS-distance bin"}
for m, mn in MODELS:
    for v, vl in VL.items():
        for g in GROUPS:
            r = b2[(b2.model == m) & (b2.version == v) & (b2.group == g)].iloc[0]
            if pd.isna(r.get("mean_percentile", np.nan)): rows.append(f"{mn} & {vl} & {g} & {int(r.n_pos_evaluated)} of {int(r.n_pos)} & -- & -- & -- \\\\"); continue
            rows.append(f"{mn} & {vl} & {g} & {int(r.n_pos_evaluated)} of {int(r.n_pos)} & {int(r.median_same_gene_comparison_variants)} & {r.mean_percentile:.3f} ({r.ci_low:.3f}--{r.ci_high:.3f}) & {100*r.share_above_all_same_gene_variants:.0f}\\% \\\\")
rows += ["\\midrule", "\\multicolumn{7}{@{}l}{\\textbf{Group differences in mean percentile (difference, 95\\% CI, bootstrap $p$)}} \\\\", "\\midrule"]
for m, mn in MODELS:
    for v, vl in VL.items():
        cells = []
        for c in ("AA_minus_NHW", "AA_minus_CH", "CH_minus_NHW"):
            x = b2d[(b2d.model == m) & (b2d.version == v) & (b2d.contrast == c)]
            cells.append("--" if x.empty else f"{x.iloc[0].diff_mean:.3f} ({x.iloc[0].ci_low:.3f} to {x.iloc[0].ci_high:.3f}; {bp(x.iloc[0].bootstrap_two_sided_p)})")
        rows.append(f"{mn} & {vl} & AA $-$ NHW: {cells[0]} & \\multicolumn{{2}}{{l}}{{AA $-$ CH: {cells[1]}}} & \\multicolumn{{2}}{{l}}{{CH $-$ NHW: {cells[2]}}} \\\\")
write("review_within_gene.tex", ["\\textbf{Model} & \\textbf{Comparison variants} & \\textbf{Group} & \\textbf{High-PIP variants evaluated} & \\textbf{Median same-gene comparison variants} & \\textbf{Mean percentile (95\\% CI)} & \\textbf{Above all same-gene variants} \\\\"], rows,
      "Within-gene evaluation. For each high-PIP variant (PIP $\\geq 0.9$) the percentile of its score magnitude among low-PIP comparison variants of the same gene (at least five, or at least three when the TSS-distance bin is also matched); ties count one half. Intervals are from 2,000 bootstrap resamples of positive genes.",
      "stab:within_gene", "@{}lllrrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")
b7 = rd("b7_random_subsets.tsv"); b7w = rd("b7_pip_weighted_auroc.tsv"); b7d = rd("b7_pip_weighted_differences.tsv"); rows = []
for m, mn in MODELS:
    for c in ("low_pip", "intermediate_pip"):
        for g in ("CH", "NHW"):
            x = b7[(b7.model == m) & (b7.comparison == c) & (b7.group == g)]
            if x.empty: continue
            r = x.iloc[0]; rows.append(f"{mn} & {COMP[c]} & {g} & {int(r.n_pos_group)} & {int(r.subset_size)} & {r.aa_auroc:.3f} & {r.subset_auroc_mean:.3f} ({r.subset_p2_5:.3f}--{r.subset_p97_5:.3f}) & {100*r.share_of_subsets_at_or_above_aa:.1f}\\% \\\\")
rows += ["\\midrule", "\\multicolumn{8}{@{}l}{\\textbf{PIP-weighted AUROC (high-PIP variants weighted by PIP; positives and comparison variants resampled, 1,000 iterations)}} \\\\",
         "\\textbf{Model} & \\textbf{Comparison} & \\textbf{Group} & \\textbf{$n$} & \\textbf{Median PIP} & \\textbf{Unweighted} & \\textbf{PIP-weighted (95\\% CI)} & \\textbf{AA $-$ group (95\\% CI; $p$)} \\\\", "\\midrule"]
for m, mn in MODELS:
    for c in ("low_pip", "intermediate_pip"):
        for g in GROUPS:
            r = b7w[(b7w.model == m) & (b7w.comparison == c) & (b7w.group == g)].iloc[0]
            d = b7d[(b7d.model == m) & (b7d.comparison == c) & (b7d.contrast == f"AA_minus_{g}")]
            dd = "--" if d.empty else f"{d.iloc[0].diff_mean:.3f} ({d.iloc[0].ci_low:.3f} to {d.iloc[0].ci_high:.3f}; {bp(d.iloc[0].bootstrap_two_sided_p)})"
            rows.append(f"{mn} & {COMP[c]} & {g} & {int(r.n_pos)} & {r.median_pip:.3f} & {r.auroc_unweighted:.3f} & {ci3(r.auroc_pip_weighted, r.weighted_ci_low, r.weighted_ci_high)} & {dd} \\\\")
write("review_subsets_pip_weighting.tex", ["\\textbf{Model} & \\textbf{Comparison} & \\textbf{Group} & \\textbf{High-PIP variants} & \\textbf{Subset size} & \\textbf{AA AUROC} & \\textbf{Subset AUROC, mean (95\\% range)} & \\textbf{Subsets at or above AA} \\\\"], rows,
      "Random subsets of the Caribbean Hispanic and Non-Hispanic White high-PIP sets of the African American set size (1,000 draws without replacement, each evaluated by distance-matched AUROC against the group's own comparison pool), and PIP-weighted AUROC.",
      "stab:subsets_pip_weighting", "@{}lllrrrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")

# ---------- NEW: label transfer (b1) ----------
b1 = rd("b1_label_transfer_summary.tsv"); b1a = rd("b1_auroc_by_sharing.tsv"); rows = []
for _, r in b1.iterrows():
    rows.append(f"{r.positives_from} & {r.status_in} & {int(r.n_positives)} & {int(r['pip_ge_0.9'])} & {int(r['pip_0.5_0.9'])} & {int(r['pip_0.01_0.5'])} & {int(r['pip_lt_0.01'])} & {int(r.variant_absent)} & {int(r.egene_no_credible_set)} & {int(r.gene_not_egene)} & {int(r.in_credible_set)} \\\\")
rows += ["\\midrule", "\\multicolumn{11}{@{}l}{\\textbf{Distance-matched AUROC by whether the variant is also high-PIP in another group (positives resampled, 1,000 iterations)}} \\\\",
         "\\textbf{Model} & \\textbf{Group} & \\textbf{Subset} & \\textbf{$n$} & \\multicolumn{7}{l}{\\textbf{AUROC (95\\% CI)}} \\\\", "\\midrule"]
SL = {"shared_high_pip_in_another_group": "High-PIP in another group too", "high_pip_in_this_group_only": "High-PIP in this group only", "all": "All"}
for m, mn in MODELS:
    for g in GROUPS:
        for s, sl in SL.items():
            x = b1a[(b1a.model == m) & (b1a.group == g) & (b1a.subset == s)]
            if x.empty: continue
            r = x.iloc[0]
            val = "--" if pd.isna(r.get("auroc_mean", np.nan)) else ci3(r.auroc_mean, r.ci_low, r.ci_high)
            rows.append(f"{mn} & {g} & {sl} & {int(r.n_pos)} & \\multicolumn{{7}}{{l}}{{{val}}} \\\\")
write("review_label_transfer.tex", ["\\textbf{High-PIP from} & \\textbf{Status in} & \\textbf{$n$} & \\textbf{PIP $\\geq$ 0.9} & \\textbf{0.5--0.9} & \\textbf{0.01--0.5} & \\textbf{$<$ 0.01} & \\textbf{Variant not tested} & \\textbf{eGene, no credible set} & \\textbf{Not an eGene} & \\textbf{In a credible set} \\\\"], rows,
      "Status of each group's high-PIP variants (PIP $\\geq 0.9$, in a credible set) in the fine-mapping of the other two groups, and distance-matched AUROC by sharing. ``Variant not tested'' means the gene has a credible set in the other group but the variant did not pass its allele-count filter.",
      "stab:label_transfer", "@{}lllrrrrrrrr@{}", extra="\\setlength{\\tabcolsep}{3pt}\n")
print("done")
