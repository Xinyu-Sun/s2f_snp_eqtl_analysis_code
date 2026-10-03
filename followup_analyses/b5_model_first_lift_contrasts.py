"""B5. Group contrasts in model-first enrichment over background (lift), from the released yield table.
Top 1% of clumped lead pairs, r2 0.2 clumping, global ranking, general gene panel. Groups are independent samples, so the
standard error of a difference is sqrt(se_a^2 + se_b^2) using the gene-bootstrap standard errors stored in the table.
Usage: python3 b5_model_first_lift_contrasts.py native_ld_yields.tsv out.tsv   (standard library only)"""
import csv, math, sys
src, out_path = sys.argv[1], sys.argv[2]
r = list(csv.DictReader(open(src), delimiter="\t"))
def f(x):
    try: return float(x)
    except ValueError: return float("nan")
names = {"index_paper_exact_association": "exact association", "index_directionally_supported": "direction-concordant association"}
out = []
for dep in ("native", "three_way_shared"):
    for ep in names:
        for m in ("borzoi", "alphagenome"):
            g = {}
            for x in r:
                if (x["panel"] == "general" and x["rank_view"] == "global" and x["cutoff_percent"] in ("1", "1.0") and x["clump_threshold_r2"] in ("0.2", "0.20")
                        and x["deployment"] == dep and x["endpoint"] == ep and x["method"] == m):
                    g[x["ancestry"]] = dict(lift=f(x["lift_over_background"]), se=f(x["gene_lift_over_background_bootstrap_se"]))
            for a, b in (("AA", "NHW"), ("AA", "CH"), ("CH", "NHW")):
                d = g[a]["lift"] - g[b]["lift"]; se = math.sqrt(g[a]["se"] ** 2 + g[b]["se"] ** 2); z = d / se
                lr = math.log(g[a]["lift"] / g[b]["lift"]); selr = math.sqrt((g[a]["se"] / g[a]["lift"]) ** 2 + (g[b]["se"] / g[b]["lift"]) ** 2)
                out.append(dict(panel=dep, endpoint=names[ep], model=m, contrast=f"{a}_minus_{b}", lift_a=g[a]["lift"], lift_b=g[b]["lift"], difference=d, se=se,
                                ci_low=d - 1.96 * se, ci_high=d + 1.96 * se, z=z, p=math.erfc(abs(z) / math.sqrt(2)),
                                lift_ratio=math.exp(lr), ratio_ci_low=math.exp(lr - 1.96 * selr), ratio_ci_high=math.exp(lr + 1.96 * selr)))
w = csv.DictWriter(open(out_path, "w"), fieldnames=list(out[0].keys()), delimiter="\t"); w.writeheader(); w.writerows(out)
print("B5 done:", len(out), "contrasts; smallest p =", round(min(x["p"] for x in out), 3))
