"""Combine two or three generated supplementary tables into one table environment with lettered parts.

Each input is either a full `table` file written by make_followup_tables.py / the table renderer
(caption + label + tabular, optionally wrapped in \\resizebox) or a bare tabular. The part heading is the
bold title of the input's caption (or the --title given for a bare tabular) and the part text is the rest of
its caption.

Usage: python combine_tables.py --out <file.tex> --label <stab:...> --caption "<bold title>" \
         --part <file.tex>[::<title for a bare tabular>::<caption text>] [--part ...]
"""
import argparse, re

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--label", required=True); ap.add_argument("--caption", required=True)
ap.add_argument("--part", action="append", required=True)
a = ap.parse_args()

parts = []
for spec in a.part:
    bits = spec.split("::")
    s = open(bits[0]).read()
    m = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", s, re.S)
    tab = m.group(0)
    cap = re.search(r"\\caption\{\\textbf\{(.*?)\}\s*(.*)\}\s*$", s, re.M)
    if len(bits) == 3:
        title, text = bits[1], bits[2]
    elif cap:
        title, text = cap.group(1), cap.group(2).strip()
    else:
        raise SystemExit(f"no caption in {bits[0]} and no title given")
    parts.append((title.rstrip("."), text, tab, "\\resizebox" in s))

letters = "ABC"
cap_parts = " ".join(f"({letters[i]}) {t}. {x}" if x else f"({letters[i]}) {t}." for i, (t, x, _, _) in enumerate(parts))
out = ["\\begin{table}[p]", "\\centering", "\\scriptsize", "\\renewcommand{\\arraystretch}{0.92}", f"\\caption{{\\textbf{{{a.caption}}} {cap_parts}}}", f"\\label{{{a.label}}}"]
for i, (t, _, tab, resize) in enumerate(parts):
    out.append(f"\\textbf{{({letters[i]})}} {t}\\\\[3pt]")
    if resize:
        out.append("\\resizebox{\\textwidth}{!}{%")
    out.append(tab)
    if resize:
        out.append("}")
    if i < len(parts) - 1:
        out.append("\\vspace{10pt}\\\\")
out.append("\\end{table}")
open(a.out, "w").write("\n".join(out) + "\n")
print("wrote", a.out, "with", len(parts), "parts")
