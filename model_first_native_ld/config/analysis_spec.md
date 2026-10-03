# Locked ancestry-specific LD-reduced analysis specification

## Scope

This LD-reduced analysis starts from a scored pair table containing 1,219,052
exact SNP–gene pairs with complete controlled Borzoi and AlphaGenome scores in
the existing general, AD Tier 1, and AD Tier 2 scientific panels. General and AD
panels remain analytically separate even when they overlap.

The primary deployment view is each ancestry's ancestry-specific regular-eQTL panel. The
existing exact three-way-shared panel is retained as a common-testability
sensitivity analysis. Results describe model agreement with ancestry-specific
eQTL evidence, not ancestry-specific model predictions.

## Cohort LD and CH sample definition

LD is calculated directly from the original MAGENTA TOPMed-imputed, hg38,
imputation-r²-filtered, no-outlier, MAC10 PLINK files under
`/mnt/vstor/Data14/metabrain_lasso/`. These are the `/mnt/vstor` copies of the
analysis genotype files.
Standard PLINK 1.9 unphased hardcall `r2` is used. PLINK sample identifiers are
matched to the expression/eQTL cohort lists by stripping only leading ASCII
letters and underscores and then requiring a unique normalized core in both
sources; in these files every requested expression-cohort identifier also
matches exactly.

CH is **IHG only**; PER-listed samples are excluded from the combined HISP PLINK
file, and CH LD uses the 209 participants of the CH QTL cohort
(`CH.exact_qtl_cohort.keep`). The PLINK files cover all requested cohort samples:
224/224 AA, 209/209 CH, and 235/235 NHW. Allele-frequency agreement with the regular-eQTL summaries
was a mandatory per-chromosome QC gate and passed for all 66 units.

## Variant matching and LD calculation

Only exact chromosome, position, REF, and ALT matches enter LD calculations.
Variants must be biallelic SNVs, native-testable in the ancestry, within 100 kb
of the gene TSS, and have finite common-complete model scores. A variant with
insufficient nonmissing genotype data or negligible genotype variance is
excluded from the LD-calculable panel and counted.

For each ancestry, squared Pearson correlation of unphased best-guess ALT
allele counts defines `r2`, as implemented by PLINK 1.9. Sparse LD edges are
retained for variant pairs no more than 200 kb apart
when `r2 >= 0.1`; this single outcome-blind edge set supports the 0.1, 0.2, 0.5,
and 0.8 thresholds. Genotype missingness and nonvariable variants are reported;
no eQTL or model outcome is used to alter or select them.

The source BIM convention is verified per chromosome as A1=ALT and A2=REF.
Because `--keep-allele-order` sets A1 to ALT, the legacy PLINK `.frq` column
named `MAF` is interpreted as ALT frequency; genotype MAF is explicitly
calculated as `min(ALT AF, 1-ALT AF)` before comparison with the regular-eQTL
MAF. The calculation passed all 66 chromosome gates.

## Direct, model-first clumping

Within ancestry × scientific panel × gene × model, variants are sorted by
descending absolute model score and exact pair key. The first remaining pair
is an index. Only variants with a direct `r2` edge to that index at or above the
clump threshold are removed. Removed variants do not remove further variants,
so no transitive LD chains are created. The primary threshold is `r2 >= 0.2`;
`0.1` and `0.5` are sensitivity thresholds.

Borzoi, AlphaGenome, and consensus can nominate different indices. Consensus
strength and direction retain the existing project definitions. After
clumping, global absolute-score rank is primary and exact within-gene rank is a
gene-balanced companion. For the stratified 500-gene general benchmark, global
percentiles use inverse gene-inclusion weights so the cutoff represents the
eligible gene/pair universe; AD uses unit weights. Within-gene percentiles are
exact and unweighted. Top 1% is primary; 0.5%, 2%, 5%, and 10% form the
secondary curve.

## Exact-index and regional endpoints

For each nominated index, association evidence uses the stored two-sided
regular-eQTL p value and absolute z. Directional evidence uses
`aligned_z = sign(model score) * eQTL z`. The stringent threshold is
`p < 2.02e-5`; the paper-exact secondary endpoint also requires `q < 0.05`.
Stringent aligned, stringent opposite, and unsupported/indeterminate outcomes
are reported separately.

The close-proxy group is the index plus same-gene ancestry-specific variants having a
direct `r2 >= 0.8` edge to the index. Regional association categories are:

1. exact-index supported;
2. index unsupported but at least one close proxy supported;
3. no detected stringent support in the group.

Regional summaries include group size, minimum p, Simes p, and ACAT p. They do
not assign one direction to an LD group, and they are not interpreted as causal
variant identification.

## Top-versus-rest p-value distributions

For each ancestry, model, deployment, rank view, and cutoff, the full p-value
distribution is compared between the model-defined top set and the rest. The
association probability-of-superiority is

`sum_i_in_top sum_j_in_rest w_i w_j [I(p_i < p_j) + 0.5 I(p_i = p_j)] /
[(sum_i_in_top w_i)(sum_j_in_rest w_j)]`.

Thus 0.5 denotes no rank separation and a value above 0.5 is the weighted
probability that a randomly drawn top index has the smaller ordinary two-sided
eQTL p value. Directional probability-of-superiority uses the same calculation
after converting each index to `Phi(-aligned_z)`. The general benchmark uses
each sampled gene's inverse inclusion weight; AD uses unit weights.

The prespecified adjusted companion separately regresses `abs(z)` or
`aligned_z` on the top-1% indicator, gene fixed effects, centered
`log(1 + abs(TSS distance))` and its square, and centered ancestry-specific MAF
and its square. Standard errors are clustered by gene and the nine ancestry by
model tests are Holm-adjusted within each deployment, rank view, and evidence
type. The adjusted regression p value tests its top-indicator coefficient; it
is not calculated from the probability-of-superiority statistic.

## Uncertainty and examples

Inference uses 1,000 deterministic paired bootstrap replicates. Gene-clustered
resampling is primary. For the general benchmark, a prespecified broader local
genomic-cluster sensitivity keeps genes with overlapping ±100 kb windows
together; for AD, the broader sensitivity uses the published compound locus
definitions. The same resampled units compare all methods.
Variant-level, regional, yield, lift, distribution, and model-difference
estimates are reported.

Regional figures are selected only after systematic analysis by prespecified
rules: highest-ranked cross-ancestry supported region, ancestry-specific
support, model-discordant index nominations, and regional-only support.
If no gene is supported in all three ancestries, the cross-ancestry rule uses
the maximum supported-ancestry count and then the outcome-blind global rank;
the achieved count is reported rather than implying three-ancestry support.
