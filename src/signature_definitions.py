"""Gene sets, weights and cutoffs for the five RNA-based signatures.

Single source of truth: classification_calculations.py scores patients with these, and the
generation/judge prompts describe them, so the model is told exactly what was computed.
Deliberately free of heavy imports (no rpy2) so prompts can use it.
"""

# Ordered risk classes every signature is cut into
RISK_CLASSES = ["Low", "Intermediate", "High"]

# Oncotype DX (Paik et al., 2004; genefu oncotypedx). Each group is
# (within-group gene weights, RSU weight, floor applied to the group score on the 0-15 scale).
ONCOTYPE_GROUPS = {
    "GRB7 group":          ({"GRB7": 0.9, "ERBB2": 0.1}, 0.47, 8.0),
    "ER group":            ({"ESR1": 0.8 / 4, "PGR": 1.2 / 4, "BCL2": 1 / 4, "SCUBE2": 1 / 4}, -0.34, None),
    "Proliferation group": ({g: 1 / 5 for g in ["BIRC5", "MKI67", "MYBL2", "CCNB1", "AURKA"]}, 1.04, 6.5),
    "Invasion group":      ({"CTSL2": 1 / 2, "MMP11": 1 / 2}, 0.10, None),
}
ONCOTYPE_SINGLE_GENES = {"CD68": 0.05, "GSTM1": -0.08, "BAG1": -0.07}
# Reference genes the published assay normalises every Oncotype gene against before scoring
# (Paik et al., 2004). Not part of the transcriptomic profile shown to the LLM: they carry no
# risk signal on their own.
ONCOTYPE_REFERENCE_GENES = ["ACTB", "GAPDH", "GUSB", "RPLP0", "TFRC"]
# Each gene is rescaled to 0-15 across the cohort before the group scores (genefu do.scaling)
ONCOTYPE_GENE_SCALE = 15
# Alternative to cohort rescaling: reference-normalised log2 value + 10, with no cohort step
ONCOTYPE_REFERENCE_OFFSET = 10
# RS = 20 * (RSU - 6.7), bounded to [0, 100]
ONCOTYPE_RSU_OFFSET, ONCOTYPE_RSU_MULTIPLIER = 6.7, 20
# Published RS cutoffs (Low < 18 <= Intermediate < 31 <= High). Not used for classification: on
# RNA-seq they put ~45-67% of ER+ patients in High (published cohorts: ~15-27%), so Oncotype DX
# is classified by cohort tertiles of the unbounded RSU instead. Kept for reference.
ONCOTYPE_CUTOFFS = (18, 31)

# PAM50 (Parker et al., 2009): nearest-centroid subtype on these 50 genes.
# NUF2 and NDC80 are listed under their published symbols (CDCA1, KNTC2) in the genefu
# centroid matrix; PAM50_ALIASES maps the centroid row names to these RNA-seq column names.
PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA",
    "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NDC80",  # published symbol: KNTC2
    "NUF2",  # published symbol: CDCA1
    "ORC6L", "PGR",
    "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]

# genefu's Parker 2009 centroid matrix ships with these older gene symbols; map centroid
# row names to the RNA-seq column names before matching against the data.
PAM50_ALIASES = {"CDCA1": "NUF2", "KNTC2": "NDC80"}

# ROR-S (Parker et al., 2009, equation 1; genefu rorS). The Normal-like centroid is used for
# the subtype call but has no ROR-S term.
ROR_S_COEFFICIENTS = {"Basal": 0.05, "HER2": 0.12, "LumA": -0.34, "LumB": 0.23}
PAM50_CUTOFFS = (29, 53)  # on the 0-100 scale: Low < 29 <= Intermediate < 53 <= High

# Breast Cancer Index (Ma et al., 2008; Habel et al., 2013). IL17RB is the current symbol
# for the gene the papers call IL17BR.
BCI_HI_GENES = ["HOXB13", "IL17RB"]
BCI_MGI_GENES = ["BUB1B", "CENPA", "NEK2", "RACGAP1", "RRM2"]
# BCI raw = 0.4431*MGI + 0.4972*HI - 0.09*HI^3, rescaled to 0-10 as 2*raw + 5 bounded to [0, 10]
BCI_MGI_WEIGHT, BCI_HI_WEIGHT, BCI_HI_CUBIC_WEIGHT = 0.4431, 0.4972, -0.09
BCI_CUTOFFS = (5, 6.4)  # Low < 5 <= Intermediate < 6.4 <= High

IRRS7_WEIGHTS = {
    "EZR": 0.040, "LIFR": -0.046, "TBC1D4": -0.138, "SAA1": -0.0105,
    "NSF": 0.0218, "RPL5": -0.0566, "PGK1": 0.464,
}

HU11_WEIGHTS = {
    "IL18": 0.115, "IL12B": 0.203, "RASGRP1": -0.142, "HPN": 0.089,
    "CLEC5A": 0.176, "SCARF1": 0.134, "TACR3": 0.212, "VIP": -0.108,
    "CCL2": 0.095, "CALCRL": 0.122, "ABCA1": -0.076,
}

# Canonical (score/class column key, display label) pair for each signature, in cohort/prompt order.
SIGNATURES = [
    ("OncotypeDX", "Oncotype DX"),
    ("Pam50", "PAM50"),
    ("BCI", "Breast Cancer Index"),
    ("IRRS7", "IRRS-7"),
    ("Hu11", "Hu-11"),
]

# Published cutoffs per signature; signatures not listed here (Oncotype DX, IRRS-7, Hu-11) are split
# at cohort tertiles.
PUBLISHED_CUTOFFS = {"Pam50": PAM50_CUTOFFS, "BCI": BCI_CUTOFFS}

def get_all_signature_genes():
    """Every gene symbol used by any signature: the full transcriptomic panel shown to the model."""
    genes = set(PAM50_GENES)
    for gene_weights, _, _ in ONCOTYPE_GROUPS.values():
        genes.update(gene_weights)
    genes.update(ONCOTYPE_SINGLE_GENES)
    genes.update(BCI_HI_GENES)
    genes.update(BCI_MGI_GENES)
    genes.update(IRRS7_WEIGHTS)
    genes.update(HU11_WEIGHTS)
    return sorted(genes)

def _format_weights(weights):
    return ", ".join(f"{gene} ({weight:+g})" for gene, weight in weights.items())

def _format_linear(weights):
    """e.g. {"GRB7": 0.9, "ERBB2": 0.1} -> '0.9*GRB7 + 0.1*ERBB2'"""
    return " + ".join(f"{weight:g}*{gene}" for gene, weight in weights.items())

def get_signature_reference_text():
    """Plain-text description of how each signature scores and classifies a patient (used in generation and judge prompts)."""
    oncotype_groups = "; ".join(
        f"{name} = {_format_linear(gene_weights)}" + (f" (floored at {floor:g})" if floor is not None else "")
        for name, (gene_weights, _, floor) in ONCOTYPE_GROUPS.items()
    )
    rsu_terms = " ".join(
        [f"{weight:+g}*{name}" for name, (_, weight, _) in ONCOTYPE_GROUPS.items()]
        + [f"{weight:+g}*{gene}" for gene, weight in ONCOTYPE_SINGLE_GENES.items()]
    )
    ror_terms = " ".join(f"{weight:+g}*{subtype}" for subtype, weight in ROR_S_COEFFICIENTS.items())
    pam50_low, pam50_high = PAM50_CUTOFFS
    bci_low, bci_high = BCI_CUTOFFS
    return "\n".join([
        "Every signature produces a continuous score from log2 RNA-seq expression and places the patient in one of "
        "three classes: Low, Intermediate or High. PAM50 and Breast Cancer Index use their published cutoffs. "
        "Oncotype DX, IRRS-7 and Hu-11 use cohort tertiles: High means the top third of this TCGA-BRCA cohort, Low the "
        "bottom third, so for these three the class is relative to this cohort, not an absolute risk. These are "
        "research re-implementations, not the proprietary commercial assays.",
        f"- Oncotype DX (Paik et al., 2004): each gene is normalised by subtracting the patient's mean expression of five "
        f"reference genes ({', '.join(ONCOTYPE_REFERENCE_GENES)}), then rescaled to 0-{ONCOTYPE_GENE_SCALE} across the "
        f"cohort. Group scores: {oncotype_groups}. Unscaled recurrence score RSU = {rsu_terms}. Classes are cohort "
        f"tertiles of RSU. The published Recurrence Score cutoffs ({ONCOTYPE_CUTOFFS[0]} and {ONCOTYPE_CUTOFFS[1]}) are "
        "not used because they do not transfer to RNA-seq.",
        f"- PAM50 (Parker et al., 2009): the 50 genes ({', '.join(PAM50_GENES)}) are median-centred using a subset with "
        "equal numbers of ER-positive and ER-negative patients, then Spearman-correlated with the five Parker centroids "
        "(Basal, HER2, LumA, LumB, Normal); the subtype is the highest correlation. Risk of recurrence "
        f"ROR-S = {ror_terms} (the correlations; Normal has no term), rescaled to 0-100 between the cohort's 2.5th and "
        f"97.5th percentiles. Low < {pam50_low}, Intermediate {pam50_low} to < {pam50_high}, High >= {pam50_high}.",
        f"- Breast Cancer Index (Ma et al., 2008; Habel et al., 2013): MGI (molecular grade index) is the first principal "
        f"component of {', '.join(BCI_MGI_GENES)}, each standardised to mean 0 and SD 1, oriented so higher means more "
        f"proliferation. H/I = {BCI_HI_GENES[0]} - {BCI_HI_GENES[1]}, standardised to mean 0 and SD 1. Means, SDs and "
        f"the principal component are fitted on ER-positive patients. BCI = {BCI_MGI_WEIGHT:g}*MGI + {BCI_HI_WEIGHT:g}*H/I "
        f"- {-BCI_HI_CUBIC_WEIGHT:g}*(H/I)^3, rescaled to 0-10 as 2*BCI + 5 (bounded to 0-10). Low < {bci_low:g}, "
        f"Intermediate {bci_low:g} to < {bci_high:g}, High >= {bci_high:g}.",
        f"- IRRS-7: weighted sum of {_format_weights(IRRS7_WEIGHTS)}. Classes are cohort tertiles.",
        f"- Hu-11: weighted sum of {_format_weights(HU11_WEIGHTS)}. Classes are cohort tertiles.",
        "An Intermediate class means that signature does not take a side: it supports neither High nor Low risk.",
    ])
