"""Gene sets and weights for the eight signatures.

Single source of truth: classification_calculations.py scores patients with these, and the
generation/judge prompts describe them, so the model is told exactly what was computed.
Deliberately free of heavy imports (no rpy2) so prompts can use it.
"""

# Oncotype DX (Paik et al., 2004): weighted sum of group means and single genes
ONCOTYPE_GROUPS = {
    "GRB7 group":          (["GRB7", "ERBB2"], 0.47),
    "ER group":            (["ESR1", "PGR", "BCL2", "SCUBE2"], -0.34),
    "Proliferation group": (["MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2"], 1.04),
    "Invasion group":      (["MMP11", "CTSL2"], 0.10),
}
ONCOTYPE_SINGLE_GENES = {"CD68": 0.05, "GSTM1": -0.08, "BAG1": -0.07}

# PAM50 (Parker et al., 2009): nearest-centroid subtype on these 50 genes
PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1",
    "CDC20", "CDC6", "CDCA1", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2",
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C",
    "KNTC2", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA",
    "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6L", "PGR",
    "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS",
    "UBE2C", "UBE2T",
]

# Breast Cancer Index proxy: HOXB13 - IL17RB (log-scale ratio) + mean of the molecular grade genes
BCI_HI_GENES = ["HOXB13", "IL17RB"]
BCI_MGI_GENES = ["BUB1B", "CENPA", "NEK2", "RACGAP1", "RRM2"]

# Mammostrat proxy: unweighted mean of these (TRMT10C may appear as HTF9C / RG9MTD1)
MAMMOSTRAT_GENES = ["TP53", "CEACAM5", "NDRG1", "SLC7A5", "TRMT10C"]

IHC4_WEIGHTS = {"ESR1": -0.100, "PGR": -0.079, "ERBB2": 0.586, "MKI67": 0.240}

KIM10_WEIGHTS = {
    "DGKH": 0.818636, "GADD45B": 0.018069, "KLF7": 0.605352,
    "LYST": 0.231666, "NR6A1": 1.305352, "PYCARD": -0.052086,
    "ROBO1": -0.196973, "SLC22A20P": 0.968759,
    "SLC24A3": 0.098331, "SLC45A4": 0.311646,
}

IRRS7_WEIGHTS = {
    "EZR": 0.040, "LIFR": -0.046, "TBC1D4": -0.138, "SAA1": -0.0105,
    "NSF": 0.0218, "RPL5": -0.0566, "PGK1": 0.464,
}

HU11_WEIGHTS = {
    "IL18": 0.115, "IL12B": 0.203, "RASGRP1": -0.142, "HPN": 0.089,
    "CLEC5A": 0.176, "SCARF1": 0.134, "TACR3": 0.212, "VIP": -0.108,
    "CCL2": 0.095, "CALCRL": 0.122, "ABCA1": -0.076,
}

def _format_weights(weights):
    return ", ".join(f"{gene} ({weight:+g})" for gene, weight in weights.items())

def get_signature_reference_text():
    """Plain-text description of how each signature scores a patient (used in generation and judge prompts)."""
    oncotype_groups = "; ".join(
        f"{name} = mean({', '.join(genes)}) x {weight:+g}" for name, (genes, weight) in ONCOTYPE_GROUPS.items()
    )
    return "\n".join([
        "Every signature produces a continuous score from log2 RNA-seq expression. In this study 'High Risk' means the "
        "score is above the cohort median for that signature (1), otherwise 'Low Risk' (0). These are research "
        "re-implementations, not the proprietary commercial assays.",
        f"- Oncotype DX: weighted sum. {oncotype_groups}; single genes {_format_weights(ONCOTYPE_SINGLE_GENES)}.",
        f"- PAM50: correlates the patient's median-centred expression of 50 genes ({', '.join(PAM50_GENES)}) with the five "
        "Parker centroids (Basal, HER2, LumA, LumB, Normal) and combines the correlations into a risk-of-recurrence score.",
        f"- Breast Cancer Index: ({BCI_HI_GENES[0]} - {BCI_HI_GENES[1]}) + mean({', '.join(BCI_MGI_GENES)}), equal weights.",
        f"- Mammostrat: unweighted mean of {', '.join(MAMMOSTRAT_GENES)}.",
        f"- IHC4: weighted sum of {_format_weights(IHC4_WEIGHTS)}.",
        f"- Kim-10: weighted sum of {_format_weights(KIM10_WEIGHTS)}.",
        f"- IRRS-7: weighted sum of {_format_weights(IRRS7_WEIGHTS)}.",
        f"- Hu-11: weighted sum of {_format_weights(HU11_WEIGHTS)}.",
    ])
