import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr
from signature_definitions import (
    ONCOTYPE_GROUPS, ONCOTYPE_SINGLE_GENES, ONCOTYPE_REFERENCE_GENES, BCI_HI_GENES, BCI_MGI_GENES,
    MAMMOSTRAT_GENES, IHC4_WEIGHTS, IRRS7_WEIGHTS, HU11_WEIGHTS, PAM50_ALIASES,
)


# Oncotype DX, ✅
# PAM50 ✅
# Breast Cancer Index (BCI), ✅
# Mammostrat ✅
# IHC4 ✅
# IRRS7 ✅
# Hu11 ✅

def calculate_oncotype_dx_score(df):
    """
    Oncotype DX 21-gene Recurrence Score (Paik et al., 2004)

    NOTE: Exact proprietary weights are not publicly available.
    This implementation uses the published formula structure from
    Paik et al. (2004) with approximated group score weights.

    Formula structure:
    RS = +0.47 * GRB7_group
        - 0.34 * ER_group
        + 1.04 * Proliferation_group
        + 0.10 * Invasion_group
        + 0.05 * CD68
        - 0.08 * GSTM1
        - 0.07 * BAG1

    Risk categories:
    Low:          RS < 18
    Intermediate: 18 <= RS < 31
    High:         RS >= 31

    Before scoring, every gene is normalised against the mean of five reference genes
    (ACTB, GAPDH, GUSB, RPLP0, TFRC), per the published assay (Paik et al., 2004). The
    data is already log2-scaled, so normalisation is a per-patient subtraction.
    """

    def group_score(row, genes):
        available = [g for g in genes if g in row.index]
        if not available:
            return 0
        return row[available].mean()

    def calculate_rs(row):
        rs = sum(weight * group_score(row, genes) for genes, weight in ONCOTYPE_GROUPS.values())
        rs += sum(weight * row.get(gene, 0) for gene, weight in ONCOTYPE_SINGLE_GENES.items())
        return rs

    all_genes = [g for genes, _ in ONCOTYPE_GROUPS.values() for g in genes] + list(ONCOTYPE_SINGLE_GENES)
    available = [g for g in all_genes if g in df.columns]
    missing   = [g for g in all_genes if g not in df.columns]

    print(f"Oncotype DX: Found {len(available)}/{len(all_genes)} genes")
    if missing:
        print(f"Missing: {missing}")

    ref_available = [g for g in ONCOTYPE_REFERENCE_GENES if g in df.columns]
    ref_missing = [g for g in ONCOTYPE_REFERENCE_GENES if g not in df.columns]
    print(f"Oncotype DX: Found {len(ref_available)}/{len(ONCOTYPE_REFERENCE_GENES)} reference genes")
    if ref_missing:
        print(f"Missing reference genes: {ref_missing}")
    reference_mean = df[ref_available].mean(axis=1)

    unnormalized_score = df[available].apply(calculate_rs, axis=1)
    normalized_score = df[available].sub(reference_mean, axis=0).apply(calculate_rs, axis=1)

    corr, _ = spearmanr(unnormalized_score, normalized_score)
    print(f"Oncotype DX: Spearman correlation between un-normalised and reference-normalised scores: {corr:.4f}")

    return normalized_score

def fetch_pam50_centroids():
    """
    Uses the rpy2 bridge to extract the official Parker 2009 
    centroid matrix from the Bioconductor genefu package.
    """
    r_script = """
    function() {
        # Load the pam50 object from genefu
        data(pam50, package="genefu")
        
        # Convert it to an R data.frame so pandas translates the gene names perfectly
        centroid_df <- as.data.frame(pam50$centroids)
        return(centroid_df)
    }
    """
    get_centroids = robjects.r(r_script)
    
    with localconverter(robjects.default_converter + pandas2ri.converter):
        # Because we are inside the localconverter block, get_centroids() 
        # returns a fully formatted pandas DataFrame automatically!
        centroid_df = get_centroids()
        
    # genefu uses 'Her2', but our Python math formula explicitly looks for 'HER2'
    # We rename the columns here to ensure the dictionary lookup doesn't crash
    centroid_df.columns = ['Basal', 'HER2', 'LumA', 'LumB', 'Normal']
    
    return centroid_df

def calculate_pam50_score(df, centroid_df):
    """
    PAM50 Nearest Centroid Classifier and ROR-S Score
    df: Patient data (rows=patients, cols=genes)
    centroid_df: Parker 2009 centroids (rows=50 genes, cols=5 subtypes)
    """
    # 1. Ensure we only have the 50 PAM50 genes. The genefu centroid matrix uses older gene
    # symbols (CDCA1, KNTC2); map them to the RNA-seq column names before matching.
    centroid_df = centroid_df.rename(index=PAM50_ALIASES)
    genes = centroid_df.index.tolist()
    available_genes = [g for g in genes if g in df.columns]

    print(f"PAM50: Found {len(available_genes)}/{len(genes)} genes")
    if len(available_genes) < len(genes):
        print(f"Missing: {sorted(set(genes) - set(available_genes))}")

    # 2. Median-Centre the data (CRITICAL step for PAM50)
    # We must centre the data against the entire cohort's median
    centered_df = df[available_genes] - df[available_genes].median(axis=0)
    
    # 3. Calculate Spearman correlation to each centroid
    correlations = pd.DataFrame(index=df.index, columns=centroid_df.columns)
    
    for patient in centered_df.index:
        for subtype in centroid_df.columns:
            # spearmanr returns the correlation coefficient and the p-value
            coef, _ = spearmanr(centered_df.loc[patient], centroid_df.loc[available_genes, subtype])
            correlations.loc[patient, subtype] = coef
            
    # 4. Assign Subtype (the centroid with the highest correlation)
    subtypes = correlations.idxmax(axis=1)
    
    # 5. Calculate ROR-S Score 
    # ROR-S = (0.05*Basal) + (0.11*HER2) - (0.25*LumA) + (0.07*LumB) + (0.11*Normal)
    ror_s = (
        (0.05 * correlations['Basal']) + 
        (0.11 * correlations['HER2']) - 
        (0.25 * correlations['LumA']) + 
        (0.07 * correlations['LumB']) + 
        (0.11 * correlations['Normal'])
    )
    
    return subtypes, ror_s

def calculate_bci_score(df):
    """
    Breast Cancer Index (BCI) Approximation
    
    NOTE: The exact Biotheranostics algorithm is proprietary. 
    This is an academic proxy based on the foundational papers (Ma et al., 2008),
    combining the HOXB13:IL17RB (H/I) ratio with the Molecular Grade Index (MGI).
    """
    
    hi_genes = BCI_HI_GENES
    mgi_genes = BCI_MGI_GENES
    
    all_genes = hi_genes + mgi_genes
    missing = [g for g in all_genes if g not in df.columns]
    if missing:
        print(f"BCI Warning: Missing genes {missing}. Continuous score will be approximated.")
        
    # Calculate the H/I Ratio
    hoxb13 = df.get('HOXB13', pd.Series(0, index=df.index))
    il17rb = df.get('IL17RB', pd.Series(0, index=df.index))
    hi_ratio = hoxb13 - il17rb
    
    # Calculate the Molecular Grade Index (MGI)
    # The MGI is the mean expression of the 5 proliferation genes
    available_mgi = [g for g in mgi_genes if g in df.columns]
    if available_mgi:
        mgi_score = df[available_mgi].mean(axis=1)
    else:
        mgi_score = pd.Series(0, index=df.index)
        
    # Combine into a continuous BCI proxy score
    bci_score = hi_ratio + mgi_score
    
    return bci_score

def calculate_mammostrat_score(df):
    """
    Mammostrat 5-Gene Proxy Score
    
    Actively searches for aliases (including RG9MTD1) and safely 
    calculates the mean index using only the present biomarkers.
    """
    
    present_genes = []
    missing_genes = []
    
    for gene in [g for g in MAMMOSTRAT_GENES if g != 'TRMT10C']:
        if gene in df.columns:
            present_genes.append(gene)
        else:
            missing_genes.append(gene)
            
    if 'TRMT10C' in df.columns:
        present_genes.append('TRMT10C')
    elif 'HTF9C' in df.columns:
        present_genes.append('HTF9C')
    elif 'RG9MTD1' in df.columns:
        present_genes.append('RG9MTD1')
    else:
        missing_genes.append('TRMT10C (or aliases)')
        
    if missing_genes:
        print(f"Mammostrat Warning: Missing {missing_genes}. Calculating mean using {len(present_genes)} genes.")
            
    score = df[present_genes].mean(axis=1)
    
    return score

def calculate_ihc4_score(df):
    weights = IHC4_WEIGHTS
    available_genes = [g for g in weights.keys() if g in df.columns]
    score = df[available_genes].apply(
        lambda row: sum(row[g] * weights[g] for g in available_genes), axis=1
    )
    return score

def calculate_irrs7_score(df):
    """
    IRRS-7 Signature (MDPI/Frontiers, 2025)
    Focus: Insulin Resistance-Related Prognostic Score.
    """
    weights = IRRS7_WEIGHTS
    
    available_genes = [g for g in weights.keys() if g in df.columns]
    
    score = df[available_genes].apply(
        lambda row: sum(row[g] * weights[g] for g in available_genes), axis=1
    )
    return score

def calculate_hu11_irg_score(df):
    """
    Hu-11 IRG Signature (Hu et al., 2024)
    Focus: Inflammation-Related Genes and immune microenvironment.
    """
    weights = HU11_WEIGHTS
    
    available_genes = [g for g in weights.keys() if g in df.columns]
    
    score = df[available_genes].apply(
        lambda row: sum(row[g] * weights[g] for g in available_genes), axis=1
    )
    return score