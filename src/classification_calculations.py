import pandas as pd
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
from scipy.stats import spearmanr


# Oncotype DX, ✅
# Mammaprint, ~ looking into. proving difficult XX
# Endopredict, XX
# PAM50 ✅
# Breast Cancer Index (BCI), ✅
# Mammostrat ✅
# IHC4 ✅
# Kim10 ✅
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
    """

    # GRB7 group
    grb7_genes   = ['GRB7', 'ERBB2']
    # ER group  
    er_genes     = ['ESR1', 'PGR', 'BCL2', 'SCUBE2']
    # Proliferation group
    prolif_genes = ['MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2']
    # Invasion group
    inv_genes    = ['MMP11', 'CTSL2']
    # Single genes
    single_genes = ['CD68', 'GSTM1', 'BAG1']

    def group_score(row, genes):
        available = [g for g in genes if g in row.index]
        if not available:
            return 0
        return row[available].mean()

    all_genes = grb7_genes + er_genes + prolif_genes + inv_genes + single_genes
    available = [g for g in all_genes if g in df.columns]
    missing   = [g for g in all_genes if g not in df.columns]

    print(f"Oncotype DX: Found {len(available)}/{len(all_genes)} genes")
    if missing:
        print(f"Missing: {missing}")

    def calculate_rs(row):
        grb7  = group_score(row, grb7_genes)
        er    = group_score(row, er_genes)
        prolif = group_score(row, prolif_genes)
        inv   = group_score(row, inv_genes)
        cd68  = row.get('CD68',  0)
        gstm1 = row.get('GSTM1', 0)
        bag1  = row.get('BAG1',  0)

        rs = (0.47  * grb7
            - 0.34  * er
            + 1.04  * prolif
            + 0.10  * inv
            + 0.05  * cd68
            - 0.08  * gstm1
            - 0.07  * bag1)
        return rs

    score = df[available].apply(calculate_rs, axis=1)
    return score

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
    # 1. Ensure we only have the 50 PAM50 genes
    genes = centroid_df.index.tolist()
    available_genes = [g for g in genes if g in df.columns]
    
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
    
    hi_genes = ['HOXB13', 'IL17RB']
    mgi_genes = ['BUB1B', 'CENPA', 'NEK2', 'RACGAP1', 'RRM2']
    
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
    
    for gene in ['TP53', 'CEACAM5', 'NDRG1', 'SLC7A5']:
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
    weights = {
        'ESR1': -0.100, 
        'PGR': -0.079,  
        'ERBB2': 0.586,  
        'MKI67': 0.240
    }
    available_genes = [g for g in weights.keys() if g in df.columns]
    score = df[available_genes].apply(
        lambda row: sum(row[g] * weights[g] for g in available_genes), axis=1
    )
    return score

def calculate_kim10_tnbc_score(df):
    """
    Kim-10 TNBC Signature (Kim et al., 2024)
    Focus: Early-stage Triple-Negative Breast Cancer recurrence risk.
    
    Risk score formula:
    (0.818636 x DGKH) + (0.018069 x GADD45B) + (0.605352 x KLF7) + 
    (0.231666 x LYST) + (1.305352 x NR6A1) + (-0.052086 x PYCARD) + 
    (-0.196973 x ROBO1) + (0.968759 x SLC22A20P) + (0.098331 x SLC24A3) + 
    (0.311646 x SLC45A4)
    
    Cut-off Value: 5.959715
    High Risk: score > 5.959715
    """
    weights = {
        'DGKH': 0.818636, 'GADD45B': 0.018069, 'KLF7': 0.605352, 
        'LYST': 0.231666, 'NR6A1': 1.305352, 'PYCARD': -0.052086, 
        'ROBO1': -0.196973, 'SLC22A20P': 0.968759, 
        'SLC24A3': 0.098331, 'SLC45A4': 0.311646
    }

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
    weights = {
        'EZR': 0.040, 'LIFR': -0.046, 'TBC1D4': -0.138, 'SAA1': -0.0105, 
        'NSF': 0.0218, 'RPL5': -0.0566, 'PGK1': 0.464
    }
    
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
    weights = {

        'IL18': 0.115, 'IL12B': 0.203, 'RASGRP1': -0.142, 'HPN': 0.089,
        'CLEC5A': 0.176, 'SCARF1': 0.134, 'TACR3': 0.212, 'VIP': -0.108,
        'CCL2': 0.095, 'CALCRL': 0.122, 'ABCA1': -0.076
    }
    
    available_genes = [g for g in weights.keys() if g in df.columns]
    
    score = df[available_genes].apply(
        lambda row: sum(row[g] * weights[g] for g in available_genes), axis=1
    )
    return score