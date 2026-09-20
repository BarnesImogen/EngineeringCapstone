import os
import sys
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('src')) 

from data_prep import load_and_merge_tcga
from classification_calculations import (
    calculate_oncotype_dx_score, 
    calculate_pam50_score, 
    calculate_bci_score, 
    fetch_pam50_centroids, 
    calculate_mammostrat_score, 
    calculate_ihc4_score, 
    calculate_kim10_tnbc_score, 
    calculate_irrs7_score, 
    calculate_hu11_irg_score
)

def generate_correlation_heatmap(df, signatures, output_dir):
    score_cols = [f'{sig}_Score' for sig in signatures]
    score_df = df[score_cols]
    
    corr_matrix = score_df.corr(method='spearman')
    
    plt.figure(figsize=(11, 9))
    sns.set_theme(style="white") 
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        annot=True,          
        fmt=".2f",           
        cmap='coolwarm',     
        vmin=-1, vmax=1,     
        square=True,         
        linewidths=1,        
        cbar_kws={"shrink": .8, "label": "Spearman Correlation (rho)"}
    )
    
    clean_labels = [col.replace('_Score', '') for col in score_cols]
    plt.xticks(ticks=np.arange(len(clean_labels)) + 0.5, labels=clean_labels, rotation=45, ha='right', fontsize=11)
    plt.yticks(ticks=np.arange(len(clean_labels)) + 0.5, labels=clean_labels, rotation=0, fontsize=11)
    
    plt.title('Spearman Correlation Across Gene Expression Signatures', fontsize=16, pad=20)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, "signature_correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    print(f"Heatmap saved to {heatmap_path}")
    plt.show()

def load_discordance_margin(config_path="config.yml"):
    """Fraction (0-0.5] of each signature's tails counted as clearly high/low; 0.5 reproduces the median split."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    margin = float(config.get("bioinformatics", {}).get("discordance_margin", 1 / 3))
    if not 0 < margin <= 0.5:
        raise ValueError(f"discordance_margin must be in (0, 0.5], got {margin}")
    return margin

def run_pipeline():    
    rna_file = "data/raw/Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct" 
    clinical_file = "data/raw/Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi"
    output_dir = "data/processed"
    
    os.makedirs(output_dir, exist_ok=True)

    print("--- STARTING BIOINFORMATICS PIPELINE ENGINE ---")
    
    #Load and Merge Data
    print("\nLoading and merging TCGA cohorts...")
    master_df = load_and_merge_tcga(rna_file, clinical_file)
    print(f"Loaded master dataset: {master_df.shape[0]} patients.")

    # Calculate Continuous Scores
    print("\nCalculating Continuous Risk Scores...")
    print("Fetching PAM50 Centroids from R...")
    pam50_centroids = fetch_pam50_centroids()
    
    print("Executing all mathematical models...")
    master_df['OncotypeDX_Score'] = calculate_oncotype_dx_score(master_df)
    master_df['Pam50_Subtype'], master_df['Pam50_Score'] = calculate_pam50_score(master_df, pam50_centroids)
    master_df['BCI_Score'] = calculate_bci_score(master_df)
    master_df['Mammostrat_Score'] = calculate_mammostrat_score(master_df)
    master_df['IHC4_Score'] = calculate_ihc4_score(master_df)
    master_df['Kim10_Score'] = calculate_kim10_tnbc_score(master_df)
    master_df['IRRS7_Score'] = calculate_irrs7_score(master_df)
    master_df['Hu11_Score'] = calculate_hu11_irg_score(master_df)

    # Stratify into Binary Classes
    signatures = ['OncotypeDX', 'Pam50', 'BCI', 'Mammostrat', 'IHC4', 'Kim10', 'IRRS7', 'Hu11']

    # A NaN score would otherwise compare False against the median and be labelled Low Risk,
    # so drop patients without a complete set of scores.
    score_cols = [f'{sig}_Score' for sig in signatures]
    master_df[score_cols] = master_df[score_cols].apply(pd.to_numeric, errors='coerce')
    n_before = len(master_df)
    missing_counts = master_df[score_cols].isna().sum()
    if missing_counts.any():
        print("Patients with missing scores per signature:\n" + missing_counts[missing_counts > 0].to_string())
    master_df = master_df.dropna(subset=score_cols)
    if master_df.empty:
        raise ValueError("All patients were dropped for missing signature scores; check the per-signature counts above.")
    print(f"\nDropped {n_before - len(master_df)} patients with missing signature scores ({len(master_df)} remain).")

    print("\nStratifying Patients (Median Split)...")

    for sig in signatures:
        score_col = f'{sig}_Score'
        class_col = f'{sig}_Class'
        median_val = master_df[score_col].median()
        master_df[class_col] = (master_df[score_col] > median_val).astype(int)
        
        high_risk_count = master_df[class_col].sum()
        low_risk_count = (master_df[class_col] == 0).sum()
        print(f"  - {sig}: Cut-off {median_val:.4f} | High: {high_risk_count}, Low: {low_risk_count}")

    # Detect Discordance
    print("\nAnalysing Cohort Discordance...")
    class_columns = [f'{sig}_Class' for sig in signatures]
    total_patients = len(master_df)

    # Any disagreement across the median-split classes (the original, noisy definition)
    master_df['Is_Discordant_Median'] = master_df[class_columns].nunique(axis=1) > 1

    # Margin rule: within each signature a patient is "clearly high" in the top `margin` fraction
    # of the cohort and "clearly low" in the bottom `margin` fraction. Discordant = at least one
    # signature clearly high AND at least one clearly low.
    margin = load_discordance_margin()
    pct_rank = master_df[[f'{sig}_Score' for sig in signatures]].rank(pct=True)
    clearly_high = (pct_rank > 1 - margin).any(axis=1)
    clearly_low = (pct_rank <= margin).any(axis=1)
    master_df['Is_Discordant'] = clearly_high & clearly_low

    # Patients who disagree by the median split but have no clear-cut conflict are borderline:
    # excluded from both the concordant and discordant sets.
    master_df['Discordance_Group'] = np.select(
        [master_df['Is_Discordant'], ~master_df['Is_Discordant_Median']],
        ['discordant', 'concordant'],
        default='borderline'
    )

    n_signatures = len(signatures)
    expected_independent = 1 - 2 * (1 - margin) ** n_signatures + (1 - 2 * margin) ** n_signatures
    counts = master_df['Discordance_Group'].value_counts()
    print(f"Margin: top/bottom {margin:.1%} of each signature counts as clearly high/low.")
    print(f"Median-split discordance (any disagreement): {master_df['Is_Discordant_Median'].mean()*100:.1f}%")
    print(f"Margin-rule discordance:                     {master_df['Is_Discordant'].mean()*100:.1f}%")
    print(f"Expected under independent signatures:       {expected_independent*100:.1f}%")
    for group in ['concordant', 'discordant', 'borderline']:
        n = counts.get(group, 0)
        print(f"  - {group}: {n} ({n/total_patients*100:.1f}%)")

    # Data Partitioning for Export
    print("\nExporting Results...")
    
    # Isolate Discordant Cases (margin rule)
    discordant_df = master_df[master_df['Discordance_Group'] == 'discordant'].copy()
    
    # Isolate Concordant Cases (all eight median-split classes agree) and add a consensus label
    concordant_df = master_df[master_df['Discordance_Group'] == 'concordant'].copy()
    concordant_df['Consensus_Risk'] = concordant_df['Kim10_Class'].map({1: 'High Risk', 0: 'Low Risk'})
    
    # Export to CSV
    master_path = os.path.join(output_dir, "tcga_master_results.csv")
    discordant_path = os.path.join(output_dir, "tcga_discordant_cases.csv")
    concordant_path = os.path.join(output_dir, "tcga_concordant_cases.csv")
    
    master_df.to_csv(master_path)
    discordant_df.to_csv(discordant_path)
    concordant_df.to_csv(concordant_path)
    
    print(f"Saved Master DataFrame to: {master_path}")
    print(f"Saved Discordant Cases to: {discordant_path}")
    print(f"Saved Concordant Cases to: {concordant_path}")

    # 7. Visualisation
    generate_correlation_heatmap(master_df, signatures, output_dir)
    print("\n--- PIPELINE EXECUTION COMPLETE ---")

if __name__ == "__main__":
    run_pipeline()