import os
import sys
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
    print("\nStratifying Patients (Median Split)...")
    signatures = ['OncotypeDX', 'Pam50', 'BCI', 'Mammostrat', 'IHC4', 'Kim10', 'IRRS7', 'Hu11']
    
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
    master_df['Is_Discordant'] = master_df[class_columns].nunique(axis=1) > 1
    
    total_patients = len(master_df)
    discordant_count = master_df['Is_Discordant'].sum()
    concordant_count = total_patients - discordant_count
    
    print(f"Discordant patients: {discordant_count} ({(discordant_count/total_patients)*100:.1f}%)")
    print(f"Perfect agreement: {concordant_count} ({(concordant_count/total_patients)*100:.1f}%)")

    # Data Partitioning for Export
    print("\nExporting Results...")
    
    # Isolate Discordant Cases
    discordant_df = master_df[master_df['Is_Discordant']].copy()
    
    # Isolate Concordant Cases and add a consensus label
    concordant_df = master_df[~master_df['Is_Discordant']].copy()
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