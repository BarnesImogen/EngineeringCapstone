import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath('src'))

from data_prep import load_and_merge_tcga
from signature_definitions import SIGNATURES, PUBLISHED_CUTOFFS
from classification_calculations import (
    infer_er_status,
    calculate_oncotype_dx_score,
    calculate_pam50_score,
    calculate_bci_score,
    fetch_pam50_centroids,
    calculate_irrs7_score,
    calculate_hu11_irg_score,
    classify_by_cutoffs,
    classify_by_tertiles,
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
    plt.close()

def assign_cohorts(class_df):
    """Discordant: at least one signature High and at least one Low. Concordant: all High or all
    Low. Borderline: everyone else, including all-Intermediate patients (the LLM's binary answer
    would have nothing to be checked against)."""
    is_high, is_low = class_df.eq('High'), class_df.eq('Low')
    return pd.Series(np.select(
        [is_high.any(axis=1) & is_low.any(axis=1), is_high.all(axis=1) | is_low.all(axis=1)],
        ['discordant', 'concordant'],
        default='borderline'
    ), index=class_df.index)

def expected_discordance(class_df):
    """Discordance rate if the signatures were independent, given each one's class proportions:
    1 - P(no High) - P(no Low) + P(all Intermediate)."""
    p_high, p_low = class_df.eq('High').mean(), class_df.eq('Low').mean()
    return 1 - np.prod(1 - p_high) - np.prod(1 - p_low) + np.prod(1 - p_high - p_low)

def lone_dissenter(class_df):
    """For each patient, the one signature that disagrees with all the others: High while every
    other is Low or Intermediate, or Low while every other is High or Intermediate. Blank if
    there is no such signature or more than one (one High, one Low, the rest Intermediate)."""
    lone = pd.DataFrame({
        sig: (class_df[sig].eq('High') & class_df.drop(columns=sig).ne('High').all(axis=1))
             | (class_df[sig].eq('Low') & class_df.drop(columns=sig).ne('Low').all(axis=1))
        for sig in class_df.columns
    })
    unique = lone.sum(axis=1) == 1
    return pd.Series(np.where(unique, lone.idxmax(axis=1), ""), index=class_df.index)

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

    # Clinical ER status is recorded for only ~10% of patients, so PAM50 centring and the BCI
    # reference set use ER status inferred from (bimodal) ESR1 expression for everyone.
    print("\nInferring ER status from ESR1 expression...")
    master_df['ER_Status_Inferred'], er_threshold, er_antimode = infer_er_status(master_df['ESR1'])
    print(f"  Mixture threshold: ESR1 >= {er_threshold:.2f} (antimode {er_antimode:.2f})")
    print("  " + master_df['ER_Status_Inferred'].value_counts().to_string().replace("\n", "\n  "))
    known = master_df['ER.Status'].isin(['Positive', 'Negative'])
    agreement = (master_df.loc[known, 'ER.Status'] == master_df.loc[known, 'ER_Status_Inferred']).mean()
    print(f"  Agreement with clinical ER status: {agreement*100:.1f}% (n={known.sum()})")
    print(pd.crosstab(master_df.loc[known, 'ER.Status'], master_df.loc[known, 'ER_Status_Inferred'],
                      rownames=['clinical'], colnames=['inferred']).to_string())

    # Calculate Continuous Scores
    print("\nCalculating Continuous Risk Scores...")
    print("Fetching PAM50 Centroids from R...")
    pam50_centroids = fetch_pam50_centroids()
    
    print("Executing all mathematical models...")
    # Oncotype DX: the unbounded RSU is the score; the bounded RS is kept for reference
    master_df['OncotypeDX_Score'], master_df['OncotypeDX_RS'] = calculate_oncotype_dx_score(master_df)
    # Sensitivity version: genes rescaled on the 1st-99th percentile instead of min-max
    master_df['OncotypeDX_Score_P1_99'], _ = calculate_oncotype_dx_score(master_df, scaling="p1_99")
    master_df['Pam50_Subtype'], master_df['Pam50_Score'] = calculate_pam50_score(
        master_df, pam50_centroids, master_df['ER_Status_Inferred'])
    master_df['BCI_Score'] = calculate_bci_score(master_df, master_df['ER_Status_Inferred'])
    master_df['IRRS7_Score'] = calculate_irrs7_score(master_df)
    master_df['Hu11_Score'] = calculate_hu11_irg_score(master_df)

    # Stratify into Risk Classes
    signatures = [key for key, _ in SIGNATURES]

    # A NaN score would otherwise compare False against every cutoff and be labelled High,
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

    # PAM50 and BCI use their published cutoffs; Oncotype DX (on RSU), IRRS-7 and Hu-11 use cohort tertiles.
    # {sig}_Class_Median keeps the old 0/1 median split for sensitivity analysis.
    print("\nStratifying Patients (Low / Intermediate / High)...")

    for sig in signatures:
        score_col = f'{sig}_Score'
        class_col = f'{sig}_Class'
        if sig in PUBLISHED_CUTOFFS:
            master_df[class_col] = classify_by_cutoffs(master_df[score_col], PUBLISHED_CUTOFFS[sig])
            rule = f"published cutoffs {PUBLISHED_CUTOFFS[sig]}"
        else:
            master_df[class_col] = classify_by_tertiles(master_df[score_col])
            rule = "cohort tertiles"
        master_df[f'{class_col}_Median'] = (master_df[score_col] > master_df[score_col].median()).astype(int)

        counts = master_df[class_col].value_counts()
        print(f"  - {sig} ({rule}): " + ", ".join(f"{c}: {counts.get(c, 0)}" for c in ['Low', 'Intermediate', 'High']))

    # Detect Discordance
    print("\nAnalysing Cohort Discordance...")
    class_df = master_df[[f'{sig}_Class' for sig in signatures]].rename(columns=lambda c: c.removesuffix('_Class'))
    total_patients = len(master_df)

    # Primary: all five signatures. Sensitivity: the four without Hu-11.
    master_df['Discordance_Group'] = assign_cohorts(class_df)
    master_df['Discordance_Group_No_Hu11'] = assign_cohorts(class_df.drop(columns='Hu11'))
    master_df['Consensus_Risk'] = np.where(master_df['Discordance_Group'] != 'concordant', None,
                                           np.where(class_df.eq('High').all(axis=1), 'High Risk', 'Low Risk'))

    for label, group_col, classes in [("Five signatures", 'Discordance_Group', class_df),
                                      ("Without Hu-11", 'Discordance_Group_No_Hu11', class_df.drop(columns='Hu11'))]:
        groups = master_df[group_col]
        n_all_high = (groups.eq('concordant') & classes.eq('High').all(axis=1)).sum()
        print(f"{label}:")
        print(f"  Discordance (>=1 High and >=1 Low):  {groups.eq('discordant').mean()*100:.1f}%")
        print(f"  Expected under independent signatures: {expected_discordance(classes)*100:.1f}%")
        for group in ['concordant', 'discordant', 'borderline']:
            n = groups.eq(group).sum()
            extra = f" (High {n_all_high}, Low {n - n_all_high})" if group == 'concordant' else ""
            print(f"    - {group}: {n} ({n/total_patients*100:.1f}%){extra}")
        print(f"    (all-Intermediate patients, counted as borderline: {classes.eq('Intermediate').all(axis=1).sum()})")

    # Discordant patients where one signature alone takes the opposite side
    is_discordant = master_df['Discordance_Group'] == 'discordant'
    master_df['Lone_Dissenter'] = ""
    master_df.loc[is_discordant, 'Lone_Dissenter'] = lone_dissenter(class_df[is_discordant])
    dissenters = master_df.loc[is_discordant, 'Lone_Dissenter']
    print(f"\nDiscordant patients with a lone dissenting signature: {(dissenters != '').sum()} of {is_discordant.sum()}")
    for sig in signatures:
        print(f"  - {sig}: {(dissenters == sig).sum()}")

    # Data Partitioning for Export
    print("\nExporting Results...")
    
    # Generation and evaluation use the five-signature cohorts
    discordant_df = master_df[master_df['Discordance_Group'] == 'discordant'].copy()
    concordant_df = master_df[master_df['Discordance_Group'] == 'concordant'].copy()
    
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