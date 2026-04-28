import pandas as pd

def load_and_merge_tcga(rna_path, clinical_path):
    print("Loading datasets...")
    # Add index_col=0 to clinical_df so the row names become column headers
    rna_df = pd.read_csv(rna_path, sep='\t', index_col=0) 
    clinical_df = pd.read_csv(clinical_path, sep='\t', index_col=0)

    # Transpose RNA data and clean barcodes
    rna_df = rna_df.T 
    rna_df.index = rna_df.index.str[:12]

    # Transpose Clinical data and clean barcodes
    clinical_df = clinical_df.T
    clinical_df.index = clinical_df.index.str[:12]

    # Merge on Patient ID
    merged_data = pd.merge(clinical_df, rna_df, left_index=True, right_index=True, how='inner')
    print(f"Merge complete. Final dataset has {merged_data.shape[0]} patients and {merged_data.shape[1]} features.")
    
    return merged_data