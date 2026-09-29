import pandas as pd

PRIMARY_TUMOUR_CODE = "01"

def load_and_merge_tcga(rna_path, clinical_path):
    print("Loading datasets...")
    # Add index_col=0 to clinical_df so the row names become column headers
    rna_df = pd.read_csv(rna_path, sep='\t', index_col=0)
    clinical_df = pd.read_csv(clinical_path, sep='\t', index_col=0)

    # Transpose RNA data
    rna_df = rna_df.T

    # If barcodes carry a sample-type code (e.g. TCGA-XX-XXXX-01A, positions 13-14), keep primary tumours only:
    # normal (11) and metastatic (06) samples would otherwise collapse onto the same patient ID.
    # Patient-level files (e.g. Firehose 'TCGA.3C.AAAU') have no such code and are used as they are.
    barcodes = rna_df.index.astype(str)
    if barcodes.str.len().min() >= 15:
        is_primary = barcodes.str[13:15] == PRIMARY_TUMOUR_CODE
        print(f"Keeping {is_primary.sum()}/{len(rna_df)} primary tumour RNA samples.")
        rna_df = rna_df[is_primary]
    rna_df.index = rna_df.index.astype(str).str[:12]

    # Transpose Clinical data and clean barcodes
    clinical_df = clinical_df.T
    clinical_df.index = clinical_df.index.astype(str).str[:12]

    # Multiple vials/portions of one tumour can remain; keep the first per patient
    rna_df = rna_df[~rna_df.index.duplicated(keep='first')]
    clinical_df = clinical_df[~clinical_df.index.duplicated(keep='first')]

    # Merge on Patient ID
    merged_data = pd.merge(clinical_df, rna_df, left_index=True, right_index=True, how='inner')
    assert merged_data.index.is_unique, "Duplicate patient IDs after merge."
    if merged_data.empty:
        raise ValueError(
            f"No patients matched between RNA and clinical files. Example RNA IDs: {list(rna_df.index[:3])}; "
            f"clinical IDs: {list(clinical_df.index[:3])}"
        )
    print(f"Merge complete. Final dataset has {merged_data.shape[0]} patients and {merged_data.shape[1]} features.")

    return merged_data
