import os
import time
import inspect
import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import math

load_dotenv()

# ==========================================
# 1. Load Configuration
# ==========================================
CONFIG_PATH = "config.yml"
OUTPUT_DIR = "data/generation_outputs"

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

model_to_use = config["pipeline"]["model_name"]
generation_temp = float(config["pipeline"].get("temperature", 0.2))
api_base = config["pipeline"].get("base_url", "http://127.0.0.1:1234/v1")
api_key = config["pipeline"].get("api_key", "lmstudio")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Loaded configuration: Local LM Studio targeting model '{model_to_use}'")

# ==========================================
# 2. LM Studio Client Initialisation
# ==========================================
client = OpenAI(base_url=api_base, api_key=api_key)

# ==========================================
# 3. Reactome Knowledge Base Integration
# ==========================================
def load_reactome_pathway_index(gmt_path="data/raw/ReactomePathways.gmt"):
    pathway_dict = {}
    if not os.path.exists(gmt_path):
        print(f"\n[WARNING] Reactome mapping file not found at {gmt_path}.")
        print("Please ensure 'ReactomePathways.gmt' is placed in 'data/raw/'.")
        return pathway_dict
        
    with open(gmt_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 2:
                pathway_name = parts[0]
                genes = parts[2:]
                pathway_dict[pathway_name] = set(genes)
    return pathway_dict

REACTOME_INDEX = load_reactome_pathway_index()

# Filter for key breast cancer oncogenic domains to keep prompts concise
TARGET_REACTOME_MODULES = [
    "Cell Cycle",
    "Estrogen-dependent gene expression",
    "Signaling by ERBB2",
    "Extracellular matrix organization",
    "Programmed Cell Death",
    "Cytokine Signaling in Immune system",
    "Metabolism"
]

def get_reactome_active_pathways(upregulated_genes, reactome_index):
    if not reactome_index:
        return "* Reactome database offline or missing."
        
    active_contexts = []
    
    for pathway_name, pathway_genes in reactome_index.items():
        # Match only pathways belonging to our curated oncology domains
        if any(target.lower() in pathway_name.lower() for target in TARGET_REACTOME_MODULES):
            overlap = set(upregulated_genes).intersection(pathway_genes)
            if overlap:
                active_contexts.append(
                    f"**Reactome Pathway:** {pathway_name}\n"
                    f"  - Overlapping Biomarkers: {', '.join(sorted(overlap))}\n"
                    f"  - Relevance: Verified active biological module from Reactome."
                )
                
    if not active_contexts:
        return "* No targeted Reactome oncology pathways uniquely triggered."
        
    return "\n".join(active_contexts[:5]) # Limit to top 5 most relevant to avoid prompt bloat

# ==========================================
# 4. System Instruction & Schema
# ==========================================
SYSTEM_INSTRUCTION = inspect.cleandoc("""
    You are an advanced bioinformatics AI specialising in genomic oncology.
    Your task is to analyse transcriptomic profiles and resolve discordant prognostic risk classifications across multi-gene signatures.

    CRITICAL ANTI-HALLUCINATION CONSTRAINT:
    Base your biological synthesis EXCLUSIVELY on data explicitly provided in this prompt: the "Transcriptomic Profile"
    values and the "Verified Active Biological Pathways" context. You may reason about any gene listed in the
    Transcriptomic Profile using its given expression value, even if that gene did not trigger an entry in the
    Verified Active Biological Pathways section.
    Do not introduce genes, biomarkers, or pathways that are not explicitly present in the provided context.
    Do not invent which specific biomarkers a named signature (e.g. Oncotype DX, PAM50, BCI) mathematically weights
    unless that mapping is stated in the Verified Active Biological Pathways context.
    Do not recommend medical treatments or clinical therapies; focus strictly on prognostic risk classification.
""")

class ArbitrationResult(BaseModel):
    final_risk_class: str = Field(description="Must strictly be either 'High Risk' or 'Low Risk'.")
    arbitration_summary: str = Field(description="The structured markdown summary executing the 3 required arbitration steps: Conflict Diagnosis, Mechanistic Root Cause, and Prognostic Resolution.")

arbitration_schema = {
    "name": "arbitration_result",
    "schema": ArbitrationResult.model_json_schema(),
    "strict": True,
}

# ==========================================
# 5. Core Generation Function
# ==========================================
def generate_patient_summary(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways):
    prompt = inspect.cleandoc(f"""
        Please act as a Bioinformatics Arbitrator to resolve the prognostic discordance for this patient.

        Patient ID: {patient_id}
        
        --- Clinical Metadata ---
        {clinical_data}

        --- Conflicting Algorithmic Risk Classifications (1 = High Risk, 0 = Low Risk) ---
        {signature_classifications}

        --- Transcriptomic Profile (Key Biomarkers & Expression Levels) ---
        {transcriptomic_data}
        
        --- Verified Active Biological Pathways (Reactome Database Extraction) ---
        {active_pathways}
        
        REQUIRED ARBITRATION STEPS:
        1. Conflict Diagnosis: Explicitly state WHICH algorithms are conflicting.
        2. Mechanistic Root Cause: Using the provided pathways, explain EXACTLY why the algorithms disagreed based on how they mathematically weight different biomarkers.
        3. Prognostic Resolution: Deliver a final, tie-breaking risk classification based on the biological evidence provided.
        
        IMPORTANT: Conclude your text with this exact phrase:
        FINAL RESOLUTION: High Risk
        or
        FINAL RESOLUTION: Low Risk
    """)

    try:
        # Step 1: Generate reasoning and extract logprobs
        reasoning_response = client.chat.completions.create(
            model=model_to_use,
            temperature=generation_temp,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            logprobs=True,
            top_logprobs=1
        )
        
        raw_summary = reasoning_response.choices[0].message.content
        
        total_entropy = 0.0
        total_prob = 0.0
        token_count = 0
        min_token_prob = 1.0
        weakest_token = ""
        decision_token_confidence = 0.0

        if reasoning_response.choices[0].logprobs and reasoning_response.choices[0].logprobs.content:
            logprob_data = reasoning_response.choices[0].logprobs.content
            token_count = len(logprob_data)
            
            for token_obj in logprob_data:
                lp = token_obj.logprob 
                prob = math.exp(lp)
                
                total_prob += prob
                total_entropy -= prob * lp
                
                if prob < min_token_prob:
                    min_token_prob = prob
                    weakest_token = token_obj.token.strip()

            # Locate the specific token for the final decision by scanning backwards
            for token_obj in reversed(logprob_data):
                clean_t = token_obj.token.strip().lower()
                if clean_t in ["high", "low"]:
                    decision_token_confidence = round(math.exp(token_obj.logprob) * 100, 2)
                    break

        mean_confidence = round((total_prob / token_count) * 100, 2) if token_count > 0 else 0.0
        mean_entropy = round(total_entropy / token_count, 4) if token_count > 0 else 0.0
        min_confidence = round(min_token_prob * 100, 2) if token_count > 0 else 0.0

        # Step 2: Enforce strict JSON schema
        format_prompt = f"Extract the final risk class and the arbitration summary from the following text:\n\n{raw_summary}"
        
        formatting_response = client.chat.completions.create(
            model=model_to_use,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a strict data extraction assistant."},
                {"role": "user", "content": format_prompt}
            ],
            response_format={"type": "json_schema", "json_schema": arbitration_schema}
        )
        
        parsed_result = ArbitrationResult.model_validate_json(formatting_response.choices[0].message.content)

        return (
            parsed_result.final_risk_class,
            parsed_result.arbitration_summary,
            mean_confidence,
            mean_entropy,
            decision_token_confidence,
            min_confidence,
            weakest_token
        )
        
    except Exception as e:
        print(f"  [ERROR] Generating summary for {patient_id}: {e}")
        return "Error", f"Error: {e}", 0.0, 0.0, 0.0, 0.0, ""

# ==========================================
# 6. Data Ingestion & Batch Execution
# ==========================================
def process_cohort(input_filename, output_filename):
    if not os.path.exists(input_filename):
        print(f"File not found: {input_filename}. Skipping cohort.")
        return

    df = pd.read_csv(input_filename, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "patient_id"})

    sig_columns = [
        ("Oncotype DX", "OncotypeDX_Class"), ("PAM50", "Pam50_Class"),
        ("Breast Cancer Index", "BCI_Class"), ("Mammostrat", "Mammostrat_Class"),
        ("IHC4", "IHC4_Class"), ("Kim-10", "Kim10_Class"),
        ("IRRS-7", "IRRS7_Class"), ("Hu-11", "Hu11_Class")
    ]

    transcriptomic_columns = [
    # Oncotype DX & IHC4 Specific
    "SCUBE2", "AURKA", "CTSL2", "CD68", "GSTM1", 
    
    # PAM50 (Includes overlapping Oncotype & IHC4 genes like ERBB2, ESR1, PGR, MKI67)
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", 
    "CDC20", "CDC6", "CDCA1", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", 
    "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1", "GPR160", "GRB7", "KIF2C", 
    "KNTC2", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", 
    "MKI67", "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "ORC6L", "PGR", 
    "PHGDH", "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TYMS", 
    "UBE2C", "UBE2T",

    # Breast Cancer Index (BCI)
    "HOXB13", "IL17RB", "BUB1B", "CENPA", "NEK2", "RACGAP1",
    
    # Mammostrat Proxy Genes
    "TP53", "CEACAM5", "NDRG1", "SLC7A5", "TRMT10C", "HTF9C", "RG9MTD1",
    
    # Kim-10 TNBC Signature
    "DGKH", "GADD45B", "KLF7", "LYST", "NR6A1", "PYCARD", "ROBO1", 
    "SLC22A20P", "SLC24A3", "SLC45A4",
    
    # IRRS-7 Insulin Resistance Signature
    "EZR", "LIFR", "TBC1D4", "SAA1", "NSF", "RPL5", "PGK1",
    
    # Hu-11 Inflammation Signature
    "IL18", "IL12B", "RASGRP1", "HPN", "CLEC5A", "SCARF1", "TACR3", 
    "VIP", "CCL2", "CALCRL", "ABCA1"
    ]
    
    clinical_fields = [("years_to_birth", "Age at diagnosis"), ("ER.Status", "ER status"), ("pathologic_stage", "Pathologic stage")]

    available_tx_cols = [g for g in transcriptomic_columns if g in df.columns]
    gene_medians = df[available_tx_cols].median()

    results = []
    # Set to sample_df = df to run the entire cohort
    sample_df = df.head(6)

    print(f"\nProcessing cohort from {input_filename} ({len(sample_df)} cases to process)...")

    for _, row in sample_df.iterrows():
        p_id = row['patient_id']
        print(f"Processing Patient: {p_id}...")

        clinical_meta = "\n".join(f"{label}: {row[col]}" for col, label in clinical_fields if col in row and pd.notna(row[col]))
        sig_classifications = "\n".join(f"{label}: {row[col]}" for label, col in sig_columns if col in row and pd.notna(row[col]))
        transcriptomics = "\n".join(f"{gene}: {row[gene]:.4f}" if isinstance(row[gene], (float, int)) else f"{gene}: {row[gene]}" for gene in available_tx_cols if pd.notna(row[gene]))

        # Identify upregulated genes
        upregulated_genes = [gene for gene in available_tx_cols if pd.notna(row[gene]) and row[gene] > gene_medians[gene]]
        
        # Get Reactome pathways for the upregulated genes
        extracted_pathways_str = get_reactome_active_pathways(upregulated_genes, REACTOME_INDEX)

        final_risk, summary, confidence, entropy, decision_conf, min_conf, weak_tok = generate_patient_summary(
            patient_id=p_id,
            clinical_data=clinical_meta,
            signature_classifications=sig_classifications,
            transcriptomic_data=transcriptomics,
            active_pathways=extracted_pathways_str
        )

        record = row.to_dict()
        record.update({
            'final_risk_class': final_risk,
            'model_confidence_percent': confidence,
            'model_entropy_score': entropy,
            'decision_token_confidence': decision_conf,
            'min_token_confidence': min_conf,
            'weakest_token': weak_tok,
            'clinical_data': clinical_meta,
            'signature_classifications': sig_classifications,
            'transcriptomic_data': transcriptomics,
            'active_pathways': extracted_pathways_str,
            'lmstudio_summary': summary
        })
        results.append(record)
        time.sleep(1)

    pd.DataFrame(results).to_csv(output_filename, index=False)
    print(f"Results saved to: {output_filename}")

if __name__ == "__main__":
    process_cohort("data/processed/tcga_concordant_cases.csv", os.path.join(OUTPUT_DIR, "concordant_results.csv"))
    process_cohort("data/processed/tcga_discordant_cases.csv", os.path.join(OUTPUT_DIR, "discordant_results.csv"))