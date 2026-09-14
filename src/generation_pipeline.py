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
INPUT_FILENAME = "data/processed/tcga_discordant_cases.csv"
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
# 3. Multi-Axis RAG Knowledge Base 
# ==========================================
ONCOLOGY_PATHWAYS = {
    "Cellular Proliferation & Mitotic Progression": {
        "genes": ["MKI67", "AURKA", "CCNB1", "BIRC5", "MYBL2", "BUB1B", "CENPA", "NEK2", "RACGAP1", "RRM2"],
        "mechanism": "Drives mitotic spindle assembly, DNA replication, and uncontrolled cell cycle progression.",
        "prognostic_consensus": "Elevated expression strongly weights signatures toward High Risk (e.g., PAM50, Oncotype DX, BCI-MGI, IHC4)."
    },
    "Estrogen Receptor (ER) & Luminal Differentiation": {
        "genes": ["ESR1", "PGR", "FOXA1", "FOXC1", "GPR160", "MAPT", "NAT1", "SLC39A6", "SCUBE2"],
        "mechanism": "Luminal-driven transcriptional activity promoting hormone-dependent tumor maintenance.",
        "prognostic_consensus": "Associated with favorable prognosis in early stages, driving signatures toward Low Risk (e.g., Oncotype DX, IHC4, PAM50 LumA)."
    },
    "HER2 / Receptor Tyrosine Kinase Signaling": {
        "genes": ["ERBB2", "GRB7", "EGFR", "FGFR4"],
        "mechanism": "Receptor tyrosine kinase amplification driving aggressive cell proliferation and downstream MAPK/AKT activation.",
        "prognostic_consensus": "Indicates high intrinsic aggressiveness, strongly pushing classifications toward High Risk."
    },
    "Apoptosis & Cell Cycle Checkpoint Arrest": {
        "genes": ["BCL2", "BAG1", "TP53", "MDM2", "GADD45B", "PYCARD"],
        "mechanism": "Regulation of programmed cell death and genomic stability checkpoint maintenance.",
        "prognostic_consensus": "High anti-apoptotic BCL2/BAG1 is favorable in ER+ disease, while mutated TP53 or aberrant GADD45B/PYCARD indicates aggressive phenotype (Mammostrat, Kim-10)."
    },
    "Immune & Inflammatory Microenvironment": {
        "genes": ["IL18", "IL12B", "RASGRP1", "HPN", "CLEC5A", "SCARF1", "TACR3", "VIP", "CCL2", "CALCRL", "ABCA1"],
        "mechanism": "Cytokine signaling, immune cell recruitment, and inflammatory modulation within the tumor stroma.",
        "prognostic_consensus": "Elevated pro-inflammatory signaling correlates with immune evasion and higher recurrence risk (Hu-11 IRG Signature)."
    },
    "Insulin Resistance & Metabolic Dysregulation": {
        "genes": ["EZR", "LIFR", "TBC1D4", "SAA1", "NSF", "RPL5", "PGK1"],
        "mechanism": "Glycolytic reprogramming, insulin signaling impairment, and metabolic adaptation.",
        "prognostic_consensus": "Upregulation of glycolytic drivers (e.g., PGK1) correlates with metabolic stress and aggressive disease (IRRS-7 Signature)."
    },
    "Tumor Invasiveness & Hox Gene Dysregulation": {
        "genes": ["HOXB13", "IL17RB", "MMP11", "CTSL2", "CEACAM5", "NDRG1", "SLC7A5"],
        "mechanism": "Extracellular matrix degradation, stromal remodeling, and aberrant homeobox transcriptional activation.",
        "prognostic_consensus": "Elevated HOXB13 over IL17RB (H/I ratio) and active stromal remodeling indicate late recurrence risk (BCI, Mammostrat)."
    }
}

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

# ==========================================
# 4. Structured Output Schema
# ==========================================
class ArbitrationResult(BaseModel):
    final_risk_class: str = Field(
        description="The final resolved consensus risk classification. Must strictly be either 'High Risk' or 'Low Risk'."
    )
    arbitration_summary: str = Field(
        description="The structured markdown summary executing the 3 required arbitration steps: Conflict Diagnosis, Mechanistic Root Cause, and Prognostic Resolution."
    )

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
        
        --- Verified Active Biological Pathways (Rule-Based RAG Extraction) ---
        {active_pathways}
        
        REQUIRED ARBITRATION STEPS (to write inside arbitration_summary):
        1. Conflict Diagnosis: Explicitly state WHICH algorithms are conflicting.
        2. Mechanistic Root Cause: Using the provided pathways, explain EXACTLY why the algorithms disagreed based on how they mathematically weight different biomarkers.
        3. Prognostic Resolution: Deliver a final, tie-breaking risk classification (High Risk vs. Low Risk) based on the biological evidence provided.
    """)

    try:
        # STEP 1: Generate the reasoning and capture the logprobs (Entropy)
        reasoning_response = client.chat.completions.create(
            model=model_to_use,
            temperature=generation_temp,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ],
            logprobs=True,
            top_logprobs=1 # Forces the LM Studio backend to process the calculations
        )
        
        raw_summary = reasoning_response.choices[0].message.content
        
        # Calculate the Entropy and Confidence Safely
        total_entropy = 0
        total_prob = 0
        token_count = 0

        # Safety check: Only process logprobs if the server actually returned them
        if reasoning_response.choices[0].logprobs and reasoning_response.choices[0].logprobs.content:
            logprob_data = reasoning_response.choices[0].logprobs.content
            token_count = len(logprob_data)
            
            for token in logprob_data:
                lp = token.logprob 
                prob = math.exp(lp) 
                
                total_prob += prob
                total_entropy -= prob * lp 

        mean_confidence = round((total_prob / token_count) * 100, 2) if token_count > 0 else 0
        mean_entropy = round(total_entropy / token_count, 4) if token_count > 0 else 0

        # STEP 2: Force the JSON structure using your preferred API schema
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

        return parsed_result.final_risk_class, parsed_result.arbitration_summary, mean_confidence, mean_entropy
        
    except Exception as e:
        print(f"  [ERROR] Generating summary for {patient_id}: {e}")
        return "Error", f"Error: {e}", 0.0, 0.0

# ==========================================
# 6. Data Ingestion & Batch Execution
# ==========================================
if not os.path.exists(INPUT_FILENAME):
    raise FileNotFoundError(f"Could not find {INPUT_FILENAME}. Run preprocessing first.")

df = pd.read_csv(INPUT_FILENAME, low_memory=False)

if "Unnamed: 0" in df.columns:
    df = df.rename(columns={"Unnamed: 0": "patient_id"})

sig_columns = [
    ("Oncotype DX", "OncotypeDX_Class"),
    ("PAM50", "Pam50_Class"),
    ("Breast Cancer Index", "BCI_Class"),
    ("Mammostrat", "Mammostrat_Class"),
    ("IHC4", "IHC4_Class"),
    ("Kim-10", "Kim10_Class"),
    ("IRRS-7", "IRRS7_Class"),
    ("Hu-11", "Hu11_Class")
]

transcriptomic_columns = [
    "MKI67", "ESR1", "ERBB2", "PGR", "AURKA", "BCL2",
    "TP53", "HOXB13", "IL17RB", "PGK1", "CCL2", "GADD45B"
]
clinical_fields = [("years_to_birth", "Age at diagnosis"), ("ER.Status", "ER status"), ("pathologic_stage", "Pathologic stage")]

available_tx_cols = [g for g in transcriptomic_columns if g in df.columns]
gene_medians = df[available_tx_cols].median()

results = []
sample_df = df.head(5)

print(f"\nInitiating Local LM Studio Pipeline ({len(sample_df)} cases to process)...\n")

for _, row in sample_df.iterrows():
    p_id = row['patient_id']
    print(f"Processing Patient: {p_id}...")

    clinical_meta = "\n".join(f"{label}: {row[col]}" for col, label in clinical_fields if col in row and pd.notna(row[col]))
    sig_classifications = "\n".join(f"{label}: {row[col]}" for label, col in sig_columns if col in row and pd.notna(row[col]))
    transcriptomics = "\n".join(f"{gene}: {row[gene]:.4f}" if isinstance(row[gene], (float, int)) else f"{gene}: {row[gene]}" for gene in available_tx_cols if pd.notna(row[gene]))

    active_contexts = []
    for gene in available_tx_cols:
        if pd.notna(row[gene]) and row[gene] > gene_medians[gene]:
            for pathway_name, pathway_data in ONCOLOGY_PATHWAYS.items():
                if gene in pathway_data["genes"]:
                    context_block = (
                        f"**Pathway:** {pathway_name} (Triggered by high {gene})\n"
                        f"  - Mechanism: {pathway_data['mechanism']}\n"
                        f"  - Prognostic Consensus: {pathway_data['prognostic_consensus']}"
                    )
                    if context_block not in active_contexts:
                        active_contexts.append(context_block)

    extracted_pathways_str = "\n".join(active_contexts) if active_contexts else "* No uniquely upregulated pathways identified."

    final_risk, summary, confidence, entropy = generate_patient_summary(
        patient_id=p_id,
        clinical_data=clinical_meta,
        signature_classifications=sig_classifications,
        transcriptomic_data=transcriptomics,
        active_pathways=extracted_pathways_str
    )

    record = {
        'patient_id': p_id,
        'OncotypeDX_Class': row.get('OncotypeDX_Class'),
        'Pam50_Class': row.get('Pam50_Class'),
        'BCI_Class': row.get('BCI_Class'),
        'Mammostrat_Class': row.get('Mammostrat_Class'),
        'IHC4_Class': row.get('IHC4_Class'),
        'Kim10_Class': row.get('Kim10_Class'),
        'IRRS7_Class': row.get('IRRS7_Class'),
        'Hu11_Class': row.get('Hu11_Class'),
        'final_risk_class': final_risk,
        'model_confidence_percent': confidence,
        'model_entropy_score': entropy,
        'clinical_data': clinical_meta,
        'signature_classifications': sig_classifications,
        'transcriptomic_data': transcriptomics,
        'active_pathways': extracted_pathways_str,
        'lmstudio_summary': summary
    }
    results.append(record)
    time.sleep(1)

output_filename = os.path.join(OUTPUT_DIR, "lmstudio_generation_results.csv")
pd.DataFrame(results).to_csv(output_filename, index=False)

print(f"\nPipeline execution complete. Results saved to: {output_filename}")