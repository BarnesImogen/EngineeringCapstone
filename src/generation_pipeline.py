import os
import inspect
import yaml
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import argparse
from run_paths import generation_path
from signature_definitions import get_signature_reference_text
import math
import re

load_dotenv()

# ==========================================
# 1. Load Configuration
# ==========================================
CONFIG_PATH = "config.yml"
OUTPUT_DIR = "data/generation_outputs"
MASTER_FILE = "data/processed/tcga_master_results.csv"
DEFAULT_LIMIT = 6  # quick-test size; pass --limit 0 for the final full run

with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)

model_to_use = config["pipeline"]["model_name"]
generation_temp = float(config["pipeline"].get("temperature", 0.2))
api_base = config["pipeline"].get("base_url", "http://127.0.0.1:1234/v1")
api_key = os.getenv("GENERATION_API_KEY") or config["pipeline"].get("api_key", "lmstudio")

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
    Base your biological synthesis EXCLUSIVELY on data explicitly provided in this prompt: the "Signature Definitions",
    and, when they are present, the "Clinical Metadata", the "Transcriptomic Profile" values and the "Verified Active
    Biological Pathways" context. You may reason about any gene listed in the Transcriptomic Profile using its given
    expression value, even if that gene did not trigger an entry in the Verified Active Biological Pathways section.
    Do not introduce genes, biomarkers, or pathways that are not explicitly present in the provided context.
    Describe how a named signature (e.g. Oncotype DX, PAM50, BCI) weights biomarkers only as stated in the
    Signature Definitions; do not invent other weights or gene memberships.
    If a section of evidence is not provided, do not make claims that would require it.
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
# Which evidence blocks the model sees. Signature classifications are always included.
ABLATIONS = {
    "full":          {"clinical": True,  "expression": True,  "pathways": True},
    "no_pathways":   {"clinical": True,  "expression": True,  "pathways": False},
    "clinical_only": {"clinical": True,  "expression": False, "pathways": False},
    "classes_only":  {"clinical": False, "expression": False, "pathways": False},
}

FINAL_RESOLUTION_PATTERN = re.compile(r"FINAL RESOLUTION:\s*\**\s*(High|Low)\s+Risk", re.IGNORECASE)

def parse_final_resolution(text):
    """Return 'High Risk' or 'Low Risk' from the last FINAL RESOLUTION line, or None if absent."""
    matches = FINAL_RESOLUTION_PATTERN.findall(text or "")
    if not matches:
        return None
    return f"{matches[-1].capitalize()} Risk"

def build_prompt(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways, ablation="full"):
    include = ABLATIONS[ablation]
    sections = [
        "Please act as a Bioinformatics Arbitrator to resolve the prognostic discordance for this patient.",
        f"Patient ID: {patient_id}",
    ]
    if include["clinical"]:
        sections.append(f"--- Clinical Metadata ---\n{clinical_data}")
    # Static description of the algorithms; shown in every ablation because it is not patient data
    sections.append(f"--- Signature Definitions (how each algorithm scores a patient) ---\n{get_signature_reference_text()}")
    sections.append(
        "--- Conflicting Algorithmic Risk Classifications (1 = High Risk, 0 = Low Risk) ---\n"
        f"{signature_classifications}"
    )
    if include["expression"]:
        sections.append(f"--- Transcriptomic Profile (Key Biomarkers & Expression Levels) ---\n{transcriptomic_data}")
    if include["pathways"]:
        sections.append(f"--- Verified Active Biological Pathways (Reactome Database Extraction) ---\n{active_pathways}")

    evidence = "the Signature Definitions and the provided pathways" if include["pathways"] else "the Signature Definitions and the provided evidence"
    sections.append(
        "REQUIRED ARBITRATION STEPS:\n"
        "1. Conflict Diagnosis: Explicitly state WHICH algorithms are conflicting.\n"
        f"2. Mechanistic Root Cause: Using {evidence}, explain EXACTLY why the algorithms disagreed based on how they mathematically weight different biomarkers.\n"
        "3. Prognostic Resolution: Deliver a final, tie-breaking risk classification based on the biological evidence provided."
    )
    sections.append(
        "IMPORTANT: Conclude your text with this exact phrase:\n"
        "FINAL RESOLUTION: High Risk\n"
        "or\n"
        "FINAL RESOLUTION: Low Risk"
    )
    return "\n\n".join(sections)

def generate_patient_summary(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways, ablation="full"):
    prompt = build_prompt(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways, ablation)

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

        # Prefer the class stated in the reasoning text itself, so the label matches the
        # tokens the decision confidence was measured on; fall back to the extraction call.
        final_risk_class = parse_final_resolution(raw_summary) or parsed_result.final_risk_class

        return (
            final_risk_class,
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
def process_cohort(input_filename, output_filename, ablation="full", limit=DEFAULT_LIMIT, resume=False):
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
    "CDC20", "CDC6", "CDCA1", "CDH3", "CENPF", "CEP55", "CXXC5", "EGFR", "ERBB2", 
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

    # Medians come from the full cohort so "upregulated" means the same thing for concordant and discordant cases
    if os.path.exists(MASTER_FILE):
        gene_medians = pd.read_csv(MASTER_FILE, usecols=available_tx_cols).median()
    else:
        print(f"[WARNING] {MASTER_FILE} not found; falling back to medians from {input_filename} only.")
        gene_medians = df[available_tx_cols].median()

    # limit=0 runs the entire cohort
    sample_df = df if limit == 0 else df.head(limit)
    include = ABLATIONS[ablation]

    # Rows are appended to the output file as they finish, so a crash loses at most one case.
    # --resume keeps the cases that already succeeded and retries the rest; otherwise start fresh.
    done_ids = set()
    if resume and os.path.exists(output_filename):
        previous = pd.read_csv(output_filename, low_memory=False)
        succeeded = previous[previous["final_risk_class"].isin(["High Risk", "Low Risk"])]
        if len(succeeded) < len(previous):
            print(f"Resuming: dropping {len(previous) - len(succeeded)} failed rows so they are retried.")
        succeeded.to_csv(output_filename, index=False)
        done_ids = set(succeeded["patient_id"])
        print(f"Resuming: {len(done_ids)} cases already done in {output_filename}.")
    elif os.path.exists(output_filename):
        print(f"[NOTE] Overwriting existing {output_filename} (use --resume to continue it instead).")
        os.remove(output_filename)

    print(f"\nProcessing cohort from {input_filename} ({len(sample_df)} cases to process, ablation='{ablation}')...")

    n_total = len(sample_df)
    for position, (_, row) in enumerate(sample_df.iterrows(), start=1):
        p_id = row['patient_id']
        if p_id in done_ids:
            continue
        print(f"[{position}/{n_total}] Processing Patient: {p_id}...")

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
            active_pathways=extracted_pathways_str,
            ablation=ablation
        )

        record = row.to_dict()
        # Log only the context the model actually saw, so the judge audits against the right evidence
        record.update({
            'ablation': ablation,
            'final_risk_class': final_risk,
            'model_confidence_percent': confidence,
            'model_entropy_score': entropy,
            'decision_token_confidence': decision_conf,
            'min_token_confidence': min_conf,
            'weakest_token': weak_tok,
            'clinical_data': clinical_meta if include["clinical"] else "",
            'signature_classifications': sig_classifications,
            'transcriptomic_data': transcriptomics if include["expression"] else "",
            'active_pathways': extracted_pathways_str if include["pathways"] else "",
            'lmstudio_summary': summary
        })
        append_result(output_filename, record)

    print(f"Results saved to: {output_filename}")

def append_result(output_filename, record):
    """Append one result row, writing the header only for a new file."""
    new_row = pd.DataFrame([record])
    if os.path.exists(output_filename):
        existing_columns = list(pd.read_csv(output_filename, nrows=0).columns)
        if existing_columns != list(new_row.columns):
            raise ValueError(f"{output_filename} was written with different columns; rerun without --resume.")
        new_row.to_csv(output_filename, mode="a", header=False, index=False)
    else:
        new_row.to_csv(output_filename, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate arbitration summaries for TCGA-BRCA cases.")
    parser.add_argument("--model", default=None,
                        help="Generation model name (defaults to pipeline.model_name in config.yml); used in output filenames.")
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="full",
                        help="Which evidence the model sees (non-full runs are saved with a suffix).")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Cases per cohort (default {DEFAULT_LIMIT} for quick tests; 0 = whole cohort).")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run: keep finished cases, retry failed or missing ones.")
    parser.add_argument("--cohorts", nargs="+", choices=["concordant", "discordant"], default=["concordant", "discordant"])
    args = parser.parse_args()
    if args.model:
        model_to_use = args.model
    print(f"Generating with model '{model_to_use}'")

    for cohort in args.cohorts:
        process_cohort(f"data/processed/tcga_{cohort}_cases.csv", generation_path(cohort, model_to_use, args.ablation),
                       ablation=args.ablation, limit=args.limit, resume=args.resume)
