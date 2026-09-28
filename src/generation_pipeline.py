import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field
import argparse
from run_paths import generation_path, load_config
from signature_definitions import get_all_signature_genes, SIGNATURES
from arbitration import SYSTEM_INSTRUCTION, ABLATIONS, build_prompt, parse_final_resolution

load_dotenv()

# ==========================================
# 1. Load Configuration
# ==========================================
CONFIG_PATH = "config.yml"
OUTPUT_DIR = "data/generation_outputs"
MASTER_FILE = "data/processed/tcga_master_results.csv"
DEFAULT_LIMIT = 6  # quick-test size; pass --limit 0 for the final full run

config = load_config(CONFIG_PATH)

model_to_use = config["pipeline"]["model_name"]
generation_temp = float(config["pipeline"].get("temperature", 0.2))
api_base = config["pipeline"]["base_url"]
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
# 4. Extraction Schema
# ==========================================
# SYSTEM_INSTRUCTION, ABLATIONS and build_prompt live in arbitration.py (pure, no side
# effects), so semantic_entropy.py can import them without pulling in this script's
# config/client/Reactome setup.
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
def generate_patient_summary(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways, ablation="full"):
    prompt = build_prompt(patient_id, clinical_data, signature_classifications, transcriptomic_data, active_pathways, ablation)

    try:
        # Step 1: Generate the reasoning text
        reasoning_response = client.chat.completions.create(
            model=model_to_use,
            temperature=generation_temp,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ]
        )

        raw_summary = reasoning_response.choices[0].message.content

        # Token-logprob confidence/entropy is disabled for now; semantic entropy
        # (src/semantic_entropy.py) is the uncertainty measure. See TODO.md and
        # git history for the previous implementation if it needs reviving.

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

        # Prefer the class stated in the reasoning text itself; fall back to the extraction call.
        final_risk_class = parse_final_resolution(raw_summary) or parsed_result.final_risk_class

        return final_risk_class, parsed_result.arbitration_summary
        
    except Exception as e:
        print(f"  [ERROR] Generating summary for {patient_id}: {e}")
        return "Error", f"Error: {e}"

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

    sig_columns = [(label, f"{key}_Class") for key, label in SIGNATURES]

    # Every gene symbol used by any signature (single source of truth: signature_definitions.py)
    transcriptomic_columns = get_all_signature_genes()

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

        final_risk, summary = generate_patient_summary(
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
            # 'model_confidence_percent', 'model_entropy_score', 'decision_token_confidence',
            # 'min_token_confidence' and 'weakest_token' (logprob fields) are disabled for now.
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
