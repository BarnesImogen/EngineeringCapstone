import os
import pandas as pd
from dotenv import load_dotenv
from pydantic import BaseModel, Field
import argparse
from typing import Literal
from run_paths import generation_path, load_config
from signature_definitions import get_all_signature_genes, get_class_cutoffs, format_signature_class, SIGNATURES
from arbitration import SYSTEM_INSTRUCTION, ABLATIONS, build_prompt, parse_final_resolution, majority_vote
from llm_requests import (FatalAPIError, RunStopped, SAVE_EVERY, add_workers_arg, chat, make_client,
                          run_cli, run_in_pool, save_csv)

load_dotenv()

# ==========================================
# 1. Load Configuration
# ==========================================
CONFIG_PATH = "config.yml"
OUTPUT_DIR = "data/generation_outputs"
MASTER_FILE = "data/processed/tcga_master_results.csv"
DEFAULT_LIMIT = 6  # quick-test size; pass --limit 0 for the final full run
PILOT_SEED = 42  # fixed so a stratified pilot draws the same patients again (and --resume works)

config = load_config(CONFIG_PATH)

model_to_use = config["pipeline"]["model_name"]
generation_temp = float(config["pipeline"].get("temperature", 0.2))
api_base = config["pipeline"]["base_url"]
api_key = os.getenv("GENERATION_API_KEY") or config["pipeline"].get("api_key", "lmstudio")

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"Loaded configuration: generation model '{model_to_use}' at {api_base}")

# ==========================================
# 2. API Client Initialisation
# ==========================================
client = make_client(api_base, api_key)

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
# Only used to recover the class when the reasoning text has no FINAL RESOLUTION line; the
# summary itself is always the model's full step-1 text.
class ArbitrationResult(BaseModel):
    final_risk_class: Literal["High Risk", "Low Risk"] = Field(description="Must strictly be either 'High Risk' or 'Low Risk'.")

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
        reasoning_response = chat(
            client,
            model=model_to_use,
            temperature=generation_temp,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt}
            ]
        )

        raw_summary = reasoning_response.choices[0].message.content
        if not (raw_summary or "").strip():
            # Nothing to store or extract from; the extraction call would invent a class
            raise ValueError("model returned an empty response")

        # Token-logprob confidence/entropy is disabled for now; semantic entropy
        # (src/semantic_entropy.py) is the uncertainty measure. See TODO.md and
        # git history for the previous implementation if it needs reviving.

        # The class stated in the reasoning text itself
        final_risk_class = parse_final_resolution(raw_summary)

        # Step 2 (fallback only): no FINAL RESOLUTION line, so extract the class with a strict JSON schema.
        # The stored summary stays the full step-1 text, so the judge audits the real reasoning.
        if final_risk_class is None:
            format_prompt = f"Extract the final risk class from the following text:\n\n{raw_summary}"

            formatting_response = chat(
                client,
                model=model_to_use,
                temperature=0.0,
                messages=[
                    {"role": "system", "content": "You are a strict data extraction assistant."},
                    {"role": "user", "content": format_prompt}
                ],
                response_format={"type": "json_schema", "json_schema": arbitration_schema}
            )

            parsed_result = ArbitrationResult.model_validate_json(formatting_response.choices[0].message.content)
            final_risk_class = parsed_result.final_risk_class

        return final_risk_class, raw_summary

    except (FatalAPIError, RunStopped):
        raise  # stop the run / leave the case undone, rather than logging an Error row
    except Exception as e:
        print(f"  [ERROR] Generating summary for {patient_id}: {e}")
        return "Error", f"Error: {e}"

# ==========================================
# 6. Data Ingestion & Batch Execution
# ==========================================
CLINICAL_FIELDS = [
    ("years_to_birth", "Age at diagnosis"),
    ("ER.Status", "ER status"),
    ("pathologic_stage", "Pathologic stage"),
    ("pathology_T_stage", "T stage"),
    ("pathology_N_stage", "N stage (nodal status)"),
    ("pathology_M_stage", "M stage"),
    ("histological_type", "Histological type"),
    ("number_of_lymph_nodes", "Positive lymph nodes"),
    ("PR.Status", "PR status"),
    ("HER2.Status", "HER2 status"),
]

SIG_COLUMNS = [(key, label, f"{key}_Class") for key, label in SIGNATURES]

def load_cohort(input_filename):
    df = pd.read_csv(input_filename, low_memory=False)
    if "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "patient_id"})
    return df

def load_gene_medians(df, available_tx_cols, input_filename):
    # Medians come from the full cohort so "upregulated" means the same thing for concordant and discordant cases
    if os.path.exists(MASTER_FILE):
        return pd.read_csv(MASTER_FILE, usecols=available_tx_cols).median()
    print(f"[WARNING] {MASTER_FILE} not found; falling back to medians from {input_filename} only.")
    return df[available_tx_cols].median()

def load_class_cutoffs():
    """Cutoffs behind each signature's class. Tertile cut points must come from the whole cohort
    (the master file), not from the concordant/discordant subset being processed."""
    if not os.path.exists(MASTER_FILE):
        raise FileNotFoundError(f"{MASTER_FILE} not found; run bioinformatics_pipeline.py first.")
    return get_class_cutoffs(pd.read_csv(MASTER_FILE, usecols=[f"{key}_Score" for key, _ in SIGNATURES]))

def clinical_lines(row):
    """Clinical metadata lines. Recorded ER.Status exists for only ~11% of patients, so when it is missing
    the ESR1-inferred status (bioinformatics_pipeline.py) is shown instead, labelled as inferred."""
    lines = []
    for col, label in CLINICAL_FIELDS:
        if col in row and pd.notna(row[col]):
            lines.append(f"{label}: {row[col]}")
        elif col == "ER.Status" and pd.notna(row.get("ER_Status_Inferred")):
            lines.append(f"ER status (inferred from ESR1 expression): {row['ER_Status_Inferred']}")
    return lines

def build_case_context(row, available_tx_cols, gene_medians, class_cutoffs):
    """The evidence blocks for one patient: (clinical, signature classes, transcriptomics, Reactome pathways)."""
    clinical_meta = "\n".join(clinical_lines(row))
    sig_classifications = "\n".join(
        format_signature_class(key, label, row[col], row[f"{key}_Score"], class_cutoffs[key])
        for key, label, col in SIG_COLUMNS if col in row and pd.notna(row[col])
    )
    transcriptomics = "\n".join(f"{gene}: {row[gene]:.4f}" if isinstance(row[gene], (float, int)) else f"{gene}: {row[gene]}" for gene in available_tx_cols if pd.notna(row[gene]))

    # Identify upregulated genes
    upregulated_genes = [gene for gene in available_tx_cols if pd.notna(row[gene]) and row[gene] > gene_medians[gene]]

    # Get Reactome pathways for the upregulated genes
    extracted_pathways_str = get_reactome_active_pathways(upregulated_genes, REACTOME_INDEX)
    return clinical_meta, sig_classifications, transcriptomics, extracted_pathways_str

# Pilot strata for the discordant cohort, from Lone_Dissenter: Hu-11, IRRS-7, any other signature, none
LONE_DISSENTER_STRATA = {"Hu11": "Hu-11", "IRRS7": "IRRS-7"}
PILOT_STRATA = ["Hu-11", "IRRS-7", "other", "none"]

def lone_dissenter_stratum(value):
    if pd.isna(value) or value == "":
        return "none"
    return LONE_DISSENTER_STRATA.get(value, "other")

def pilot_strata(df, cohort):
    """Stratum per patient for a pilot draw: the lone dissenting signature for discordant cases,
    the consensus class for concordant cases."""
    if cohort == "discordant":
        return df["Lone_Dissenter"].map(lone_dissenter_stratum), PILOT_STRATA
    return df["Consensus_Risk"], ["High Risk", "Low Risk"]

def stratified_sample(df, n, cohort, seed=PILOT_SEED):
    """Up to n patients spread as evenly as possible across the cohort's pilot strata (a stratum
    smaller than its share gives its remainder to the others), drawn at random within each
    stratum with a fixed seed."""
    strata, names = pilot_strata(df, cohort)
    groups = {name: df[strata == name] for name in names}
    quota = dict.fromkeys(groups, 0)
    remaining = n
    while remaining > 0 and any(quota[name] < len(group) for name, group in groups.items()):
        for name, group in groups.items():
            if remaining > 0 and quota[name] < len(group):
                quota[name] += 1
                remaining -= 1
    print(f"Stratified pilot sample ({cohort}): "
          + ", ".join(f"{name} {quota[name]}/{len(group)}" for name, group in groups.items()))
    return pd.concat([group.sample(quota[name], random_state=seed) for name, group in groups.items()])

def process_cohort(input_filename, output_filename, cohort, ablation="full", limit=DEFAULT_LIMIT, resume=False, sampler="first", workers=1):
    if not os.path.exists(input_filename):
        print(f"File not found: {input_filename}. Skipping cohort.")
        return

    df = load_cohort(input_filename)

    # Every gene symbol used by any signature (single source of truth: signature_definitions.py)
    available_tx_cols = [g for g in get_all_signature_genes() if g in df.columns]
    gene_medians = load_gene_medians(df, available_tx_cols, input_filename)
    class_cutoffs = load_class_cutoffs()

    # limit=0 runs the entire cohort; otherwise the first `limit` cases, or a stratified pilot draw
    if limit == 0:
        sample_df = df
    elif sampler == "stratified":
        sample_df = stratified_sample(df, limit, cohort)
    else:
        sample_df = df.head(limit)
    include = ABLATIONS[ablation]

    # --resume keeps the cases that already succeeded and retries the rest (including rows that
    # recorded a failed request as "Error"); otherwise start fresh.
    records = []
    if resume and os.path.exists(output_filename):
        previous = pd.read_csv(output_filename, low_memory=False)
        succeeded = previous[previous["final_risk_class"].isin(["High Risk", "Low Risk"])]
        if len(succeeded) < len(previous):
            print(f"Resuming: dropping {len(previous) - len(succeeded)} failed rows so they are retried.")
        records = succeeded.to_dict("records")
        print(f"Resuming: {len(records)} cases already done in {output_filename}.")
    elif os.path.exists(output_filename):
        print(f"[NOTE] Overwriting existing {output_filename} (use --resume to continue it instead).")
        os.remove(output_filename)
    done_ids = {record["patient_id"] for record in records}
    columns = list(records[0]) if records else None

    pending = [row for _, row in sample_df.iterrows() if row["patient_id"] not in done_ids]
    print(f"\nProcessing cohort from {input_filename} ({len(pending)} of {len(sample_df)} cases to process, "
          f"ablation='{ablation}', workers={workers})...")

    def run_case(row):
        """One patient, run on a worker thread. Returns the output row; touches no shared state."""
        clinical_meta, sig_classifications, transcriptomics, extracted_pathways_str = build_case_context(
            row, available_tx_cols, gene_medians, class_cutoffs)

        final_risk, summary = generate_patient_summary(
            patient_id=row['patient_id'],
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
            # Simple baselines to compare the model's call against: High vs Low counts, Intermediate ignored
            'majority_vote': majority_vote(row[col] for _, _, col in SIG_COLUMNS),
            'majority_vote_without_hu11': majority_vote(row[col] for key, _, col in SIG_COLUMNS if key != "Hu11"),
            # 'model_confidence_percent', 'model_entropy_score', 'decision_token_confidence',
            # 'min_token_confidence' and 'weakest_token' (logprob fields) are disabled for now.
            'clinical_data': clinical_meta if include["clinical"] else "",
            'signature_classifications': sig_classifications,
            'transcriptomic_data': transcriptomics if include["expression"] else "",
            'active_pathways': extracted_pathways_str if include["pathways"] else "",
            'lmstudio_summary': summary
        })
        return record

    # Only this (main) thread writes the file: the whole CSV, in cohort order, every SAVE_EVERY
    # cases and on exit, so a crash or Ctrl+C loses at most the unsaved cases.
    order = {p_id: k for k, p_id in enumerate(sample_df["patient_id"])}
    def save():
        if records:
            save_csv(pd.DataFrame(sorted(records, key=lambda r: order.get(r["patient_id"], len(order)))), output_filename)

    n_done = 0
    try:
        for row, record, error in run_in_pool(pending, run_case, workers):
            n_done += 1
            if error is not None:
                print(f"[{n_done}/{len(pending)}] [ERROR] Patient {row['patient_id']}: {error}")
                continue
            if columns is None:
                columns = list(record)
            elif list(record) != columns:
                raise ValueError(f"{output_filename} was written with different columns; rerun without --resume.")
            records.append(record)
            print(f"[{n_done}/{len(pending)}] Patient {row['patient_id']}: {record['final_risk_class']}")
            if n_done % SAVE_EVERY == 0:
                save()
    finally:
        save()

    print(f"Results saved to: {output_filename}")

def main():
    global model_to_use
    parser = argparse.ArgumentParser(description="Generate arbitration summaries for TCGA-BRCA cases.")
    parser.add_argument("--model", default=None,
                        help="Generation model name (defaults to pipeline.model_name in config.yml); used in output filenames.")
    parser.add_argument("--ablation", choices=list(ABLATIONS), default="full",
                        help="Which evidence the model sees (non-full runs are saved with a suffix).")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT,
                        help=f"Cases per cohort (default {DEFAULT_LIMIT} for quick tests; 0 = whole cohort).")
    parser.add_argument("--sampler", choices=["first", "stratified"], default="first",
                        help="How --limit picks cases. 'stratified' draws evenly, with a fixed seed, across the lone "
                             "dissenting signature (Hu-11, IRRS-7, other, none) for discordant cases and across "
                             "Consensus High / Low for concordant cases.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue an interrupted run: keep finished cases, retry failed or missing ones.")
    parser.add_argument("--cohorts", nargs="+", choices=["concordant", "discordant"], default=["concordant", "discordant"])
    add_workers_arg(parser, config)
    args = parser.parse_args()
    if args.model:
        model_to_use = args.model
    print(f"Generating with model '{model_to_use}'")

    for cohort in args.cohorts:
        process_cohort(f"data/processed/tcga_{cohort}_cases.csv", generation_path(cohort, model_to_use, args.ablation),
                       cohort, ablation=args.ablation, limit=args.limit, resume=args.resume, sampler=args.sampler,
                       workers=args.workers)

if __name__ == "__main__":
    run_cli(main)
