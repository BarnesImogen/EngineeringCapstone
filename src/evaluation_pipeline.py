import os
import json
import argparse
import yaml
import inspect
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from run_paths import generation_path, evaluation_path
from signature_definitions import get_signature_reference_text

load_dotenv()

# ==========================================
# 1. Load Configuration & Paths
# ==========================================
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

model_to_use = config["evaluation"]["judge_model"]
generation_temp = float(config["evaluation"].get("judge_temperature", 0.1))
eval_base_url = config["evaluation"].get("base_url", "http://127.0.0.1:1234/v1")
eval_api_key = os.getenv("JUDGE_API_KEY") or config["evaluation"].get("api_key", "lmstudio")
success_threshold = float(config["evaluation"].get("success_threshold", 4.0))
metrics = config["evaluation"]["metrics"]

parser = argparse.ArgumentParser(description="Judge generated summaries with the judge model in config.yml.")
parser.add_argument("--model", default=None, help="GENERATION model whose outputs to judge (default: config pipeline.model_name).")
parser.add_argument("--ablation", default="full", help="Which generation ablation run to judge.")
args = parser.parse_args()
generated_by = args.model or config["pipeline"]["model_name"]

generation_input_column = "lmstudio_summary"
input_filename = generation_path("discordant", generated_by, args.ablation)
output_filename = evaluation_path("discordant", generated_by, args.ablation)
print(f"Judging outputs of '{generated_by}' (ablation: {args.ablation}) from {input_filename}")

if model_to_use == generated_by:
    print("[WARNING] Judge model is the same as the generation model; scores may be biased toward its own style.")

os.makedirs("data/evaluation_outputs", exist_ok=True)
print(f"Loaded configuration: Local LM Studio targeting judge model '{model_to_use}'")

# ==========================================
# 2. LM Studio Client Initialisation
# ==========================================
client = OpenAI(base_url=eval_base_url, api_key=eval_api_key)

# ==========================================
# 3. Structured Pydantic Schema
# ==========================================
class GradingReport(BaseModel):
    # Field order matters: the model audits claims and writes each justification BEFORE committing to its score.
    ungrounded_claims: list[str] = Field(description="Every specific claim in the AI output (gene, value, pathway, signature weight or mechanism) that is NOT supported by the provided context. Empty list if none.")
    biological_synthesis_justification: str = Field(description="Audit explaining the biological synthesis score, quoting the output and the context line checked.")
    biological_synthesis_score: int = Field(description="Integer score from 1 to 5.")
    systematic_reasoning_justification: str = Field(description="Audit explaining the systematic reasoning score.")
    systematic_reasoning_score: int = Field(description="Integer score from 1 to 5.")
    prognostic_resolution_justification: str = Field(description="Audit explaining the prognostic resolution score.")
    prognostic_resolution_score: int = Field(description="Integer score from 1 to 5.")

# Deterministic cap so the score cannot ignore the judge's own list of ungrounded claims
def cap_biological_score(score, n_ungrounded):
    if n_ungrounded >= 3:
        return min(score, 2)
    if n_ungrounded >= 1:
        return min(score, 3)
    return score

grading_report_schema = {
    "name": "grading_report",
    "schema": GradingReport.model_json_schema(),
    "strict": True,
}

# ==========================================
# 4. Evidence-Based Auditor System Prompt
# ==========================================
evaluator_instruction = inspect.cleandoc("""
    You are an expert, independent academic auditor specialising in precision oncology and bioinformatics.
    Your role is to audit and evaluate AI summaries generated to resolve discordant multi-gene prognostic risk signatures.

    You will be provided with:
    1. The FULL CONTEXT given to the AI during generation: Signature Definitions (gene sets and weights for each
       algorithm), Clinical Metadata, Algorithmic Risk Classifications, Transcriptomic Profile, and Verified Active
       Pathways. This is everything the AI actually saw.
    2. The AI'S GENERATED OUTPUT.

    Some context sections may be marked "Not provided to the AI in this run". That is intentional. Missing evidence is
    not a fault of the AI, but any claim that would require evidence it was not given (for example a specific gene
    expression value when no Transcriptomic Profile was provided) is UNGROUNDED.

    WHAT COUNTS AS GROUNDED: any gene, expression value, pathway, clinical value, or signature weight/gene membership
    that appears in the FULL CONTEXT and is stated correctly. A gene mentioned with its correct expression value is
    grounded even if it triggered no pathway entry. Signature weighting claims are grounded only if they match the
    Signature Definitions. Claims that are simply wrong (wrong value, wrong direction, wrong weight) are ungrounded.

    STEP 1: List every ungrounded claim in `ungrounded_claims` (quote or paraphrase it precisely). Be exhaustive and
    literal; do not list claims that are grounded.
    STEP 2: For each domain write the justification first, quoting the output and the context line you checked, then
    give the score.

    Score on a 1 to 5 scale, judged in absolute terms (do NOT grade on a curve for how much evidence was provided):

    1. Biological Synthesis (Grounding & Anti-Hallucination):
       * Score 1: Multiple ungrounded or contradicted claims; core reasoning rests on invented biology.
       * Score 2: Several ungrounded claims, or grounded facts misused. Example: cites a gene expression value that is not in the profile.
       * Score 3: Broadly grounded but generic or imprecise; may contain a minor ungrounded claim.
       * Score 4: Accurate and specific use of the provided context with no ungrounded claims, but some available evidence left unused.
       * Score 5: Every claim is grounded and precisely integrates the available context, with no external claims.
       Cap: 1 to 2 ungrounded claims means the score cannot exceed 3; 3 or more means it cannot exceed 2.

    2. Systematic Reasoning (Mechanistic Root-Cause Analysis):
       * Score 1: Fails to explain why the signatures disagree, or makes contradictory logical leaps.
       * Score 2: Restates that signatures disagree without a mechanism.
       * Score 3: Superficial explanation of discordance, not tied to specific weights or expression levels.
       * Score 4: Explains the disagreement using specific weights or gene sets from the Signature Definitions, but only partly anchored in the patient's values.
       * Score 5: Identifies the specific mechanistic root cause (e.g. which weighted gene groups push each signature in opposite directions) using the Signature Definitions and anchored in the patient's own evidence.

    3. Prognostic Resolution (Conflict Arbitration):
       * Score 1: No final risk consensus, or arbitrarily picks a side without linking it to the reasoning.
       * Score 3: Provides a final risk consensus, but the justification is generic or weakly linked to the evidence.
       * Score 5: A definitive, well-justified consensus that logically follows from the interpreted evidence.

    SCORING PRINCIPLES:
    * Cross-reference the AI output against the FULL CONTEXT only. Judge support from the context, not whether the
      conclusion seems clinically plausible.
    * Do not reward length, formatting, or a confident tone.
    * Do not favour either risk class. The correct class is unknown to you.
""")

# ==========================================
# 5. Execution Loop (LLM-as-a-Judge)
# ==========================================
if not os.path.exists(input_filename):
    raise FileNotFoundError(f"Could not find {input_filename}. Please run the generation script first!")

df = pd.read_csv(input_filename)
evaluation_results = []

signature_reference = get_signature_reference_text()

print("Initiating Local LLM-as-a-Judge Evaluation Pipeline...\n")

for index, row in df.iterrows():
    patient_id = row['patient_id']
    generated_text = row.get(generation_input_column)
    # Ablation runs log blank fields for evidence the model was not shown
    not_shown = '* Not provided to the AI in this run.'
    clinical_data = row.get('clinical_data') if pd.notna(row.get('clinical_data')) else not_shown
    signature_classifications = row.get('signature_classifications') if pd.notna(row.get('signature_classifications')) else not_shown
    transcriptomic_data = row.get('transcriptomic_data') if pd.notna(row.get('transcriptomic_data')) else not_shown
    active_pathways = row.get('active_pathways') if pd.notna(row.get('active_pathways')) else not_shown

    print(f"Auditing Report for Patient: {patient_id}...")
    
    if pd.isna(generated_text) or row.get('final_risk_class') == "Error":
        print(f"  [SKIPPED] Missing or invalid text for {patient_id}.")
        continue

    prompt = inspect.cleandoc(f"""
        Please audit and score the following generated Bioinformatics summary.
        --- FULL CONTEXT PROVIDED TO THE AI DURING GENERATION ---
        Signature Definitions:
        {signature_reference}
        Clinical Metadata:
        {clinical_data}
        Conflicting Algorithmic Risk Classifications (1 = High Risk, 0 = Low Risk):
        {signature_classifications}
        Transcriptomic Profile (Key Biomarkers & Expression Levels):
        {transcriptomic_data}
        Verified Active Biological Pathways (Rule-Based RAG Extraction):
        {active_pathways}
        --- START OF AI OUTPUT ---
        {generated_text}
        --- END OF AI OUTPUT ---
        Audit the output against the 3 domains.
        List the ungrounded claims first, then return justifications and integer scores (1-5) strictly matching the requested JSON schema.
    """)

    try:
        response = client.chat.completions.create(
            model=model_to_use,
            temperature=generation_temp,
            messages=[
                {"role": "system", "content": evaluator_instruction},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_schema", "json_schema": grading_report_schema},
        )

        parsed_report = GradingReport.model_validate_json(response.choices[0].message.content)
        
        n_ungrounded = len(parsed_report.ungrounded_claims)
        bio_score = cap_biological_score(parsed_report.biological_synthesis_score, n_ungrounded)

        evaluation_results.append({
            'patient_id': patient_id,
            'ablation': args.ablation,
            'generation_model': generated_by,
            'judge_model': model_to_use,
            'clinical_data': clinical_data,
            'signature_classifications': signature_classifications,
            'transcriptomic_data': transcriptomic_data,
            'active_pathways': active_pathways,
            generation_input_column: generated_text,
            'model_confidence_percent': row.get('model_confidence_percent'), 
            'model_entropy_score': row.get('model_entropy_score'),           
            'ungrounded_claim_count': n_ungrounded,
            'ungrounded_claims': json.dumps(parsed_report.ungrounded_claims),
            'bio_synthesis_score': bio_score,
            'bio_synthesis_score_raw': parsed_report.biological_synthesis_score,
            'bio_synthesis_justification': parsed_report.biological_synthesis_justification,
            'sys_reasoning_score': parsed_report.systematic_reasoning_score,
            'sys_reasoning_justification': parsed_report.systematic_reasoning_justification,
            'prognostic_resolution_score': parsed_report.prognostic_resolution_score,
            'prognostic_resolution_justification': parsed_report.prognostic_resolution_justification
        })
        print(f"  [SUCCESS] Scored {patient_id}")
        
    except Exception as e:
        print(f"  [ERROR] Evaluating patient {patient_id}: {e}")

# ==========================================
# 6. Save Structured Audit Outputs
# ==========================================
final_df = pd.DataFrame(evaluation_results)
final_df.to_csv(output_filename, index=False)
print(f"\nStructural evaluation complete. Results stored successfully in: {output_filename}")

if not final_df.empty:
    print("\n--- Structured Audit Metrics ---")
    print(f"Mean Generation Confidence:    {final_df['model_confidence_percent'].mean():.2f}%")
    print(f"Mean Biological Synthesis:     {final_df['bio_synthesis_score'].mean():.2f} / 5.0 (raw {final_df['bio_synthesis_score_raw'].mean():.2f})")
    print(f"Mean Ungrounded Claims:        {final_df['ungrounded_claim_count'].mean():.2f} per summary "
          f"({(final_df['ungrounded_claim_count'] > 0).mean() * 100:.1f}% of summaries have at least one)")
    print(f"Mean Systematic Reasoning:     {final_df['sys_reasoning_score'].mean():.2f} / 5.0")
    print(f"Mean Prognostic Resolution:    {final_df['prognostic_resolution_score'].mean():.2f} / 5.0")

    score_columns = {
        "biological_synthesis": "bio_synthesis_score",
        "systematic_reasoning": "sys_reasoning_score",
        "prognostic_resolution": "prognostic_resolution_score",
    }
    print(f"\n--- Pass rate (score >= {success_threshold}) ---")
    for metric in metrics:
        col = score_columns[metric]
        print(f"{metric}: {(final_df[col] >= success_threshold).mean() * 100:.1f}%")