import os
import yaml
import inspect
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field

# Ragas & LangChain Imports
from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness
from ragas import evaluate

# ==========================================
# 1. Load Configuration & Paths
# ==========================================
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

model_to_use = config["evaluation"]["judge_model"]
generation_temp = float(config["evaluation"].get("judge_temperature", 0.1))
eval_base_url = config["evaluation"].get("base_url", "http://127.0.0.1:1234/v1")
eval_api_key = config["evaluation"].get("api_key", "lmstudio")

generation_input_column = "lmstudio_summary"
input_filename = "data/generation_outputs/lmstudio_generation_results.csv"
output_filename = "data/evaluation_outputs/lmstudio_evaluated_results.csv"

os.makedirs("data/evaluation_outputs", exist_ok=True)
print(f"Loaded configuration: Local LM Studio targeting judge model '{model_to_use}'")

# ==========================================
# 2. LM Studio Client Initialisation (Standard & Ragas)
# ==========================================
client = OpenAI(base_url=eval_base_url, api_key=eval_api_key)

ragas_chat_model = ChatOpenAI(
    base_url=eval_base_url,
    api_key=eval_api_key,
    model=model_to_use,
    temperature=generation_temp,
)
ragas_local_llm = LangchainLLMWrapper(ragas_chat_model)
faithfulness.llm = ragas_local_llm

# ==========================================
# 3. Structured Pydantic Schema
# ==========================================
class GradingReport(BaseModel):
    biological_synthesis_score: int = Field(description="Integer score from 1 to 5.")
    biological_synthesis_justification: str = Field(description="Detailed audit explaining the biological synthesis score.")
    systematic_reasoning_score: int = Field(description="Integer score from 1 to 5.")
    systematic_reasoning_justification: str = Field(description="Detailed audit explaining the systematic reasoning score.")
    prognostic_resolution_score: int = Field(description="Integer score from 1 to 5.")
    prognostic_resolution_justification: str = Field(description="Detailed audit explaining the prognostic resolution score.")

grading_report_schema = {
    "name": "grading_report",
    "schema": GradingReport.model_json_schema(),
    "strict": True,
}

# ==========================================
# 4. Evidence-Based Auditor System Prompt
# ==========================================
evaluator_instruction = inspect.cleandoc("""
    You are an expert, independent academic auditor specializing in precision oncology and bioinformatics.
    Your role is to audit and evaluate AI summaries generated to resolve discordant multi-gene prognostic risk signatures.

    You will be provided with:
    1. The FULL CONTEXT given to the AI during generation (Clinical Metadata, Algorithmic Risk Classifications, Transcriptomic Profile, and Verified Active Pathways). This is everything the AI actually saw.
    2. The AI'S GENERATED OUTPUT.

    IMPORTANT: The AI is permitted to reason about ANY gene or value that appears in the FULL CONTEXT, not only the
    genes listed under Verified Active Pathways. A gene mentioned with its correct, given expression value from the
    Transcriptomic Profile is grounded, not hallucinated, even if that gene did not trigger a Verified Active Pathway
    entry. Only flag hallucination for genes, pathways, expression values, or signature-specific claims (e.g. "BCI
    weights ER and HER2") that do NOT appear anywhere in the provided FULL CONTEXT.

    Evaluate the generated summary across three distinct domains using a 1 to 5 scale:

    1. Biological Synthesis (Grounding & Anti-Hallucination):
       * Score 1: Hallucinates biological mechanisms, introduces genes/values not present anywhere in the FULL CONTEXT, or contradicts it.
       * Score 3: Broad or generic descriptions; loosely grounded in the provided context but lacks precision.
       * Score 5: Flawless synthesis. Exclusively and accurately integrates the provided context without introducing external claims.

    2. Systematic Reasoning (Mechanistic Root-Cause Analysis):
       * Score 1: Fails to explain why algorithmic signatures disagree or makes contradictory logical leaps.
       * Score 3: Superficial explanation of discordance without tying it directly to specific biomarker expression levels.
       * Score 5: Diagnoses the exact mechanistic root cause of signature disagreement (e.g., proliferation-weighted vs. ER-driven signature biases) anchored directly in the patient's transcriptomic profile.

    3. Prognostic Resolution (Conflict Arbitration):
       * Score 1: Fails to provide a final risk consensus, or arbitrarily picks a side without linking it to the biological synthesis.
       * Score 3: Provides a final risk consensus, but the justification is generic or weakly linked to the transcriptomic data.
       * Score 5: Delivers a definitive, well-justified prognostic consensus (e.g., High Risk vs. Low Risk) that logically flows from the interpreted data.

    SCORING PRINCIPLES:
    * Cross-reference the AI output directly against the provided FULL CONTEXT (not only the Verified Active Pathways).
    * Deduct marks immediately for genes, values, mechanisms, or signature-weighting claims that appear nowhere in the FULL CONTEXT.
    * Award 4 or 5 only to outputs demonstrating strong mechanistic rigor and clear resolution logic.
""")

# ==========================================
# 5. Execution Loop (LLM-as-a-Judge)
# ==========================================
if not os.path.exists(input_filename):
    raise FileNotFoundError(f"Could not find {input_filename}. Please run the generation script first!")

df = pd.read_csv(input_filename)
evaluation_results = []

ragas_data = {"question": [], "answer": [], "contexts": []}

print("Initiating Local LLM-as-a-Judge Evaluation Pipeline...\n")

for index, row in df.iterrows():
    patient_id = row['patient_id']
    generated_text = row.get(generation_input_column)
    clinical_data = row.get('clinical_data', '* Not available in log.')
    signature_classifications = row.get('signature_classifications', '* Not available in log.')
    transcriptomic_data = row.get('transcriptomic_data', '* Not available in log.')
    active_pathways = row.get('active_pathways', '* Reference pathways not available in log.')
    
    ragas_data["question"].append(f"Resolve prognostic discordance for patient {patient_id}.")
    ragas_data["answer"].append(str(generated_text))
    full_context_str = f"Pathways:\n{active_pathways}\n\nClinical Data:\n{clinical_data}\n\nTranscriptomic Profile:\n{transcriptomic_data}"
    ragas_data["contexts"].append([full_context_str])

    print(f"Auditing Report for Patient: {patient_id}...")
    
    if pd.isna(generated_text) or "Error" in str(generated_text):
        print(f"  [SKIPPED] Missing or invalid text for {patient_id}.")
        continue

    prompt = inspect.cleandoc(f"""
        Please audit and score the following generated Bioinformatics summary.
        --- FULL CONTEXT PROVIDED TO THE AI DURING GENERATION ---
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
        Return the integer scores (1-5) and explicit justifications strictly matching the requested JSON schema.
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
        
        evaluation_results.append({
            'patient_id': patient_id,
            'clinical_data': clinical_data,
            'signature_classifications': signature_classifications,
            'transcriptomic_data': transcriptomic_data,
            'active_pathways': active_pathways,
            generation_input_column: generated_text,
            'model_confidence_percent': row.get('model_confidence_percent'), 
            'model_entropy_score': row.get('model_entropy_score'),           
            'bio_synthesis_score': parsed_report.biological_synthesis_score,
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
# 6. Execution Phase (Ragas Faithfulness)
# ==========================================
print("\nInitiating Ragas Faithfulness Evaluation...")
ragas_dataset = Dataset.from_dict(ragas_data)
ragas_results = evaluate(dataset=ragas_dataset, metrics=[faithfulness])
ragas_df = ragas_results.to_pandas()

final_df = pd.DataFrame(evaluation_results)
final_df['ragas_faithfulness_score'] = ragas_df['faithfulness']

# ==========================================
# 7. Save Outputs & Metrics
# ==========================================
final_df.to_csv(output_filename, index=False)
print(f"\nEvaluation pipeline complete. Matrix scores stored successfully in: {output_filename}")

if not final_df.empty:
    print("\n--- Current Performance Metrics ---")
    print(f"Mean Generation Confidence:    {final_df['model_confidence_percent'].mean():.2f}%")
    print(f"Mean Ragas Faithfulness:       {final_df['ragas_faithfulness_score'].mean():.2f} / 1.0")
    print(f"Mean Biological Synthesis:     {final_df['bio_synthesis_score'].mean():.2f} / 5.0")
    print(f"Mean Systematic Reasoning:     {final_df['sys_reasoning_score'].mean():.2f} / 5.0")
    print(f"Mean Prognostic Resolution:    {final_df['prognostic_resolution_score'].mean():.2f} / 5.0")