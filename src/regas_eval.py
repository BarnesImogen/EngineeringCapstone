import os
import sys
import types
import yaml
import pandas as pd

# Work around ragas importing a VertexAI class from a removed langchain-community path
_vertexai_stub = types.ModuleType("langchain_community.chat_models.vertexai")
_vertexai_stub.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = _vertexai_stub

from datasets import Dataset
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness
from ragas import RunConfig, evaluate

# ==========================================
# 1. Load Configuration & Paths
# ==========================================
with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

model_to_use = config["evaluation"]["judge_model"]
generation_temp = float(config["evaluation"].get("judge_temperature", 0.1))
eval_base_url = config["evaluation"].get("base_url", "http://127.0.0.1:1234/v1")
eval_api_key = config["evaluation"].get("api_key", "lmstudio")

input_filename = "data/evaluation_outputs/lmstudio_evaluated_results.csv"
final_output_filename = "data/evaluation_outputs/lmstudio_fully_evaluated_results.csv"

print(f"Loaded configuration for Ragas: Targeting model '{model_to_use}'")

# ==========================================
# 2. LM Studio LangChain Client Initialisation
# ==========================================
ragas_chat_model = ChatOpenAI(
    base_url=eval_base_url,
    api_key=eval_api_key,
    model=model_to_use,
    temperature=generation_temp,
)
ragas_local_llm = LangchainLLMWrapper(ragas_chat_model)
faithfulness.llm = ragas_local_llm

# ==========================================
# 3. Execution Phase (Ragas Faithfulness)
# ==========================================
if not os.path.exists(input_filename):
    raise FileNotFoundError(f"Could not find {input_filename}. Please run evaluation_pipeline.py first!")

df = pd.read_csv(input_filename)
ragas_data = {"question": [], "answer": [], "contexts": []}

print("Preparing dataset for Ragas semantic faithfulness...")

for _, row in df.iterrows():
    patient_id = row['patient_id']
    generated_text = row.get("lmstudio_summary")
    active_pathways = row.get('active_pathways', '')
    clinical_data = row.get('clinical_data', '')
    transcriptomic_data = row.get('transcriptomic_data', '')
    
    ragas_data["question"].append(f"Resolve prognostic discordance for patient {patient_id}.")
    ragas_data["answer"].append(str(generated_text))
    full_context_str = f"Pathways:\n{active_pathways}\n\nClinical Data:\n{clinical_data}\n\nTranscriptomic Profile:\n{transcriptomic_data}"
    ragas_data["contexts"].append([full_context_str])

print("\nInitiating Ragas Faithfulness Evaluation...")
ragas_dataset = Dataset.from_dict(ragas_data)

ragas_results = evaluate(
    dataset=ragas_dataset,
    metrics=[faithfulness],
    run_config=RunConfig(max_workers=1, timeout=600),
)
ragas_df = ragas_results.to_pandas()

df['ragas_faithfulness_score'] = ragas_df['faithfulness']

# ==========================================
# 4. Save Fully Evaluated Output & Metrics
# ==========================================
df.to_csv(final_output_filename, index=False)
print(f"\nRagas evaluation complete. Fully evaluated results stored in: {final_output_filename}")

if not df.empty:
    print("\n--- Final Consolidated Performance Metrics ---")
    print(f"Mean Generation Confidence:    {df['model_confidence_percent'].mean():.2f}%")
    print(f"Mean Ragas Faithfulness:       {df['ragas_faithfulness_score'].mean():.2f} / 1.0")
    print(f"Mean Biological Synthesis:     {df['bio_synthesis_score'].mean():.2f} / 5.0")
    print(f"Mean Systematic Reasoning:     {df['sys_reasoning_score'].mean():.2f} / 5.0")
    print(f"Mean Prognostic Resolution:    {df['prognostic_resolution_score'].mean():.2f} / 5.0")