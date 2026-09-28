import re
import yaml

GENERATION_DIR = "data/generation_outputs"
EVALUATION_DIR = "data/evaluation_outputs"

def load_config(path="config.yml"):
    with open(path, "r") as file:
        return yaml.safe_load(file)

def model_slug(model_name):
    """Filesystem-safe version of a model name (e.g. 'openai/gpt-oss-20b' -> 'openai-gpt-oss-20b')."""
    return re.sub(r"[^A-Za-z0-9.]+", "-", model_name).strip("-")

def run_tag(model_name, ablation="full"):
    return f"{model_slug(model_name)}__{ablation}"

def generation_path(cohort, model_name, ablation="full"):
    return f"{GENERATION_DIR}/{cohort}_results__{run_tag(model_name, ablation)}.csv"

def evaluation_path(cohort, model_name, ablation="full"):
    return f"{EVALUATION_DIR}/{cohort}_evaluated__{run_tag(model_name, ablation)}.csv"
