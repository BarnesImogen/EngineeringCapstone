import os
import argparse
from datetime import date
import yaml
import pandas as pd
from run_paths import generation_path, EVALUATION_DIR

SUMMARY_PATH = os.path.join(EVALUATION_DIR, "concordant_check_summary.csv")
DEFAULT_MIN_AGREEMENT = 0.8
MAX_FAILED_FRACTION = 0.1

def load_min_agreement():
    with open("config.yml", "r") as file:
        config = yaml.safe_load(file)
    return float(config.get("evaluation", {}).get("min_concordant_agreement", DEFAULT_MIN_AGREEMENT))

def save_summary(row):
    """One row per (model, ablation); rerunning a check replaces its earlier row."""
    os.makedirs(EVALUATION_DIR, exist_ok=True)
    new = pd.DataFrame([row])
    if os.path.exists(SUMMARY_PATH):
        old = pd.read_csv(SUMMARY_PATH)
        keep = ~((old["model"] == row["model"]) & (old["ablation"] == row["ablation"]))
        new = pd.concat([old[keep], new], ignore_index=True)
    new.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSummary saved to {SUMMARY_PATH}")

# Sanity check: on concordant cases all eight signatures agree, so the correct label is known.
# Usage: python src/concordant_check.py [--model NAME] [--ablation full|no_pathways|...]

def run_concordant_check(model, ablation="full"):
    path = generation_path("concordant", model, ablation)
    if not os.path.exists(path):
        print(f"{path} not found. Run generation_pipeline.py on the concordant cohort first.")
        return None

    df = pd.read_csv(path, low_memory=False)
    print(f"--- Concordant Sanity Check (model: {model}, ablation: {ablation}) ---")

    valid = df[df["final_risk_class"].isin(["High Risk", "Low Risk"])].copy()
    n_failed = len(df) - len(valid)
    print(f"Cases: {len(df)} | valid LLM labels: {len(valid)} | failed/invalid: {n_failed}")
    if valid.empty:
        print("[WARNING] No valid LLM labels; this model failed the sanity check.")
        save_summary({"model": model, "ablation": ablation, "n_cases": len(df), "n_valid": 0, "n_failed": n_failed,
                      "agreement": float("nan"), "min_required": load_min_agreement(), "passed": False, "date": date.today().isoformat()})
        return None

    valid["agrees"] = valid["final_risk_class"] == valid["Consensus_Risk"]
    agreement = valid["agrees"].mean()
    print(f"Agreement with unanimous signature consensus: {agreement*100:.1f}% ({valid['agrees'].sum()}/{len(valid)})")

    print("\nConfusion (rows = consensus, columns = LLM):")
    print(pd.crosstab(valid["Consensus_Risk"], valid["final_risk_class"]).to_string())

    for consensus, group in valid.groupby("Consensus_Risk"):
        print(f"  Consensus {consensus}: LLM agrees on {group['agrees'].mean()*100:.1f}% (n={len(group)})")

    # DISABLED (logprobs): do the model's own confidence signals separate its right answers from its wrong ones?
    # Semantic entropy is validated in src/semantic_entropy.py --validate instead.
    # signals = [c for c in ["model_confidence_percent", "decision_token_confidence", "model_entropy_score"] if c in valid.columns]
    # if signals and valid["agrees"].nunique() == 2:
    #     print("\nMean confidence signals when the LLM agrees vs disagrees:")
    #     print(valid.groupby("agrees")[signals].mean().rename(index={True: "agrees", False: "disagrees"}).round(3).to_string())

    # Gate: flag (do not block) a model that fails on cases where the correct answer is known
    min_required = load_min_agreement()
    failed_fraction = n_failed / len(df)
    passed = bool(agreement >= min_required and failed_fraction <= MAX_FAILED_FRACTION)
    if agreement < min_required:
        print(f"\n[WARNING] Agreement {agreement*100:.1f}% is below the required {min_required*100:.0f}%. "
              "Treat this model's discordant results with caution.")
    if failed_fraction > MAX_FAILED_FRACTION:
        print(f"\n[WARNING] {failed_fraction*100:.0f}% of concordant cases produced no valid label.")
    if passed:
        print(f"\nSanity check PASSED (agreement {agreement*100:.1f}% >= {min_required*100:.0f}%).")

    by_consensus = valid.groupby("Consensus_Risk")["agrees"].mean()
    save_summary({
        "model": model, "ablation": ablation, "n_cases": len(df), "n_valid": len(valid), "n_failed": n_failed,
        "agreement": round(float(agreement), 4),
        "agreement_when_consensus_high": round(float(by_consensus.get("High Risk", float("nan"))), 4),
        "agreement_when_consensus_low": round(float(by_consensus.get("Low Risk", float("nan"))), 4),
        "min_required": min_required, "passed": passed, "date": date.today().isoformat(),
    })
    return agreement

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Generation model whose outputs to check (default: config pipeline.model_name).")
    parser.add_argument("--ablation", default="full")
    args = parser.parse_args()
    with open("config.yml", "r") as file:
        default_model = yaml.safe_load(file)["pipeline"]["model_name"]
    run_concordant_check(args.model or default_model, args.ablation)
