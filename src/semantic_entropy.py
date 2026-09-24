"""Semantic entropy for the arbitration outputs, following Farquhar et al. (2024).

Paper: "Detecting hallucinations in large language models using semantic entropy",
Nature 630, 625-630. Reference code: https://github.com/jlko/semantic_uncertainty

No logprobs needed. This is the paper's *discrete* semantic entropy, which they
propose for models that do not return token probabilities.

For each patient:
  1. Rebuild the exact evidence the generation model saw (same system prompt and
     evidence sections as generation_pipeline.py).
  2. Ask the same focused question M times at temperature 1.0 (paper: M = 10,
     T = 1.0, nucleus sampling p = 0.9). Each answer is one brief sentence
     (the paper's sentence-length template) plus a FINAL RESOLUTION line.
  3. Cluster the sentences by bidirectional entailment, conditioned on the
     question, using the paper's entailment prompt (entailment / neutral /
     contradiction) and its default strict rule: both directions must be
     "entailment".
  4. Discrete semantic entropy = entropy of the share of answers in each
     cluster, in nats (their `cluster_assignment_entropy`).
  5. Risk entropy = the same calculation over the High/Low labels, where
     clusters are exact label matches (max ln 2 = 0.693).

The answer being scored for correctness is the pipeline's own low-temperature
output (config temperature 0.1), which plays the role of the paper's
low-temperature "best generation".

Validation (--validate), mirroring the paper's AUROC and rejection accuracy:
  * Concordant cohort: ground truth is the unanimous signature consensus, so
    "incorrect" = final_risk_class != Consensus_Risk.
  * Discordant cohort: labels come from the judge (evaluation_pipeline.py):
    any ungrounded claim, or a mean judge score below success_threshold.

Usage (from the repository root):
  python src/semantic_entropy.py --model NAME [--cohorts concordant discordant] [--limit N]
  python src/semantic_entropy.py --model NAME --validate
  Options: --ablation, --num-generations 10, --temperature 1.0,
           --entailment llm|deberta, --resume
"""

import argparse
import hashlib
import json
import os
from collections import Counter

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from generation_pipeline import SYSTEM_INSTRUCTION, build_prompt, parse_final_resolution
from run_paths import EVALUATION_DIR, evaluation_path, generation_path, run_tag

load_dotenv()

with open("config.yml", "r") as file:
    config = yaml.safe_load(file)

DEFAULT_LIMIT = 6
ENTAILMENT_CACHE = os.path.join(EVALUATION_DIR, "entailment_cache.json")
VALIDATION_SUMMARY = os.path.join(EVALUATION_DIR, "semantic_entropy_validation.csv")

# One focused question per cohort. The paper's method is built for answers that
# make a single claim, so the long arbitration is replaced by one sentence here.
QUESTIONS = {
    "discordant": "What is the main biological reason these signature risk classifications disagree for this patient?",
    "concordant": "What is the main biological reason for this patient's prognostic risk classification?",
}


def sampling_prompt(evidence_prompt, question):
    # Paper's sentence-length template ("Answer the following question in a single
    # brief but complete sentence."), plus the pipeline's FINAL RESOLUTION line.
    return (
        f"{evidence_prompt}\n\n"
        "Answer the following question in a single brief but complete sentence.\n"
        f"Question: {question}\n\n"
        "Reply in exactly this format:\n"
        "ANSWER: <one sentence>\n"
        "FINAL RESOLUTION: High Risk or Low Risk"
    )


# ==========================================
# Clients
# ==========================================
def make_clients():
    gen = OpenAI(base_url=config["pipeline"].get("base_url", "http://127.0.0.1:1234/v1"),
                 api_key=os.getenv("GENERATION_API_KEY") or config["pipeline"].get("api_key", "lmstudio"))
    judge = OpenAI(base_url=config["evaluation"].get("base_url", "http://127.0.0.1:1234/v1"),
                   api_key=os.getenv("JUDGE_API_KEY") or config["evaluation"].get("api_key", "lmstudio"))
    return gen, judge


# ==========================================
# 1-2. Rebuild the evidence and sample answers
# ==========================================
def text_or_blank(value):
    return "" if pd.isna(value) else str(value)


def evidence_from_row(row, ablation):
    """The generation prompt up to (not including) its arbitration instructions."""
    full = build_prompt(
        row["patient_id"],
        text_or_blank(row.get("clinical_data")),
        text_or_blank(row.get("signature_classifications")),
        text_or_blank(row.get("transcriptomic_data")),
        text_or_blank(row.get("active_pathways")),
        ablation,
    )
    return full.split("REQUIRED ARBITRATION STEPS:")[0].strip()


def parse_answer(text):
    for line in (text or "").splitlines():
        if line.strip().upper().startswith("ANSWER:"):
            return line.split(":", 1)[1].strip()
    # Fall back to the first non-empty line that is not the resolution line
    for line in (text or "").splitlines():
        if line.strip() and "FINAL RESOLUTION" not in line.upper():
            return line.strip()
    return ""


def sample_answers(gen_client, model, evidence, question, n, temperature, top_p):
    prompt = sampling_prompt(evidence, question)
    samples = []
    for _ in range(n):
        response = gen_client.chat.completions.create(
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "system", "content": SYSTEM_INSTRUCTION},
                      {"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        samples.append({"answer": parse_answer(text),
                        "risk": parse_final_resolution(text) or "Unparsed"})
    return samples


# ==========================================
# 3. Entailment models (0 = contradiction, 1 = neutral, 2 = entailment, as in the repo)
# ==========================================
class EntailmentLLM:
    """Paper's LLM entailment prompt, with an on-disk cache like the reference code."""

    def __init__(self, client, model):
        self.client, self.model = client, model
        self.cache = {}
        if os.path.exists(ENTAILMENT_CACHE):
            with open(ENTAILMENT_CACHE) as f:
                self.cache = json.load(f)

    def check_implication(self, text1, text2, question):
        prompt = (f'We are evaluating answers to the question "{question}"\n'
                  "Here are two possible answers:\n"
                  f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
                  "Does Possible Answer 1 semantically entail Possible Answer 2? "
                  "Respond only with entailment, contradiction, or neutral.\n"
                  "Response:")
        key = f"{self.model}|" + hashlib.md5(prompt.encode()).hexdigest()
        if key not in self.cache:
            response = self.client.chat.completions.create(
                model=self.model, temperature=0.02,
                messages=[{"role": "user", "content": prompt}])
            self.cache[key] = response.choices[0].message.content or ""
        reply = self.cache[key].lower()[:30]
        if "entailment" in reply:
            return 2
        if "contradiction" in reply:
            return 0
        return 1  # neutral, or an unclear reply (the repo does the same)

    def save_cache(self):
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        with open(ENTAILMENT_CACHE, "w") as f:
            json.dump(self.cache, f)


class EntailmentDeberta:
    """The repo's default entailment model. Needs `pip install torch transformers`."""

    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        name = "microsoft/deberta-v2-xlarge-mnli"
        self.device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModelForSequenceClassification.from_pretrained(name).to(self.device).eval()
        labels = {v.lower(): k for k, v in self.model.config.id2label.items()}
        self.to_code = {labels["contradiction"]: 0, labels["neutral"]: 1, labels["entailment"]: 2}

    def check_implication(self, text1, text2, question):
        # The repo conditions DeBERTa on the question by prepending it to each answer
        inputs = self.tokenizer(f"{question} {text1}", f"{question} {text2}",
                                return_tensors="pt", truncation=True).to(self.device)
        with self.torch.no_grad():
            predicted = int(self.model(**inputs).logits.argmax(dim=1).item())
        return self.to_code[predicted]

    def save_cache(self):
        pass


def get_semantic_ids(answers, entailment, question, strict=True):
    """Port of the repo's get_semantic_ids: compare each new answer to each cluster's first member."""
    def equivalent(a, b):
        forward = entailment.check_implication(a, b, question)
        backward = entailment.check_implication(b, a, question)
        if strict:
            return forward == 2 and backward == 2
        return 0 not in (forward, backward) and (forward, backward) != (1, 1)

    ids = [-1] * len(answers)
    next_id = 0
    for i, a in enumerate(answers):
        if ids[i] == -1:
            ids[i] = next_id
            for j in range(i + 1, len(answers)):
                if ids[j] == -1 and equivalent(a, answers[j]):
                    ids[j] = next_id
            next_id += 1
    return ids


# ==========================================
# 4-5. Entropy
# ==========================================
def cluster_assignment_entropy(ids):
    """Discrete semantic entropy in nats (the repo's cluster_assignment_entropy)."""
    counts = np.bincount(ids)
    p = counts[counts > 0] / len(ids)
    return abs(float(-(p * np.log(p)).sum()))  # abs() avoids printing -0.0


def label_entropy(labels):
    codes = {label: k for k, label in enumerate(dict.fromkeys(labels))}
    return cluster_assignment_entropy([codes[label] for label in labels])


def score_row(row, cohort, args, gen_client, entailment):
    question = QUESTIONS[cohort]
    evidence = evidence_from_row(row, args.ablation)
    samples = sample_answers(gen_client, args.model, evidence, question,
                             args.num_generations, args.temperature, args.top_p)
    risks = [s["risk"] for s in samples]
    # An empty answer (format failure) is not a claim, so it must not be clustered as one
    answers = [s["answer"] for s in samples if s["answer"]]
    if len(answers) < 2:
        raise ValueError(f"only {len(answers)} usable answers out of {len(samples)} samples")
    ids = get_semantic_ids(answers, entailment, question, strict=not args.non_strict)
    majority_risk, majority_count = Counter(risks).most_common(1)[0]
    return {
        "patient_id": row["patient_id"],
        "cohort": cohort,
        "ablation": args.ablation,
        "generation_model": args.model,
        "entailment_model": args.entailment_name,
        "num_generations": args.num_generations,
        "temperature": args.temperature,
        "semantic_entropy": round(cluster_assignment_entropy(ids), 4),
        "n_semantic_clusters": len(set(ids)),
        "risk_entropy": round(label_entropy(risks), 4),
        "sampled_majority_risk": majority_risk,
        "sampled_majority_share": round(majority_count / len(risks), 3),
        "n_answers_clustered": len(answers),
        "n_unparsed_risk": risks.count("Unparsed"),
        "final_risk_class": row.get("final_risk_class"),  # the pipeline's low-temperature answer
        "Consensus_Risk": row.get("Consensus_Risk"),
        "semantic_ids": json.dumps(ids),
        "sampled_answers": json.dumps(answers),
        "sampled_risks": json.dumps(risks),
    }


def entropy_path(cohort, model, ablation):
    return os.path.join(EVALUATION_DIR, f"{cohort}_semantic_entropy__{run_tag(model, ablation)}.csv")


def run_cohort(cohort, args, gen_client, entailment):
    source = generation_path(cohort, args.model, args.ablation)
    if not os.path.exists(source):
        print(f"{source} not found. Run generation_pipeline.py for this model first. Skipping {cohort}.")
        return
    df = pd.read_csv(source, low_memory=False)
    df = df[df["final_risk_class"].isin(["High Risk", "Low Risk"])]
    if args.limit:
        df = df.head(args.limit)

    out = entropy_path(cohort, args.model, args.ablation)
    done = set()
    if args.resume and os.path.exists(out):
        done = set(pd.read_csv(out)["patient_id"])
        print(f"Resuming: {len(done)} {cohort} cases already scored.")
    elif os.path.exists(out):
        os.remove(out)

    print(f"\n--- {cohort}: {len(df)} cases, {args.num_generations} samples each at T={args.temperature} ---")
    for position, (_, row) in enumerate(df.iterrows(), start=1):
        if row["patient_id"] in done:
            continue
        print(f"[{position}/{len(df)}] {row['patient_id']}")
        try:
            result = score_row(row, cohort, args, gen_client, entailment)
        except Exception as e:
            print(f"  [ERROR] {e}")
            continue
        print(f"  semantic entropy {result['semantic_entropy']:.3f} "
              f"({result['n_semantic_clusters']} clusters), risk entropy {result['risk_entropy']:.3f}")
        pd.DataFrame([result]).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
        entailment.save_cache()
    print(f"Saved to {out}")


# ==========================================
# Validation: does entropy predict mistakes?
# ==========================================
def auroc(is_wrong, scores):
    from sklearn.metrics import roc_auc_score
    is_wrong, scores = np.asarray(is_wrong, int), np.asarray(scores, float)
    keep = ~np.isnan(scores)
    if keep.sum() < 2 or len(set(is_wrong[keep])) < 2:
        return float("nan")  # needs both right and wrong cases
    return float(roc_auc_score(is_wrong[keep], scores[keep]))


def rejection_accuracy(is_wrong, scores, keep_fraction=0.8):
    """Accuracy on the most-confident keep_fraction of cases (paper's 'rejection accuracy')."""
    order = np.argsort(np.asarray(scores, float), kind="stable")
    kept = order[:max(1, int(round(len(order) * keep_fraction)))]
    return float(1 - np.asarray(is_wrong, int)[kept].mean())


def summarise(label_name, is_wrong, df, rows):
    measures = {"semantic_entropy": df["semantic_entropy"], "risk_entropy": df["risk_entropy"]}
    # (Logprob entropy baseline disabled along with logprobs in generation_pipeline.py.)
    print(f"\n{label_name}: n = {len(df)}, error rate = {np.mean(is_wrong):.1%}")
    for name, scores in measures.items():
        row = {"label": label_name, "measure": name, "n": len(df),
               "error_rate": round(float(np.mean(is_wrong)), 3),
               "auroc": round(auroc(is_wrong, scores), 3),
               "accuracy_all": round(1 - float(np.mean(is_wrong)), 3),
               "rejection_accuracy_80": round(rejection_accuracy(is_wrong, scores.fillna(np.inf)), 3)}
        rows.append(row)
        print(f"  {name:28s} AUROC {row['auroc']:.3f} | accuracy {row['accuracy_all']:.3f} "
              f"-> {row['rejection_accuracy_80']:.3f} after rejecting the least confident 20%")


def validate(args):
    rows = []
    conc_path = entropy_path("concordant", args.model, args.ablation)
    if os.path.exists(conc_path):
        conc = pd.read_csv(conc_path)
        is_wrong = (conc["final_risk_class"] != conc["Consensus_Risk"]).astype(int)
        summarise("concordant: risk class wrong vs consensus", is_wrong, conc, rows)
    else:
        print(f"No concordant entropy file at {conc_path}.")

    disc_path = entropy_path("discordant", args.model, args.ablation)
    judge_path = evaluation_path("discordant", args.model, args.ablation)
    if os.path.exists(disc_path) and os.path.exists(judge_path):
        disc = pd.read_csv(disc_path).merge(pd.read_csv(judge_path), on="patient_id", how="inner",
                                            suffixes=("", "_judge"))
        threshold = float(config["evaluation"].get("success_threshold", 4.0))
        score_cols = ["bio_synthesis_score", "sys_reasoning_score", "prognostic_resolution_score"]
        summarise("discordant: any ungrounded claim", (disc["ungrounded_claim_count"] > 0).astype(int), disc, rows)
        summarise(f"discordant: mean judge score < {threshold}",
                  (disc[score_cols].mean(axis=1) < threshold).astype(int), disc, rows)
    else:
        print(f"\nNeed both {disc_path} and {judge_path} for the discordant validation.")

    if rows:
        out = pd.DataFrame(rows).assign(model=args.model, ablation=args.ablation)
        out.to_csv(VALIDATION_SUMMARY, mode="a", header=not os.path.exists(VALIDATION_SUMMARY), index=False)
        print(f"\nValidation summary appended to {VALIDATION_SUMMARY}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Discrete semantic entropy (Farquhar et al., 2024).")
    parser.add_argument("--model", default=config["pipeline"]["model_name"], help="Generation model (must match the generation run).")
    parser.add_argument("--ablation", default="full")
    parser.add_argument("--cohorts", nargs="+", choices=["concordant", "discordant"], default=["concordant", "discordant"])
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help="Cases per cohort (0 = all).")
    parser.add_argument("--num-generations", type=int, default=10, help="Paper default: 10.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Paper default: 1.0.")
    parser.add_argument("--top-p", type=float, default=0.9, help="Paper default: 0.9.")
    parser.add_argument("--entailment", choices=["llm", "deberta"], default="llm",
                        help="llm = judge model from config.yml; deberta = microsoft/deberta-v2-xlarge-mnli.")
    parser.add_argument("--non-strict", action="store_true", help="Use the repo's looser equivalence rule.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Compute AUROC from existing entropy and judge files.")
    args = parser.parse_args()

    if args.validate:
        validate(args)
    else:
        gen_client, judge_client = make_clients()
        if args.entailment == "deberta":
            entailment = EntailmentDeberta()
            args.entailment_name = "deberta-v2-xlarge-mnli"
        else:
            args.entailment_name = config["evaluation"]["judge_model"]
            entailment = EntailmentLLM(judge_client, args.entailment_name)
        for cohort in args.cohorts:
            run_cohort(cohort, args, gen_client, entailment)
