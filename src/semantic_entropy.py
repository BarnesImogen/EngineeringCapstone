"""Semantic entropy for the arbitration outputs, following Farquhar et al. (2024).

Paper: "Detecting hallucinations in large language models using semantic entropy",
Nature 630, 625-630. Reference code: https://github.com/jlko/semantic_uncertainty

No logprobs needed. This is the paper's *discrete* semantic entropy, which they
propose for models that do not return token probabilities.

For each patient:
  1. Rebuild the exact evidence the generation model saw (same system prompt and
     evidence sections as generation_pipeline.py).
  2. Ask the same focused question M times at temperature 1.0 (paper: M = 10,
     T = 1.0, nucleus sampling p = 0.9). Each answer is a short standard label
     of at most 6 words naming the single main biological program (e.g.
     "cell-cycle proliferation"), plus a FINAL RESOLUTION line. This is shorter
     than the paper's answers (see sampling_prompt for why).
  3. Cluster the answers by bidirectional entailment, conditioned on the
     question, using the paper's entailment prompt (entailment / neutral /
     contradiction). By default this uses the reference repo's looser rule:
     two answers match unless either direction is "contradiction" or both are
     "neutral". Pass --strict for the paper's rule, where both directions must
     be "entailment".
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
    Class counts are printed for each; a target with only one class is
    reported and its AUROC skipped.
  * Discordant cohort, continuous targets: hallucination_rate and mean judge
    score, scored with a concordance AUROC (Harrell's C) oriented so that
    > 0.5 means higher entropy goes with a higher hallucination rate or a
    lower judge score.

Usage (from the repository root):
  python src/semantic_entropy.py --model NAME [--cohorts concordant discordant] [--limit N]
  python src/semantic_entropy.py --model NAME --validate
  Options: --ablation, --num-generations 10, --temperature 1.0,
           --entailment llm|deberta, --strict, --resume, --workers N
"""

import argparse
import hashlib
import json
import os
import threading
import traceback
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from arbitration import SYSTEM_INSTRUCTION, build_prompt, parse_final_resolution
from run_paths import EVALUATION_DIR, evaluation_path, generation_path, run_tag, load_config
from llm_requests import SAVE_EVERY, add_workers_arg, chat, make_client, run_cli, run_in_pool, save_csv

load_dotenv()

config = load_config()

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
    # Deviation from Farquhar et al. (2024): the paper asks for a single sentence. Here each
    # answer is a short standard label (at most 6 words) naming one biological program. Even
    # one-sentence answers about mechanisms vary in wording and in added detail (genes,
    # qualifiers, hedges), which the entailment judge reads as "neutral" rather than
    # "entailment", so answers giving the same explanation split into many clusters and
    # entropy stays high even on concordant cases. A fixed, standard-terminology label keeps
    # one claim per answer, so clusters reflect different explanations, not different
    # phrasing. Plus the pipeline's FINAL RESOLUTION line.
    return (
        f"{evidence_prompt}\n\n"
        "Answer the following question with a short standard label of AT MOST 6 words that names "
        "the single main biological program (for example \"cell-cycle proliferation\" or "
        "\"estrogen-driven luminal differentiation\"). Use standard terminology. Do not write a "
        "sentence, do not list gene names, and do not add qualifiers or hedges.\n"
        f"Question: {question}\n\n"
        "Reply in exactly this format:\n"
        "ANSWER: <label of at most 6 words>\n"
        "FINAL RESOLUTION: High Risk or Low Risk"
    )


# ==========================================
# Clients
# ==========================================
def make_clients():
    gen = make_client(config["pipeline"]["base_url"],
                      os.getenv("GENERATION_API_KEY") or config["pipeline"].get("api_key", "lmstudio"))
    judge = make_client(config["evaluation"]["base_url"],
                        os.getenv("JUDGE_API_KEY") or config["evaluation"].get("api_key", "lmstudio"))
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


def sample_answers(gen_client, model, evidence, question, n, temperature, top_p, pool=None, limiter=None):
    """n samples, in order. With `pool` they run concurrently; `limiter` caps requests in flight."""
    prompt = sampling_prompt(evidence, question)

    def one_sample(_):
        response = chat(
            gen_client, limiter,
            model=model,
            temperature=temperature,
            top_p=top_p,
            messages=[{"role": "system", "content": SYSTEM_INSTRUCTION},
                      {"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content
        return {"answer": parse_answer(text),
                "risk": parse_final_resolution(text) or "Unparsed"}

    if pool is None:
        return [one_sample(i) for i in range(n)]
    futures = [pool.submit(one_sample, i) for i in range(n)]
    try:
        return [f.result() for f in futures]
    finally:
        for f in futures:
            f.cancel()  # one sample failed: do not spend requests on the rest of this case


# ==========================================
# 3. Entailment models (0 = contradiction, 1 = neutral, 2 = entailment, as in the repo)
# ==========================================
class EntailmentLLM:
    """Paper's LLM entailment prompt, with an on-disk cache like the reference code."""

    def __init__(self, client, model, limiter=None):
        self.client, self.model, self.limiter = client, model, limiter
        self.lock = threading.Lock()  # cases share the cache across threads
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
        reply = self.cache.get(key)
        if reply is None:
            response = chat(
                self.client, self.limiter,
                model=self.model, temperature=0.02,
                messages=[{"role": "user", "content": prompt}])
            reply = response.choices[0].message.content or ""
            with self.lock:
                self.cache[key] = reply
        reply = reply.lower()[:30]
        if "entailment" in reply:
            return 2
        if "contradiction" in reply:
            return 0
        return 1  # neutral, or an unclear reply (the repo does the same)

    def save_cache(self):
        os.makedirs(EVALUATION_DIR, exist_ok=True)
        with self.lock:
            snapshot = dict(self.cache)
        with open(f"{ENTAILMENT_CACHE}.tmp", "w") as f:
            json.dump(snapshot, f)
        os.replace(f"{ENTAILMENT_CACHE}.tmp", ENTAILMENT_CACHE)


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
        self.lock = threading.Lock()  # one forward pass at a time: the torch model is not thread-safe

    def check_implication(self, text1, text2, question):
        # The repo conditions DeBERTa on the question by prepending it to each answer
        with self.lock, self.torch.no_grad():
            inputs = self.tokenizer(f"{question} {text1}", f"{question} {text2}",
                                    return_tensors="pt", truncation=True).to(self.device)
            predicted = int(self.model(**inputs).logits.argmax(dim=1).item())
        return self.to_code[predicted]

    def save_cache(self):
        pass


def get_semantic_ids(answers, entailment, question, strict=False):
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


def score_row(row, cohort, args, gen_client, entailment, sample_pool=None, limiter=None):
    question = QUESTIONS[cohort]
    evidence = evidence_from_row(row, args.ablation)
    samples = sample_answers(gen_client, args.model, evidence, question,
                             args.num_generations, args.temperature, args.top_p, sample_pool, limiter)
    risks = [s["risk"] for s in samples]
    # An empty answer (format failure) is not a claim, so it must not be clustered as one
    answers = [s["answer"] for s in samples if s["answer"]]
    if len(answers) < 2:
        raise ValueError(f"only {len(answers)} usable answers out of {len(samples)} samples")
    ids = get_semantic_ids(answers, entailment, question, strict=args.strict)
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


def run_cohort(cohort, args, gen_client, entailment, sample_pool=None, limiter=None):
    source = generation_path(cohort, args.model, args.ablation)
    if not os.path.exists(source):
        print(f"{source} not found. Run generation_pipeline.py for this model first. Skipping {cohort}.")
        return
    df = pd.read_csv(source, low_memory=False)
    df = df[df["final_risk_class"].isin(["High Risk", "Low Risk"])]
    if args.limit:
        df = df.head(args.limit)

    out = entropy_path(cohort, args.model, args.ablation)
    # Only successfully scored cases are ever written, so every row in an existing file is done.
    results = []
    if args.resume and os.path.exists(out):
        results = pd.read_csv(out).to_dict("records")
        print(f"Resuming: {len(results)} {cohort} cases already scored.")
    elif os.path.exists(out):
        os.remove(out)
    done = {r["patient_id"] for r in results}

    pending = [row for _, row in df.iterrows() if row["patient_id"] not in done]
    print(f"\n--- {cohort}: {len(pending)} of {len(df)} cases to score, {args.num_generations} samples each "
          f"at T={args.temperature}, workers={args.workers} ---")

    # Only this (main) thread writes the CSV and the entailment cache, every SAVE_EVERY cases and on exit.
    order = {p_id: k for k, p_id in enumerate(df["patient_id"])}
    def save():
        if results:
            save_csv(pd.DataFrame(sorted(results, key=lambda r: order.get(r["patient_id"], len(order)))), out)
        entailment.save_cache()

    n_done = 0
    try:
        for row, result, error in run_in_pool(
                pending, lambda row: score_row(row, cohort, args, gen_client, entailment, sample_pool, limiter),
                args.workers):
            n_done += 1
            if error is not None:
                print(f"[{n_done}/{len(pending)}] [ERROR] {row['patient_id']}: {error}")
                traceback.print_exception(error)
                continue
            answers = json.loads(result["sampled_answers"])
            result["mean_answer_words"] = round(float(np.mean([len(a.split()) for a in answers])), 1)
            print(f"[{n_done}/{len(pending)}] {row['patient_id']}: semantic entropy {result['semantic_entropy']:.3f} "
                  f"({result['n_semantic_clusters']} clusters), risk entropy {result['risk_entropy']:.3f}, "
                  f"mean answer length {result['mean_answer_words']} words")
            results.append(result)
            if n_done % SAVE_EVERY == 0:
                save()
    finally:
        save()
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


def entropy_measures(df):
    # (Logprob entropy baseline disabled along with logprobs in generation_pipeline.py.)
    return {"semantic_entropy": df["semantic_entropy"], "risk_entropy": df["risk_entropy"]}


def summarise(label_name, is_wrong, df, rows):
    """Binary target: AUROC of each entropy measure for predicting is_wrong (1 = wrong)."""
    is_wrong = np.asarray(is_wrong, int)
    n_wrong = int(is_wrong.sum())
    n_right = len(is_wrong) - n_wrong
    print(f"\n{label_name}: n = {len(df)} ({n_wrong} wrong, {n_right} right), error rate = {np.mean(is_wrong):.1%}")
    one_class = n_wrong == 0 or n_right == 0
    if one_class:
        print(f"  Only one class ({'all wrong' if n_right == 0 else 'all right'}): AUROC needs both right "
              "and wrong cases, so it is skipped for this target.")
    for name, scores in entropy_measures(df).items():
        row = {"label": label_name, "target_type": "binary", "measure": name, "n": len(df),
               "n_wrong": n_wrong, "n_right": n_right,
               "error_rate": round(float(np.mean(is_wrong)), 3),
               "auroc": float("nan") if one_class else round(auroc(is_wrong, scores), 3),
               "accuracy_all": round(1 - float(np.mean(is_wrong)), 3),
               "rejection_accuracy_80": round(rejection_accuracy(is_wrong, scores.fillna(np.inf)), 3)}
        rows.append(row)
        auroc_text = "AUROC skipped (one class)" if one_class else f"AUROC {row['auroc']:.3f}"
        print(f"  {name:28s} {auroc_text} | accuracy {row['accuracy_all']:.3f} "
              f"-> {row['rejection_accuracy_80']:.3f} after rejecting the least confident 20%")


def concordance_auroc(badness, scores):
    """AUROC generalised to a continuous target (Harrell's C): over every pair of cases with
    different badness, the share where the case with higher entropy is the worse one (ties in
    entropy count 0.5). 0.5 = no signal, 1 = entropy perfectly ranks the cases worst-first.
    With a binary target this equals the ordinary AUROC."""
    badness, scores = np.asarray(badness, float), np.asarray(scores, float)
    keep = ~(np.isnan(badness) | np.isnan(scores))
    badness, scores = badness[keep], scores[keep]
    d_bad = np.sign(badness[:, None] - badness[None, :])
    d_score = np.sign(scores[:, None] - scores[None, :])
    usable = d_bad > 0  # each pair once, ordered so the first case is the worse one
    if not usable.any():
        return float("nan")
    return float(((d_score[usable] > 0) + 0.5 * (d_score[usable] == 0)).mean())


def summarise_continuous(label_name, target, higher_is_worse, df, rows):
    """Continuous target: concordance AUROC with the sign set so that 'entropy predicts worse
    outputs' always scores above 0.5 (higher hallucination rate, or lower judge score)."""
    target = pd.to_numeric(target, errors="coerce")
    badness = target if higher_is_worse else -target
    valid = target.dropna()
    direction = "higher = worse" if higher_is_worse else "lower = worse"
    print(f"\n{label_name} ({direction}): n = {len(valid)}"
          + (f" of {len(target)} (others missing)" if len(valid) < len(target) else "")
          + (f", mean {valid.mean():.3f}, range {valid.min():.3f} to {valid.max():.3f}, "
             f"{valid.nunique()} distinct values" if len(valid) else ""))
    one_class = valid.nunique() < 2
    if one_class:
        print("  Only one distinct value: there is nothing to rank, so AUROC is skipped for this target.")
    for name, scores in entropy_measures(df).items():
        rho = float("nan") if one_class else float(pd.Series(scores).corr(badness, method="spearman"))
        row = {"label": label_name, "target_type": "continuous", "measure": name, "n": len(valid),
               "auroc": float("nan") if one_class else round(concordance_auroc(badness, scores), 3),
               "spearman_vs_badness": round(rho, 3)}
        rows.append(row)
        if one_class:
            print(f"  {name:28s} AUROC skipped (one value)")
        else:
            print(f"  {name:28s} AUROC (concordance) {row['auroc']:.3f} | Spearman rho with badness {rho:+.3f}")


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
        mean_score = disc[score_cols].mean(axis=1)
        summarise("discordant: any ungrounded claim", (disc["ungrounded_claim_count"] > 0).astype(int), disc, rows)
        summarise(f"discordant: mean judge score < {threshold}", (mean_score < threshold).astype(int), disc, rows)
        summarise_continuous("discordant: hallucination rate (%)", disc["hallucination_rate"], True, disc, rows)
        summarise_continuous("discordant: mean judge score", mean_score, False, disc, rows)
    else:
        print(f"\nNeed both {disc_path} and {judge_path} for the discordant validation.")

    if rows:
        out = pd.DataFrame(rows).assign(model=args.model, ablation=args.ablation)
        # Rewrite rather than append, so older files with fewer columns stay aligned. Rows from an
        # earlier validation of the same model and ablation are replaced, not duplicated.
        if os.path.exists(VALIDATION_SUMMARY):
            old = pd.read_csv(VALIDATION_SUMMARY)
            old = old[~((old["model"] == args.model) & (old["ablation"] == args.ablation))]
            out = pd.concat([old, out], ignore_index=True)
        save_csv(out, VALIDATION_SUMMARY)
        print(f"\nValidation summary saved to {VALIDATION_SUMMARY}")


def main():
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
    parser.add_argument("--strict", action="store_true", help="Use the paper's strict bidirectional-entailment rule instead of the repo's looser default.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--validate", action="store_true", help="Compute AUROC from existing entropy and judge files.")
    add_workers_arg(parser, config)
    args = parser.parse_args()

    if args.validate:
        validate(args)
        return
    # One limit on requests in flight, shared by every case's samples and entailment calls
    limiter = threading.BoundedSemaphore(args.workers)
    sample_pool = ThreadPoolExecutor(max_workers=args.workers) if args.workers > 1 else None
    gen_client, judge_client = make_clients()
    if args.entailment == "deberta":
        entailment = EntailmentDeberta()
        args.entailment_name = "deberta-v2-xlarge-mnli"
    else:
        args.entailment_name = config["evaluation"]["judge_model"]
        entailment = EntailmentLLM(judge_client, args.entailment_name, limiter)
    for cohort in args.cohorts:
        run_cohort(cohort, args, gen_client, entailment, sample_pool, limiter)
    if sample_pool:
        sample_pool.shutdown()


if __name__ == "__main__":
    run_cli(main)
