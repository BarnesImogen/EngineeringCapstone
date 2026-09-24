# Interpreting Gene Expression Signatures using Large Language Models

**Research question:** Can large language models be used to resolve and interpret conflicting prognostic gene expression classifications in breast invasive carcinoma?

This project takes TCGA-BRCA patients on whom eight published gene expression signatures disagree, asks an LLM to arbitrate the conflict, and measures how well it does. *Resolve* means reaching a definitive High or Low Risk call. *Interpret* means explaining, from the patient's own data, why the signatures disagree.

## How the question is answered

| Part of the question | Evidence | Script |
|---|---|---|
| Conflicting classifications exist and are real | Margin-rule discordance rate, compared with the rate expected if the signatures were independent | `bioinformatics_pipeline.py` |
| The LLM can **resolve** conflicts | Accuracy on concordant cases, where the answer is known; stability of the High/Low call across repeated samples | `concordant_check.py`, `semantic_entropy.py` |
| The LLM can **interpret** conflicts | Judge audit of grounding, mechanistic reasoning and resolution; ungrounded-claim counts; ablations that remove evidence | `evaluation_pipeline.py`, `generation_pipeline.py --ablation` |
| The interpretation can be trusted (or flagged as unreliable) | Discrete semantic entropy (Farquhar et al., Nature 2024), validated by AUROC and rejection accuracy | `semantic_entropy.py` |

## Workflow

All scripts are run from the repository root.

1. Put the raw TCGA BRCA files and `ReactomePathways.gmt` in `data/raw/`.
2. `python src/bioinformatics_pipeline.py` scores the eight signatures and writes the concordant, discordant and borderline case tables to `data/processed/`. Requires R with the Bioconductor `genefu` package for `rpy2`. The discordance margin is `bioinformatics.discordance_margin` in `config.yml`.
3. `python src/generation_pipeline.py --model NAME [--ablation full|no_pathways|clinical_only|classes_only] [--limit N]` produces the LLM arbitration summaries. `--limit` defaults to 6 for quick tests; use `--limit 0` for the full cohort. Outputs are named `{cohort}_results__{model}__{ablation}.csv`. Use `--resume` to continue an interrupted run.
4. `python src/concordant_check.py --model NAME` checks the LLM against the unanimous consensus on concordant cases (the sanity check for "resolve").
5. `python src/evaluation_pipeline.py --model NAME [--ablation ...]` has the judge model in `config.yml` audit and score the discordant summaries. `--model` is the generation model being judged.
6. `python src/semantic_entropy.py --model NAME [--limit N]` measures semantic entropy: 10 samples per case at temperature 1.0, clustered by bidirectional entailment using the judge model (or `--entailment deberta`). Then run `python src/semantic_entropy.py --model NAME --validate` to test whether entropy predicts wrong risk classes (concordant cohort) and poor judge scores (discordant cohort).

API keys can be set with `GENERATION_API_KEY` and `JUDGE_API_KEY` in `.env`, which override `api_key` in `config.yml`. Keep the judge model different from the generation model.

## What each part does

### Building the conflict
- `data_prep.py` loads and merges the TCGA-BRCA RNA-seq and clinical files (primary tumours only, one row per patient).
- `signature_definitions.py` holds the gene sets and weights for the eight signatures (Oncotype DX, PAM50, Breast Cancer Index, Mammostrat, IHC4, Kim-10, IRRS-7, Hu-11). It is the single source of truth for both the scoring code and the prompts, so the LLM is told exactly what was computed.
- `classification_calculations.py` computes each signature's continuous score. PAM50 uses the Parker centroids from the R `genefu` package. Oncotype DX, BCI and Mammostrat are research re-implementations, not the proprietary assays.
- `bioinformatics_pipeline.py` median-splits each score into High (1) or Low (0), then defines the cohorts:
  - **Discordant:** at least one signature puts the patient in its top `margin` fraction and another puts them in its bottom `margin` fraction (a clear-cut conflict, not noise near the median).
  - **Concordant:** all eight median-split classes agree. The shared class is the known answer (`Consensus_Risk`).
  - **Borderline:** mild disagreement only; excluded from both.

### Resolving and interpreting: the LLM arbitrator
`generation_pipeline.py` builds one prompt per patient from the clinical metadata, the signature definitions, the eight conflicting classes, the expression values and Reactome pathways matched to the patient's upregulated genes. The model works through three steps: Conflict Diagnosis, Mechanistic Root Cause (interpret) and Prognostic Resolution (resolve). A system prompt restricts it to the supplied evidence. The output is extracted into a validated JSON schema.

Ablations remove evidence blocks (`no_pathways`, `clinical_only`, `classes_only`) to test what the model needs to see in order to interpret conflicts well.

### Evaluating resolution
- `concordant_check.py` measures agreement with the unanimous consensus, prints a confusion matrix and flags any model below `min_concordant_agreement` (0.8). This shows the model is not a random label generator. It is a sanity check, not proof of resolving real conflicts, because concordant cases are easy.
- Semantic entropy's `risk_entropy` and `sampled_majority_share` show how stable the High/Low call is across samples.

### Evaluating interpretation
`evaluation_pipeline.py` uses a separate judge model. It receives the full context the generator saw and first lists every ungrounded claim (invented genes or values, wrong weights, unsupported mechanisms). It then writes justifications and scores 1-5 for biological synthesis, systematic reasoning and prognostic resolution. A deterministic cap keeps the biological score at 3 or below with any ungrounded claim, and at 2 or below with three or more. It reports the share of summaries meeting `success_threshold`.

### Trusting the interpretation: semantic entropy
`semantic_entropy.py` implements discrete semantic entropy. For each case it asks one focused question 10 times, clusters the one-sentence answers by meaning and computes the entropy of the cluster shares. Low entropy means the model repeatedly gives the same explanation; high entropy means the explanations differ, a known sign of confabulation. `risk_entropy` is the same calculation over the High/Low labels. `--validate` reports AUROC and rejection accuracy.

Token-logprob confidence is switched off for now (commented out in the generation, concordant check and evaluation scripts).

## Limitations
- Discordant cases have no ground truth. "Resolve" is evaluated through the concordant benchmark, the stability of the call across samples and the judge's rating.
- The judge is an LLM. Its scores are also the validation target for semantic entropy on discordant cases.
- High and Low are relative to the cohort median, not absolute risk.
- Oncotype DX, BCI and Mammostrat are approximations of the proprietary assays.

## Repository Layout

```text
Capstone/
├── config.yml
├── README.md
├── TODO.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generation_outputs/
│   ├── evaluation_outputs/
│   └── archive/          (old outputs from earlier framework versions, not tracked)
├── notebooks/
└── src/
    ├── bioinformatics_pipeline.py
    ├── classification_calculations.py
    ├── concordant_check.py
    ├── data_prep.py
    ├── evaluation_pipeline.py
    ├── generation_pipeline.py
    ├── run_paths.py
    ├── semantic_entropy.py
    └── signature_definitions.py
```
