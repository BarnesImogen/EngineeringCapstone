# Interpreting Gene Expression Signatures using Large Language Models

This project is a clinical decision support system for TCGA-BRCA breast cancer data. It combines a bioinformatics pipeline with large language model generation and evaluation to study discordant prognostic risk classifications across multiple gene expression signatures.

The workflow is:
1. Upload the raw TCGA BRCA files and Reactome pathway mappings into the expected `data/raw/` folder.
2. Run `python src/bioinformatics_pipeline.py` to preprocess the cohort and write the concordant, discordant and borderline case tables (margin set by `bioinformatics.discordance_margin` in `config.yml`). Requires R with the Bioconductor `genefu` package for `rpy2`.
3. Run `python src/generation_pipeline.py --model NAME [--ablation full|no_pathways|clinical_only|classes_only] [--limit N]` to create model-generated summaries. `--limit` defaults to 6 for quick tests; use `--limit 0` for the full cohort. Outputs are named `{cohort}_results__{model}__{ablation}.csv`.
4. Run `python src/concordant_check.py --model NAME` to check the LLM against the unanimous consensus on concordant cases.
5. Run `python src/survival_analysis.py --model NAME` to compare the LLM's risk class against individual signatures, a majority vote, a clinical model and cross-validated Cox models, plus any ablation runs.
6. Run `python src/evaluation_pipeline.py --model NAME [--ablation ...]` to have the judge model in `config.yml` grade the summaries. `--model` is the generation model being judged.

All scripts are run from the repository root.

## What The Project Does

The bioinformatics pipeline loads TCGA-BRCA clinical and RNA-seq data, calculates risk scores for eight gene expression signatures, splits patients into concordant and discordant groups, and writes the processed results to CSV files.

The generation pipeline uses the OpenAI-compatible endpoint and model defined in `config.yml` to produce a structured clinical summary for each case. It leverages an offline Reactome database index to cross-reference upregulated patient biomarkers against verified oncology domains, providing the model with biologically grounded context.

The survival analysis pipeline uses empirical clinical data to validate the generated predictions, computing Kaplan-Meier survival curves, Cox Proportional Hazards regression, and Mann-Whitney U tests to assess statistical significance.

The evaluation pipeline uses the judge model defined in `config.yml` (which should differ from the generation model) to score each generated summary from 1 to 5 across three dimensions: biological synthesis, systematic reasoning, and prognostic resolution. It reports the share of summaries meeting `success_threshold`.

API keys can be set with `GENERATION_API_KEY` and `JUDGE_API_KEY` in `.env`, which override `api_key` in `config.yml`.

## Repository Layout

```text
Capstone/
├── config.yml
├── README.md
├── requirements.txt
├── data/
│   ├── raw/
│   ├── processed/
│   ├── generation_outputs/
│   └── evaluation_outputs/
├── notebooks/
└── src/
    ├── bioinformatics_pipeline.py
    ├── classification_calculations.py
    ├── concordant_check.py
    ├── data_prep.py
    ├── generation_pipeline.py
    ├── run_paths.py
    ├── signature_definitions.py
    ├── survival_analysis.py
    └── evaluation_pipeline.py