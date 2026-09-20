# Interpreting Gene Expression Signatures using Large Language Models

This project is a clinical decision support system for TCGA-BRCA breast cancer data. It combines a bioinformatics pipeline with large language model generation and evaluation to study discordant prognostic risk classifications across multiple gene expression signatures.

The workflow is:
1. Upload the raw TCGA BRCA files and Reactome pathway mappings into the expected `data/raw/` folder.
2. Run `src/bioinformatics_pipeline.py` to preprocess the cohort and generate concordant and discordant case tables.
3. Run `src/generation_pipeline.py` to create model-generated clinical summaries for the discordant cases.
4. Run `src/survival_analysis.py` to validate clinical predictive superiority and token entropy calibration.
5. Run `src/evaluation_pipeline.py` to grade the generated summaries.

## What The Project Does

The bioinformatics pipeline loads TCGA-BRCA clinical and RNA-seq data, calculates risk scores for eight gene expression signatures, splits patients into concordant and discordant groups, and writes the processed results to CSV files.

The generation pipeline uses the provider and model defined in `config.yml` to produce a structured clinical summary for each case. It leverages an offline Reactome database index to cross-reference upregulated patient biomarkers against verified oncology domains, providing the model with biologically grounded context.

The survival analysis pipeline uses empirical clinical data to validate the generated predictions, computing Kaplan-Meier survival curves, Cox Proportional Hazards regression, and Mann-Whitney U tests to assess statistical significance.

The evaluation pipeline uses the judge provider defined in `config.yml` to score each generated summary across three dimensions: biological synthesis, systematic reasoning, and clinical actionability.

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
    ├── data_prep.py
    ├── generation_pipeline.py
    ├── survival_analysis.py
    └── evaluation_pipeline.py