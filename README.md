# Interpreting Gene Expression Signatures using Large Language Models

This project is a clinical decision support system for TCGA-BRCA breast cancer data. It combines a bioinformatics pipeline with large language model generation and evaluation to study discordant prognostic risk classifications across multiple gene expression signatures.

The workflow is:
1. Upload the raw TCGA BRCA files into the expected `data/raw/` folder.
2. Run `src/bioinformatics_pipeline.py` to preprocess the cohort and generate concordant and discordant case tables.
3. Run `src/generation_pipeline.py` to create model-generated clinical summaries for the discordant cases.
4. Run `src/evaluation_pipeline.py` to grade the generated summaries.

## What The Project Does

The bioinformatics pipeline loads TCGA-BRCA clinical and RNA-seq data, calculates risk scores for eight gene expression signatures, splits patients into concordant and discordant groups, and writes the processed results to CSV files.

The generation pipeline uses the provider and model defined in `config.yml` to produce a structured clinical summary for each case.

The evaluation pipeline uses Gemini as a judge to score each generated summary across three dimensions: biological synthesis, systematic reasoning, and clinical actionability.

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
    └── evaluation_pipeline.py
```

## Setup
Add the raw TCGA BRCA files to `data/raw/`:

- `Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi`
- `Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct`

Retrieve these datasets from this link: https://linkedomics.org/data_download/TCGA-BRCA/ 
Ensure that you download the Clinical and RNAseq (HiSeq, Gene level) datasets.

Create a `.env` file in the project root before using any LLM provider that needs API access. Put the relevant keys in that file, for example:

```bash
GEMINI_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

If you want to change the model or provider, update `config.yml`. The generation pipeline reads the `pipeline` section, so you can switch providers or model names there. The `.env` file must already exist and contain the matching API key for the provider you choose.

### Windows and R Setup

If you are running the project on Windows, use PowerShell rather than the macOS/Linux activation commands.

Prerequisites:

- Python 3.x
- R Statistical Software from CRAN, installed in a path such as `C:\Program Files\R\R-4.x.x`

Set up the Python environment in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks script execution, run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`, confirm with `Y`, and activate the environment again.

The bioinformatics pipeline uses `rpy2` to communicate with R. On Windows, `rpy2` may not find the R installation automatically, so the `R_HOME` and `PATH` variables should be set in the Python file where `rpy2` is first imported, such as `src/classification_calculations.py`:

```python
import os

r_home = r'C:\Program Files\R\R-4.x.x'
os.environ['R_HOME'] = r_home

r_dll_path = os.path.join(r_home, 'bin', 'x64')
os.environ['PATH'] = r_dll_path + ';' + os.environ.get('PATH', '')

import rpy2.robjects as robjects
```

Do not add `\bin` to the `r_home` path. Keep the `r` prefix so Python reads the Windows backslashes correctly.

Install the required R packages from the R Console, not through `pip`:

```r
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("genefu")
```

Once Python, R, and the R packages are installed, the pipeline scripts can run normally.

## Installation

Install the project dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running The Pipelines

Run the scripts in this order from the project root:

```bash
python src/bioinformatics_pipeline.py
python src/generation_pipeline.py
python src/evaluation_pipeline.py
```

### 1. Bioinformatics Pipeline

`src/bioinformatics_pipeline.py` reads the raw TCGA BRCA clinical and RNA-seq files from `data/raw/`, calculates the signature scores, identifies discordant and concordant cases, and writes:

- `data/processed/tcga_master_results.csv`
- `data/processed/tcga_discordant_cases.csv`
- `data/processed/tcga_concordant_cases.csv`

It also saves a signature correlation heatmap into `data/processed/`.

### 2. Generation Pipeline

`src/generation_pipeline.py` loads the provider and model from `config.yml`, reads API credentials from `.env`, and generates clinical summaries for the selected cases. The output is written to `data/generation_outputs/<provider>_generation_results.csv`.

### 3. Evaluation Pipeline

`src/evaluation_pipeline.py` reads the generated summaries and grades them with Gemini. The final scored results are written to `data/evaluation_outputs/gemini_evaluated_results.csv`.

## Configuration Notes

- `config.yml` controls the active provider, model name, and temperature settings.
- If you change the provider in `config.yml`, make sure the matching API key is present in `.env`.
- The evaluation script currently expects Gemini output in `data/generation_outputs/gemini_generation_results.csv`.

## Outputs

After running the full workflow, you should have:

- processed cohort tables in `data/processed/`
- model-generated summaries in `data/generation_outputs/`
- graded evaluation results in `data/evaluation_outputs/`

## Notes

- The notebooks in `notebooks/` are for exploration and visualisation.
- The raw TCGA files are required before the bioinformatics pipeline can run successfully.
- Do not commit `.env` or the raw data files to GitHub.
