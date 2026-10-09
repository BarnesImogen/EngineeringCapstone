# Interpreting Gene Expression Signatures using Large Language Models

**Research question:** Can large language models be used to resolve and interpret conflicting prognostic gene expression classifications in breast invasive carcinoma?

This project takes TCGA-BRCA patients on whom five published gene expression signatures disagree, asks an LLM to arbitrate the conflict, and measures how well it does. *Resolve* means reaching a definitive High or Low Risk call. *Interpret* means explaining, from the patient's own data, why the signatures disagree.

## How the question is answered

| Part of the question | Evidence | Script |
|---|---|---|
| Conflicting classifications exist and are real | Discordance rate (at least one signature High and one Low), compared with the rate expected if the signatures were independent, with and without Hu-11 | `bioinformatics_pipeline.py` |
| The LLM can **resolve** conflicts | Accuracy on concordant cases, where the answer is known; stability of the High/Low call across repeated samples | `concordant_check.py`, `semantic_entropy.py` |
| The LLM can **interpret** conflicts | Judge audit of grounding, mechanistic reasoning and resolution; ungrounded-claim counts; ablations that remove evidence | `evaluation_pipeline.py`, `generation_pipeline.py --ablation` |
| The interpretation can be trusted (or flagged as unreliable) | Discrete semantic entropy (Farquhar et al., Nature 2024), validated by AUROC and rejection accuracy | `semantic_entropy.py` |

## Setup

### 1. Raw data

Add the raw TCGA BRCA files to `data/raw/`:

- `Human__TCGA_BRCA__MS__Clinical__Clinical__01_28_2016__BI__Clinical__Firehose.tsi`
- `Human__TCGA_BRCA__UNC__RNAseq__HiSeq_RNA__01_28_2016__BI__Gene__Firehose_RSEM_log2.cct`

Retrieve these from https://linkedomics.org/data_download/TCGA-BRCA/ — download the Clinical and RNAseq (HiSeq, Gene level) datasets.

Also put `ReactomePathways.gmt` in `data/raw/` — it's used by the generation pipeline to match a patient's upregulated genes to pathways.

### 2. Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, use PowerShell instead:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks script execution, run `Set-ExecutionPolicy Unrestricted -Scope CurrentUser`, confirm with `Y`, and activate the environment again.

### 3. R (required for `bioinformatics_pipeline.py`)

The bioinformatics pipeline uses `rpy2` to call R's Bioconductor `genefu` package for the PAM50 Parker centroids. Install R from CRAN, then from the R console:

```r
if (!require("BiocManager", quietly = TRUE)) install.packages("BiocManager")
BiocManager::install("genefu")
```

**Windows only:** `rpy2` may not find the R installation automatically. Set `R_HOME` and `PATH` in the Python file where `rpy2` is first imported (`src/classification_calculations.py`):

```python
import os

r_home = r'C:\Program Files\R\R-4.x.x'
os.environ['R_HOME'] = r_home

r_dll_path = os.path.join(r_home, 'bin', 'x64')
os.environ['PATH'] = r_dll_path + ';' + os.environ.get('PATH', '')

import rpy2.robjects as robjects
```

Do not add `\bin` to `r_home`, and keep the `r` prefix so Python reads the Windows backslashes correctly.

### 4. Model access

Both the generation and judge stages talk to any OpenAI-compatible endpoint (LM Studio, OpenAI, vLLM, ...); `base_url`/`model_name` for each live in `config.yml`.

Create a `.env` file in the project root:

```bash
GENERATION_API_KEY=your_key_here
JUDGE_API_KEY=your_key_here
```

These override `api_key` in `config.yml` when set. With a local LM Studio server, `config.yml`'s default `api_key: "lmstudio"` is enough and `.env` can be left out. Keep the judge model different from the generation model to avoid self-preference bias. Never commit `.env`.

## Running the pipelines

All commands are run from the repository root, in this order.

**1. Score signatures & build cohorts**
```bash
python src/bioinformatics_pipeline.py
```
Needs R + Bioconductor `genefu` via `rpy2`. Scores the five signatures and writes the concordant, discordant and borderline case tables to `data/processed/` (`tcga_master_results.csv`, `tcga_discordant_cases.csv`, `tcga_concordant_cases.csv`), plus a signature correlation heatmap. It also prints the expected discordance under independence and, for the discordant cohort, how often each signature is the lone dissenter.

**2. LLM arbitration**
```bash
python src/generation_pipeline.py --model NAME --limit 6
```
Quick test with 6 cases; use `--limit 0` for the full cohort, `--resume` to continue an interrupted run. `--sampler stratified` makes a pilot draw with a fixed seed instead of taking the first N: discordant cases spread evenly across the lone dissenting signature (Hu-11, IRRS-7, other, none), concordant cases evenly across Consensus High and Low. `--ablation no_pathways|clinical_only|classes_only` exists but per [TODO.md](TODO.md) is being left out of the main flow for now. Outputs are written to `data/generation_outputs/` as `{cohort}_results__{model}__{ablation}.csv`.

**3. Sanity check on concordant cases**
```bash
python src/concordant_check.py --model NAME
```
Checks the LLM against the consensus on concordant cases (all five signatures High, or all five Low) — the sanity check for "resolve".

**4. Judge audit of discordant summaries**
```bash
python src/evaluation_pipeline.py --model NAME
```
The judge model from `config.yml` audits and scores the discordant summaries. `--model` is the generation model being judged, not the judge itself. Results go to `data/evaluation_outputs/`.

**5. Semantic entropy (trust signal)**
```bash
python src/semantic_entropy.py --model NAME --limit N
```
10 samples per case at temperature 1.0, clustered by bidirectional entailment using the judge model (or `--entailment deberta`). Then run:
```bash
python src/semantic_entropy.py --model NAME --validate
```
to test whether entropy predicts wrong risk classes (concordant cohort) and poor judge scores (discordant cohort).

`NAME` should match whichever model is loaded at `pipeline.base_url` in `config.yml`.

## What each part does

### Building the conflict
- `data_prep.py` loads and merges the TCGA-BRCA RNA-seq and clinical files (primary tumours only, one row per patient).
- `signature_definitions.py` holds the gene sets and weights for the five RNA-based signatures (Oncotype DX, PAM50, Breast Cancer Index, IRRS-7, Hu-11). IHC4 and Mammostrat were removed because they are protein (IHC) assays and cannot follow their papers on RNA-seq; Kim-10 was removed earlier because it is TNBC-only. It is the single source of truth for both the scoring code and the prompts, so the LLM is told exactly what was computed.
- `classification_calculations.py` computes each signature's continuous score and its Low / Intermediate / High class. Clinical ER status exists for only ~10% of patients, so ER status is inferred for everyone from ESR1 expression (two-component Gaussian mixture; 98% agreement with the known labels) and saved as `ER_Status_Inferred`.
  - **Oncotype DX** (Paik et al., 2004; genefu `oncotypedx`): genes are normalised against the mean of five reference genes (ACTB, GAPDH, GUSB, RPLP0, TFRC), rescaled to 0-15 across the cohort, combined into the published group scores (with floors) and the unscaled score RSU. RSU is the continuous score and is classified by cohort tertiles. The published RS cutoffs (18 / 31) are not used: they need gene-wise calibration against commercial Recurrence Scores from the same tumours (Buus et al., 2021, npj Breast Cancer 7:15), and TCGA has no commercial scores. Applied without calibration, the algorithm overestimates RS about twofold and places most patients in a higher risk group (Bartlett et al., 2020, PLoS ONE 15:e0238593). On this data the cutoffs put 45-67% of ER+ patients in High across three scalings, against 27% in Paik et al. (2004)'s validation cohort. RS (bounded 0-100) is kept as `OncotypeDX_RS` for reference.
  - **PAM50** (Parker et al., 2009; genefu `rorS`): Parker centroids from the R `genefu` package, ER-balanced median centring (fixed seed), Spearman correlation to each centroid, ROR-S = 0.05·Basal + 0.12·HER2 − 0.34·LumA + 0.23·LumB, rescaled to 0-100 on the cohort's 2.5th-97.5th percentiles. Published cutoffs: Low < 29, Intermediate < 53, High ≥ 53. Two PAM50 genes (CDCA1, KNTC2) are listed under symbols the RNA-seq file doesn't use, so `signature_definitions.py` aliases them to NUF2 and NDC80.
  - **Breast Cancer Index** (Ma et al., 2008; Habel et al., 2013): MGI is the first principal component of five standardised proliferation genes; H/I = HOXB13 − IL17RB, standardised. Standardisation and the PCA are fitted on ER+ patients (the one deviation from the published assay, which used a reference set that isn't available). BCI = 0.4431·MGI + 0.4972·H/I − 0.09·H/I³, rescaled to 0-10. Published cutoffs: Low < 5, Intermediate < 6.4, High ≥ 6.4.
  - **IRRS-7** (Wei et al., 2025) and **Hu-11** (Hu et al., 2024): weighted sums of log2 expression with the published coefficients (Hu-11's read from the paper's Figure 3C, two decimals), classified by cohort tertiles (their papers used a median split; tertiles keep the panel consistent). Both were developed on TCGA-BRCA, so their survival association in this cohort is in-sample.
  - Oncotype DX and BCI are research re-implementations, not the proprietary assays. Each signature also keeps the old 0/1 median split as `{sig}_Class_Median` for sensitivity analysis.
- `bioinformatics_pipeline.py` defines the cohorts from the five classes:
  - **Discordant:** at least one signature High and at least one Low. `Lone_Dissenter` names the signature that alone takes the opposite side, if there is one.
  - **Concordant:** all five High or all five Low. The shared class is the known answer (`Consensus_Risk`: High Risk / Low Risk).
  - **Borderline:** everyone else, including patients with all five Intermediate (the LLM's binary answer would have nothing to be checked against). Excluded from both.
  - `Discordance_Group` is the five-signature definition used by generation and evaluation; `Discordance_Group_No_Hu11` repeats it without Hu-11 as a sensitivity analysis.

### Resolving and interpreting: the LLM arbitrator
`generation_pipeline.py` builds one prompt per patient from the clinical metadata, the signature definitions, the five conflicting classes, the expression values and Reactome pathways matched to the patient's upregulated genes. The same prompt is used for concordant and discordant patients. The model works through three steps: Classification Assessment (identify any High-against-Low conflict), Mechanistic Explanation (interpret) and Prognostic Resolution (resolve). Each signature line shows the class, the patient's score and the cutoffs that produced the class. The prompt explains that an Intermediate signature does not take a side, and the model must still finish with a binary `FINAL RESOLUTION: High Risk` or `Low Risk`. A system prompt restricts it to the supplied evidence. The output is extracted into a validated JSON schema. Each output row also records `majority_vote` and `majority_vote_without_hu11` (High against Low calls, Intermediate ignored, `Tie` when equal) to compare the model's call against.

Ablations remove evidence blocks (`no_pathways`, `clinical_only`, `classes_only`) to test what the model needs to see in order to interpret conflicts well.

### Evaluating resolution
- `concordant_check.py` measures agreement with the consensus (overall and per consensus class), prints a confusion matrix and flags any model below `min_concordant_agreement` (0.8). This shows the model is not a random label generator. It is a sanity check, not proof of resolving real conflicts, because concordant cases are easy.
- Semantic entropy's `risk_entropy` and `sampled_majority_share` show how stable the High/Low call is across samples.

### Evaluating interpretation
`evaluation_pipeline.py` uses a separate judge model. It receives the full context the generator saw and first lists every ungrounded claim (invented genes or values, wrong weights, misquoted formulas, cutoffs or classes, unsupported mechanisms). It is told that Intermediate signatures do not take a side. It then writes justifications and scores 1-5 for biological synthesis, systematic reasoning and prognostic resolution. A deterministic cap keeps the biological score at 3 or below with any ungrounded claim, and at 2 or below with three or more. It reports the share of summaries meeting `success_threshold`.

### Trusting the interpretation: semantic entropy
`semantic_entropy.py` implements discrete semantic entropy. For each case it asks one focused question 10 times, clusters the one-sentence answers by meaning and computes the entropy of the cluster shares. Low entropy means the model repeatedly gives the same explanation; high entropy means the explanations differ, a known sign of confabulation. `risk_entropy` is the same calculation over the High/Low labels. `--validate` reports AUROC and rejection accuracy.

Token-logprob confidence is switched off for now (commented out in the generation, concordant check and evaluation scripts).

## Configuration notes

- `config.yml` controls the active model names, base URLs, temperatures and thresholds for both the `pipeline` (generation) and `evaluation` (judge) stages.
- `GENERATION_API_KEY` / `JUDGE_API_KEY` in `.env` override `api_key` in `config.yml` — set them when pointing at a hosted provider rather than a local server.
- Keep the judge model different from the generation model to avoid self-preference bias.

## Outputs

After running the full workflow, you should have:

- processed cohort tables (+ correlation heatmap) in `data/processed/`
- LLM arbitration summaries in `data/generation_outputs/`
- judge scores in `data/evaluation_outputs/`
- semantic entropy results alongside the generation/evaluation outputs they were computed from

## Limitations
- Discordant cases have no ground truth. "Resolve" is evaluated through the concordant benchmark, the stability of the call across samples and the judge's rating.
- The judge is an LLM. Its scores are also the validation target for semantic entropy on discordant cases.
- For Oncotype DX, IRRS-7 and Hu-11 the classes are cohort tertiles, so they are relative to TCGA-BRCA, not absolute risk. PAM50's 0-100 scale is also rescaled on this cohort.
- Hu-11 is nearly uncorrelated with the other four signatures and is the lone dissenter in most discordant cases; `Discordance_Group_No_Hu11` and `majority_vote_without_hu11` exist to test its influence.
- IRRS-7 and Hu-11 were trained on TCGA-BRCA, so their survival associations here are in-sample.
- Oncotype DX and BCI are approximations of the proprietary assays.

## Notes
- The notebooks in `notebooks/` are for exploration and visualisation. They predate the five-signature panel and will not run against the current code without changes.
- The raw TCGA files and `ReactomePathways.gmt` are required before the bioinformatics/generation pipelines can run successfully.
- Do not commit `.env` or the raw data files to GitHub.

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
