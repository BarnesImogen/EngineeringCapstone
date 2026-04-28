# Interpreting Gene Expression Signatures using LLMs and ML

## Overview
This is the repository for my final year engineering capstone project. 
The goal is to investigate the discordance between various prognostic gene expression signatures (e.g., Oncotype DX, PAM50) in Breast Invasive Carcinoma (BRCA) and build an interpretable reasoning layer using Machine Learning and Large Language Models (LLMs).

## Data
Data is sourced from [LinkedOmics](https://linkedomics.org/) (TCGA-BRCA cohort).
* **RNA-seq Data:** `RNAseq (HiSeq, Gene level)`
* **Clinical Data:** `Clinical (Phenotype)`

## Project Structure
* `notebooks/`: Jupyter notebooks for experimentation.
* `src/`: Reusable Python modules for data processing and signature scoring.