# TODO

- [ ] **Logprob confidence is switched off** (commented out in `src/generation_pipeline.py`, `concordant_check.py`, `evaluation_pipeline.py`). Semantic entropy is the main uncertainty signal for now. If it comes back, first check LM Studio actually returns logprobs (the fields silently become 0.0 otherwise).

## Maybe / circle back to

- [ ] **Ablation runs** (`--ablation no_pathways|clinical_only|classes_only`): the code exists in `src/generation_pipeline.py`. Left out of the main flow for now. Circle back to test whether Reactome and expression evidence actually help, and whether the judge should score `no_pathways` runs.
