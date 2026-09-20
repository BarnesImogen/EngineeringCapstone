# TODO

- [ ] **Check LM Studio returns logprobs** for each generation model before any long run.
  - Run `python src/generation_pipeline.py --model NAME --limit 3` and open the output CSV.
  - `model_confidence_percent`, `model_entropy_score`, `decision_token_confidence` and `min_token_confidence` should be non-zero.
  - If they are all 0.0, the backend is not returning logprobs. Nothing errors in that case (see `generate_patient_summary` in `src/generation_pipeline.py`), so the confidence and entropy analyses would be silently empty.
  - Consider making the pipeline warn (or stop) when logprobs are missing.

## Maybe / circle back to

- [ ] **Ablation runs** (`--ablation no_pathways|clinical_only|classes_only`): the code exists in `src/generation_pipeline.py` and the survival comparison picks up any ablation files. Left out of the main flow for now. Circle back to test whether Reactome and expression evidence actually help, and whether the judge should score `no_pathways` runs.
