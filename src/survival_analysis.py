import os
import glob
import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import mannwhitneyu, chi2
from run_paths import GENERATION_DIR, EVALUATION_DIR, generation_path, run_tag, model_slug

OUTPUT_DIR = EVALUATION_DIR

SIGNATURE_CLASS_COLUMNS = {
    "Oncotype DX": "OncotypeDX_Class", "PAM50": "Pam50_Class", "BCI": "BCI_Class",
    "Mammostrat": "Mammostrat_Class", "IHC4": "IHC4_Class", "Kim10": "Kim10_Class",
    "IRRS7": "IRRS7_Class", "Hu11": "Hu11_Class",
}
STAGE_PATTERN = r"stage\s*(iv|iii|ii|i)"
STAGE_ORDINAL = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
N_BOOTSTRAPS = 500
MASTER_FILE = "data/processed/tcga_master_results.csv"
SCORE_COLUMNS = [col.replace("_Class", "_Score") for col in SIGNATURE_CLASS_COLUMNS.values()]
CLINICAL_COLUMNS = ["years_to_birth", "stage_ordinal", "er_positive"]

def add_clinical_covariates(df):
    """Adds numeric stage and ER covariates used for the clinical baseline."""
    df = df.copy()
    stage = df["pathologic_stage"].astype(str).str.lower().str.extract(STAGE_PATTERN)[0]
    df["stage_ordinal"] = stage.map(STAGE_ORDINAL)
    df["er_positive"] = df["ER.Status"].astype(str).str.lower().map({"positive": 1, "negative": 0})
    return df

def bootstrap_c_index(durations, events, risk, n_boot=N_BOOTSTRAPS, seed=0):
    """95% percentile CI for the C-index (higher risk = shorter survival)."""
    rng = np.random.default_rng(seed)
    durations, events, risk = np.asarray(durations), np.asarray(events), np.asarray(risk)
    values = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(durations), len(durations))
        if events[idx].sum() == 0:
            continue
        try:
            values.append(concordance_index(durations[idx], -risk[idx], events[idx]))
        except ZeroDivisionError:
            continue
    if not values:
        return np.nan, np.nan
    return tuple(np.percentile(values, [2.5, 97.5]))

def evaluate_binary_predictor(name, df, risk_high, risk_score=None):
    """Log-rank p, Cox HR and C-index for a High(1)/Low(0) label on the shared patient set.
    If a continuous risk_score is given the C-index uses it instead of the binary label."""
    risk_high = pd.Series(risk_high, index=df.index).astype(float)
    c_index_risk = risk_high if risk_score is None else pd.Series(risk_score, index=df.index).astype(float)
    row = {"predictor": name, "n_high": int(risk_high.sum()), "n_low": int((risk_high == 0).sum()),
           "logrank_p": np.nan, "hazard_ratio": np.nan, "c_index": np.nan, "c_index_ci_low": np.nan, "c_index_ci_high": np.nan}
    if risk_high.nunique() < 2:
        return row

    high, low = df[risk_high == 1], df[risk_high == 0]
    row["logrank_p"] = logrank_test(
        high["overall_survival"], low["overall_survival"],
        event_observed_A=high["status"], event_observed_B=low["status"]
    ).p_value

    try:
        cox_df = pd.DataFrame({"T": df["overall_survival"], "E": df["status"], "risk": risk_high})
        cph = CoxPHFitter(penalizer=0.01).fit(cox_df, duration_col="T", event_col="E")
        row["hazard_ratio"] = float(np.exp(cph.params_["risk"]))
    except Exception:
        pass

    row["c_index"] = concordance_index(df["overall_survival"], -c_index_risk, df["status"])
    row["c_index_ci_low"], row["c_index_ci_high"] = bootstrap_c_index(df["overall_survival"], df["status"], c_index_risk)
    return row

def cross_validated_cox_risk(master, feature_cols, k=5, penalizer=0.1, seed=0):
    """Out-of-fold Cox log-hazard for every patient in the full cohort (each patient is scored by a
    model that never saw them). Features are standardised with cohort statistics (no outcome used)."""
    data = master[["overall_survival", "status"] + feature_cols].dropna()
    features = data[feature_cols]
    data = data.assign(**{c: (features[c] - features[c].mean()) / features[c].std() for c in feature_cols})

    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(data)), k)
    risk = pd.Series(np.nan, index=data.index)
    for test_idx in folds:
        train = data.iloc[np.setdiff1d(np.arange(len(data)), test_idx)]
        test = data.iloc[test_idx]
        cph = CoxPHFitter(penalizer=penalizer).fit(train, duration_col="overall_survival", event_col="status")
        risk.iloc[test_idx] = cph.predict_log_partial_hazard(test).to_numpy()
    return risk

def load_ablation_labels(model):
    """Map ablation name -> Series(patient_id -> valid High/Low label) for this model's non-full discordant runs."""
    labels = {}
    prefix = f"discordant_results__{model_slug(model)}__"
    for path in sorted(glob.glob(os.path.join(GENERATION_DIR, f"{prefix}*.csv"))):
        name = os.path.basename(path)[len(prefix):-len(".csv")]
        if name == "full":
            continue
        run = pd.read_csv(path, low_memory=False)
        run = run[run["final_risk_class"].isin(["High Risk", "Low Risk"])]
        labels[name] = run.drop_duplicates("patient_id").set_index("patient_id")["final_risk_class"]
    return labels

def build_competitor_risks():
    """Cross-validated Cox models on the full cohort: non-LLM competitors to the LLM arbitration."""
    if not os.path.exists(MASTER_FILE):
        print(f"{MASTER_FILE} not found; skipping non-LLM competitor models.")
        return {}
    master = add_clinical_covariates(pd.read_csv(MASTER_FILE, index_col=0, low_memory=False))
    for col in ["overall_survival", "status", "years_to_birth"] + SCORE_COLUMNS:
        master[col] = pd.to_numeric(master[col], errors="coerce")
    return {
        "Cox on 8 signature scores (5-fold CV)": cross_validated_cox_risk(master, SCORE_COLUMNS),
        "Cox on scores + clinical (5-fold CV)": cross_validated_cox_risk(master, SCORE_COLUMNS + CLINICAL_COLUMNS),
    }

def run_baseline_comparison(df_clean, model):
    print("--- 4. Baseline Comparison (LLM vs Signatures vs Clinical) ---")
    df = add_clinical_covariates(df_clean)
    class_cols = list(SIGNATURE_CLASS_COLUMNS.values())
    required = ["overall_survival", "status", "years_to_birth", "stage_ordinal", "er_positive"] + class_cols
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        print(f"Missing columns for baseline comparison: {missing_cols}")
        return

    # Every predictor is scored on exactly the same patients
    df = df.dropna(subset=required)
    df = df[df["final_risk_class"].isin(["High Risk", "Low Risk"])]

    # Restrict to patients every extra predictor can score, so all rows use the same patients
    ablation_labels = load_ablation_labels(model)
    competitor_risks = build_competitor_risks()
    df = df.set_index("patient_id", drop=False)
    for labels in ablation_labels.values():
        df = df[df.index.isin(labels.index)]
    for risk in competitor_risks.values():
        df = df[df.index.isin(risk.dropna().index)]
    n_events = int(df["status"].sum())
    print(f"Shared patient set: n={len(df)}, events={n_events}")
    if len(df) < 10 or n_events < 3:
        print("Too few patients/events for a meaningful comparison.\n")
        return

    rows = [evaluate_binary_predictor("LLM arbitration", df, df["final_risk_class"] == "High Risk")]

    # Majority vote: High when at least half of the eight signatures say High
    vote_fraction = df[class_cols].mean(axis=1)
    rows.append(evaluate_binary_predictor("Majority vote (>=4/8 High)", df, vote_fraction >= 0.5))

    for label, col in SIGNATURE_CLASS_COLUMNS.items():
        rows.append(evaluate_binary_predictor(label, df, df[col]))

    # LLM ablations: same prompt with evidence removed
    for name, labels in ablation_labels.items():
        rows.append(evaluate_binary_predictor(f"LLM ablation: {name}", df, labels.reindex(df.index) == "High Risk"))

    # Non-LLM competitors trained on the whole cohort; High = above the cohort median out-of-fold risk
    for name, risk in competitor_risks.items():
        threshold = risk.median()
        rows.append(evaluate_binary_predictor(name, df, risk.reindex(df.index) > threshold, risk_score=risk.reindex(df.index)))

    # Clinical baseline: age + stage + ER in a Cox model, scored by its linear predictor
    clin_cols = ["years_to_birth", "stage_ordinal", "er_positive"]
    clin_df = df[["overall_survival", "status"] + clin_cols]
    clin_cph = None
    try:
        clin_cph = CoxPHFitter(penalizer=0.1).fit(clin_df, duration_col="overall_survival", event_col="status")
        clin_risk = clin_cph.predict_partial_hazard(clin_df)
        rows.append({
            "predictor": "Clinical only (age+stage+ER, in-sample)", "n_high": np.nan, "n_low": np.nan,
            "logrank_p": np.nan, "hazard_ratio": np.nan,
            "c_index": concordance_index(df["overall_survival"], -clin_risk, df["status"]),
            "c_index_ci_low": bootstrap_c_index(df["overall_survival"], df["status"], clin_risk)[0],
            "c_index_ci_high": bootstrap_c_index(df["overall_survival"], df["status"], clin_risk)[1],
        })
    except Exception as e:
        print(f"Clinical baseline failed: {e}")

    results = pd.DataFrame(rows)
    print(results.round(4).to_string(index=False))
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    table_path = os.path.join(OUTPUT_DIR, f"survival_baseline_comparison__{model_slug(model)}.csv")
    results.to_csv(table_path, index=False)
    print(f"Comparison table saved to {table_path}")

    # Does the LLM label add information beyond the clinical variables? (likelihood-ratio test)
    if clin_cph is not None:
        try:
            full_df = clin_df.copy()
            full_df["llm_high_risk"] = (df["final_risk_class"] == "High Risk").astype(int)
            full_cph = CoxPHFitter(penalizer=0.1).fit(full_df, duration_col="overall_survival", event_col="status")
            lr_stat = 2 * (full_cph.log_likelihood_ - clin_cph.log_likelihood_)
            lr_p = chi2.sf(lr_stat, df=1)
            print(f"\nLLM label added to clinical model: HR={np.exp(full_cph.params_['llm_high_risk']):.3f}, "
                  f"likelihood-ratio p={lr_p:.4f}")
        except Exception as e:
            print(f"Incremental value test failed: {e}")

    print(f"\nNote: {len(rows)} predictors compared on n={len(df)}; p-values are uncorrected for multiple comparisons, "
          "and the clinical C-index is in-sample (optimistic).\n")

def report_sanity_check_status(model):
    """Remind the reader if this model has not passed the concordant sanity check."""
    path = os.path.join(EVALUATION_DIR, "concordant_check_summary.csv")
    if not os.path.exists(path):
        print("[NOTE] Concordant sanity check has not been run yet (python src/concordant_check.py --model NAME).\n")
        return
    summary = pd.read_csv(path)
    row = summary[(summary["model"] == model) & (summary["ablation"] == "full")]
    if row.empty:
        print(f"[NOTE] No concordant sanity check recorded for '{model}'.\n")
    elif not bool(row.iloc[0]["passed"]):
        print(f"[WARNING] '{model}' did NOT pass the concordant sanity check "
              f"(agreement {row.iloc[0]['agreement']}). Interpret these results with caution.\n")

def run_survival_analysis(model):
    discordant_file = generation_path("discordant", model)
    concordant_file = generation_path("concordant", model)
    print(f"Survival analysis for model '{model}'")
    report_sanity_check_status(model)

    if not os.path.exists(discordant_file) or not os.path.exists(concordant_file):
        print("Required generation outputs not found. Please run the generation pipeline first.")
        return

    df_discordant = pd.read_csv(discordant_file)
    df_concordant = pd.read_csv(concordant_file)

    print("--- 1. Entropy Baseline Calibration ---")
    u_stat, p_val = mannwhitneyu(df_concordant['model_entropy_score'], df_discordant['model_entropy_score'])
    print(f"Mean Entropy (Concordant): {df_concordant['model_entropy_score'].mean():.4f}")
    print(f"Mean Entropy (Discordant): {df_discordant['model_entropy_score'].mean():.4f}")
    print(f"Mann-Whitney U p-value: {p_val:.4f}\n")

    print("--- 2. Kaplan-Meier Survival Analysis ---")
    # Clean and convert the target clinical columns to numeric
    df_discordant['overall_survival'] = pd.to_numeric(df_discordant['overall_survival'], errors='coerce')
    df_discordant['status'] = pd.to_numeric(df_discordant['status'], errors='coerce')
    
    # Drop rows missing survival time, vital status, or the LLM's final risk classification
    df_clean = df_discordant.dropna(subset=['overall_survival', 'status', 'final_risk_class'])
    df_clean = df_clean[df_clean['final_risk_class'].isin(['High Risk', 'Low Risk'])]

    high_risk = df_clean[df_clean['final_risk_class'] == 'High Risk']
    low_risk = df_clean[df_clean['final_risk_class'] == 'Low Risk']

    if not high_risk.empty and not low_risk.empty:
        # One fitter per group; reusing a single fitter would overwrite the first curve
        kmf_high = KaplanMeierFitter()
        kmf_low = KaplanMeierFitter()
        kmf_high.fit(high_risk['overall_survival'], high_risk['status'], label='LLM High Risk')
        kmf_low.fit(low_risk['overall_survival'], low_risk['status'], label='LLM Low Risk')

        fig, ax = plt.subplots(figsize=(8, 6))
        kmf_high.plot_survival_function(ax=ax)
        kmf_low.plot_survival_function(ax=ax)
        ax.set_title('Kaplan-Meier Survival by LLM Risk Class (Discordant Cases)')
        ax.set_xlabel('Overall survival (days)')
        ax.set_ylabel('Survival probability')
        fig.tight_layout()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        km_path = os.path.join(OUTPUT_DIR, f"km_llm_risk_class__{model_slug(model)}.png")
        fig.savefig(km_path, dpi=300)
        plt.close(fig)
        print(f"Kaplan-Meier plot saved to {km_path}")

        results = logrank_test(
            high_risk['overall_survival'], low_risk['overall_survival'],
            event_observed_A=high_risk['status'], event_observed_B=low_risk['status']
        )
        print(f"Kaplan-Meier Log-Rank p-value: {results.p_value:.4f}\n")
    else:
        print("Not enough survival data to compute Kaplan-Meier curves.\n")

    print("--- 3. Cox Proportional Hazards Regression ---")
    # Include patient age alongside the LLM risk classification
    cox_df = df_clean[['overall_survival', 'status', 'years_to_birth']].copy()
    cox_df['llm_risk_binary'] = df_clean['final_risk_class'].apply(lambda x: 1 if x == 'High Risk' else 0)
    
    cox_df = cox_df.dropna()

    if len(cox_df) > 5:
        cph = CoxPHFitter()
        try:
            cph.fit(cox_df, duration_col='overall_survival', event_col='status')
            cph.print_summary()
        except Exception as e:
            print(f"Cox Regression failed (possibly due to collinearity or sparse data): {e}")
    else:
        print("Insufficient data for Cox regression.")

    run_baseline_comparison(df_clean, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Generation model whose outputs to analyse (default: config pipeline.model_name).")
    args = parser.parse_args()
    with open("config.yml", "r") as file:
        default_model = yaml.safe_load(file)["pipeline"]["model_name"]
    run_survival_analysis(args.model or default_model)
