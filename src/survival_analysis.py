import os
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
from scipy.stats import mannwhitneyu

def run_survival_analysis():
    discordant_file = "data/generation_outputs/discordant_results.csv"
    concordant_file = "data/generation_outputs/concordant_results.csv"

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

    high_risk = df_clean[df_clean['final_risk_class'] == 'High Risk']
    low_risk = df_clean[df_clean['final_risk_class'] == 'Low Risk']

    kmf = KaplanMeierFitter()
    if not high_risk.empty and not low_risk.empty:
        # Fit the curves using 'status' as the event observed
        kmf.fit(high_risk['overall_survival'], high_risk['status'], label='LLM High Risk')
        kmf.fit(low_risk['overall_survival'], low_risk['status'], label='LLM Low Risk')

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

if __name__ == "__main__":
    run_survival_analysis()