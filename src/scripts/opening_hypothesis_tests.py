import pandas as pd
import numpy as np
from scipy import stats

# Import your analysis function
from src.scripts.get_opening_confidence import analyze_confidences_with_stats

def run_statistical_tests():
    """
    Run hypothesis tests on initial confidence data as specified.
    """
    # Get the data using your existing analysis function
    stats_df, combined_pivot, all_confidences = analyze_confidences_with_stats()

    # Create a mapping of experiment names to shorter labels
    experiment_mapping = {
    "Cross-model": "cross_model",
    "Debate against same model": "standard_self",
    "Debate against same model informed with 50% probability": "informed_self",
    "Public Bets": "public_bets",
    "Redteam": "redteam"  # Add this line
}



    # Extract raw data for each experiment from stats_df
    raw_data = {}
    model_means = {}

    # Get all confidence values for each experiment
    for exp_name, exp_short in experiment_mapping.items():
        exp_data = stats_df[stats_df["Experiment"] == exp_name]

        # Store all raw confidence values
        raw_confidence_values = []
        for _, row in exp_data.iterrows():
            model = row["Model"]
            mean_conf = row["Mean Initial Confidence"]
            sample_size = row["Sample Size"]

            # Calculate model means for each experiment
            if exp_name not in model_means:
                model_means[exp_name] = {}
            model_means[exp_name][model] = mean_conf

        # Get all raw confidence values for this experiment type
        all_data = stats_df[stats_df["Experiment"] == exp_name]
        all_models = all_data["Model"].unique()

        # Extract raw data from the original DataFrame
        from_df = pd.DataFrame(all_confidences)  # You'll need to access this from the original function
        raw_data[exp_short] = from_df[from_df["Experiment"] == exp_name]["Confidence"].values

    # 1. PRIMARY OVERCONFIDENCE: One-Sample t-tests against 50
    print("\n===== PRIMARY OVERCONFIDENCE TESTS =====")
    print("One-Sample tests against null hypothesis of 50% confidence")

    overconfidence_results = []
    for exp_name, exp_short in experiment_mapping.items():
        if exp_short in ["cross_model", "standard_self", "public_bets", "redteam"]:
            data = raw_data[exp_short]
            mean_val = np.mean(data)

            # Parametric test (t-test)
            t_stat, p_value_t = stats.ttest_1samp(data, 50, alternative='greater')

            # Non-parametric test (Wilcoxon)
            w_stat, p_value_w = stats.wilcoxon(data - 50, alternative='greater')

            overconfidence_results.append({
                "Experiment": exp_name,
                "N": len(data),
                "Mean": mean_val,
                "t-statistic": t_stat,
                "p-value (t-test)": p_value_t,
                "Wilcoxon statistic": w_stat,
                "p-value (Wilcoxon)": p_value_w,
                "Significant (t-test)": p_value_t < 0.05,
                "Significant (Wilcoxon)": p_value_w < 0.05
            })

    # Display results as a table
    print(pd.DataFrame(overconfidence_results))

    # 2. EFFECT OF INSTRUCTION: Paired Sample t-test between Standard Self and Informed Self
    print("\n===== EFFECT OF INSTRUCTION =====")
    print("Paired Sample tests between Standard Self and Informed Self")

    # Extract paired model means
    standard_means = []
    informed_means = []

    models = list(model_means["Debate against same model"].keys())
    for model in models:
        if model in model_means["Debate against same model informed with 50% probability"]:
            standard_means.append(model_means["Debate against same model"][model])
            informed_means.append(model_means["Debate against same model informed with 50% probability"][model])

    # Parametric test (paired t-test)
    t_stat, p_value_t = stats.ttest_rel(standard_means, informed_means, alternative='greater')

    # Non-parametric test (Wilcoxon signed-rank)
    w_stat, p_value_w = stats.wilcoxon(np.array(standard_means) - np.array(informed_means), alternative='greater')

    print("Standard Self vs Informed Self (H1: Standard > Informed):")
    print(f"  Mean difference: {np.mean(standard_means) - np.mean(informed_means):.2f}")
    print(f"  Paired t-test: t={t_stat:.2f}, p={p_value_t:.4f}, significant: {p_value_t < 0.05}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.2f}, p={p_value_w:.4f}, significant: {p_value_w < 0.05}")

    # 3. EFFECT OF PUBLIC REPORTING VS STANDARD SELF
    print("\n===== EFFECT OF PUBLIC REPORTING VS STANDARD SELF =====")
    print("Paired Sample tests between Standard Self and Public Bets")

    # Extract paired model means
    standard_means = []
    public_means = []

    for model in models:
        if model in model_means["Public Bets"]:
            standard_means.append(model_means["Debate against same model"][model])
            public_means.append(model_means["Public Bets"][model])

    # Parametric test (paired t-test, two-sided)
    t_stat, p_value_t = stats.ttest_rel(standard_means, public_means)

    # Non-parametric test (Wilcoxon signed-rank, two-sided)
    w_stat, p_value_w = stats.wilcoxon(np.array(standard_means) - np.array(public_means))

    print("Standard Self vs Public Bets (H1: Standard ≠ Public):")
    print(f"  Mean difference: {np.mean(standard_means) - np.mean(public_means):.2f}")
    print(f"  Paired t-test: t={t_stat:.2f}, p={p_value_t:.4f}, significant: {p_value_t < 0.05}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.2f}, p={p_value_w:.4f}, significant: {p_value_w < 0.05}")

    # 4. EFFECT OF PUBLIC REPORTING VS INFORMED SELF
    print("\n===== EFFECT OF PUBLIC REPORTING VS INFORMED SELF =====")
    print("Paired Sample tests between Informed Self and Public Bets")

    # Extract paired model means
    informed_means = []
    public_means = []

    for model in models:
        if model in model_means["Public Bets"] and model in model_means["Debate against same model informed with 50% probability"]:
            informed_means.append(model_means["Debate against same model informed with 50% probability"][model])
            public_means.append(model_means["Public Bets"][model])

    # Parametric test (paired t-test, two-sided)
    t_stat, p_value_t = stats.ttest_rel(informed_means, public_means)

    # Non-parametric test (Wilcoxon signed-rank, two-sided)
    w_stat, p_value_w = stats.wilcoxon(np.array(informed_means) - np.array(public_means))

    print("Informed Self vs Public Bets (H1: Informed ≠ Public):")
    print(f"  Mean difference: {np.mean(informed_means) - np.mean(public_means):.2f}")
    print(f"  Paired t-test: t={t_stat:.2f}, p={p_value_t:.4f}, significant: {p_value_t < 0.05}")
    print(f"  Wilcoxon signed-rank: W={w_stat:.2f}, p={p_value_w:.4f}, significant: {p_value_w < 0.05}")

    # 5. GRANULAR PER-MODEL OVERCONFIDENCE
    print("\n===== GRANULAR PER-MODEL OVERCONFIDENCE =====")
    print("One-Sample tests against null hypothesis of 50% confidence for each model in each configuration")

    per_model_results = []

    # Get unique models
    all_models = set()
    for exp_data in model_means.values():
        all_models.update(exp_data.keys())

    # For each model in each experiment type, run one-sample test
    for exp_name, exp_short in experiment_mapping.items():
        for model in all_models:
            if model in model_means[exp_name]:
                # Get raw data for this model in this experiment
                model_data = from_df[(from_df["Experiment"] == exp_name) & (from_df["Model"] == model)]["Confidence"].values

                if len(model_data) > 0:
                    mean_val = np.mean(model_data)

                    # Only run if we have enough data points
                    if len(model_data) >= 3:  # Minimal requirement for statistical test
                        # Parametric test (t-test)
                        t_stat, p_value_t = stats.ttest_1samp(model_data, 50, alternative='greater')

                        # Non-parametric test (Wilcoxon) if enough data
                        if len(model_data) >= 6:  # Wilcoxon needs more data points
                            try:
                                w_stat, p_value_w = stats.wilcoxon(model_data - 50, alternative='greater')
                            except:
                                w_stat, p_value_w = np.nan, np.nan
                        else:
                            w_stat, p_value_w = np.nan, np.nan

                        per_model_results.append({
                            "Experiment": exp_name,
                            "Model": model,
                            "N": len(model_data),
                            "Mean": mean_val,
                            "p-value (t-test)": p_value_t,
                            "Significant (t-test)": p_value_t < 0.05,
                            "p-value (Wilcoxon)": p_value_w,
                            "Significant (Wilcoxon)": p_value_w < 0.05 if not np.isnan(p_value_w) else np.nan
                        })

    # Create a DataFrame for per-model results
    per_model_df = pd.DataFrame(per_model_results)

    # Summarize results
    print(f"Total per-model tests: {len(per_model_df)}")
    print(f"Significant results (t-test): {per_model_df['Significant (t-test)'].sum()} / {len(per_model_df)}")

    # Save detailed per-model results to CSV
    per_model_df.to_csv("per_model_overconfidence_tests.csv", index=False)
    print("Detailed per-model results saved to 'per_model_overconfidence_tests.csv'")

    return {
        "overconfidence_results": pd.DataFrame(overconfidence_results),
        "per_model_results": per_model_df
    }

if __name__ == "__main__":
    results = run_statistical_tests()
