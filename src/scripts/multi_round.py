import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict
import statistics
from tabulate import tabulate
from scipy import stats
import json
from enum import Enum


# Define enums to match the ones in your src.core.models
class Side(Enum):
    PROPOSITION = "proposition"
    OPPOSITION = "opposition"


class SpeechType(Enum):
    OPENING = "opening"
    REBUTTAL_1 = "rebuttal_1"
    REBUTTAL_2 = "rebuttal_2"
    CLOSING = "closing"


# Create classes to match your src.core.models structure
class DebatorBet:
    def __init__(self, side, speech_type, amount, thoughts):
        self.side = Side(side) if isinstance(side, str) else side
        self.speech_type = SpeechType(speech_type) if isinstance(speech_type, str) else speech_type
        self.amount = amount
        self.thoughts = thoughts


class Motion:
    def __init__(self, topic_description):
        self.topic_description = topic_description


class DebateTotal:
    @classmethod
    def load_from_json(cls, file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)

        debate = cls()

        # Extract basic information
        if "motion" in data:
            debate.motion = Motion(data["motion"]["topic_description"])
        else:
            debate.motion = Motion("Unknown topic")

        debate.proposition_model = data.get("proposition_model", "Unknown")
        debate.opposition_model = data.get("opposition_model", "Unknown")

        # Extract bets
        debate.debator_bets = []
        if "debator_bets" in data:
            for bet_data in data["debator_bets"]:
                side = bet_data.get("side")
                speech_type = bet_data.get("speech_type")
                amount = bet_data.get("amount", 0)
                thoughts = bet_data.get("thoughts", "")

                bet = DebatorBet(side, speech_type, amount, thoughts)
                debate.debator_bets.append(bet)

        return debate


def load_debate_totals(directory_path: str) -> List[DebateTotal]:
    """
    Load all JSON files in the directory as DebateTotal objects.
    """
    debate_totals = []
    directory = Path(directory_path)
    for file_path in directory.glob("*.json"):
        try:
            debate_total = DebateTotal.load_from_json(file_path)
            debate_totals.append(debate_total)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return debate_totals


def extract_model_name(full_model_path: str) -> str:
    """
    Extract a shorter model name from the full model path.
    """
    # Handle qwen/qwq-32b:free format or other formats
    if '/' in full_model_path:
        parts = full_model_path.split('/')
        return parts[-1]
    return full_model_path


def create_multi_round_bet_dataset():
    """
    Process multi_round_experiments debate files and create a dataset of all non-zero bets.
    """
    base_dir = "experiments"
    exp_dir = "multi_round_experiments"
    path = os.path.join(base_dir, exp_dir)

    all_bets = []

    debate_totals = load_debate_totals(path)
    print(f"Processing {len(debate_totals)} debates from multi_round_experiments")

    for debate in debate_totals:
        # Extract base information from the debate
        prop_model = extract_model_name(debate.proposition_model)
        opp_model = extract_model_name(debate.opposition_model)
        topic = debate.motion.topic_description

        # Generate debate ID
        topic_hash = hash(topic) % 10000
        debate_id = f"multi_round_{prop_model}_vs_{opp_model}_{topic_hash}"

        # Process all bets from this debate
        if debate.debator_bets:
            for bet in debate.debator_bets:
                # Skip bets with amount == 0
                if bet.amount == 0:
                    continue

                # Determine the betting model based on the side
                betting_model = prop_model if bet.side == Side.PROPOSITION else opp_model

                # Extract bet details
                bet_data = {
                    "debate_id": debate_id,
                    "experiment_type": "multi_round",
                    "betting_model": betting_model,
                    "proposition_model": prop_model,
                    "opposition_model": opp_model,
                    "side": bet.side.value,
                    "speech_type": bet.speech_type.value,
                    "bet_amount": bet.amount,
                    "topic": topic,
                    "bet_reasoning": bet.thoughts
                }

                all_bets.append(bet_data)

    # Convert to DataFrame and save to CSV
    if all_bets:
        bets_df = pd.DataFrame(all_bets)
        print(f"Total non-zero bets extracted: {len(bets_df)}")

        # Basic statistics
        print(f"Unique debates: {bets_df['debate_id'].nunique()}")
        print(f"Unique models: {bets_df['betting_model'].nunique()}")
        print(f"Speech types: {bets_df['speech_type'].value_counts().to_dict()}")

        # Save to CSV
        bets_df.to_csv("multi_round_bets.csv", index=False)
        print("Data saved to multi_round_bets.csv")

        return bets_df
    else:
        print("No non-zero bets found in the multi_round_experiments data!")
        return None


def get_initial_confidences(debate_total: DebateTotal) -> List[Dict]:
    """
    Extract initial confidence (first bet) for each model in a debate.
    Returns a list of dictionaries containing model name and confidence value.
    This preserves all bets including multiple bets from the same model.
    """
    confidences = []

    if debate_total.debator_bets:
        # Filter bets for the first speech type (OPENING)
        first_bets = [bet for bet in debate_total.debator_bets
                      if bet.speech_type == SpeechType.OPENING and bet.amount > 0]  # Ignore zero bets

        # Associate bets with their respective models
        if first_bets:
            for bet in first_bets:
                model = debate_total.proposition_model if bet.side == Side.PROPOSITION else debate_total.opposition_model
                confidences.append({
                    "model": model,
                    "side": bet.side.value,
                    "confidence": bet.amount
                })

    return confidences


def analyze_opening_confidences():
    """
    Analyze initial confidence means and standard deviations by model for multi_round_experiments.
    """
    base_dir = "experiments"
    exp_dir = "multi_round_experiments"
    path = os.path.join(base_dir, exp_dir)
    debate_totals = load_debate_totals(path)

    all_confidences = []

    # Extract all confidence values
    for debate in debate_totals:
        model_confidences = get_initial_confidences(debate)
        for item in model_confidences:
            all_confidences.append({
                "Experiment": "multi_round",
                "Model": item["model"],
                "Side": item["side"],
                "Confidence": item["confidence"]
            })

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_confidences)

    # Get count of unique models
    unique_models = df["Model"].nunique() if not df.empty else 0
    print(f"Found {unique_models} unique models in multi_round experiments")

    # Group data by model to calculate statistics
    stats_data = []
    if not df.empty:
        grouped = df.groupby(["Model"])

        # Calculate statistics for each group
        for model, group in grouped:
            confidences = group["Confidence"].tolist()
            if confidences:  # Check if list is not empty
                mean_confidence = statistics.mean(confidences)
                std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
                sample_size = len(confidences)

                stats_data.append({
                    "Model": model,
                    "Experiment": "multi_round",
                    "Mean Initial Confidence": mean_confidence,
                    "SD Initial Confidence": std_confidence,
                    "Sample Size": sample_size
                })

    # Convert statistics to DataFrame
    stats_df = pd.DataFrame(stats_data)

    # Calculate overall statistics
    if not df.empty:
        overall_mean = df["Confidence"].mean()
        overall_sd = df["Confidence"].std()
        overall_sample = len(df)

        print("\n== Summary of Initial Confidence in Multi-Round Experiments ==")
        print(f"Overall: {overall_mean:.2f} ± {overall_sd:.2f} (n={overall_sample})")
    else:
        print("\nNo confidence data found.")

    # Save results to CSV
    if not stats_df.empty:
        stats_df.to_csv("multi_round_initial_confidence_stats.csv", index=False)

    return stats_df, df


def run_statistical_tests(all_confidences):
    """
    Run hypothesis tests on initial confidence data for multi_round_experiments.
    """
    if all_confidences.empty:
        print("No confidence data to run statistical tests.")
        return None

    print("\n===== PRIMARY OVERCONFIDENCE TESTS =====")
    print("One-Sample tests against null hypothesis of 50% confidence")

    # Get raw confidence values
    data = all_confidences["Confidence"].values
    mean_val = np.mean(data)

    # Parametric test (t-test)
    t_stat, p_value_t = stats.ttest_1samp(data, 50, alternative='greater')

    # Non-parametric test (Wilcoxon)
    w_stat, p_value_w = stats.wilcoxon(data - 50, alternative='greater')

    overconfidence_result = {
        "Experiment": "multi_round",
        "N": len(data),
        "Mean": mean_val,
        "t-statistic": t_stat,
        "p-value (t-test)": p_value_t,
        "Wilcoxon statistic": w_stat,
        "p-value (Wilcoxon)": p_value_w,
        "Significant (t-test)": p_value_t < 0.05,
        "Significant (Wilcoxon)": p_value_w < 0.05
    }

    # Display results
    print(pd.DataFrame([overconfidence_result]))

    # Per-model overconfidence tests
    print("\n===== GRANULAR PER-MODEL OVERCONFIDENCE =====")
    print("One-Sample tests against null hypothesis of 50% confidence for each model")

    per_model_results = []
    models = all_confidences["Model"].unique()

    for model in models:
        model_data = all_confidences[all_confidences["Model"] == model]["Confidence"].values

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
    if not per_model_df.empty:
        print(f"Total per-model tests: {len(per_model_df)}")
        print(f"Significant results (t-test): {per_model_df['Significant (t-test)'].sum()} / {len(per_model_df)}")

        # Save detailed per-model results to CSV
        per_model_df.to_csv("multi_round_per_model_overconfidence_tests.csv", index=False)
        print("Detailed per-model results saved to 'multi_round_per_model_overconfidence_tests.csv'")

    return {
        "overconfidence_result": pd.DataFrame([overconfidence_result]),
        "per_model_results": per_model_df
    }


def analyze_confidence_escalation(df):
    """
    Analyze how confidence changes across speech types.
    """
    if df is None or df.empty:
        print("No bet data to analyze confidence escalation.")
        return None

    print("\n===== ANALYZING CONFIDENCE ESCALATION =====")

    # Function to format mean with standard deviation and sample size
    def format_stat(values):
        values = [v for v in values if not np.isnan(v)]  # Filter out NaN values
        if len(values) == 0:
            return "N/A"
        mean = np.mean(values)
        std = np.std(values)
        count = len(values)
        return f"{mean:.2f} (±{std:.2f}, N={count})"

    # Function to perform paired statistical tests and return formatted results
    def paired_test(earlier_values, later_values, alpha=0.05):
        # Create pairs without NaN values
        pairs = [(e, l) for e, l in zip(earlier_values, later_values) if not np.isnan(e) and not np.isnan(l)]
        if not pairs:
            return "N/A"

        earlier = [p[0] for p in pairs]
        later = [p[1] for p in pairs]

        if len(earlier) != len(later) or len(earlier) == 0:
            return "N/A"

        # Check if data is paired correctly
        if len(earlier) < 2:
            return f"N={len(earlier)}, insufficient data"

        # Calculate mean difference
        mean_diff = np.mean(np.array(later) - np.array(earlier))

        # Perform paired t-test (one-sided, testing if later > earlier)
        t_stat, p_value = stats.ttest_rel(earlier, later)

        # Convert to one-sided p-value if mean difference is positive
        if mean_diff > 0:
            p_value = p_value / 2
        else:
            # If mean difference is negative, one-sided p-value is 1 - p_value/2
            p_value = 1 - (p_value / 2)

        # Determine significance
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < alpha else ""

        return f"Δ={mean_diff:.2f}, p={p_value:.4f}{significance}"

    # Map the speech types to their respective columns for pivoting
    speech_type_mapping = {
        "opening": "opening_bet",
        "rebuttal_1": "rebuttal_bet",
        "rebuttal_2": "rebuttal_bet",  # Combine rebuttal_1 and rebuttal_2 for now
        "closing": "closing_bet"
    }

    # Reshape the data so each row is a single debate instance per model regardless of side
    debate_data = []

    # Group by debate_id, betting_model, and side to get all speech types
    for (debate_id, model, side), group in df.groupby(['debate_id', 'betting_model', 'side']):
        # Extract data for each speech type if available
        opening_data = group[group['speech_type'] == 'opening']['bet_amount'].values
        rebuttal_data = group[(group['speech_type'] == 'rebuttal_1') |
                             (group['speech_type'] == 'rebuttal_2')]['bet_amount'].values
        closing_data = group[group['speech_type'] == 'closing']['bet_amount'].values

        # For each individual speech set
        for i in range(max(len(opening_data), len(rebuttal_data), len(closing_data))):
            debate_data.append({
                'debate_id': debate_id,
                'betting_model': model,
                'experiment_type': 'multi_round',
                'side': side,
                'topic': group['topic'].iloc[0],
                'opening_bet': opening_data[i] if i < len(opening_data) else np.nan,
                'rebuttal_bet': rebuttal_data[i] if i < len(rebuttal_data) else np.nan,
                'closing_bet': closing_data[i] if i < len(closing_data) else np.nan
            })

    # Create a DataFrame with one row per debate-model-side instance
    pivot_df = pd.DataFrame(debate_data)

    if pivot_df.empty:
        print("No escalation data after pivoting.")
        return None

    pivot_df.to_csv("multi_round_pivoted_bets.csv", index=False)

    # Get unique models
    models = sorted(pivot_df['betting_model'].unique())
    print(f"\nFound {len(models)} models in the multi_round pivoted data")

    # Analyze escalation for multi_round experiments
    print("\n\n=== MULTI_ROUND EXPERIMENTS: CONFIDENCE ESCALATION ===")

    # Prepare data for tabulate
    table_data = []

    # For collecting overall paired data
    all_opening_bets = []
    all_rebuttal_bets = []
    all_closing_bets = []

    for model in models:
        # Filter data for this model
        model_data = pivot_df[pivot_df['betting_model'] == model]

        if not model_data.empty:
            # Get all bet values for each speech type
            opening_bets = model_data['opening_bet'].tolist()
            rebuttal_bets = model_data['rebuttal_bet'].tolist()
            closing_bets = model_data['closing_bet'].tolist()

            # Add to overall stats
            all_opening_bets.extend(opening_bets)
            all_rebuttal_bets.extend(rebuttal_bets)
            all_closing_bets.extend(closing_bets)

            # Format descriptive stats
            opening_stats = format_stat(opening_bets)
            rebuttal_stats = format_stat(rebuttal_bets)
            closing_stats = format_stat(closing_bets)

            # Perform paired tests
            open_vs_rebuttal = paired_test(opening_bets, rebuttal_bets)
            rebuttal_vs_closing = paired_test(rebuttal_bets, closing_bets)
            open_vs_closing = paired_test(opening_bets, closing_bets)

            # Add row to table
            table_data.append([
                model,
                opening_stats,
                rebuttal_stats,
                closing_stats,
                open_vs_rebuttal,
                rebuttal_vs_closing,
                open_vs_closing
            ])
        else:
            # No data for this model
            table_data.append([model, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])

    # Add overall row with paired tests across all models
    overall_opening_stats = format_stat(all_opening_bets)
    overall_rebuttal_stats = format_stat(all_rebuttal_bets)
    overall_closing_stats = format_stat(all_closing_bets)

    # Perform overall paired tests
    overall_open_vs_rebuttal = paired_test(all_opening_bets, all_rebuttal_bets)
    overall_rebuttal_vs_closing = paired_test(all_rebuttal_bets, all_closing_bets)
    overall_open_vs_closing = paired_test(all_opening_bets, all_closing_bets)

    table_data.append([
        "OVERALL",
        overall_opening_stats,
        overall_rebuttal_stats,
        overall_closing_stats,
        overall_open_vs_rebuttal,
        overall_rebuttal_vs_closing,
        overall_open_vs_closing
    ])

    # Print table with headers
    headers = [
        "Model",
        "Opening Bet",
        "Rebuttal Bet",
        "Closing Bet",
        "Open→Rebuttal",
        "Rebuttal→Closing",
        "Open→Closing"
    ]
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Save escalation results to CSV
    escalation_df = pd.DataFrame(table_data, columns=headers)
    escalation_df.to_csv("multi_round_confidence_escalation.csv", index=False)
    print("Confidence escalation data saved to 'multi_round_confidence_escalation.csv'")

    # Count models with significant escalation
    print("\n\n=== SUMMARY OF MODELS WITH SIGNIFICANT ESCALATION ===")

    # Initialize counters
    sig_open_to_rebuttal = 0
    sig_rebuttal_to_closing = 0
    sig_open_to_closing = 0
    total_models_with_data = 0

    for model in models:
        model_data = pivot_df[pivot_df['betting_model'] == model]

        if not model_data.empty:
            # Get bet values
            opening_bets = model_data['opening_bet'].tolist()
            rebuttal_bets = model_data['rebuttal_bet'].tolist()
            closing_bets = model_data['closing_bet'].tolist()

            # Only count if we have paired data
            pairs_or_rebuttal = [(o, r) for o, r in zip(opening_bets, rebuttal_bets) if not np.isnan(o) and not np.isnan(r)]
            pairs_rc = [(r, c) for r, c in zip(rebuttal_bets, closing_bets) if not np.isnan(r) and not np.isnan(c)]
            pairs_oc = [(o, c) for o, c in zip(opening_bets, closing_bets) if not np.isnan(o) and not np.isnan(c)]

            if len(pairs_or_rebuttal) >= 2 or len(pairs_rc) >= 2 or len(pairs_oc) >= 2:
                total_models_with_data += 1

                # Check for significance in each transition
                if paired_test(opening_bets, rebuttal_bets).find("*") != -1:
                    sig_open_to_rebuttal += 1

                if paired_test(rebuttal_bets, closing_bets).find("*") != -1:
                    sig_rebuttal_to_closing += 1

                if paired_test(opening_bets, closing_bets).find("*") != -1:
                    sig_open_to_closing += 1

    # Print summary
    summary_data = [
        [
            "multi_round",
            f"{sig_open_to_rebuttal}/{total_models_with_data}" if total_models_with_data > 0 else "N/A",
            f"{sig_rebuttal_to_closing}/{total_models_with_data}" if total_models_with_data > 0 else "N/A",
            f"{sig_open_to_closing}/{total_models_with_data}" if total_models_with_data > 0 else "N/A"
        ]
    ]

    # Print summary table
    summary_headers = [
        "Experiment Type",
        "Models with Sig. Open→Rebuttal",
        "Models with Sig. Rebuttal→Closing",
        "Models with Sig. Open→Closing"
    ]
    print(tabulate(summary_data, headers=summary_headers, tablefmt="grid"))

    # Save summary to CSV
    summary_df = pd.DataFrame(summary_data, columns=summary_headers)
    summary_df.to_csv("multi_round_escalation_summary.csv", index=False)

    return pivot_df


if __name__ == "__main__":
    # Step 1: Create the dataset from multi_round_experiments, filtering out zero bets
    bets_df = create_multi_round_bet_dataset()

    if bets_df is not None and not bets_df.empty:
        print("\nSample of the extracted data:")
        print(bets_df.head())

        # Additional stats about models
        print("\nModel participation:")
        print(bets_df['betting_model'].value_counts())

        # Step 2: Analyze initial confidences
        stats_df, confidence_df = analyze_opening_confidences()

        # Step 3: Run statistical tests on initial confidences
        if not confidence_df.empty:
            test_results = run_statistical_tests(confidence_df)

        # Step 4: Analyze confidence escalation
        pivot_df = analyze_confidence_escalation(bets_df)

        print("\nAnalysis complete. All results saved to CSV files.")
    else:
        print("No data to analyze. Check if there are any non-zero bets in the multi_round_experiments directory.")
