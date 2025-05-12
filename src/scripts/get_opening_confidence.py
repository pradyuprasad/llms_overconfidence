import os
from typing import Dict, List
from pathlib import Path
import statistics
import pandas as pd

from src.core.models import DebateTotal, Side, SpeechType


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


def get_experiment_names() -> Dict[str, str]:
    """
    Returns a mapping from directory names to formatted experiment names.
    """
    return {
        "private_bet_experiments_diff_models": "Cross-model",
        "private_self_bet": "Debate against same model",
        "private_self_bet_anchored": "Debate against same model informed with 50% probability",
        "public_bets": "Public Bets"
    }


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
                      if bet.speech_type == SpeechType.OPENING]

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


def analyze_confidences_with_stats():
    """
    Analyze initial confidence means and standard deviations by model for each experiment type.
    Include sample size in the output. Add overall average row for each experiment.
    """
    base_dir = "experiments"  # Updated base directory path
    experiment_dirs = {
        "private_bet_experiments_diff_models",
        "private_self_bet",
        "private_self_bet_anchored",
        "public_bets"
    }

    experiment_names = get_experiment_names()
    all_confidences = []

    # Process each experiment directory
    for exp_dir in experiment_dirs:
        path = os.path.join(base_dir, exp_dir)
        debate_totals = load_debate_totals(path)
        experiment_name = experiment_names[exp_dir]

        # Extract all confidence values
        for debate in debate_totals:
            model_confidences = get_initial_confidences(debate)
            for item in model_confidences:
                all_confidences.append({
                    "Experiment": experiment_name,
                    "Model": item["model"],
                    "Side": item["side"],
                    "Confidence": item["confidence"]
                })

    # Convert to DataFrame for analysis
    df = pd.DataFrame(all_confidences)

    # Get count of unique models
    unique_models = df["Model"].nunique()
    print(f"Found {unique_models} unique models across all experiments")

    # Group data by model and experiment to calculate statistics
    grouped = df.groupby(["Model", "Experiment"])

    # Calculate statistics for each group
    stats_data = []
    for (model, experiment), group in grouped:
        confidences = group["Confidence"].tolist()
        mean_confidence = statistics.mean(confidences)
        std_confidence = statistics.stdev(confidences) if len(confidences) > 1 else 0
        sample_size = len(confidences)

        stats_data.append({
            "Model": model,
            "Experiment": experiment,
            "Mean Initial Confidence": mean_confidence,
            "SD Initial Confidence": std_confidence,
            "Sample Size": sample_size
        })

    # Convert statistics to DataFrame
    stats_df = pd.DataFrame(stats_data)

    # Create pivot tables for mean, SD, and sample size
    mean_pivot = stats_df.pivot_table(
        index="Model",
        columns="Experiment",
        values="Mean Initial Confidence"
    )

    sd_pivot = stats_df.pivot_table(
        index="Model",
        columns="Experiment",
        values="SD Initial Confidence"
    )

    sample_pivot = stats_df.pivot_table(
        index="Model",
        columns="Experiment",
        values="Sample Size"
    )

    # Create combined mean±SD (n=sample) table
    combined_pivot = pd.DataFrame(index=mean_pivot.index, columns=mean_pivot.columns)

    for model in mean_pivot.index:
        for exp in mean_pivot.columns:
            mean_val = mean_pivot.loc[model, exp]
            sd_val = sd_pivot.loc[model, exp]
            sample_size = int(sample_pivot.loc[model, exp])
            combined_pivot.loc[model, exp] = f"{mean_val:.2f} ± {sd_val:.2f} (n={sample_size})"

    # Calculate overall statistics for each experiment
    overall_stats = {}

    for experiment in df["Experiment"].unique():
        exp_data = df[df["Experiment"] == experiment]["Confidence"]
        overall_mean = exp_data.mean()
        overall_sd = exp_data.std()
        overall_sample = len(exp_data)
        overall_stats[experiment] = f"{overall_mean:.2f} ± {overall_sd:.2f} (n={overall_sample})"

    # Add overall row to the combined pivot
    combined_pivot.loc["OVERALL AVERAGE"] = pd.Series(overall_stats)

    print("\n== Summary (Mean ± SD Initial Confidence by Model and Experiment) ==")
    print(combined_pivot)

    return stats_df, combined_pivot, df


if __name__ == "__main__":
    stats_df, combined_pivot = analyze_confidences_with_stats()

    # Save results to CSV if desired
    stats_df.to_csv("initial_confidence_detailed_stats.csv", index=False)
    combined_pivot.to_csv("initial_confidence_mean_sd_sample_summary.csv")

    print("\nResults saved to CSV files.")
