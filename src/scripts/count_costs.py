import os
import json
from pathlib import Path
from typing import Dict, Any, Set

from src.core.models import DebateTotal


def load_debate_totals(directory_path: str) -> list:
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


def analyze_token_usage_and_cost():
    """
    Analyze token usage across all debates and calculate costs.
    """
    # Define pricing information (cost per million tokens: [prompt_cost, completion_cost])
    pricing = {
        "openai/o3-mini": [1.1, 4.4],
        "google/gemini-2.0-flash-001": [0.1, 0.4],
        "anthropic/claude-3.7-sonnet": [3, 15],
        "deepseek/deepseek-chat": [0.7, 2.5],
        "qwen/qwq-32b:free": [0, 0],
        "openai/gpt-4o-mini": [0.15, 0.6],
        "google/gemma-3-27b-it": [0.1, 0.2],
        "anthropic/claude-3.5-haiku": [0.8, 4],
        "deepseek/deepseek-r1-distill-qwen-14b:free": [0, 0],
        "qwen/qwen-max": [1.6, 6.4]
    }

    # Base directory containing experiment folders
    base_dir = "experiments"
    experiment_dirs = {
        "private_bet_experiments_diff_models",
        "private_self_bet",
        "private_self_bet_anchored",
        "public_bets",
        "SELF_REDTEAM_DEBATE",
        # Adding the new directory but we'll ignore debates in it
        "multi_round_experiments"
    }

    # Dictionary to track token usage and cost by model
    model_stats = {}

    # Tracking grand totals
    grand_totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "total_cost": 0.0,
        "unknown_cost_models": set()  # Track models without pricing info
    }

    # Process each experiment directory
    total_debates = 0

    for exp_dir in experiment_dirs:
        path = os.path.join(base_dir, exp_dir)

        # Skip if directory doesn't exist
        if not os.path.exists(path):
            print(f"Directory not found: {path}")
            continue

        # Skip processing debates in multi_round_experiments as instructed
        if exp_dir == "multi_round_experiments":
            print(f"Skipping debates in {exp_dir} as instructed")
            continue

        debate_totals = load_debate_totals(path)
        total_debates += len(debate_totals)
        print(f"Processing {len(debate_totals)} debates from {exp_dir}")

        for debate in debate_totals:
            # Process proposition model
            prop_model = debate.proposition_model
            if prop_model not in model_stats:
                model_stats[prop_model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "debate_count": 0
                }

            # Process opposition model
            opp_model = debate.opposition_model
            if opp_model not in model_stats:
                model_stats[opp_model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "debate_count": 0
                }

            # Update debate counts
            model_stats[prop_model]["debate_count"] += 1
            model_stats[opp_model]["debate_count"] += 1

            # Extract token counts from debate token data
            for model, usage in debate.debator_token_counts.model_usages.items():
                if model == prop_model or model == opp_model:
                    if model not in model_stats:
                        model_stats[model] = {
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "debate_count": 0
                        }

                    # Add successful and failed tokens (both matter for cost)
                    prompt_tokens = usage.total_prompt_tokens
                    completion_tokens = usage.total_completion_tokens
                    total_tokens = usage.total_tokens

                    model_stats[model]["prompt_tokens"] += prompt_tokens
                    model_stats[model]["completion_tokens"] += completion_tokens
                    model_stats[model]["total_tokens"] += total_tokens

                    # Add to grand totals
                    grand_totals["prompt_tokens"] += prompt_tokens
                    grand_totals["completion_tokens"] += completion_tokens
                    grand_totals["total_tokens"] += total_tokens

    # Calculate costs for each model
    for model, stats in model_stats.items():
        # Calculate cost based on pricing
        if model in pricing:
            prompt_cost = (stats["prompt_tokens"] / 1_000_000) * pricing[model][0]
            completion_cost = (stats["completion_tokens"] / 1_000_000) * pricing[model][1]
            total_cost = prompt_cost + completion_cost
            stats["prompt_cost"] = round(prompt_cost, 2)
            stats["completion_cost"] = round(completion_cost, 2)
            stats["total_cost"] = round(total_cost, 2)

            # Add to grand total cost
            grand_totals["total_cost"] += total_cost
        else:
            stats["prompt_cost"] = "unknown"
            stats["completion_cost"] = "unknown"
            stats["total_cost"] = "unknown"
            grand_totals["unknown_cost_models"].add(model)

    # Sort models by total tokens (or cost if available)
    sorted_models = sorted(
        model_stats.items(),
        key=lambda x: x[1]["total_tokens"],
        reverse=True
    )

    sorted_model_stats = {model: stats for model, stats in sorted_models}

    # Prepare output with summary
    output = {
        "summary": {
            "total_debates": total_debates,
            "unique_models": len(model_stats),
            "grand_totals": {
                "prompt_tokens": grand_totals["prompt_tokens"],
                "completion_tokens": grand_totals["completion_tokens"],
                "total_tokens": grand_totals["total_tokens"],
                "total_cost_usd": round(grand_totals["total_cost"], 2),
                "unknown_cost_models": list(grand_totals["unknown_cost_models"])
            }
        },
        "model_stats": sorted_model_stats
    }

    # Save to JSON
    with open("model_token_usage_and_cost.json", "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"\nAnalysis complete. Processed {total_debates} debates with {len(model_stats)} unique models.")
    print("Model token usage and cost summary:")
    print("-" * 100)
    print(f"{'Model':<40} {'Total Tokens':<15} {'Cost (USD)':<12} {'Debates':<8}")
    print("-" * 100)

    for model, stats in sorted_models:
        cost_str = f"${stats['total_cost']}" if isinstance(stats['total_cost'], (int, float)) else stats['total_cost']
        print(f"{model:<40} {stats['total_tokens']:<15,} {cost_str:<12} {stats['debate_count']:<8}")

    print("-" * 100)
    print(f"GRAND TOTALS: {grand_totals['total_tokens']:,} tokens, ${grand_totals['total_cost']:.2f}")

    if grand_totals["unknown_cost_models"]:
        print(f"Note: Cost calculation excludes these models with unknown pricing: "
              f"{', '.join(grand_totals['unknown_cost_models'])}")

    print("-" * 100)
    print(f"Results saved to model_token_usage_and_cost.json")

    return output


if __name__ == "__main__":
    analyze_token_usage_and_cost()
