import os
import pandas as pd
from pathlib import Path
from typing import List, Dict

from src.core.models import DebateTotal, Side


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
        "private_bet_experiments_diff_models": "cross_model",
        "private_self_bet": "self_debate",
        "private_self_bet_anchored": "informed_self",
        "public_bets": "public_bets",
        "SELF_REDTEAM_DEBATE": "self_redteam_debate"
    }


def extract_model_name(full_model_path: str) -> str:
    """
    Extract a shorter model name from the full model path.
    """
    # Extract just the model name part (may need to be adjusted based on your naming conventions)
    return full_model_path.split('/')[-1] if '/' in full_model_path else full_model_path


def create_bet_dataset():
    """
    Process all debate files and create a comprehensive dataset of all bets.
    """
    base_dir = "experiments"
    experiment_dirs = {
        "private_bet_experiments_diff_models",
        "private_self_bet",
        "private_self_bet_anchored",
        "public_bets",
        "SELF_REDTEAM_DEBATE"
    }

    experiment_names = get_experiment_names()
    all_bets = []

    # Keep track of debate IDs and counter
    debate_id_counter = {}

    for exp_dir in experiment_dirs:
        path = os.path.join(base_dir, exp_dir)
        formatted_exp_name = experiment_names[exp_dir]
        debate_totals = load_debate_totals(path)

        print(f"Processing {len(debate_totals)} debates from {formatted_exp_name}")

        for debate in debate_totals:
            # Extract base information from the debate
            prop_model = extract_model_name(debate.proposition_model)
            opp_model = extract_model_name(debate.opposition_model)
            topic = debate.motion.topic_description

            # Create base debate ID
            base_debate_id = f"{formatted_exp_name}_{prop_model}_vs_{opp_model}"

            # Add topic hash to make it more unique
            topic_hash = hash(topic) % 10000  # Use a small hash to keep ID readable
            base_debate_id = f"{base_debate_id}_{topic_hash}"

            # Check if we've seen this base ID before and increment counter if needed
            if base_debate_id in debate_id_counter:
                debate_id_counter[base_debate_id] += 1
                debate_id = f"{base_debate_id}_{debate_id_counter[base_debate_id]}"
            else:
                debate_id_counter[base_debate_id] = 0
                debate_id = base_debate_id

            # Process all bets from this debate
            if debate.debator_bets:
                for bet in debate.debator_bets:
                    # Determine the betting model based on the side
                    betting_model = prop_model if bet.side == Side.PROPOSITION else opp_model

                    # Extract bet details
                    bet_data = {
                        "debate_id": debate_id,
                        "experiment_type": formatted_exp_name,
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
        print(f"Total bets extracted: {len(bets_df)}")

        # Basic statistics
        print(f"Unique debates: {bets_df['debate_id'].nunique()}")
        print(f"Unique models: {bets_df['betting_model'].nunique()}")
        print(f"Speech types: {bets_df['speech_type'].value_counts().to_dict()}")

        # Save to CSV
        bets_df.to_csv("all_bets.csv", index=False)
        print("Data saved to all_bets.csv")

        return bets_df
    else:
        print("No bets found in the data!")
        return None


if __name__ == "__main__":
    bets_df = create_bet_dataset()
    num_unique_debates_original = bets_df['debate_id'].nunique()
    print(f"Number of unique debate IDs in the original unfiltered dataset: {num_unique_debates_original}")

    # Display sample of the data
    if bets_df is not None:
        print("\nSample of the extracted data:")
        print(bets_df.head())
