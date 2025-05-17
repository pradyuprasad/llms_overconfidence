import json
from pathlib import Path
from tqdm import tqdm
import time
from dotenv import load_dotenv

from src.core.config import Config
from src.core.models import DebateTopic, DebateType

def sanitize_model_name(model_name: str) -> str:
    """Convert model name to a valid filename by replacing / with _"""
    return model_name.replace("/", "_")

def main():
    # Load environment variables
    load_dotenv()

    # Initialize config
    config = Config()
    logger = config.logger.get_logger()

    # Create output directory
    output_dir = Path("replicate_cross_model_experiments")
    output_dir.mkdir(exist_ok=True)

    # Load pairings
    with open("experiments/private_bet_experiment_pairings.json", "r") as f:
        pairings = json.load(f)

    logger.info(f"Loaded {len(pairings)} debate pairings")

    # Set debate type
    debate_type = DebateType.PRIVATE_BET

    # Process each pairing
    for i, pairing in enumerate(tqdm(pairings, desc="Running debates")):
        prop_model = pairing["proposition"]
        opp_model = pairing["opposition"]
        topic_dict = pairing["topic"]

        # Create topic object
        topic = DebateTopic(**topic_dict)

        # Create sanitized file name
        prop_name = sanitize_model_name(prop_model)
        opp_name = sanitize_model_name(opp_model)
        debate_filename = f"{prop_name}_vs_{opp_name}_{debate_type.value}.json"
        path_to_store = output_dir / debate_filename

        logger.info(f"Starting debate {i+1}/{len(pairings)}")
        logger.info(f"Proposition: {prop_model}")
        logger.info(f"Opposition: {opp_model}")
        logger.info(f"Topic: {topic.topic_description}")

        try:
            # Run the debate
            debate = config.debate_service.run_debate(
                proposition_model=prop_model,
                opposition_model=opp_model,
                motion=topic,
                path_to_store=path_to_store,
                debate_type=debate_type
            )

            # Add judgment using the opposition model as judge
            # Using opposition model as judge for simplicity
            judge_model = opp_model
            logger.info(f"Getting judgment from {judge_model}")

            judge_result = config.judgement_processor.process_judgment(
                debate=debate,
                model=judge_model
            )

            logger.info(f"Debate completed. Winner: {judge_result.winner} with confidence: {judge_result.confidence}%")

            # Save the debate
            debate.save_to_json()

            # Add a small delay between debates to avoid rate limits
            time.sleep(5)

        except Exception as e:
            logger.error(f"Error in debate {prop_model} vs {opp_model}: {str(e)}")
            continue

if __name__ == "__main__":
    main()
