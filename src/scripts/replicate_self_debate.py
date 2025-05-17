from dotenv import load_dotenv
from pathlib import Path
import json
import datetime
from itertools import cycle

from src.core.config import Config
from src.core.models import DebateTopic, DebateType
from src.core.utils import sanitize_model_name

def run_single_model_debate(
    model_name: str,
    topic: DebateTopic,
    config: Config,
    output_dir: Path,
    debate_type: DebateType
) -> None:
    """Run a debate where a model debates against itself with the specified debate type"""
    model_name_for_path = sanitize_model_name(model_name)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_to_store = output_dir / f"{model_name_for_path}_vs_self_{debate_type.value}_{timestamp_str}.json"

    logger = config.logger.get_logger()
    logger.info(f"Running debate: {model_name} vs self under {debate_type.value}")
    logger.info(f"Topic: {topic.topic_description}")

    try:
        config.debate_service.run_debate(
            proposition_model=model_name,
            opposition_model=model_name,
            motion=topic,
            path_to_store=path_to_store,
            debate_type=debate_type
        )
        logger.info(f"Successfully completed debate and saved to {path_to_store}")
    except Exception as e:
        logger.error(f"Error running debate: {str(e)}")

def main():
    # Load environment variables
    load_dotenv()

    # Initialize config
    config = Config()
    logger = config.logger.get_logger()
    logger.info("Starting self-debate experiment script")

    # Configure debate types and their output directories
    debate_configs = [
        (DebateType.PRIVATE_SAME_DEBATOR, "private_same_debator"),
        (DebateType.PUBLIC_SAME_DEBATOR, "public_same_debator"),
        (DebateType.PRIVATE_SAME_DEBATOR_INFORMED, "private_same_debator_informed")
    ]

    # Create output directories
    output_dirs = {}
    for debate_type, dir_name in debate_configs:
        output_dir = Path(f"experiments/{dir_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[debate_type] = output_dir

    # Load topics
    with open(config.topic_list_path, 'r') as f:
        topic_list_raw = json.load(f)
        topic_list = [DebateTopic(**topic) for topic in topic_list_raw]
        logger.info(f"Loaded {len(topic_list)} topics")
        infinite_topic_list = cycle(topic_list)

    # Load models
    with open(config.debate_models_list_path, 'r') as f:
        models = list(json.load(f).keys())
        logger.info(f"Loaded {len(models)} models")

    # Number of debates per model per type
    num_debates_per_model = 6

    # Run debates for each model, for each debate type
    for model_idx, model in enumerate(models):
        logger.info(f"Processing model {model_idx+1}/{len(models)}: {model}")

        for debate_type, _ in debate_configs:
            output_dir = output_dirs[debate_type]
            logger.info(f"Running {num_debates_per_model} debates for debate type: {debate_type.value}")

            for debate_count in range(num_debates_per_model):
                topic = next(infinite_topic_list)
                logger.info(f"Starting debate {debate_count+1}/{num_debates_per_model}")

                run_single_model_debate(
                    model_name=model,
                    topic=topic,
                    config=config,
                    output_dir=output_dir,
                    debate_type=debate_type
                )

    logger.info("All debates completed")

if __name__ == "__main__":
    main()
