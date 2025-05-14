from typing import List
from dotenv import load_dotenv
from pathlib import Path
from src.core.config import Config
from src.core.models import DebateTopic, DebateTotal, DebateType
import json
from src.core.utils import sanitize_model_name
import datetime
from itertools import cycle



load_dotenv()
# 1. load all models

num_debates_per_model = 6



def run_single_model_debate_redteam(model_name: str, topic:DebateTopic, config:Config, main_dir:Path) -> DebateTotal:
    model_name_for_path = sanitize_model_name(model_name)
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path_to_store = main_dir / Path(f"{model_name_for_path}_vs_self_{timestamp_str}.json")
    debate_service = config.debate_service
    debate_service.run_debate(
        proposition_model=model_name,
        opposition_model=model_name,
        motion=topic,
        path_to_store=path_to_store,
        debate_type=DebateType.SELF_REDTEAM_DEBATE
    )




def main():
    config = Config(log_file_name="readteam.log")
    with open(config.topic_list_path, 'r') as f:
        topic_list_raw: List[str] = json.load(f)
        print
        topic_list: List[DebateTopic] = [DebateTopic(**topic) for topic in topic_list_raw]
        print("there are", len(topic_list), "topics")
        infinite_topic_list = cycle(topic_list)


    logger = config.logger.get_logger(log_file="readteam.log")


    # 2. load all models
    with open(config.debate_models_list_path, 'r') as f:
        models = list(json.load(f).keys())

    output_dir = Path("experiments/SELF_REDTEAM_DEBATE")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_idx, model in enumerate(models):
        logger.info(f"Starting debates for model: {model_idx+1}/{len(models)}")
        for debate_count in range(num_debates_per_model):
            logger.info(f"Starting debate {debate_count+1}/{num_debates_per_model} for model {model}")
            topic = next(infinite_topic_list)


            run_single_model_debate_redteam(model_name=model,
                                            topic=topic,
                                            config=config,
                                            main_dir=output_dir)






main()



