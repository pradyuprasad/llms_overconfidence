from __future__ import annotations
import datetime
import json
from pathlib import Path

from dotenv import load_dotenv

from src.core.config import Config
from src.core.models import DebateTopic, DebateType

"""Quick smoke-test: run one self-debate between two copies of Claude-3.7-Sonnet.

Usage (from repo root)::

    python -m src.scripts.test_single_debate

The script will pick the *first* motion in `config_data/topic_list.json`,
run a PRIVATE_SAME_DEBATOR debate, and save the resulting JSON under
`experiments/test_single/`.
"""

MODEL_ID = "anthropic/claude-3.7-sonnet"


def main() -> None:
    load_dotenv()  # picks up OPENROUTER_API_KEY from .env or env vars

    cfg = Config()

    # Select the first topic from the canonical list ---------------------------------
    with open(cfg.topic_list_path, "r", encoding="utf-8") as f:
        topics_json = json.load(f)
    if not topics_json:
        raise RuntimeError("No topics found in config_data/topic_list.json")
    topic = DebateTopic(**topics_json[0])

    # Output path --------------------------------------------------------------------
    out_dir = Path("experiments/test_single")
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"claude_sonnet_vs_self_{timestamp}.json"

    print(f"Running single self-debate on topic: {topic.topic_description[:60]}…")
    print(f"Proposition & Opposition model: {MODEL_ID}")

    cfg.debate_service.run_debate(
        proposition_model=MODEL_ID,
        opposition_model=MODEL_ID,
        motion=topic,
        path_to_store=out_path,
        debate_type=DebateType.PRIVATE_SAME_DEBATOR,
    )

    print(f"Saved debate JSON → {out_path}\nDone ✔︎")


if __name__ == "__main__":
    main() 