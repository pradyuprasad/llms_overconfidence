# LLM Debate Bets Dataset Documentation

## Overview
This dataset contains confidence bets made by large language models (LLMs) across different debate formats. Each row represents a single bet made by a model during a structured debate.

## Experiment Types
The dataset includes bets from four distinct experimental configurations:

1. **cross_model**: Debates between different language models, each taking one side
2. **self_debate**: Self-debates where the same model argues both sides
3. **informed_self**: Self-debates with explicit instructions that the win probability is 50%
4. **public_bets**: Debates where confidence scores are publicly declared to the opponent

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `debate_id` | string | Unique identifier for each debate in format: `[experiment_type]_[proposition_model]_vs_[opposition_model]` |
| `experiment_type` | string | The type of experiment: "cross_model", "self_debate", "informed_self", or "public_bets" |
| `betting_model` | string | The specific model that made this bet |
| `proposition_model` | string | The model arguing the proposition side of the debate |
| `opposition_model` | string | The model arguing the opposition side of the debate |
| `side` | string | Which side the betting model is arguing (`proposition` or `opposition`) |
| `speech_type` | string | Stage of debate when the bet was placed (`opening`, `rebuttal`, or `closing`) |
| `bet_amount` | integer | Confidence level expressed as a value from 0-100 |
| `topic` | string | The debate topic/motion being argued |
| `bet_reasoning` | string | The model's reasoning or justification for its confidence bet |

## Values for Key Columns

### experiment_type
- "cross_model"
- "self_debate"
- "informed_self"
- "public_bets"

### side
- "proposition"
- "opposition"

### speech_type
- "opening"
- "rebuttal"
- "closing"

## Models Included
The dataset includes bets from 10 different language models:
- anthropic/claude-3.5-haiku
- anthropic/claude-3.7-sonnet
- deepseek/deepseek-chat
- deepseek/deepseek-r1-distill-qwen-14b
- google/gemini-2.0-flash-001
- google/gemma-3-27b-it
- openai/gpt-4o-mini
- openai/o3-mini
- qwen/qwen-max
- qwen/qwq-32b

## Usage Notes
- In self-debate conditions (self_debate, informed_self), the proposition_model and opposition_model will be identical
- Opening speech bets represent initial confidence before seeing the opponent's arguments
- The bet_amount column can be used to measure overconfidence when compared to actual debate outcomes
- All debates follow a structured format with opening, rebuttal, and closing phases

## File Format
The data is provided as a CSV file with headers, compatible with most data analysis tools and languages.
