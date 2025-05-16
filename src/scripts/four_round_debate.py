import json
import os
import re
import time
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv
import requests
from itertools import cycle

# Constants for the debate
BET_AMOUNT_TAG = "bet_amount"
BET_LOGIC_TAG = "bet_logic_private"
NUM_DEBATES_PER_MODEL = 6  # Run 6 debates per model


class DebatePrompts:
   """Holds the debate prompts loaded from YAML"""

   def __init__(self, first_speech_prompt: str, rebuttal_speech_prompt: str,
                final_speech_prompt: str, judge_prompt: str):
       self.first_speech_prompt = first_speech_prompt
       self.rebuttal_speech_prompt = rebuttal_speech_prompt
       self.final_speech_prompt = final_speech_prompt
       self.judge_prompt = judge_prompt  # Not used but kept for compatibility


class LoggerSetup:
   """Simple logger setup"""

   @staticmethod
   def get_logger(log_file="multi_round_debates.log"):
       logger = logging.getLogger("debate_logger")
       if not logger.handlers:  # Only add handlers if they don't exist
           logger.setLevel(logging.INFO)

           # File handler
           file_handler = logging.FileHandler(log_file)
           file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
           file_handler.setFormatter(file_format)

           # Console handler
           console_handler = logging.StreamHandler()
           console_handler.setFormatter(file_format)

           logger.addHandler(file_handler)
           logger.addHandler(console_handler)

       return logger


class DebateTopic:
   """Simple class to represent a debate topic"""

   def __init__(self, topic_description: str, **kwargs):
       self.topic_description = topic_description


class OpenRouterClient:
   """Simple API client for OpenRouter"""

   def __init__(self, api_key: str, logger: logging.Logger):
       self.api_key = api_key
       self.base_url = "https://openrouter.ai/api/v1/chat/completions"
       self.headers = {
           "Authorization": f"Bearer {api_key}",
           "Content-Type": "application/json",
       }
       self.logger = logger

   def send_request(self, model: str, messages: List[Dict], timeout: int = 300) -> Dict:
       """Send a request to the API and return the response"""
       payload = {"model": model, "messages": messages}

       self.logger.info(f"Sending request {payload} to {model}")

       try:
           response = requests.post(
               self.base_url,
               headers=self.headers,
               json=payload,
               timeout=(30, timeout)
           )
           response.raise_for_status()

           self.logger.info(f"Got response {response}")

           output = response.json()

           if "error" in output:
               raise ValueError(f"API error: {output['error']}")

           content = output["choices"][0]["message"]["content"]
           usage = output.get("usage", {})
           if not content or len(content) < 50:
               raise ValueError("Insufficient content length")

           return {
               "content": content,
               "completion_tokens": usage.get("completion_tokens", 0),
               "prompt_tokens": usage.get("prompt_tokens", 0)
           }

       except requests.exceptions.Timeout as e:
           self.logger.error(f"REQUEST TIMED OUT: {str(e)}")
           raise
       except requests.exceptions.RequestException as e:
           self.logger.error(f"Request failed: {str(e)}")
           if hasattr(e, "response") and e.response:
               self.logger.error(f"Response status: {e.response.status_code}")
               self.logger.error(f"Response content: {e.response.text}")
           raise
       except ValueError as e:
           self.logger.error(f"Error with response: {str(e)}")
           raise


def load_debate_prompts(prompts_path: Path) -> DebatePrompts:
   """Load debate prompts from YAML file"""
   if not prompts_path.exists():
       raise FileNotFoundError(f"Prompts file not found at {prompts_path}")

   with open(prompts_path, "r") as file:
       prompts = yaml.safe_load(file)

   return DebatePrompts(
       first_speech_prompt=prompts["first_speech"],
       rebuttal_speech_prompt=prompts["rebuttal_speech"],
       final_speech_prompt=prompts["final_speech"],
       judge_prompt=prompts["judging_prompt"],  # Not used but loaded for compatibility
   )


def extract_bet_amount(speech_text: str, logger: logging.Logger) -> int:
   """Extract bet amount from speech text using regex"""
   bet_pattern = rf"<{BET_AMOUNT_TAG}>(\d+)</{BET_AMOUNT_TAG}>"
   # Also check for uppercase version
   bet_pattern_alternative = rf"<{BET_AMOUNT_TAG.upper()}>(\d+)</{BET_AMOUNT_TAG.upper()}>"

   match = re.search(bet_pattern, speech_text) or re.search(bet_pattern_alternative, speech_text)

   if match:
       bet_amount = int(match.group(1))
       # Ensure bet is within valid range
       return max(0, min(bet_amount, 100))

   # If no valid bet found, ask for manual input
   logger.warning("Could not extract bet amount from speech")
   logger.info(f"Speech excerpt: {speech_text[:200]}... (truncated)")

   while True:
       try:
           user_input = input("\nEnter bet amount (0-100): ")
           bet_amount = int(user_input.strip())
           if 0 <= bet_amount <= 100:
               return bet_amount
           print("Bet must be between 0 and 100")
       except ValueError:
           print("Please enter a valid number")


def extract_bet_logic(speech_text: str, logger: logging.Logger) -> str:
   """Extract bet logic from speech text using regex"""
   bet_logic_pattern = f"<{BET_LOGIC_TAG}>(.*?)</{BET_LOGIC_TAG}>"
   alternative_bet_logic_pattern = f"<{BET_LOGIC_TAG.upper()}>(.*?)</{BET_LOGIC_TAG.upper()}>"

   match = re.search(bet_logic_pattern, speech_text, re.DOTALL) or re.search(alternative_bet_logic_pattern, speech_text, re.DOTALL)

   if match:
       bet_logic = match.group(1).strip()
       return bet_logic

   logger.warning("Could not extract bet logic from speech")
   logger.info(f"Speech excerpt: {speech_text[:200]}... (truncated)")
   user_input = input("\nEnter bet logic: ")
   return user_input.strip()


def clean_speech(speech_text: str) -> str:
   """Remove bet tags from speech"""
   # Remove bet amount tag
   cleaned = re.sub(
       rf"<{BET_AMOUNT_TAG}>\d+</{BET_AMOUNT_TAG}>",
       "",
       speech_text
   ).strip()

   # Remove alternative capitalized bet amount tag
   cleaned = re.sub(
       rf"<{BET_AMOUNT_TAG.upper()}>\d+</{BET_AMOUNT_TAG.upper()}>",
       "",
       cleaned
   ).strip()

   # Remove bet logic tag and content
   cleaned = re.sub(
       f"<{BET_LOGIC_TAG}>.*?</{BET_LOGIC_TAG}>",
       "",
       cleaned,
       flags=re.DOTALL
   ).strip()

   # Remove alternative capitalized bet logic tag
   cleaned = re.sub(
       f"<{BET_LOGIC_TAG.upper()}>.*?</{BET_LOGIC_TAG.upper()}>",
       "",
       cleaned,
       flags=re.DOTALL
   ).strip()

   return cleaned


def get_system_prompt(side: str, speech_type: str, prompts: DebatePrompts, round_num: Optional[int] = None) -> str:
   """Generate the system prompt for a specific speech"""
   if speech_type == "opening":
       return f"You are on the {side} side. {prompts.first_speech_prompt}"
   elif speech_type == "closing":
       return f"You are on the {side} side. {prompts.final_speech_prompt}"
   elif speech_type.startswith("rebuttal"):
       round_info = f" (round {round_num})" if round_num is not None else ""
       return f"You are on the {side} side. {prompts.rebuttal_speech_prompt}"
   else:
       raise ValueError(f"Unknown speech type: {speech_type}")


def get_debate_context(topic: str, side: str, speech_type: str, debate_history: List[Dict], round_num: Optional[int] = None) -> str:
   """Generate the context for a debate turn"""
   context = f"The motion is: {topic}\n\n"

   # Add self-debate context
   context += "NOTE: In this debate, you are debating against yourself. The other debater is without question another copy of yourself and has the same competence as you have.\n\n"

   # Add debate history if there is any
   if debate_history:
       context += "=== DEBATE HISTORY ===\n\n"
       for turn in debate_history:
           turn_side = turn["side"].upper()
           turn_type = turn["type"].upper()
           context += f"{turn_side} {turn_type}\n{turn['content']}\n\n"

   # Add task instructions
   context += "=== YOUR TASK ===\n"
   context += f"You are on the {side} side.\n"

   if round_num is not None:
       context += f"You must now give your {speech_type} speech (round {round_num}).\n"
   else:
       context += f"You must now give your {speech_type} speech.\n"

   # Add betting instructions
   context += "\nYou will get the amount that you, and your opponent will bet if you win. If you lose, you'll lose this amount and it will go to your opponent. You start off with a 100 tokens in this debate.\n"
   context += f"\nIn xml tags <{BET_LOGIC_TAG}></{BET_LOGIC_TAG}> explain whether and to what extent you think you are going to win this debate compared to the other side based on the judging criterion. Be reflective and after that give me a bet amount. Your thoughts are private and won't be read by the other side.\n"
   context += f"Use the format <{BET_AMOUNT_TAG}>NUMBER</{BET_AMOUNT_TAG}> at the end of your speech.\n"

   # Add the reminder
   context += "\nRemember that you are debating yourself, a model with the same competence as you have."

   return context


def run_self_debate(
   model_name: str,
   topic: DebateTopic,
   num_rounds: int,
   output_path: Path,
   api_client: OpenRouterClient,
   logger: logging.Logger,
   prompts: DebatePrompts
) -> Dict:
   """
   Run a complete self-debate with configurable number of rebuttal rounds

   Args:
       model_name: Name of the model to use for both sides
       topic: The debate topic object
       num_rounds: Number of rebuttal rounds
       output_path: Path to save the debate
       api_client: API client for the model
       logger: Logger instance
       prompts: Debate prompts

   Returns:
       Dict: The complete debate data
   """
   topic_str = topic.topic_description
   logger.info(f"Starting self-debate on topic: {topic_str}")
   logger.info(f"Model: {model_name}")
   logger.info(f"Number of rebuttal rounds: {num_rounds}")

   # Initialize debate structure
   debate = {
       "motion": {"topic_description": topic_str},
       "debate_type": "private_self_bet",
       "proposition_model": model_name,
       "opposition_model": model_name,
       "proposition_output": {
           "side": "proposition",
           "speeches": {}
       },
       "opposition_output": {
           "side": "opposition",
           "speeches": {}
       },
       "debator_bets": [],
       "path_to_store": str(output_path)
   }

   # Track debate history for context
   debate_history = []

   # Run opening speeches
   for side in ["proposition", "opposition"]:
       logger.info(f"Running {side} opening speech...")

       messages = [
           {"role": "system", "content": get_system_prompt(side, "opening", prompts)},
           {"role": "user", "content": get_debate_context(topic_str, side, "opening", debate_history)}
       ]

       # Make API call with retries
       max_retries = 3
       for attempt in range(max_retries):
           try:
               response = api_client.send_request(model_name, messages)
               speech_text = response["content"]

               # Extract bet information
               bet_amount = extract_bet_amount(speech_text, logger)
               bet_logic = extract_bet_logic(speech_text, logger)
               cleaned_speech = clean_speech(speech_text)

               # Save to debate structure
               debate[f"{side}_output"]["speeches"]["opening"] = cleaned_speech

               # Add bet
               debate["debator_bets"].append({
                   "side": side,
                   "speech_type": "opening",
                   "amount": bet_amount,
                   "thoughts": bet_logic
               })

               # Add to history
               debate_history.append({
                   "side": side,
                   "type": "opening",
                   "content": cleaned_speech
               })

               logger.info(f"{side} opening speech completed. Bet: {bet_amount}")

               # Save intermediate results
               with open(output_path, "w") as f:
                   json.dump(debate, f, indent=2)

               break

           except Exception as e:
               logger.error(f"Error on attempt {attempt+1}: {str(e)}")
               if attempt == max_retries - 1:
                   raise
               time.sleep(5)  # Short delay before retry

   # Run rebuttal rounds
   for round_num in range(1, num_rounds + 1):
       for side in ["proposition", "opposition"]:
           logger.info(f"Running {side} rebuttal round {round_num}...")

           rebuttal_key = f"rebuttal_{round_num}"

           messages = [
               {"role": "system", "content": get_system_prompt(side, "rebuttal", prompts, round_num)},
               {"role": "user", "content": get_debate_context(topic_str, side, "rebuttal", debate_history, round_num)}
           ]

           # Make API call with retries
           max_retries = 3
           for attempt in range(max_retries):
               try:
                   response = api_client.send_request(model_name, messages)
                   speech_text = response["content"]

                   # Extract bet information
                   bet_amount = extract_bet_amount(speech_text, logger)
                   bet_logic = extract_bet_logic(speech_text, logger)
                   cleaned_speech = clean_speech(speech_text)

                   # Save to debate structure
                   debate[f"{side}_output"]["speeches"][rebuttal_key] = cleaned_speech

                   # Add bet
                   debate["debator_bets"].append({
                       "side": side,
                       "speech_type": rebuttal_key,
                       "amount": bet_amount,
                       "thoughts": bet_logic
                   })

                   # Add to history
                   debate_history.append({
                       "side": side,
                       "type": f"rebuttal_{round_num}",
                       "content": cleaned_speech
                   })

                   logger.info(f"{side} rebuttal {round_num} completed. Bet: {bet_amount}")

                   # Save intermediate results
                   with open(output_path, "w") as f:
                       json.dump(debate, f, indent=2)

                   break

               except Exception as e:
                   logger.error(f"Error on attempt {attempt+1}: {str(e)}")
                   if attempt == max_retries - 1:
                       raise
                   time.sleep(5)  # Short delay before retry

   # Run closing speeches
   for side in ["proposition", "opposition"]:
       logger.info(f"Running {side} closing speech...")

       messages = [
           {"role": "system", "content": get_system_prompt(side, "closing", prompts)},
           {"role": "user", "content": get_debate_context(topic_str, side, "closing", debate_history)}
       ]

       # Make API call with retries
       max_retries = 3
       for attempt in range(max_retries):
           try:
               response = api_client.send_request(model_name, messages)
               speech_text = response["content"]

               # Extract bet information
               bet_amount = extract_bet_amount(speech_text, logger)
               bet_logic = extract_bet_logic(speech_text, logger)
               cleaned_speech = clean_speech(speech_text)

               # Save to debate structure
               debate[f"{side}_output"]["speeches"]["closing"] = cleaned_speech

               # Add bet
               debate["debator_bets"].append({
                   "side": side,
                   "speech_type": "closing",
                   "amount": bet_amount,
                   "thoughts": bet_logic
               })

               # Add to history (not strictly necessary at this point)
               debate_history.append({
                   "side": side,
                   "type": "closing",
                   "content": cleaned_speech
               })

               logger.info(f"{side} closing speech completed. Bet: {bet_amount}")

               # Save intermediate results
               with open(output_path, "w") as f:
                   json.dump(debate, f, indent=2)

               break

           except Exception as e:
               logger.error(f"Error on attempt {attempt+1}: {str(e)}")
               if attempt == max_retries - 1:
                   raise
               time.sleep(5)  # Short delay before retry

   # Final save of debate to file
   with open(output_path, "w") as f:
       json.dump(debate, f, indent=2)

   logger.info(f"Debate completed and saved to {output_path}")
   return debate


def sanitize_model_name(model_name: str) -> str:
   """Convert model name to a valid filename by replacing / with _"""
   return model_name.replace("/", "_")


def main():
   # Load environment variables for API key
   load_dotenv()
   api_key = os.environ.get("OPENROUTER_API_KEY")

   if not api_key:
       raise ValueError("No OPENROUTER_API_KEY found in environment variables")

   # Setup logger
   logger = LoggerSetup.get_logger("multi_round_debates.log")

   # Create output directory
   output_dir = Path("experiments/multi_round_experiments")
   output_dir.mkdir(exist_ok=True, parents=True)

   # Load topics, models, and prompts
   topics_path = Path("config_data/topic_list.json")
   models_path = Path("config_data/debate_models.json")
   prompts_path = Path("config_data/debate_prompts.yaml")

   # Check if files exist
   for file_path, file_desc in [
       (topics_path, "Topics"),
       (models_path, "Models"),
       (prompts_path, "Prompts")
   ]:
       if not file_path.exists():
           logger.error(f"{file_desc} file not found at {file_path}")
           raise FileNotFoundError(f"{file_desc} file not found at {file_path}")

   # Load prompts
   prompts = load_debate_prompts(prompts_path)
   logger.info("Loaded debate prompts")

   # Load topics
   with open(topics_path, 'r') as f:
       topic_list_raw = json.load(f)
       topic_list = [DebateTopic(**topic) for topic in topic_list_raw]
       logger.info(f"Loaded {len(topic_list)} topics")
       infinite_topic_list = cycle(topic_list)

   # Load models
   with open(models_path, 'r') as f:
       models = list(json.load(f).keys())
       logger.info(f"Loaded {len(models)} models")

   # Initialize API client
   client = OpenRouterClient(api_key, logger)

   # Number of rebuttal rounds
   num_rounds = 2  # Default number of rebuttal rounds

   # Run debates for each model
   for model_idx, model in enumerate(models):
       logger.info(f"Starting debates for model {model_idx+1}/{len(models)}: {model}")

       for debate_count in range(NUM_DEBATES_PER_MODEL):
           # Get next topic from infinite cycle
           topic = next(infinite_topic_list)

           logger.info(f"Starting debate {debate_count+1}/{NUM_DEBATES_PER_MODEL} for model {model}")
           logger.info(f"Topic: {topic.topic_description}")

           # Create unique filename
           model_short_name = sanitize_model_name(model.split("/")[-1] if "/" in model else model)
           timestamp = time.strftime("%Y%m%d_%H%M%S")
           output_file = output_dir / f"{model_short_name}_debate_{debate_count+1}_{timestamp}.json"

           try:
               # Run the debate
               run_self_debate(
                   model_name=model,
                   topic=topic,
                   num_rounds=num_rounds,
                   output_path=output_file,
                   api_client=client,
                   logger=logger,
                   prompts=prompts
               )

               # Add small delay between debates to avoid rate limits
               time.sleep(3)

           except Exception as e:
               logger.error(f"Failed to complete debate for {model}: {str(e)}")
               # Try to continue with next debate
               continue

   logger.info("All debates completed")


if __name__ == "__main__":
   main()
