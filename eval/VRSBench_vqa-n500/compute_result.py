"""
Compute the result of the VRSBench VQA 5Q dataset.

Usage:
- Make sure Phi-4 is running on Ollama
- Activate conda environment: agent
- Navigate to /ISRAgent folder
- Run: python eval/vrsbench_vqa_5q/compute_result.py
"""

# Standard library imports
import os
import json

# Third-party imports
from dotenv import load_dotenv
import wandb
import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

# Local application imports
from logger import logger

# TODO: change parameters
WANDB_ENTITY = "chungenyu6-uwf"
WANDB_PROJECT = "vrsbench_vqa_5q"
WANDB_NAME = "agent-r1.1-rm" # geochat # agent-r3 # llava1.5 # TESTING

OLLAMA_BASE_URL = "127.0.0.1:11434"
OLLAMA_MODEL = "phi4"

# TESTING
# RESULT_FILE = "results/vrsbench_vqa_5q/ans_geochat.json"
# RESULT_FILE = "results/vrsbench_vqa_5q/ans_llava1.5.json"
RESULT_FILE = "results/vrsbench_vqa_5q/ans_agent.json"
QUESTION_TYPES = ["object quantity", "object position", "object direction", "object size", "reasoning"] # select question types based on your needs # "object quantity", "object position", "object direction", "object size", "reasoning" 

def init_wandb():
    """Initialize wandb"""

    logger.info("Initializing wandb")

    load_dotenv()
    wandb_api_key = os.getenv("WANDB_API_KEY")
    if not wandb_api_key:
        logger.error("WANDB_API_KEY not found in environment variables")
        raise ValueError("WANDB_API_KEY not found in environment variables")
    wandb.login(key=wandb_api_key)
    wandb.init(
        entity=WANDB_ENTITY,                        
        project=WANDB_PROJECT,
        name=WANDB_NAME,
    )

    logger.info("Wandb initialized")

def init_llm():
    """Initialize the LLM"""

    logger.info("Initializing LLM")

    # Initialize the ChatOllama model
    model = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL, 
        temperature=0.1,
        max_tokens=100
    )

    # Define the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human", 
                "Question: {question}\nGround Truth Answer: {ground_truth}\nPredicted Answer: {prediction}\nDoes the predicted answer match the ground truth? Answer 1 for match and 0 for not match. Use semantic meaning not exact match. Synonyms are also treated as a match, e.g., football and soccer, playground and ground track field, building and rooftop, pond and swimming pool. Do not explain the reason.\n"
            ),
        ]
    )

    # Chain the prompt and the model
    chain = prompt | model

    logger.info("LLM initialized")
    return chain

def check_match_with_llm(chain, question, ground_truth, prediction):
    """Check if the predicted answer matches the ground truth using LLM"""

    response = chain.invoke(
        {
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
        }
    )

    # Extract only leading '1' or '0'
    raw_content = response.content
    raw_content = raw_content.strip()
    if raw_content and raw_content[0] in ('1', '0'):
        match = raw_content[0]
    else:
        # Fallback if LLM response is unexpected # NOTE
        match = '0'

    # Remove the leading '1' or '0' and any whitespace/newlines
    reason = raw_content[1:].lstrip().replace('\n', ' ')

    return match, reason

def compute_result():
    """Compute the result"""

    # Initialize wandb
    init_wandb()

    # Initialize LLM
    chain = init_llm()

    # Load all results from the output file
    logger.info(f"Loading results from {RESULT_FILE}")
    with open(RESULT_FILE, "r") as f:
        data = json.load(f)

    # Initialize counters
    correct_per_type = {qtype: 0 for qtype in QUESTION_TYPES}
    total_per_type = {qtype: 0 for qtype in QUESTION_TYPES}
    unknown_type = 0
    total = 0
    correct = 0

    # Prepare list of all numbers as strings for matching
    all_numbers = [str(i) for i in range(100)]

    # Iterate through each record and tally counts using new logic
    logger.info("Starting to process item")
    logger.info("\n" + "="*25 + " Processing items " + "="*25)
    for item in data:
        logger.info(f"\n===== Processing item: {total+1} =====")

        # Get question type
        qtype = item.get("type", "").lower()
        # Standardize synonyms
        if qtype in ("image", "rural or urban"):
            qtype = "scene type"
        total += 1

        if qtype in QUESTION_TYPES:
            total_per_type[qtype] += 1
        else:
            unknown_type += 1

        # Extract required fields
        ground_truth = str(item.get("ground_truth", "")).strip()
        prediction = str(item.get("prediction", "")).strip()
        question = item.get("question", "")

        logger.info(f"Question Type: \n{qtype}")
        logger.info(f"Question: \n{question}")
        logger.info(f"Ground Truth: \n{ground_truth}")
        logger.info(f"Prediction: \n{prediction}")

        # Matching logic
        match = None
        if ground_truth in prediction:
            match = '1'
            logger.info(f"Match (ground truth in prediction): \n{match}")
        elif ground_truth in (['yes', 'no'] + all_numbers):
            match = '1' if ground_truth == prediction else '0'
            logger.info(f"Match (ground truth yes/no/number): \n{match}")
        elif ('correct' not in item) or (item['correct'] not in ['1', '0']):
            match, reason = check_match_with_llm(chain, question, ground_truth, prediction)
            logger.info(f"Match (LLM): \n{match}")
            logger.info(f"LLM (un)match reason: \n{reason}")
        else:
            match = item['correct']
            logger.info(f"Match (correct): \n{match}")
        
        # Ensure match is string
        match = str(match)
        item['correct'] = match

        if match == "1":
            correct += 1
            if qtype in QUESTION_TYPES:
                correct_per_type[qtype] += 1

    logger.info("\n" + "="*67)
    logger.info("End of processing")

    # Print result summary
    logger.info("Printing result summary")
    logger.info("\n" + "="*25 + " Result summary " + "="*26 + "\n")
    # Print overall accuracy
    logger.info(f"Total questions: {total}, Unknown types: {unknown_type}")
    logger.info(f"Overall accuracy: {correct/total*100:.2f}% ({correct}/{total})")
    # Print per-type accuracies
    logger.info("Per-type accuracy")
    for qtype in QUESTION_TYPES:
        if total_per_type[qtype] > 0:
            acc = correct_per_type[qtype] / total_per_type[qtype] * 100
            logger.info(f"  > {qtype}: \t{acc:.2f}% ({correct_per_type[qtype]}/{total_per_type[qtype]})")
        else:
            logger.info(f"  > {qtype}: \tNo questions")
    logger.info("\n" + "="*67 + "\n")
    logger.info("Result summary printed")

    # Save result summary to wandb
    logger.info("Saving result summary to wandb")
    wandb.log({
        "total_questions": total,
        "unknown_types": unknown_type,
        "overall_accuracy": correct/total*100,
        "per_type_accuracy": {qtype: correct_per_type[qtype] / total_per_type[qtype] * 100 for qtype in QUESTION_TYPES}
    })
    logger.info("Result summary saved to wandb")
    
if __name__ == "__main__":
    compute_result()
