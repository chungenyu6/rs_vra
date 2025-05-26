"""
Compute the result of the VRSBench VQA 5Q dataset.

Usage:
- Make sure Phi-4 is running on Ollama
- Activate conda environment: agent
- Navigate to /ISRAgent folder
- Run: python eval/vrsbench_vqa_5q/compute_result.py
"""

########################################################################################
# Standard library imports
import os
import json

# Third-party imports
import numpy as np
from dotenv import load_dotenv
import wandb
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama

# Local application imports
from logger import logger
########################################################################################

# TODO: change if needed
OLLAMA_BASE_URL = "127.0.0.1:11434"
OLLAMA_MODEL = "phi4"

def init_eval_llm():
    """Initialize the LLM"""

    logger.info("Initializing evaluation LLM")

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
        # Fallback if LLM response is unexpected # NOTE: unfair here?
        match = '0'

    # Remove the leading '1' or '0' and any whitespace/newlines
    reason = raw_content[1:].lstrip().replace('\n', ' ')

    return match, reason

def compute_result(args, argw, RESULT_FILE, EVAL_RESULT_FILE):
    """Compute the result"""

    # Initialize LLM
    chain = init_eval_llm()

    # Load all results from the output file
    logger.info("Loading answers from results file")
    with open(RESULT_FILE, "r") as f:
        data = json.load(f)

    # Initialize counters
    correct = 0
    total = 0
    unknown_type = 0

    # Prepare list of all numbers as strings for matching
    all_numbers = [str(i) for i in range(100)]

    # Iterate through each record and tally counts using new logic
    logger.info("Starting to compute results")
    for item in data:
        logger.info("="*15 + f" Computing result ({total+1}/{len(data)}) " + "="*15)

        # Get question type
        qtype = item.get("type", "").lower()
        # Standardize synonyms
        if qtype in ("image", "rural or urban"):
            qtype = "scene type"
        total += 1

        # Convert both types to a common format for comparison
        normalized_qtype = qtype.replace(" ", "_")
        normalized_args_qtype = args.qtype.replace(" ", "_")

        if normalized_qtype != normalized_args_qtype:
            unknown_type += 1
            continue

        # Extract required fields
        ground_truth = str(item.get("ground_truth", "")).strip()
        prediction = str(item.get("prediction", "")).strip()
        question = item.get("question", "")

        logger.info(f"Question Type: {qtype}")
        logger.info(f"Question: {question}")
        logger.info(f"Ground Truth: {ground_truth}")
        logger.info(f"Prediction: {prediction}")

        # Matching logic
        match = None
        if ground_truth in prediction:
            match = '1'
            logger.info(f"Match (ground truth in prediction): {match}")
        elif ground_truth in (['yes', 'no'] + all_numbers):
            match = '1' if ground_truth == prediction else '0'
            logger.info(f"Match (ground truth yes/no/number): {match}")
        elif ('correct' not in item) or (item['correct'] not in ['1', '0']):
            match, reason = check_match_with_llm(chain, question, ground_truth, prediction)
            logger.info(f"Match (LLM): {match}")
            logger.info(f"LLM (un)match reason: {reason}")
        else:
            match = item['correct']
            logger.info(f"Match (correct): {match}")
        
        # Ensure match is string
        match = str(match)
        item['correct'] = match

        if match == "1":
            correct += 1

    logger.info("="*55)
    logger.info("End of computing result")

    # Print result summary
    logger.info("Printing result summary")
    logger.info("="*25 + " Result summary " + "="*25)
    # Print overall accuracy
    logger.info(f"Total questions: {total}, Unknown types: {unknown_type}")
    if total > 0:
        logger.info(f"Accuracy for {args.qtype}: {correct/total*100:.2f}% ({correct}/{total})")
    else:
        logger.info(f"No questions found for type: {args.qtype}")
    logger.info("="*67)
    logger.info("Result summary printed")

    # Save result summary to txt file, eval_result.txt
    logger.info("Saving result summary to txt file")
    with open(EVAL_RESULT_FILE, "w") as f:
        f.write(f"Total questions: {total}, Unknown types: {unknown_type}\n")
        f.write(f"Accuracy for {args.qtype}: {correct/total*100:.2f}% ({correct}/{total})\n")
    logger.info(f"Result summary saved to {EVAL_RESULT_FILE}")

    # Save result summary to wandb
    if args.wandb == True:
        logger.info("Saving result summary to wandb")
        wandb.log(
            {
                "total_questions": total,
                "unknown_qtypes": unknown_type,
                "accuracy": correct/total*100 if total > 0 else 0,
                "total_runtime (min)": argw.total_runtime,
            }
        )
        logger.info("Result summary saved to wandb")
        wandb.finish()
