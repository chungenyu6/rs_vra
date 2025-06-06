# FIX LATER: change words here
"""
This script asks model to answer questions on MME-RealWorld-Lite.

Usage:
- 

Note:
- 
"""

########################################################################################
# Standard imports
import os
import json
import time
import sys
import asyncio
from pathlib import Path
# Add parent directory to Python path (for exec_model.py)
sys.path.append(str(Path(__file__).parent.parent))

# Local imports
## Initialization
from init import initialize
## Utils
from utils import *
## Model
import exec_model as exec_model
## Compute result
from compute_result import compute_result
########################################################################################

async def evaluate():
    """Evaluate the performance on MME-RealWorld-Lite."""

    # Start timing
    start_time = time.time()

    # Initialize everything in the correct order
    args, argw, logger = initialize()

    # Get file paths
    IMG_FOLDER = "/home/vol-llm/datasets/ISR/MME-RealWorld-Lite/img" # NOTE: change if needed
    LABEL_FILE = "datasets/MME-RealWorld-Lite/json/label.json"
    RESULTS_FILE = f"results/MME-RealWorld-Lite/{args.model}/{args.version}/{args.l2category}/sam{args.sample}/answer.json"
    EVAL_RESULT_FILE = f"results/MME-RealWorld-Lite/{args.model}/{args.version}/{args.l2category}/sam{args.sample}/eval_result.txt"
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True) # make folder if not exist
    os.makedirs(os.path.dirname(EVAL_RESULT_FILE), exist_ok=True)

    # Load label file
    with open(LABEL_FILE, "r") as f:
        labels = json.load(f)
    results = []

    # Get the list of question index in this l2category
    ids = get_indices(args.l2category, args.sample) # in utils.py
    count = 0

    # Process each image-question pair within the range of NUM_QUESTIONS
    logger.info(f"Processing {len(ids)} questions in l2-category: {args.l2category}")
    for id in ids:
        # Find the corresponding entry in labels
        entry = next(item for item in labels if item["index"] == id and item["l2_category"].lower() == args.l2category.lower())
        category = entry["category"]
        img_path = entry["image_path"]
        question = entry["question"]
        options = entry["options"] # a list of strings
        label = entry["label"]

        # Assemble question and options in utils.py
        usr_msg = assemble_question(question, options)

        # Query model to answer the question
        logger.info("="*15 + f" Model answering question idex: {id} ({count+1}/{len(ids)}) " + "="*15)
        logger.info(f"l2-category: {args.l2category}")
        logger.info(f"Question: {question}")
        logger.info(f"Options: {options}")
        logger.info(f"Question image path: {img_path}")
        logger.info(f"Querying {args.model} to answer the question")
        # NOTE: add more models here
        if args.model == "agent":
            prediction = exec_model.query_agent(args, usr_msg, img_path)
        elif args.model == "geochat":
            prediction = exec_model.query_geochat(args, usr_msg, img_path)
        elif args.model == "llava15":
            prediction = await exec_model.query_llava15(args, usr_msg, img_path)
        elif args.model == "gemma3":
            prediction = await exec_model.query_gemma3(args, usr_msg, img_path)
        elif args.model == "mistral31":
            prediction = await exec_model.query_mistral31(args, usr_msg, img_path)
        elif args.model == "gemini25-flash":
            prediction = await exec_model.query_gemini25flash(args, usr_msg, img_path)
        else:
            raise ValueError(f"Invalid model: {args.model}")

        # Save the response
        results.append({
            "category": category,
            "l2_category": args.l2category,
            "index": id,
            "img_path": img_path,
            "question": question,
            "options": options,
            "label": label,
            "prediction": prediction,
        })
        count += 1

    logger.info("="*67)
    # End timing
    end_time = time.time()
    argw.total_runtime = (end_time - start_time)/60
    logger.info(f"Total time: {argw.total_runtime:.2f} minutes")

    # Write results to JSON file
    logger.info(f"Saving the response to {RESULTS_FILE}")
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # Print result summary
    logger.info(f"Results saved to {RESULTS_FILE}")

    # Compute result
    compute_result(args, argw, RESULTS_FILE, EVAL_RESULT_FILE)


if __name__ == "__main__":
    asyncio.run(evaluate())