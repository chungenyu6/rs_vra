"""
This script asks GeoChat to answer questions on VRSBench VQA 5Q.
# FIX LATER: change words here
Usage:
- Make sure GeoChat is running on port 8000
- Activate conda environment: agent
- Navigate to /ISRAgent folder
- Run: python eval/vrsbench_vqa_5q/eval_geochat.py 
- The results are saved in the results/vrsbench_vqa_5q/ans_geochat.json file,
which contains both predicted answers and ground truth answers. We will use this file
to evaluate the performance of GeoChat.

Note:
- The results are saved in the results/vrsbench_vqa_5q/ans_geochat.json file,
which contains both predicted answers and ground truth answers. We will use this file
to evaluate the performance of GeoChat.
"""

########################################################################################
# Standard imports
import os
import json
import time
import sys
from pathlib import Path
import asyncio

# Add parent directory to Python path (for exec_model.py)
sys.path.append(str(Path(__file__).parent.parent))

# Local imports
## Initialization
from init import init_arg, init_login
## Utils
from utils import *
## Model
import exec_model as exec_model
## Compute result
from compute_result import compute_result
## Logger
from logger import logger
########################################################################################

async def evaluate():
    """Evaluate the performance on VRSBench_vqa-n1000."""

    # Start timing
    start_time = time.time()

    # Initialize arguments and login
    args, argw = init_arg()
    init_login()

    # Get file paths
    IMG_FOLDER = "/home/vol-llm/datasets/ISR/VRSBench/validation/Images_val" # NOTE: change if needed
    LABEL_FILE = "datasets/VRSBench_vqa-n1000/json/label.json"
    RESULTS_FILE = f"results/VRSBench_vqa-n1000/{args.model}/{args.version}/{args.qtype}/sam{args.sample}/answer.json"
    EVAL_RESULT_FILE = f"results/VRSBench_vqa-n1000/{args.model}/{args.version}/{args.qtype}/sam{args.sample}/eval_result.txt"
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True) # make folder if not exist
    os.makedirs(os.path.dirname(EVAL_RESULT_FILE), exist_ok=True)

    # Load label file
    with open(LABEL_FILE, "r") as f:
        labels = json.load(f)
    results = []

    # Get the list of question_ids in this question type and number of questions
    ids = get_question_ids(args.qtype, args.sample)
    count = 0

    # Process each image-question pair within the range of NUM_QUESTIONS
    logger.info(f"Processing {len(ids)} questions in question type: {args.qtype}")
    for id in ids:
        logger.info("-"*15 + f" Model answering question id: {id} ({count+1}/{len(ids)}) " + "-"*15)
        
        # Find the corresponding entry in labels
        entry = next(item for item in labels if item["question_id"] == id and item["type"].lower() == args.qtype.replace("_", " ").lower())
        image_id = entry["image_id"]
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        img_path = os.path.join(IMG_FOLDER, image_id)

        # Query geochat to answer the question
        logger.info(f"Querying {args.model} to answer the question")
        # NOTE: add more models here
        if args.model == "agent":
            prediction = exec_model.query_agent(args, question, img_path)
        elif args.model == "geochat":
            prediction = exec_model.query_geochat(args, question, img_path)
        elif args.model == "llava1.5":
            prediction = await exec_model.query_llava(args, question, img_path)
        elif args.model == "gemma3":
            prediction = await exec_model.query_gemma3(args, question, img_path)
        else:
            raise ValueError(f"Invalid model: {args.model}")

        # Save the response
        results.append({
            "image_id": image_id,
            "question": question,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "question_id": id,
            "type": entry["type"]
        })
        count += 1

    logger.info("-"*67)
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