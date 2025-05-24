"""
This script is to evaluate on benchmark POPE-n900.

Usage:
- Navigate to the root directory of the project
- Run the following command:
    python ./eval/POPE-n900/evaluate.py --subset random --sample 1 --model llava1.5 --version v1 --max_reflexion_iters 1 --wandb False
- The results will be saved in the results/POPE-n900/MODEL/VERSION/SUBSET/SAMPLE directory
- The logs will be saved in the results/POPE-n900/MODEL/VERSION/SUBSET/SAMPLE/log_qid%d.json file
- The answers will be saved in the results/POPE-n900/MODEL/VERSION/SUBSET/SAMPLE/answer.json file
- The evaluation results will be saved in the results/POPE-n900/MODEL/VERSION/SUBSET/SAMPLE/eval_result.txt file
"""

######################################################################################
# Standard library imports
import os
import json
import time
import asyncio
import sys
from pathlib import Path
# Add parent directory to Python path (for exec_model.py)
sys.path.append(str(Path(__file__).parent.parent))

# Third-party imports
import wandb

# Local application imports
## Initialization
from init import init_arg, init_login
## Logger
from logger import logger
## Evaluation
from compute_result import compute_result
## Model
import exec_model as exec_model
######################################################################################

async def evaluate():
    """Main evaluation function"""

    # Start timing
    start_time = time.time()
    
    # Initialize arguments and login
    args, argw = init_arg()
    init_login()

    # Load image and question from POPE-n900
    output_folder = f"./results/POPE-n900/{args.model}/{args.version}/{args.subset}/sam{str(args.sample)}/"
    question_file = f"./datasets/POPE-n900/json/coco_50_pope_{args.subset}.json"    # benchmark questions
    answer_file = output_folder + "answer.json"                                     # predictions
    log_file = output_folder + "log_qid%d.json"                                     # filename dynamically changes with question id
    img_folder = f"/home/vol-llm/datasets/POPE300/img50/NoAttack/NoDefense/{args.subset}/"           # NOTE: change if needed
    os.makedirs(output_folder, exist_ok=True)
    
    question_list = [json.loads(q) for q in open(question_file, 'r')]
    answer_list = []

    # Torture list to track which samples entered torture
    tortureList = []

    # NOTE: for testing
    # question_list = [
    #     {"question_id": 1, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a bag in the image?", "label": "yes"},
    #     {"question_id": 2, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a truck in the image?", "label": "yes"},
    #     {"question_id": 3, "image": "COCO_val2014_000000127153.jpg", "text": "Are there only two people in the image?", "label": "yes"},
    #     {"question_id": 4, "image": "COCO_val2014_000000127153.jpg", "text": "Is there two people in the image?", "label": "yes"},
    #     {"question_id": 5, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a total of two umbrella in the image?", "label": "yes"},
    #     {"question_id": 6, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a total of five bag in the image?", "label": "yes"},
    #     {"question_id": 7, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a yellow umbrella with black handle in the image?", "label": "yes"},
    #     {"question_id": 8, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a red umbrella in the image?", "label": "yes"},
    #     {"question_id": 9, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a black bottle in the image?", "label": "yes"},
    #     {"question_id": 10, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a car at the left side of a bag in the image?", "label": "yes"},
    #     {"question_id": 11, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a car above a bag in the image?", "label": "yes"},
    #     {"question_id": 12, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a person in a car in the image?", "label": "yes"},
    #     {"question_id": 13, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a dog in the middle of cars in the image?", "label": "no"},
    #     {"question_id": 14, "image": "COCO_val2014_000000127153.jpg", "text": "Is there a car behind a laptop in the image?", "label": "no"},
    # ]

    # Start asking questions
    logger.info("Start asking questions")
    for idx, question in enumerate(question_list):
        if idx >= args.sample:
            break

        # Set up a sample (image, question, label), serves as the memory for the system
        logger.info("-"*10 + "Question Id: %d"%(idx+1) + "-"*10)
        qid = question["question_id"]
        img_filename = question["image"]
        bench_query = question["text"]          # benchmark question
        bench_label = question["label"]         # benchmark label
        img_path = os.path.join(img_folder, img_filename)
        sample = {
            "idx": idx, # start with 0
            "qid": qid, # start with 1
            "img_path": img_path,
            "bench_query": bench_query,
            "bench_label": bench_label,
        }

        # TODO: add more models here
        # Model answers the benchmark question
        if args.model == "agent":
            model_answer = exec_model.query_agent(args, bench_query, img_path)
        elif args.model == "llava1.5":
            model_answer = await exec_model.query_llava(args, bench_query, img_path)
        else:
            raise ValueError(f"Invalid model name: {args.model}")
        sample["final_answer"] = model_answer

        # Save answer per question
        answer = {
            "quesion_id (start with 1)": qid, 
            "bench_query": bench_query, 
            'bench_label': bench_label,
            "answer": sample["final_answer"],
            "img_path": img_path,
        }
        answer_list.append(answer)
        
        # Write logs to file per question
        with open(log_file % (idx+1), 'w') as f:
            json_sample = json.dumps(sample, indent=4)
            f.write(json_sample+"\n")
    logger.info("Finish asking questions")

    # Write answers to file for evaluation
    with open(answer_file, 'w') as f:
        for answer in answer_list:
            json_str = json.dumps(answer)
            f.write(json_str + "\n")
    logger.info(f"Answers and logs are saved to the folder: {output_folder}")

    # Calculate runtime
    end_time = time.time()
    elapsed_time = (end_time-start_time) / 60 # in minutes
    argw.total_runtime = elapsed_time

    # Evaluate +torture list
    compute_result(args, argw, output_folder, answer_file, question_file, tortureList)
    logger.info(f"Evaluation results are saved to the folder: {output_folder}")

    # Log to wandb
    if args.wandb == True:
        wandb.log(
            {
                "total runtime (min)": argw.total_runtime,
                "TP": argw.tp,
                "FP": argw.fp,
                "FN": argw.fn,
                "TN": argw.tn,
                "acc": argw.acc,
                "precision_p": argw.precision_p,
                "recall_p": argw.recall_p,
                "f1_p": argw.f1_p,
                "precision_n": argw.precision_n,
                "recall_n": argw.recall_n,
                "f1_n": argw.f1_n,
                "f1_macro": argw.f1_macro,
                "yes_ratio": argw.yes_ratio,
            }
        )
        wandb.finish()
    logger.info("System finished!")

if __name__ == '__main__':
    asyncio.run(evaluate())