"""
This script is to evaluate HalluAgent on AMBER100 benchmark.
# FIXME: Partial code from LogicCheckGPT
"""

######################################################################################
# Standard library imports
import re
import os
import sys
import json
import time
import argparse

# Third-party imports
from dotenv import load_dotenv
import torch
import spacy
import numpy as np
from typing import List
from PIL import Image
from transformers import AutoTokenizer
import openai
from openai import OpenAI
import wandb
from huggingface_hub import login

# Local application imports
from logger import logger
## Initialization
from init import init_arg, init_login
## Agent backbone
# FIXME: maybe integrate to agent py file
from agent_backbone.gpt35turbo import load_llm_gpt, get_llm_res, get_gpt_assistant_res
from agent_backbone.llama32 import load_llm_llama, get_llm_res_llama
## Agent
# from agent.critic
# from agent.planner
## Visual tools
from visual_tool.obj_detector import load_obj_detector, use_obj_detector
from visual_tool.vlm_mplugowl import load_vlm_owl, get_owl_res
from visual_tool.vlm_paligemma import load_vlm_paligemma, get_paligemma_res
from visual_tool.vlm_blipvqa import load_vlm_blipvqa, get_blipvqa_res
## Evaluation
from eval.eval_amber50 import evaluate_amber


from hydra_caption import exec_hydra_caption
######################################################################################

def main_amber():
    """
    Main function
    """

    start_time = time.time()
    
    # Initialize arguments
    args, argp, argw = init_arg() # function in init.py # FIXME: remove argp later
    init_login() # function in init.py

    # Load models
    if args.sys == 0:
        # Load VLM
        load_vlm_owl() # function in vlm_mplugowl.py
    elif args.sys == 1:
        # Load agent backbone
        # load_llm_gpt() # function in gpt35turbo.py
        load_llm_llama() # function in llama32.py
        # Load visual tools
        load_obj_detector() # function in obj_detector.py
        load_vlm_owl() # function in vlm_mplugowl.py
        # Load extra visual tools
        load_vlm_paligemma()
        load_vlm_blipvqa()

    end_time_load_model = time.time()
    argw.load_model_time = (end_time_load_model-start_time) / 60

    # Load image and question from POPE300
    output_folder = f"/home/vol-llm/proj-AgenticAI/20250131-OfficialPaperResults/AMBER50/{args.name}/"  # Johnny
    # output_folder = f"./results/POPE300/{args.name}/"                                                 # public
    os.makedirs(output_folder, exist_ok=True)                                       # create directory if it doesn't exist
    question_file = "/home/vol-llm/datasets/AMBER50/query/generative_query.json"     # benchmark questions
    benchmark_file = "/home/HalluAgent-Private/datasets/AMBER50/utils/annotations.json" # annotations.json
    
    answer_file = output_folder + "answer.json"                                     # my answers for evaluation
    log_file = output_folder + "log_qid%d.json"                                     # filename dynamically changes with question id

    img_folder = f"/home/vol-llm/datasets/AMBER50/image/{args.attack}/{args.defense}/{args.subset}/"  # Johnny
    
    
    with open(question_file, 'r') as file:
        question_list = json.load(file)  # Load the entire JSON content

    with open(benchmark_file, 'r') as file:
        benchmark_list = json.load(file)  # Load the entire JSON content
    
    # Convert benchmark_list into a dictionary for quick lookup
    benchmark_dict = {benchmark["id"]: benchmark for benchmark in benchmark_list}

    answer_list = []

    # Torture list to track which samples entered torture
    tortureList = []

    # Start asking questions
    logger.info("\n[HalluAgent] [INFO] Start benchmark querying.")
    for idx, question in enumerate(question_list):
    # for idx, (question, benchmark) in enumerate(zip(question_list, benchmark_list)):
        if idx >= args.sample:
            break

        # Set up a sample (image, question, label), serves as the memory for the system
        qid = question["id"]
        logger.info("\n" + "-"*20 + "Question Id: %d"%(qid) + "-"*20) # NOTE: use `qid` not `idx+1`
        img_filename = question["image"]
        bench_query = question["query"]        # benchmark question
        img_path = os.path.join(img_folder, img_filename)

        # Fetch corresponding benchmark labels using qid
        if qid in benchmark_dict:
            truth_label = benchmark_dict[qid]["truth"]
            hallu_label = benchmark_dict[qid]["hallu"]
        else:
            logger.warning(f"[HalluAgent] [WARNING] No benchmark found for Question ID: {qid}")
            truth_label, hallu_label = None, None  # Handle missing cases gracefully

        sample = {  
            "idx": idx, # start with 0
            "qid": qid, # start with 1
            "img_path": img_path,
            "bench_query": bench_query,
            "truth_label": truth_label,
            "hallu_label": hallu_label,
        }

        start_time_hydra = time.time() # TODO: change a name

        # Enter conversation
        if args.sys == 0:
            """VLM answers the benchmark question"""
            owl_res = get_owl_res(img_path, bench_query)
            sample["final_answer"] = owl_res

        elif args.sys == 1:
            """HalluAgent answers the benchmark question"""
            # NOTE: entering HalluAgent, code in agent_sys.py ...
            sample = exec_hydra_caption(argp, sample, tortureList)
        else:
            logger.error(f"[HalluAgent] [Error] Invalid system part: {args.sys}, should be 0 or 1")
            raise ValueError(f"[HalluAgent] [Error] Invalid system part: {args.sys}, should be 0 or 1")

        end_time_hydra = time.time()
        argw.hydra_time = (end_time_hydra-start_time_hydra) / 60

        # Save answer per question
        answer = {
            "id": qid, 
            "response": sample["final_answer"],
        }
        answer_list.append(answer)
        
        # Write logs to file per question # NOTE: use `qid` not `idx+1`
        with open(log_file % (qid), 'w') as f:
            json_sample = json.dumps(sample, indent=4)
            f.write(json_sample+"\n")
    logger.info("\n[HalluAgent] [INFO] Finish asking questions")

    # Write answers to file for evaluation
    with open(answer_file, 'w') as f:
        json.dump(answer_list, f, indent=4)

    logger.info(f"\n[HalluAgent] [INFO] Answers and logs are saved to the folder: {output_folder}")

    # Calculate runtime
    end_time = time.time()
    argw.total_runtime = (end_time-start_time) / 60 # in minutes

    # Evaluate +torture list
    evaluate_amber(args, argw, output_folder, answer_file, question_file, tortureList)
    logger.info(f"\n[HalluAgent] [INFO] Evaluation results are saved to the folder: {output_folder}")

    # Log to wandb
    if args.wandb:
        wandb.log(
            {
                "attack": args.attack,
                "defense": args.defense,
                "subset": args.subset,
                "chair": argw.chair,
                "cover": argw.cover,
                "hal": argw.hal,
                "cog": argw.cog,
                "total runtime (min)": argw.total_runtime,
                "load model time (min)": argw.load_model_time,
                "hydra time (min)": argw.hydra_time,
            }
        )
        wandb.finish()
    else:
        logger.info("\n[HalluAgent] ========= System runtime information =========")
        logger.info(f"[HalluAgent] [INFO] load model time (min): {argw.load_model_time}")
        logger.info(f"[HalluAgent] [INFO] hydra time (min): {argw.hydra_time}")
        logger.info(f"[HalluAgent] [INFO] total runtime (min): {argw.total_runtime}")
        logger.info("[HalluAgent] [INFO] System finished!")
        logger.info("\n")

if __name__ == '__main__':
    main_amber()