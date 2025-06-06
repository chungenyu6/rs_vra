from ast import arg
import json
import os
import re
import argparse
import wandb
from logger import logger

def extract_choice(s):
    """
    Extracts the first letter (A-E), case-insensitive, from a string.
    Handles patterns like 'A', '(a)', 'the answer is C)', etc.
    """
    s = str(s).strip()
    # Use re.IGNORECASE to handle both 'a' and 'A'
    # Searches for a letter A-E that might be surrounded by parentheses
    match = re.search(r'\(?([A-E])\)?', s, re.IGNORECASE)
    
    if match:
        # Return the matched letter in uppercase for consistent comparison
        return match.group(1).upper()
        
    # If no match is found for A-E, return an empty string
    return ""

def compute_result(args, argw, results_file, eval_results_file):
    """
    Evaluates a single L2-category from a results file.
    """
    target_l2_category = args.l2category
    # file_path = os.path.expanduser(results_file)

    try:
        if results_file.endswith(".jsonl"):
            with open(results_file, "r") as f:
                data = [json.loads(line) for line in f]
        elif results_file.endswith(".json"):
            with open(results_file, "r") as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {results_file}. Please use .json or .jsonl.")
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing file: {e}")
        return

    correct_count = 0
    total_count = 0
    e_choice_count = 0

    for record in data:
        if record.get('l2_category') == target_l2_category:
            total_count += 1
            ground_truth = str(record.get('label', '')).strip().upper()
            
            # Use the new robust function
            prediction_clean = extract_choice(record.get('prediction', ''))

            if ground_truth == prediction_clean:
                correct_count += 1
            if prediction_clean == 'E':
                e_choice_count += 1

    if total_count == 0:
        logger.error(f"Could not find any records for L2-Category: '{target_l2_category}' in the file.")
        return

    accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
    e_choice_pct = (e_choice_count / total_count * 100) if total_count > 0 else 0

    logger.info("="*60)
    logger.info(f"Evaluation Report for L2-Category: '{target_l2_category}'")
    logger.info("="*60)
    logger.info(f"  - Accuracy:   {accuracy:>6.2f}% ({correct_count}/{total_count})")
    logger.info(f"  - E-Choices:  {e_choice_pct:>6.2f}% ({e_choice_count}/{total_count})")
    logger.info("="*60)

    # Save the evaluation results to a new file
    with open(eval_results_file, 'w') as f:
        f.write(f"Evaluation Report for L2-Category: '{target_l2_category}'\n")
        f.write("="*60 + "\n")
        f.write(f"  - Accuracy:   {accuracy:>6.2f}% ({correct_count}/{total_count})\n")
        f.write(f"  - E-Choices:  {e_choice_pct:>6.2f}% ({e_choice_count}/{total_count})\n")
        f.write("="*60 + "\n")
        f.write("\n")

    # Save result summary to wandb
    if args.wandb == True:
        logger.info("Saving result summary to wandb")
        wandb.log(
            {
                "total_questions": total_count,
                "correct": correct_count,
                "e_choices": e_choice_count,
                "accuracy": accuracy,
                "e_choice_percent": e_choice_pct,
                "total_runtime (min)": argw.total_runtime,
            }
        )
        logger.info("Result summary saved to wandb")
        wandb.finish()
