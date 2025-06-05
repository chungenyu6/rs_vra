"""
Co-author: Gemini-2.5-Pro
Original evaluation code: EarthVQA/train_earthvqa.py
"""

import json
import logging
import numpy as np # Only used if you were to adapt the original metric more directly

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Official question types from data/earthvqa.py - EarthVQADataset.QUESTION_TYPES
# This ensures all types are accounted for in the report, even if not in the JSON.
OFFICIAL_QUESTION_TYPES = [
    'Basic Judging', 'Reasoning-based Judging', 'Basic Counting',
    'Reasoning-based Counting', 'Object Situation Analysis', 'Comprehensive Analysis'
]

class VQAAccMetric:
    """
    Adapted VQA Overall Accuracy metric class.
    It mirrors the logic of VQA_OA_Metric from utils/metric.py
    but is simplified for string inputs and avoids external dependencies like prettytable.
    """
    def __init__(self, ques_classes_list: list, logger=None):
        self.ques_classes_list = ques_classes_list  # Ordered list of all known types
        self.logger = logger if logger else logging.getLogger(__name__)

        # Initialize counts for each question type
        self.cls_true_num = [0] * len(self.ques_classes_list)
        self.cls_total_num = [0] * len(self.ques_classes_list)

    def add_prediction(self, predicted_answer: str, ground_truth_answer: str, question_type_str: str):
        """
        Processes a single prediction.
        Compares string answers directly.
        """
        try:
            # Find the index for this question type to update the correct list entry
            cls_idx = self.ques_classes_list.index(question_type_str)
        except ValueError:
            self.logger.warning(
                f"Question type '{question_type_str}' not in known OFFICIAL_QUESTION_TYPES. "
                f"This prediction will be ignored. Please check your JSON or OFFICIAL_QUESTION_TYPES list."
            )
            return

        self.cls_total_num[cls_idx] += 1
        if predicted_answer == ground_truth_answer:
            self.cls_true_num[cls_idx] += 1

    def get_report(self):
        """
        Calculates and returns the accuracy report as text and a dictionary.
        """
        report_lines = ["\n--- VQA Accuracy Report ---"]
        
        class_accuracies_details = {}
        for idx, ques_type_name in enumerate(self.ques_classes_list):
            correct_count = self.cls_true_num[idx]
            total_count = self.cls_total_num[idx]
            
            accuracy = 0.0
            if total_count > 0:
                accuracy = (correct_count / total_count) * 100
            
            report_lines.append(f"  {ques_type_name}: {accuracy:.2f}% ({correct_count}/{total_count})")
            class_accuracies_details[ques_type_name] = {
                "accuracy_percent": accuracy,
                "correct": correct_count,
                "total": total_count
            }

        overall_correct = sum(self.cls_true_num)
        overall_total = sum(self.cls_total_num)
        
        overall_accuracy_percent = 0.0
        if overall_total > 0:
            overall_accuracy_percent = (overall_correct / overall_total) * 100
        
        report_lines.append(f"  -------------------------")
        report_lines.append(f"  Overall Accuracy (OA): {overall_accuracy_percent:.2f}% ({overall_correct}/{overall_total})")
        report_lines.append("--- End of Report ---\n")

        full_report_text = "\n".join(report_lines)
        
        detailed_results_dict = {
            "class_accuracies": class_accuracies_details,
            "overall_accuracy": {
                "accuracy_percent": overall_accuracy_percent,
                "correct": overall_correct,
                "total": overall_total
            }
        }
        return full_report_text, detailed_results_dict

    def print_summary(self):
        """
        Prints the summary report using the logger.
        """
        report_text, _ = self.get_report()
        # Log each line of the report
        for line in report_text.strip().split('\n'):
            if line.strip(): # Avoid logging empty lines
                 self.logger.info(line)


def calculate_accuracy_from_json(json_file_path: str):
    """
    Calculates VQA accuracy from a JSON file.

    Args:
        json_file_path (str): Path to the input JSON file.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: JSON file not found at {json_file_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {json_file_path}")
        return

    if not isinstance(data, list):
        logging.error("Error: JSON data should be a list of objects.")
        return

    # Initialize the metric calculator with the official question types
    vqa_metric = VQAAccMetric(ques_classes_list=OFFICIAL_QUESTION_TYPES)

    processed_items = 0
    for item in data:
        if not isinstance(item, dict):
            logging.warning(f"Skipping invalid item (not a dictionary): {item}")
            continue

        # Extract necessary fields from your JSON structure
        # Adjust these keys if your JSON has different names
        try:
            image_name = item.get("Image Name", "N/A") # For context in logs
            question_type = item["Question Type"]
            ground_truth_label = item["Label"]
            model_answer = item["Answer"]
        except KeyError as e:
            logging.warning(f"Skipping item due to missing key {e} in item for image '{image_name}'. Item: {item}")
            continue
        
        if not all(isinstance(val, str) for val in [question_type, ground_truth_label, model_answer]):
            logging.warning(
                f"Skipping item for image '{image_name}' due to non-string value in critical fields "
                f"(Question Type, Label, or Answer). GT: '{ground_truth_label}', Pred: '{model_answer}'"
            )
            continue


        vqa_metric.add_prediction(
            predicted_answer=model_answer,
            ground_truth_answer=ground_truth_label,
            question_type_str=question_type
        )
        processed_items += 1
    
    logging.info(f"Processed {processed_items} items from the JSON file.")

    # Print the summary report
    vqa_metric.print_summary()
    
    # You can also get the detailed results as a dictionary if needed elsewhere
    _, detailed_results = vqa_metric.get_report()
    # For example, to save detailed results to another JSON:
    # with open('accuracy_report_details.json', 'w') as f_out:
    #     json.dump(detailed_results, f_out, indent=2)
    # logging.info("Detailed accuracy report saved to accuracy_report_details.json")


if __name__ == "__main__":
    # Replace 'your_vqa_results.json' with the actual path to your JSON file
    json_file_to_evaluate = 'your_vqa_results.json'
    
    # Example: Create a dummy JSON file for testing
    dummy_data = [
        {
            "Image Name": "3582.png",
            "Question Type": "Object Situation Analysis",
            "Question": "What are the types of residential buildings?",
            "Label": "There are commercial buildings",
            "Question ID": 600,
            "Answer": "There are commercial houses"
        },
        {
            "Image Name": "3583.png",
            "Question Type": "Object Situation Analysis",
            "Question": "Another question.",
            "Label": "Correct Answer",
            "Question ID": 601,
            "Answer": "Wrong Answer"
        },
        {
            "Image Name": "3584.png",
            "Question Type": "Basic Judging",
            "Question": "Is there a building?",
            "Label": "Yes",
            "Question ID": 602,
            "Answer": "Yes"
        },
        {
            "Image Name": "3585.png",
            "Question Type": "Unknown Type", # This will be logged as a warning
            "Question": "Some other question.",
            "Label": "A",
            "Question ID": 603,
            "Answer": "A"
        }
    ]
    with open(json_file_to_evaluate, 'w', encoding='utf-8') as f:
        json.dump(dummy_data, f, indent=2)
    logging.info(f"Created a dummy JSON file for testing: {json_file_to_evaluate}")
    
    calculate_accuracy_from_json(json_file_to_evaluate)