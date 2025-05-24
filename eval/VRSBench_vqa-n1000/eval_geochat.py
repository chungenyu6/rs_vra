"""
This script asks GeoChat to answer questions on VRSBench VQA 5Q.

Usage:
- Make sure GeoChat is running on port 8000
- Activate conda environment: agent
- Navigate to /ISRAgent folder
- Run: python eval/vrsbench_vqa_5q/eval_geochat.py 

Note:
- The results are saved in the results/vrsbench_vqa_5q/ans_geochat.json file,
which contains both predicted answers and ground truth answers. We will use this file
to evaluate the performance of GeoChat.
"""

import os
import requests
import json
from typing import List, Optional, Any
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage, HumanMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun


# TODO: Set parameters
QUESTION_TYPES = ["object quantity", "object position", "object direction", "object size", "reasoning"] # select question types based on your needs # "object quantity", "object position", "object direction", "object size", "reasoning" 
NUM_QUESTIONS = 20 # range from 1 - 100 (for each type)  # TESTING
IMG_FOLDER = "/home/vol-llm/datasets/ISR/VRSBench/validation/Images_val"
LABEL_FILE = "eval/vrsbench_vqa_5q/label.json"
RESULTS_FILE = "results/vrsbench_vqa_5q/ans_geochat.json"

# Each question type has 100 examples (question_id lists)
object_quantity_list = [1, 4, 9, 14, 18, 26, 30, 36, 40, 43, 46, 49, 54, 59, 62, 67, 71, 74, 78, 80, 85, 89, 90, 102, 103, 106, 108, 111, 124, 127, 131, 134, 135, 138, 144, 146, 152, 155, 159, 161, 168, 171, 174, 180, 185, 189, 194, 197, 204, 206, 210, 215, 217, 223, 226, 229, 234, 235, 243, 248, 255, 256, 261, 265, 268, 269, 272, 278, 283, 287, 291, 295, 300, 304, 311, 315, 321, 327, 331, 342, 345, 357, 360, 365, 368, 373, 378, 382, 386, 387, 390, 397, 405, 411, 425, 433, 444, 448, 451, 459]

object_position_list = [3, 5, 11, 16, 20, 21, 31, 33, 41, 44, 45, 48, 50, 52, 58, 61, 64, 66, 75, 76, 79, 83, 84, 86, 92, 105, 107, 110, 113, 116, 119, 123, 129, 139, 140, 149, 153, 160, 173, 176, 177, 178, 192, 195, 198, 202, 207, 208, 211, 212, 213, 218, 224, 227, 228, 230, 233, 236, 238, 244, 246, 249, 251, 254, 257, 259, 260, 263, 271, 276, 277, 281, 282, 284, 293, 301, 302, 307, 309, 317, 333, 336, 343, 350, 352, 361, 369, 371, 372, 375, 381, 385, 388, 389, 396, 410, 429, 430, 446, 447]

object_direction_list = [13, 290, 346, 358, 362, 393, 406, 662, 841, 869, 916, 1542, 1550, 1564, 1697, 1719, 1797, 1885, 1971, 2060, 2085, 2273, 2402, 2480, 2520, 2583, 2588, 2674, 2718, 2762, 2817, 2833, 2989, 2999, 3021, 3547, 3922, 4447, 5155, 5157, 5297, 5363, 5374, 5494, 5549, 5568, 5704, 5714, 5794, 6147, 6154, 6267, 6380, 6396, 6439, 6443, 6457, 6470, 6508, 6510, 6519, 6645, 6746, 6831, 6836, 6879, 6905, 6953, 7012, 7049, 7266, 7408, 7671, 7766, 7769, 7799, 7815, 7861, 7866, 7894, 7939, 7948, 7991, 8050, 8059, 8231, 8239, 8252, 8267, 8282, 8343, 8390, 8463, 8479, 8489, 8511, 8528, 8549, 8787, 8799]

object_size_list = [22, 96, 120, 187, 188, 221, 286, 399, 537, 584, 604, 611, 622, 627, 629, 650, 667, 709, 711, 725, 796, 828, 900, 938, 962, 1092, 1117, 1162, 1171, 1196, 1210, 1264, 1300, 1309, 1364, 1405, 1482, 1568, 1876, 1926, 1936, 1944, 1959, 1960, 1970, 1975, 1977, 1981, 1999, 2005, 2097, 2098, 2122, 2130, 2133, 2155, 2178, 2181, 2206, 2212, 2215, 2223, 2230, 2242, 2484, 2528, 2541, 2546, 2547, 2552, 2558, 2671, 2679, 2708, 2716, 2730, 2747, 2816, 2821, 2887, 2933, 2937, 2955, 3003, 3015, 3063, 3129, 3160, 3221, 3225, 3226, 3245, 3267, 3337, 3387, 3413, 3440, 3468, 3473, 3504]

reasoning_list = [57, 63, 109, 128, 220, 367, 455, 461, 518, 522, 551, 588, 672, 721, 730, 736, 749, 757, 774, 840, 883, 890, 894, 1005, 1036, 1044, 1076, 1113, 1131, 1223, 1334, 1388, 1500, 1504, 1702, 1749, 2042, 2088, 2126, 2276, 2338, 2464, 2467, 2699, 2702, 2713, 2808, 2858, 3077, 3169, 3177, 3375, 3398, 3469, 3487, 3562, 3564, 3600, 3728, 3899, 3911, 3914, 3915, 3992, 4001, 4201, 4532, 4899, 4911, 5030, 5031, 5055, 5071, 5084, 5095, 5099, 5134, 5142, 5163, 5164, 5176, 5197, 5199, 5205, 5206, 5207, 5270, 5359, 5362, 5400, 5401, 5417, 5418, 5426, 5434, 5458, 5570, 5612, 5619, 5768]

def get_question_ids(question_type, num_questions):
    """Get the list of question_ids in each question type."""

    if question_type == "object quantity":
        return object_quantity_list[:num_questions]
    elif question_type == "object position":
        return object_position_list[:num_questions]
    elif question_type == "object direction":
        return object_direction_list[:num_questions]
    elif question_type == "object size":
        return object_size_list[:num_questions]
    elif question_type == "reasoning":
        return reasoning_list[:num_questions]

class CustomGeoChatModel(BaseChatModel):
    """A custom LangChain chat model that interfaces with the GeoChat FastAPI endpoint."""
    
    api_url: str = "http://localhost:8000/generate"  # URL of your FastAPI endpoint

    @property
    def _llm_type(self) -> str:
        """Return the type of the language model."""
        return "geochat"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a response by sending text and image to the GeoChat API.
        
        Args:
            messages: List of messages, expecting a single HumanMessage with text and image.
            stop: Optional list of stop tokens (not used here).
            **kwargs: Additional parameters like temperature, max_new_tokens, etc.
        """
        # Validate input: expect exactly one HumanMessage
        if len(messages) != 1 or not isinstance(messages[0], HumanMessage):
            raise ValueError("Expected a single HumanMessage with text and image")
        
        message = messages[0]
        
        # Extract text and image from the message content
        text_prompt = None
        image_path = None
        for content in message.content:
            if content["type"] == "text":
                text_prompt = content["text"]
            elif content["type"] == "image_url":
                image_path = content["image_url"]["url"]
        
        if text_prompt is None or image_path is None:
            raise ValueError("Message must contain both text and an image path")
        
        # Extract optional generation parameters from kwargs
        temperature = kwargs.get("temperature", 0.1)
        top_p = kwargs.get("top_p", None)
        max_new_tokens = kwargs.get("max_new_tokens", 256)
        
        # Prepare and send the API request
        try:
            with open(image_path, "rb") as image_file:
                files = {"image": image_file}
                data = {
                    "prompt": text_prompt,
                    "temperature": temperature,
                    "top_p": top_p,
                    "max_new_tokens": max_new_tokens,
                }
                response = requests.post(self.api_url, files=files, data=data)
            
            # Check for successful response
            if response.status_code != 200:
                raise Exception(f"API request failed with status {response.status_code}: {response.text}")
            
            # Parse the JSON response
            result = response.json()
            generated_text = result["response"]
        
        except Exception as e:
            raise Exception(f"Error during API call: {str(e)}")
        
        # Create the LangChain response
        ai_message = AIMessage(content=generated_text)
        generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[generation])

def ask_geochat(usr_msg, img_path):
    """Ask geochat to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    chat_model = CustomGeoChatModel() # add temperature if needed (default is 0.1)
    
    # Create a HumanMessage with text and image
    message = HumanMessage(content=[
        {"type": "text", "text": usr_msg},
        {"type": "image_url", "image_url": {"url": img_path}}
    ])
    
    # Invoke the model and print the response
    response = chat_model.invoke([message]) # add temperature if needed
    return response.content

def evaluate():
    """Evaluate the performance on VRSBench VQA 5Q."""

    # Load label file
    with open(LABEL_FILE, "r") as f:
        labels = json.load(f)
    results = []

    # Evaluate (loop through each question type)
    print(f"Evaluating {len(QUESTION_TYPES)} question types")
    for question_type in QUESTION_TYPES:
        # Get the list of question_ids in this question type and number of questions
        ids = get_question_ids(question_type, NUM_QUESTIONS)

        # Process each image-question pair within the range of NUM_QUESTIONS
        print(f"Processing {len(ids)} questions in question type: {question_type}")
        for id in ids:
            # Find the corresponding entry in labels
            entry = next(item for item in labels if item["question_id"] == id and item["type"].lower() == question_type.lower())
            image_id = entry["image_id"]
            question = entry["question"]
            ground_truth = entry["ground_truth"]
            img_path = os.path.join(IMG_FOLDER, image_id)

            # Ask geochat to answer the question
            prediction = ask_geochat(question, img_path)

            # Save the response
            results.append({
                "image_id": image_id,
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
                "question_id": id,
                "type": entry["type"]
            })

    # Write results to JSON file
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

    # Print result summary
    print(f"Results saved to {RESULTS_FILE}")

# Example usage
if __name__ == "__main__":
    evaluate()