"""Send a message to the assistant (threadless run) in async mode
- Reference: https://langchain-ai.github.io/langgraph/tutorials/langgraph-platform/local-server/#test-the-api
"""

import pprint
from langgraph_sdk import get_sync_client
from typing import List, Tuple
import time
from dataclasses import replace

from react_agent.configuration import Configuration


# TODO: Replace with your own image-question pairs
# Define image-question pairs here
msg = """\
Question: {question}
The choices are listed below: 
{choice_a}
{choice_b}
{choice_c}
{choice_d}
{choice_e}
Select the best answer to the above multiple-choice question based on the image. Respond with only the letter (A, B, C, D, or E) of the correct option. 
The best answer is:
"""
usr_msg = msg.format(question="Where is the yellow crane in the picture?", 
                     choice_a="(A) In the upper right area of the picture", 
                     choice_b="(B) In the upper left area of the picture", 
                     choice_c="(C) In the lower right area of the picture", 
                     choice_d="(D) In the lower left area of the picture", 
                     choice_e="(E) This image doesn't feature the position.")

IMAGE_QUESTION_PAIRS: List[Tuple[str, str]] = [
    # ("Is it a urban area?", "/home/ISRAgent/src/tests/demo_img/05864_0000.png"), # free-form question
    (usr_msg, "/home/vol-llm/datasets/ISR/MME-RealWorld-Lite/img/23204.jpg"), # multiple-choice question
    # ("How many bridges are in the image?", "/home/ISRAgent/src/tests/demo_img/05865_0000.png"),
    # ("Is there a car in the image?", "/home/ISRAgent/src/tests/demo_img/05863_0000.png"),
]

MAX_REFLEXION_ITERS = 1

def query_agent(question: str, img_path: str):
    """Process a single image-question pair."""

    client = get_sync_client(url="http://localhost:2024")
    pp = pprint.PrettyPrinter(indent=2)
    
    print(f"\n{'='*50}")
    print(f"Processing image: {img_path}")
    print(f"Question: {question}")
    print(f"{'='*50}\n")
    
    for chunk in client.runs.stream(
        None,       # threadless run
        "agent",    # name of assistant, defined in langgraph.json
        input={     # send user message to the assistant
            "messages": [{
                "role": "human",
                "content": question,
            }],
        },
        config={    # update the configuration
            "configurable": {
                "img_path": img_path,
                "max_reflexion_iters": MAX_REFLEXION_ITERS
            }
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        pp.pprint(chunk.data)
        print("\n")
    
    # NOTE: only if the last agent is revisor
    # Extract the final response
    final_response = chunk.data["final_response"]["messages"][-1]["content"]
    print("\n----- Extracted final response of revisor -----")
    print(final_response)
    print("-----------------------------------------------\n")

    # Add a small delay between requests to avoid overwhelming the server
    time.sleep(1)

def main():
    """Query the agent with the image-question pairs."""

    for question, img_path in IMAGE_QUESTION_PAIRS:
        query_agent(question, img_path)

if __name__ == "__main__":
    main()