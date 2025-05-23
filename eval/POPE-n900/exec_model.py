"""
Execute the model to answer questions with an image-question pair.
"""

######################################################################################
# Standard library imports
import pprint
import time

# Third-party imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph_sdk import get_sync_client

# Local application imports
## Utils
import react_agent.utils as utils
######################################################################################

def query_agent(args, question: str, img_path: str) -> str:
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
                "max_reflexion_iters": args.max_reflexion_iters
            }
        },
        stream_mode="updates",
    ):
        print(f"Receiving new event of type: {chunk.event}...")
        pp.pprint(chunk.data)
        print("\n")
    
    # Extract the last response from spokesman
    last_response = chunk.data["spokesman"]["messages"][-1]["content"]
    print("\n----- Extracted last response of spokesman -----")
    print(last_response)
    print("-----------------------------------------------\n")

    # Add a small delay between requests to avoid overwhelming the server
    time.sleep(1)

    return last_response

async def query_llava(args, usr_msg: str, img_path: str) -> str:
    """Ask llava to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    chat_model = ChatOllama( 
        base_url="127.0.0.1:11436", # depend on ollama server
        model="llava:7b-v1.5-fp16",
        temperature=0.1            # dynamic temperature based on the need
    ) # add temperature if needed (default is 0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(usr_msg, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await chat_model.ainvoke(
        [HumanMessage(content=multimodal_content)],
    )

    # Return the assistant's reply text
    return response.content