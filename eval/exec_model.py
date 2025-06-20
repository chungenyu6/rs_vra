"""
Execute the model to answer questions with an image-question pair.
"""

########################################################################################
# Standard imports
import pprint
import time
from typing import List, Optional, Any
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage, HumanMessage
from langchain.callbacks.manager import CallbackManagerForLLMRun

# Third-party imports
import requests
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph_sdk import get_sync_client
import json

# Local imports
import react_agent.utils as utils
from logger import logger
########################################################################################

def query_agent(args, usr_msg: str, img_path: str) -> str:
    """Process a single image-question pair."""

    client = get_sync_client(url="http://localhost:2024")
    
    for chunk in client.runs.stream(
        None,       # threadless run
        "agent",    # name of assistant, defined in langgraph.json
        input={     # send user message to the assistant
            "messages": [{
                "role": "human",
                "content": usr_msg,
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
        logger.info(f"Receiving new event of type: {chunk.event}...")
        # Convert chunk.data to a formatted string for logging
        formatted_data = json.dumps(chunk.data, indent=2)
        logger.info(f"Event data:\n{formatted_data}\n")
    
    # Extract the final response
    last_response = chunk.data["final_response"]["messages"][-1]["content"]
    logger.info("----- Extracted final response of agent -----")
    logger.info(last_response)
    logger.info("------------------------------------------------")

    # Add a small delay between requests to avoid overwhelming the server
    time.sleep(1)

    return last_response

async def query_llava15(args, usr_msg: str, img_path: str) -> str:
    """Ask llava1.5 to answer questions with an image-question pair."""

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

    logger.info("----- Extracted last response of llava -----")
    logger.info(response.content)
    logger.info("------------------------------------------------")

    # Return the assistant's reply text
    return response.content

async def query_gemma3(args, usr_msg: str, img_path: str) -> str:
    """Ask gemma3 to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    chat_model = ChatOllama( 
        base_url="127.0.0.1:11433", # depend on ollama server
        model="gemma3:12b-it-fp16",
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

    logger.info("----- Extracted last response of gemma3 -----")
    logger.info(response.content)
    logger.info("------------------------------------------------")

    # Return the assistant's reply text
    return response.content

async def query_mistral31(args, usr_msg: str, img_path: str) -> str:
    """Ask mistral31 to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    chat_model = ChatOllama( 
        base_url="127.0.0.1:11432", # depend on ollama server
        model="mistral-small3.1:24b-instruct-2503-q8_0",
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

    logger.info("----- Extracted last response of mistral31 -----")
    logger.info(response.content)
    logger.info("------------------------------------------------")

    # Return the assistant's reply text
    return response.content

async def query_gemini25flash(args, usr_msg: str, img_path: str) -> str:
    """Ask gemini2.5-flash to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    model = utils.load_commercial_model(
        provider="google_genai",
        model="gemini-2.5-flash-preview-05-20",
        temp=0.1
    )

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(usr_msg, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [HumanMessage(content=multimodal_content)],
    )

    logger.info("----- Extracted last response of gemini -----")
    logger.info(response.content)
    logger.info("------------------------------------------------")

    # Return the assistant's reply text
    return response.content

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

def query_geochat(args, usr_msg, img_path):
    """Query geochat to answer questions with an image-question pair."""

    # Instantiate the custom chat model
    chat_model = CustomGeoChatModel() # add temperature if needed (default is 0.1)
    
    # Create a HumanMessage with text and image
    message = HumanMessage(content=[
        {"type": "text", "text": usr_msg},
        {"type": "image_url", "image_url": {"url": img_path}}
    ])
    
    # Invoke the model and print the response
    response = chat_model.invoke([message]) # add temperature if needed

    logger.info("----- Extracted last response of geochat -----")
    logger.info(response.content)
    logger.info("------------------------------------------------")
    
    return response.content