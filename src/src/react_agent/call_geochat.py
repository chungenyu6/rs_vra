"""
Co-Author: Grok 3
"""

import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import BaseMessage, ChatResult, ChatGeneration, AIMessage, HumanMessage
from typing import List, Optional, Any, Dict
from langchain.callbacks.manager import CallbackManagerForLLMRun

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
        """Generate a response by sending text and image to the GeoChat API.
        
        Args:
            messages: List of messages, expecting a single HumanMessage with text and image.
            stop: Optional list of stop tokens (not used here).
            run_manager: Optional callback manager for the LLM run. (for LangChain callbacks)
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
