"""Utility & helper functions."""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

import base64
from io import BytesIO
from PIL import Image
import asyncio

from react_agent.state import HistoryRecord
from typing import List

def get_message_text(msg: BaseMessage) -> str:
    """Get the text content of a message from commercial model."""

    content = msg.content
    if isinstance(content, str):
        return content
    elif isinstance(content, dict):
        return content.get("text", "")
    else:
        txts = [c if isinstance(c, str) else (c.get("text") or "") for c in content]
        return "".join(txts).strip()

# TESTING
# NOTE: the model must has the ability to use tools
def load_reasoning_model() -> ChatOllama:
    """Load a chat model from ollama."""

    model = ChatOllama( 
        # base_url="127.0.0.1:11438", # depend on ollama server
        # model="granite3.2:8b-instruct-fp16",
        # model="qwen2.5:7b-instruct-fp16", # chat model
        base_url="127.0.0.1:11439", # depend on ollama server (qwq)
        model="qwq", # reasoning model
        temperature=0.5,
        # tool_choice="vision_model"  # Forces the model to use one of the provided tools
    )

    return model

# TODO: remove this function
def load_tool_model() -> ChatOllama:
    """Load a chat model from ollama."""

    model = ChatOllama( 
        base_url="127.0.0.1:11437", # depend on ollama server
        model="granite3.2:8b",
        temperature=0.5
    )
    
    return model

def load_llava15(temp=0.1) -> ChatOllama:
    """Load a vision model from ollama."""

    model = ChatOllama( 
        base_url="127.0.0.1:11436", # depend on ollama server
        model="llava:7b-v1.5-fp16", 
        temperature=temp            # dynamic temperature based on the need
    )
    
    return model

def load_gemma3(temp=0.1) -> ChatOllama:
    """Load a vision model from ollama."""

    model = ChatOllama(
        base_url="127.0.0.1:11433", # depend on ollama server
        model="gemma3:12b-it-fp16", 
        temperature=temp            # dynamic temperature based on the need
    )
    
    return model

def load_mistral31(temp=0.1) -> ChatOllama:
    """Load a vision model from ollama."""

    model = ChatOllama(
        base_url="127.0.0.1:11432", # depend on ollama server
        model="mistral-small3.1:24b-instruct-2503-q8_0", 
        temperature=temp            # dynamic temperature based on the need
    )
    
    return model

def load_commercial_model(provider: str, model: str, temp=0.1) -> BaseChatModel:
    """Load a commercial chat model from a fully specified name.

    Args:
        provider (str): Provider of the model.
        model (str): Model name.
        temp (float): Temperature for the model.
    """
    
    return init_chat_model(model, model_provider=provider, temperature=temp)

# NOTE: Functions for vision models
class VLMPromptTools:
    def __init__(self, question, img_path):
        self.question = question
        self.img_path = img_path
        self.image_b64 = None  # Will be set asynchronously
        
    async def convert_to_base64(self):
        """Convert PIL image to base64 string asynchronously using a thread."""

        def _blocking_image_encode():
            pil_image = Image.open(self.img_path)
            buffered = BytesIO()
            pil_image.save(buffered, format="JPEG")

            return base64.b64encode(buffered.getvalue()).decode("utf-8")

        self.image_b64 = await asyncio.to_thread(_blocking_image_encode)

    def get_multimodal_content(self):
        """Generate the content parts for multimodal input to a vision model."""

        multimodal_content = []
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{self.image_b64}",
        }
        text_part = {
                "type": "text",
                "text": self.question,
        }
        multimodal_content.append(image_part)
        multimodal_content.append(text_part)

        return multimodal_content

# NOTE: functions for reflection
from pydantic import BaseModel, Field

# TODO: maybe remove this
# class Reflection(BaseModel):
#     missing: str = Field(description="Critique of what is missing.")         # describes missing info
#     superfluous: str = Field(description="Critique of what is superfluous.") # describes excess info

class AnswerQuestion(BaseModel):
    """Defines the expected function output schema:
        - answer: ~50 word detailed answer (str)
        - critique: a critique on the answer
        - query: a follow-up question
        - reflection: a reflection object
        - queries: a list of follow-up questions
    """

    answer: str = Field(description="~50 word detailed answer to the question.")
    critique: str = Field(description="Critique of what is missing or superfluous in the answer.")
    query: str = Field(description="1 query for researching improvements to address the critique of your current answer.")
    # reflection: Reflection = Field(description="Your reflection on the initial answer.") # using class Reflection
    # queries: list[str] = Field(
    #     description="3 queries for researching improvements to address the critique of your current answer."
    # )

class ReviseAnswer(AnswerQuestion):
    """Defines the expected function output schema:
        - answer: ~50 word detailed answer (str)
        - critique: a critique on the answer
        - query: a follow-up question
        - reflection: a reflection object
        - queries: a list of follow-up questions
        - references: list of citations for your updated answer

    Note: It is a subset of the AnswerQuestion class
    """

    # answer: str = Field(description="~50 word detailed answer to the question.")
    # critique: str = Field(description="Critique of what is missing or superfluous in the answer.")
    # query: str = Field(description="1 query for researching improvements to address the critique of your current answer.")
    # references: list[str] = Field(description="Citations for your revised answer.")
    # reflection: Reflection = Field(description="Your reflection on the revised answer.")
    # queries: list[str] = Field(
    #     description="1 query for researching improvements to address the critique of your current answer."
    # )

class FinalAnswer(BaseModel):
    """Defines the expected function output schema:
        - answer: ~50 word detailed answer to the question.
    """

    answer: str = Field(description="~50 word detailed answer to the question.")

def format_history_for_prompt(history: List[HistoryRecord]) -> str:
    """Formats the structured history into a string for the LLM prompt."""

    if not history:
        return "No history yet."

    formatted_history = ""
    # The first record is always the initial draft
    init_record = history[0]
    formatted_history += "--- Initial Draft ---\n"
    formatted_history += f"Image Caption: {init_record.visual_info}\n"
    formatted_history += f"Initial Answer based on Caption: {init_record.answer}\n"
    formatted_history += f"Critique of Initial Answer: {init_record.critique}\n"
    formatted_history += "---------------------\n"

    # Subsequent records are from the revision loops
    if len(history) > 1:
        for i, record in enumerate(history[1:]):
            formatted_history += f"--- Revised Draft {i+1} ---\n"
            formatted_history += f"Question to Vision Models: {record.query}\n"
            formatted_history += "Visual Models Responses:\n"
            for model, response in record.visual_info.items():
                formatted_history += f"  - {model}: {response}\n"
            formatted_history += f"Revised Answer: {record.answer}\n"
            formatted_history += f"Critique of Revision: {record.critique}\n"
            # if record.references:
            #     formatted_history += f"References: {', '.join(record.references)}\n"
            formatted_history += "---------------------\n"
            
    return formatted_history