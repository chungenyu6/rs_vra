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

def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a commercial chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """

    provider, model = fully_specified_name.split("/", maxsplit=1)
    
    return init_chat_model(model, model_provider=provider)


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
        tool_choice="vision_model"  # Forces the model to use one of the provided tools

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

def load_vision_model(temp=0.1) -> ChatOllama:
    """Load a vision model from ollama."""

    model = ChatOllama( 
        base_url="127.0.0.1:11436", # depend on ollama server
        model="llava:7b-v1.5-fp16",
        temperature=temp            # dynamic temperature based on the need
    )
    
    return model


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
# reflection models
from pydantic import BaseModel, Field

class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing.")         # describes missing info
    superfluous: str = Field(description="Critique of what is superfluous.") # describes excess info

class AnswerQuestion(BaseModel):
    """Defines the expected function output schema:
        - answer: ~250 word detailed answer (str)
        - reflection: Reflection object
        - search_queries: list of query strings
    """

    answer: str = Field(description="~50 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer.")
    queries: list[str] = Field(
        description="1 query for researching improvements to address the critique of your current answer."
    )

class ReviseAnswer(AnswerQuestion):
    """Defines the expected function output schema:
        - answer: ~50 word detailed answer to the question
        - reflection: Reflection object
        - references: list of citations for your updated answer
        - search_queries: list of query strings
    """

    answer: str = Field(description="~50 word detailed answer to the question.")
    reflection: Reflection = Field(description="Your reflection on the revised answer.")
    references: list[str] = Field(description="Citations for your revised answer.")
    queries: list[str] = Field(
        description="1 query for researching improvements to address the critique of your current answer."
    )

# NOTE: original prompt in ReviseAnswer
"""Revise your original answer to your question. Provide an answer, reflection, 
cite your reflection with references, and finally
add search queries to improve the answer.
"""

class FinalAnswer(BaseModel):
    """Defines the expected function output schema:
        - answer: ~50 word detailed answer to the question.
    """

    answer: str = Field(description="~50 word detailed answer to the question.")
