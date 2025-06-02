"""This module provides example tools for web scraping and search functionality.

It includes a basic Tavily search function (as an example)

These tools are intended as free examples to get started. For production use,
consider implementing more robust and specialized tools tailored to your needs.
"""

from typing import Any, Callable, List, Optional, cast

from langchain_tavily import TavilySearch  # type: ignore[import-not-found]

from react_agent.configuration import Configuration
import react_agent.utils as utils
import react_agent.call_geochat as call_geochat

from langchain_core.messages import HumanMessage
from langchain_core.tools import InjectedToolArg
from langchain_core.runnables import RunnableConfig
from typing_extensions import Annotated


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results.

    This function performs a search using the Tavily search engine, which is designed
    to provide comprehensive, accurate, and trusted results. It's particularly useful
    for answering questions about current events.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# Use chat model as tool
async def commonsense_reasoner(
    question: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Provides commonsense reasoning for the given question."""

    # Instantiate a chat model for this tool
    model = utils.load_tool_model()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [HumanMessage(content=question)],
        config=config
    )
    # Return the assistant's reply text
    return response.content

# NOTE: llava 1.5
async def llava15(
    question: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Provides visual information for the given question."""

    img_path = Configuration.from_context().img_path

    # Instantiate a chat model for this tool
    model = utils.load_llava15(temp=0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(question, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [HumanMessage(content=multimodal_content)],
        config=config
    )

    # Return the assistant's reply text
    return response.content

# NOTE: gemma3
async def gemma3(
    question: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Provides visual information for the given question."""

    img_path = Configuration.from_context().img_path

    # Instantiate a chat model for this tool
    model = utils.load_gemma3(temp=0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(question, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [HumanMessage(content=multimodal_content)],
        config=config
    )

    # Return the assistant's reply text
    return response.content

# NOTE: mistral small 3.1
async def mistral31(
    question: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Provides visual information for the given question."""

    img_path = Configuration.from_context().img_path

    # Instantiate a chat model for this tool
    model = utils.load_mistral31(temp=0.1)

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(question, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [HumanMessage(content=multimodal_content)],
        config=config
    )

    # Return the assistant's reply text
    return response.content

# NOTE: geochat
async def geochat(
    question: str,
    *,
    config: Annotated[RunnableConfig, InjectedToolArg]
) -> str:
    """Provides RS-related visual information for the given question."""

    img_path = Configuration.from_context().img_path

    # Instantiate a chat model for this tool
    model = call_geochat.CustomGeoChatModel()

    # Create a HumanMessage with text and image
    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": img_path}}
    ])
    
    # Invoke with a list of BaseMessage, supplying the injected config
    response = await model.ainvoke(
        [message],
        config=config
    )

    # Return the assistant's reply text
    return response.content


# Add tools in this list
# TOOLS: List[Callable[..., Any]] = [search]
# TOOLS: List[Callable[..., Any]] = [commonsense_reasoner]
# TOOLS: List[Callable[..., Any]] = [vision_model]
# TOOLS: List[Callable[..., Any]] = [geochat]
TOOLS: List[Callable[..., Any]] = [mistral31, geochat] # TESTING
# TOOLS: List[Callable[..., Any]] = [llava15, gemma3, geochat]
