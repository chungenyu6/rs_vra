"""
Define a custom Reasoning and Action agent.
Works with a chat model with tool calling support.
"""

from datetime import UTC, datetime
from typing import Dict, List, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from react_agent.configuration import Configuration
from react_agent.state import InputState, State
from react_agent.tools import TOOLS, vision_model
import react_agent.utils as utils
import react_agent.call_geochat as call_geochat

GRAPH_NAME = "GeoChat-Reflexion-React"

async def get_rs_caption(state: State) -> Dict[str, List[AIMessage]]:
    """rs_captioner node: Ask RS-VLM to generate a caption."""

    model = call_geochat.CustomGeoChatModel()

    # TODO: change this later
    question = "Describe every details in the image."
    img_path = Configuration.from_context().img_path 
    message = HumanMessage(content=[
        {"type": "text", "text": question},
        {"type": "image_url", "image_url": {"url": img_path}}
    ])
    
    # Invoke with a list of BaseMessage, supplying the injected config
    response = cast(
        AIMessage, 
        await model.ainvoke([message])
    )

    return {"messages": [response]}

async def get_caption(state: State) -> Dict[str, List[AIMessage]]:
    """captioner node: Ask VLM to generate a caption."""

    model = utils.load_vision_model(temp=0.1)

    # TODO: change this later
    question = "Describe every details in the image."
    img_path = Configuration.from_context().img_path

    # Get multimodal message 
    vlm_prompt_tools = utils.VLMPromptTools(question, img_path)
    await vlm_prompt_tools.convert_to_base64()  # async base64 encoding to avoid BlockingError
    multimodal_content = vlm_prompt_tools.get_multimodal_content()

    # Invoke with a list of BaseMessage
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "user", "content": multimodal_content}
            ]
        ),
    )

    return {"messages": [response]}

async def draft_respond(state: State) -> Dict[str, List[AIMessage]]:
    """drafter node: Ask the LLM to generate an initial response based on the caption."""

    config = Configuration.from_context()
    model = utils.load_reasoning_model()

    sys_msg = config.drafter_sys_prompt.format(
        time=datetime.now(tz=UTC).isoformat()
    )
    usr_msg = config.drafter_usr_prompt.format(
        function_name=utils.AnswerQuestion.__name__
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg}, 
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    return {"messages": [response]}

async def send_query(state: State) -> Dict[str, List[AIMessage]]:
    """inquirer node: Send a question from the latest response to the geochat tool."""
    
    model = utils.load_reasoning_model().bind_tools(TOOLS, tool_choice="geochat") # tool_choice enforces the model to use one of the provided tools

    usr_msg = "Extract one question from the latest response and invoke the tool with the question.<tool_call>" # TEST: not sure if need <tool_call> tag

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    return {"messages": [response]}

async def revise_respond(state: State) -> Dict[str, List[AIMessage]]:
    """reviser node: Ask the LLM to critique the last draft given
    the tool outputs, enumerate missing/superfluous aspects,
    and produce a refined response.
    """

    config = Configuration.from_context()
    model = utils.load_reasoning_model().bind_tools(TOOLS, tool_choice="vision_model")

    sys_msg = config.revisor_sys_prompt.format(
        time=datetime.now(tz=UTC).isoformat()
    )
    usr_msg = config.revisor_usr_prompt.format(
        function_name=utils.ReviseAnswer.__name__
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg}, 
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )
    response.name = "revisor" # set name for thiis response (used in loop_or_end)

    return {"messages": [response]}

# TODO: add spokesman node
async def finalize_response(state: State) -> Dict[str, List[AIMessage]]:
    """spokesman node: Ask the LLM to speak the latest response."""

    config = Configuration.from_context()
    model = utils.load_reasoning_model()

    sys_msg = config.spokesman_sys_prompt.format(
        time=datetime.now(tz=UTC).isoformat()
    )
    # usr_msg = config.spokesman_usr_prompt
    usr_msg = config.spokesman_usr_prompt.format(
        function_name=utils.FinalAnswer.__name__
    )

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg}, 
                *state.messages,
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    return {"messages": [response]}

# Build the Reflexion graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)
builder.add_node("rs_captioner", get_rs_caption)
# builder.add_node("captioner", get_caption)
builder.add_node("drafter", draft_respond)
builder.add_node("inquirer", send_query)
builder.add_node("vision_model", ToolNode(TOOLS))
builder.add_node("revisor", revise_respond)
builder.add_node("spokesman", finalize_response)

builder.add_edge("__start__", "rs_captioner")
# builder.add_edge("__start__", "captioner")
builder.add_edge("rs_captioner", "drafter")
builder.add_edge("drafter", "inquirer")
builder.add_edge("inquirer", "vision_model")
builder.add_edge("vision_model", "revisor")

# Decide whether to loop or finish
def loop_or_end(state: State) -> Literal["inquirer", "spokesman"]:
    # Count how many revise steps have happened so far
    config = Configuration.from_context()
    rev_count = sum(1 for m in state.messages if getattr(m, "name", None) == "revisor")
    print(f"rev_count: {rev_count}")
    return "inquirer" if rev_count < config.max_reflexion_iters else "spokesman"

builder.add_conditional_edges("revisor", loop_or_end)
builder.add_edge("spokesman", "__end__")

# TEST: geochat
# builder.add_edge("rs_captioner", "__end__")
# builder.add_edge("vision_model", "__end__")

# Compile into an executable graph
graph = builder.compile(name=GRAPH_NAME, debug=True)
