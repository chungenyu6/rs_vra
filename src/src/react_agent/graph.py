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
from react_agent.state import InputState, State, HistoryRecord
from react_agent.tools import TOOLS
import react_agent.utils as utils
import react_agent.call_geochat as call_geochat

GRAPH_NAME = "GeoChat-Reflexion-React" 
# GRAPH_NAME = "LLaVA1.5-Reflexion-React"
# GRAPH_NAME = "Gemma3-Reflexion-React"
# GRAPH_NAME = "MistralSmall3.1-Reflexion-React"
# GRAPH_NAME = "GeoChat_LLaVA1.5-Reflexion-React"
# GRAPH_NAME = "GeoChat_LLaVA1.5_Gemma3-Reflexion-React"

# NOTE: if agent see from state.messages that includes "<think>", it might perturb its behavior
# To avoid this, we use history to record the necessary information
# NOTE: llm agent not always output structured output
def record_history(state: State) -> dict:
    """
    Archives the data from the current loop into a HistoryRecord,
    appends it to the main history, and clears the transient fields.
    """

    # Assemble the record from the clean, current state fields
    new_record = HistoryRecord(
        query=state.current_query,
        visual_info=state.current_visual_info,
        answer=state.current_answer,
        critique=state.current_critique,
        # references=state.current_references
    )

    updated_history = state.history + [new_record]

    # Return the updated history AND clear the current fields
    return {
        "history": updated_history,
        "current_query": "",
        "current_visual_info": {},
        "current_answer": "",
        "current_critique": "",
        # "current_references": [],
    }

async def get_caption(state: State) -> Dict[str, List[AIMessage]]:
    """captioner node: Ask LVLM to generate a caption."""

    model = call_geochat.CustomGeoChatModel() # NOTE: the captioner is RS-LVLM

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

    # Extract visual information from the response
    visual_info = response.content

    return {
        "messages": [response],
        "current_visual_info": visual_info  # for history record
    }

async def draft_respond(state: State) -> Dict[str, List[AIMessage]]:
    """drafter node: Ask the LLM to generate an initial response based on the caption."""

    config = Configuration.from_context()
    model = utils.load_reasoning_model().bind_tools(
        [utils.AnswerQuestion], # like TOOLS list format, make sure structured output
        tool_choice=utils.AnswerQuestion.__name__
    )
    
    sys_msg = config.drafter_sys_prompt.format(
        function_name=utils.AnswerQuestion.__name__,
        time=datetime.now(tz=UTC).isoformat()
    )
    usr_msg = config.drafter_usr_prompt.format(
        usr_question=state.messages[0], # initial user question is always the first message
        image_caption=state.current_visual_info
    )
    
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg}, 
                # *state.messages, # pass the entire noisy message
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    # Extract answer and critique from the response
    if response.tool_calls:
        print("Drafter successfully called the AnswerQuestion tool.")
        tool_call = response.tool_calls[0]['args']
        answer = tool_call.get('answer', '')
        critique = tool_call.get('critique', '')
    else:
        print("Warning: Drafter failed to call AnswerQuestion tool.")
        answer = response.content # Use text content as fallback
        critique = "No critique generated due to tool call failure."

    # Record to history after this
    return {
        "messages": [response],
        "current_answer": answer,        # for history record
        "current_critique": critique     # for history record
    }

async def send_query(state: State) -> Dict[str, List[AIMessage]]:
    """questioner node: Send a question from the latest response to the geochat tool."""
    
    config = Configuration.from_context()
    model = utils.load_reasoning_model().bind_tools(TOOLS)

    # Gather the query from the last ToolMessages
    last_message = state.messages[-1]
    current_query = ""
    if last_message.tool_calls:
        tool_args = last_message.tool_calls[0]['args']
        current_query = tool_args.get('query')
    if not current_query:
        # This can happen if the last tool call had neither key, or no tool call.
        # We need a fallback. We could stop, or ask the model to generate a query.
        usr_query = state.messages[0].content
        current_query = f"Describe the image in detail, focusing on elements that might be relevant to the question: {usr_query}"

    sys_msg = config.questioner_sys_prompt.format(
        time=datetime.now(tz=UTC).isoformat(),
    )
    usr_msg = config.questioner_usr_prompt.format(query=current_query)

    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}
            ]
        ),
    )

    return {
        "messages": [response],
        "current_query": current_query      # for history record
    }

async def revise_respond(state: State) -> Dict[str, List[AIMessage]]:
    """reviser node: Ask the LLM to critique the last draft given
    the outputs from vision tools, enumerate missing/superfluous aspects,
    and produce a refined response.
    """

    config = Configuration.from_context()
    
    model = utils.load_reasoning_model().bind_tools(
        [utils.ReviseAnswer], # like TOOLS list format, make sure structured output
        tool_choice=utils.ReviseAnswer.__name__
    )

    # --- 1. LOAD THE HISTORY ---
    # This is the crucial step. We format our structured memory.
    draft_history = utils.format_history_for_prompt(state.history)

    # --- 2. GATHER VISUAL INFO FOR THE *CURRENT* STEP ---
    visual_info = {}
    draft_history += "--- Latest Draft ---\n"
    draft_history += f"Query to Vision Models: {state.current_query}\n"
    for msg in reversed(state.messages):
        if isinstance(msg, ToolMessage):
            visual_info[msg.name] = msg.content
            draft_history += "Visual Information Received:\n"
            draft_history += f"  - {msg.name}: {msg.content}\n"
        elif isinstance(msg, AIMessage):
            break
    draft_history += "---------------------\n"

    # --- 3. CREATE A CLEAN, CONTEXT-RICH PROMPT ---
    sys_msg = config.revisor_sys_prompt.format(
        function_name=utils.ReviseAnswer.__name__,
        time=datetime.now(tz=UTC).isoformat()
    )
    # The user message now contains the clean history.
    usr_msg = config.revisor_usr_prompt.format(
        usr_question=state.messages[0], # initial user question is always the first message
        draft=draft_history,
    )

    # --- 4. INVOKE THE MODEL (WITHOUT the noisy *state.messages) ---
    response = cast(
        AIMessage,
        await model.ainvoke(
            [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": usr_msg}
            ]
        ),
    )
    response.name = "revisor" # set response name for stopping loop condition

    # --- 5. SAVE THE OUTPUT TO THE CURRENT STATE FOR ARCHIVING ---
    # Extract the results from the tool call
    if response.tool_calls:
        # The model called the tool as expected
        print("Revisor successfully called the ReviseAnswer tool.")
        tool_call = response.tool_calls[0]['args']
        answer = tool_call.get('answer', '')
        critique = tool_call.get('critique', '')
        # references = tool_call.get('references', [])
    else:
        # The model failed to call the tool, provide default values and log a warning.
        print("Warning: Revisor failed to call ReviseAnswer tool. Using response content as answer.")
        answer = response.content # Use the text content as a fallback
        critique = "No critique generated due to tool call failure."

    # Record to history after this
    return {
        "messages": [response],
        "current_visual_info": visual_info,
        "current_answer": answer,
        "current_critique": critique,
        # "current_references": references
    }

def finalize_response(state: State) -> Dict[str, List[AIMessage]]:
    """Provide a final response to the user."""

    # Extract the answer from the last revision
    final_answer = state.history[-1].answer
    
    # Create an AIMessage with the final answer
    response = AIMessage(content=final_answer)

    return {"messages": [response]}

# Build the Reflexion graph
builder = StateGraph(State, input=InputState, config_schema=Configuration)

####################################################
# NOTE: agent version: 
# rs_vra-rm1-vm1/2/3/4-aa1-ri3
# rs_vra-rm2-vm12-aa2-ri3
# rs_vra-rm2_1-vm123-aa2-ri3
# rs_vra-rm1-vm1/2/3/4-aa3-ri3
####################################################
builder.add_node("init_record", record_history)
builder.add_node("record_history", record_history)
builder.add_node("captioner", get_caption)
builder.add_node("drafter", draft_respond)
builder.add_node("questioner", send_query)
builder.add_node("vision_model", ToolNode(TOOLS))
builder.add_node("revisor", revise_respond)
builder.add_node("final_response", finalize_response)

builder.add_edge("__start__", "captioner")
builder.add_edge("captioner", "drafter")
builder.add_edge("drafter", "init_record")
builder.add_edge("init_record", "questioner") # loop start
builder.add_edge("questioner", "vision_model")
builder.add_edge("vision_model", "revisor")
builder.add_edge("revisor", "record_history")

# Decide whether to loop or finish
def loop_or_end(state: State) -> Literal["questioner", "final_response"]:
    # Count how many revise steps have happened so far
    config = Configuration.from_context()
    rev_count = sum(1 for m in state.messages if getattr(m, "name", None) == "revisor")
    print(f"rev_count: {rev_count}")
    return "questioner" if rev_count < config.max_reflexion_iters else "final_response"

builder.add_conditional_edges("record_history", loop_or_end)
builder.add_edge("final_response", "__end__")

# Compile into an executable graph
graph = builder.compile(name=GRAPH_NAME, debug=True)
