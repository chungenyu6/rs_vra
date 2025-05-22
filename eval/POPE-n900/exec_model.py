"""
Execute the model to answer questions with an image-question pair.
"""

######################################################################################
# Third-party imports
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage

# Local application imports
## Utils
import react_agent.utils as utils
######################################################################################

async def ask_llava(usr_msg, img_path):
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