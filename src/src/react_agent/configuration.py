"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Annotated

from langchain_core.runnables import ensure_config
from langgraph.config import get_config

from react_agent import prompts


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the agent."""

    # NOTE: change image path if needed
    img_path: str = field(
        default="ISRAgent/src/tests/demo_img/05863_0000.png",
        metadata={
            "description": "The path to the image file."
        },
    )
    max_reflexion_iters: int = field(
        default=1,
        metadata={
            "description": "The maximum number of reflexion iterations to perform."
        },
    )
    max_search_results: int = field(
        default=5,
        metadata={
            "description": "The maximum number of search results to return for each search query."
        },
    )
    reasoning_model_system_prompt: str = field(
        default=prompts.REASONING_MODEL_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt to use for the agent's interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    drafter_sys_prompt: str = field(
        default=prompts.DRAFTER_SYS_PROMPT,
        metadata={
            "description": "The prompt to use for the agent's initial interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    drafter_usr_prompt: str = field(
        default=prompts.DRAFTER_USR_PROMPT,
        metadata={
            "description": "The prompt to use for the agent's initial interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    revisor_sys_prompt: str = field(
        default=prompts.REVISOR_SYS_PROMPT,
        metadata={
            "description": "The prompt to use for the agent's revision interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    revisor_usr_prompt: str = field(
        default=prompts.REVISOR_USR_PROMPT,
        metadata={
            "description": "The prompt to use for the agent's revision interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    spokesman_sys_prompt: str = field(
        default=prompts.SPOKESMAN_SYS_PROMPT,
        metadata={
            "description": "The prompt to use for the agent's spokesman interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    spokesman_usr_prompt: str = field(
        default=prompts.SPOKESMAN_USR_PROMPT,
        metadata={
            "description": "The prompt to use for the agent's spokesman interactions. "
            "This prompt sets the context and behavior for the agent."
        },
    )
    commercial_model_api: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="anthropic/claude-3-5-sonnet-20240620",
        metadata={
            "description": "The name of the language model to use for the agent's main interactions. "
            "Should be in the form: provider/model-name."
        },
    )

    @classmethod
    def from_context(cls) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        
        try:
            config = get_config()
        except RuntimeError:
            config = None
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        _fields = {f.name for f in fields(cls) if f.init}

        return cls(**{k: v for k, v in configurable.items() if k in _fields})
