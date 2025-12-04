"""LLM (Large Language Model) integration package.

This package provides utilities to work with web LLM providers via
OpenRouter-compatible clients, including request helpers and high-level
translation functions.
"""

from .client import (
    CONFIG_PATH,
    get_picked_model,
    get_openrouter_client,
    chat_completion,
    test_model_health,
)
from .translate import (
    translate_text_openrouter,
    translate_batch_openrouter,
    translate_blocks_openrouter,
    translate_document_blocks,
)

__all__ = [
    "CONFIG_PATH",
    "get_picked_model",
    "get_openrouter_client",
    "chat_completion",
    "test_model_health",
    "translate_text_openrouter",
    "translate_batch_openrouter",
    "translate_blocks_openrouter",
    "translate_document_blocks",
]
