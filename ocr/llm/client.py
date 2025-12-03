"""Client utilities for interacting with OpenRouter-compatible LLMs.

This module contains configuration loading and convenience helpers to
create a client and perform simple chat completions.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from openai import OpenAI

# Path to the JSON configuration file with models and keys
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "models.json")


def _load_config(path: str = CONFIG_PATH) -> Dict:
    """Load and return the JSON configuration.

    Doxygen:
    - @param path: Absolute path to the JSON configuration file.
    - @return: Parsed configuration dictionary.
    - @throws FileNotFoundError: If the file is missing.
    - @throws json.JSONDecodeError: If the file content is not valid JSON.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_picked_model(path: str = CONFIG_PATH) -> Tuple[str, str]:
    """Return the selected model id and its API key as a tuple.

    The configuration file must contain the following structure:
    - model_number_picked: integer index into the "models" array
    - models: list of items with fields:
      - provider: string (e.g., "openrouter")
      - model: string (e.g., "openai/gpt-4o")
      - api_key: string

    Doxygen:
    - @param path: Absolute path to the JSON configuration file.
    - @return: (model, api_key) pair.
    - @throws ValueError: If index is invalid or fields are missing.
    """
    cfg = _load_config(path)
    models: List[Dict] = cfg.get("models", [])
    idx = cfg.get("model_number_picked")

    if not isinstance(idx, int):
        raise ValueError("Config must include integer 'model_number_picked'.")
    if idx < 0 or idx >= len(models):
        raise ValueError("'model_number_picked' is out of range for available models.")

    item = models[idx]
    model = item.get("model")
    api_key = item.get("api_key")
    if not model or not api_key:
        raise ValueError("Selected model entry must include both 'model' and 'api_key'.")
    return model, api_key


def get_openrouter_client(api_key: str) -> OpenAI:
    """Create an OpenAI client configured to use OpenRouter.

    Doxygen:
    - @param api_key: API key for the selected model/provider.
    - @return: Configured `OpenAI` client instance.
    """
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )


def chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    timeout: float | None = 60.0,
) -> str:
    """Send a chat completion request and return text content.

    Doxygen:
    - @param client: OpenAI instance created by `get_openrouter_client`.
    - @param model: Target model identifier.
    - @param messages: List of role/content dictionaries for the chat.
    - @param timeout: Request timeout in seconds; None disables timeout.
    - @return: Text content of the first completion choice.
    """
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
    )
    return completion.choices[0].message.content


def test_model_health(client: OpenAI, model: str, timeout: float | None = 10.0) -> None:
    """Perform a lightweight health check request.

    Doxygen:
    - @param client: OpenAI instance.
    - @param model: Model identifier.
    - @param timeout: Request timeout in seconds.
    - @throws RuntimeError: If the request fails.
    """
    try:
        _ = chat_completion(client, model, messages=[{"role": "user", "content": "ping"}], timeout=timeout)
    except Exception as e:
        raise RuntimeError(f"Model health check failed: {e}")
