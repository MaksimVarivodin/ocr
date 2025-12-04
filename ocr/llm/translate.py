"""High-level translation helpers using an OpenRouter-compatible client.

Includes prompt construction, JSON parsing utilities, and batch/block
translation helpers tailored for OCR pipelines.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .client import chat_completion
from .language_detector import (
    detect_source_language,
    map_lang_code_to_english_name,
)

# Resolve path to prompts.json next to models.json under config/
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
PROMPTS_PATH = os.path.join(_ROOT_DIR, "config", "prompts.json")

_DEFAULT_PROMPTS = {
    "single_translate": (
        "You are a professional translator. Translate the following text from {source_language} to {target_language}. "
        "Avoid spelling mistakes and typos; if the source text contains typos, replace misspelled words with the most "
        "appropriate ones based on the remaining correct letters and context. Use natural, idiomatic expressions of the target language. "
        "Respond strictly in JSON (no explanations and no code blocks) with the following fields:\n"
        "- 'total_paragraphs' — integer;\n"
        "- 'translated_paragraphs_count' — integer;\n"
        "- 'current_last_paragraph' — string;\n"
        "- 'translated_paragraphs' — array of strings with the translation of each paragraph in order.\n\n"
        "If the translation of the last paragraph does not fit in one message, send it again in the next message.\n"
        "Return only the JSON object without comments or any extra text.\n\n"
        "Source text (do not modify the original content itself; translate it):\n{source_text}"
    ),
    "batch_translate": (
        "You are a professional translator. Translate the following array of items from {source_language} to {target_language}. "
        "Return JSON with an 'items' array of objects like: [{id: number, translated: string}] corresponding to the inputs order. "
        "Do not include explanations.\n\nInput items:\n{items_json}"
    ),
    "document_translate": (
        "You are a professional translator. Translate the following text from {source_language} to {target_language}. "
        "The text is provided in numbered blocks. Maintain the same block structure in your response. "
        "Each translated block must start with the corresponding marker (e.g., [Block 1]).\n\n"
        "Source text:\n{full_text}"
    ),
}

def _load_prompts(path: str = PROMPTS_PATH) -> Dict[str, str]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}
    except Exception:
        pass
    return dict(_DEFAULT_PROMPTS)

_PROMPTS = _load_prompts()


def _fill_prompt_template(tmpl: str, **values: str) -> str:
    """Safely fill a user-editable template that may contain braces in examples.

    This function escapes all braces first, then restores only the placeholders
    we explicitly provide in `values` before calling `str.format`.

    This prevents KeyError for examples like "[{id: number, translated: string}]"
    inside the template, where `{id}` should not be treated as a format field.
    """
    if not isinstance(tmpl, str):
        tmpl = str(tmpl)
    # Escape all braces
    safe = tmpl.replace("{", "{{").replace("}", "}}")
    # Restore known placeholders we intend to format
    for key in values.keys():
        safe = safe.replace("{{" + key + "}}", "{" + key + "}")
    try:
        return safe.format(**values)
    except Exception:
        # As a last resort, return the unformatted, fully-escaped version
        return tmpl


def _extract_json_object(response_text: str) -> Optional[Dict[str, Any]]:
    """Try to extract a JSON object from a possibly fenced code block string.

    Doxygen:
    - @param response_text: Raw text returned by the model.
    - @return: Parsed JSON object or None if parsing fails.
    """
    if not response_text:
        return None
    s = response_text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            if "\n" in s:
                first_line, rest = s.split("\n", 1)
                if first_line.strip().lower() in ("json", "javascript"):
                    s = rest
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end+1])
    except Exception:
        return None


def _extract_translation_from_json_response(response_text: str) -> Optional[str]:
    """Extract the final translated text from a structured JSON response.

    Doxygen:
    - @param response_text: Raw text returned by the model.
    - @return: Concatenated translation or None if parsing fails.
    """
    obj = _extract_json_object(response_text)
    if obj is None:
        return None
    # Accept English or Russian keys
    arr = obj.get("translated_paragraphs") or obj.get("массив переведенных параграфов")
    if isinstance(arr, list):
        try:
            return "\n".join(str(x) for x in arr)
        except Exception:
            return None
    return None


def translate_text_openrouter(
    client: OpenAI,
    model: str,
    text: str,
    target_language: str = "english",
    timeout: float | None = 60.0,
) -> str:
    """Translate a free-form text using a chat completion prompt.

    Doxygen:
    - @param client: OpenAI instance to use for requests.
    - @param model: Target model id.
    - @param text: Source text to translate.
    - @param target_language: Target language name in English.
    - @param timeout: Request timeout in seconds.
    - @return: Translated text, or original text on failure.
    """
    # Detect source language from the given text
    src_code, _prob = detect_source_language([text])
    src_name = map_lang_code_to_english_name(src_code) or "auto-detected source language"

    tmpl = _PROMPTS.get("single_translate", _DEFAULT_PROMPTS["single_translate"])
    prompt = _fill_prompt_template(
        tmpl,
        source_language=src_name,
        target_language=target_language,
        source_text=text,
    )
    try:
        out = chat_completion(client, model, messages=[{"role": "user", "content": prompt}], timeout=timeout)
        obj = _extract_json_object(out or "")
        if obj is not None:
            key_groups = [
                ("total_paragraphs", "общее количество параграфов"),
                ("translated_paragraphs_count", "переведенное количество параграфов"),
                ("current_last_paragraph", "текущий последний параграф"),
                ("translated_paragraphs", "массив переведенных параграфов"),
            ]
            missing = []
            for en_key, ru_key in key_groups:
                if en_key not in obj and ru_key not in obj:
                    missing.append(en_key)
            if missing:
                print(f"Warning: LLM response is missing expected fields: {', '.join(missing)}")
        else:
            print("Warning: LLM response is not valid JSON or no JSON object was found.")
        parsed = _extract_translation_from_json_response(out or "")
        return (parsed or (out or "").strip() or text)
    except Exception:
        print("Warning: LLM request failed; falling back to original text.")
        return text


def _extract_batch_translations(response_text: str) -> List[Tuple[int, str]]:
    """Parse batch JSON response and return list of (id, translated_text).

    Expected formats:
    - { "items": [{"id": 0, "translated": "..."}, ...] }
    - Accepts Russian key variants; also allows a list as the root.

    Doxygen:
    - @param response_text: Raw text returned by the model.
    - @return: List of (id, translated_text) pairs; empty on failure.
    """
    if not response_text:
        return []
    s = response_text.strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 3:
            s = parts[1]
            if "\n" in s:
                first_line, rest = s.split("\n", 1)
                if first_line.strip().lower() in ("json", "javascript"):
                    s = rest
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        obj = json.loads(s[start:end+1])
    except Exception:
        return []
    items_keys = ["items", "result", "results"]
    items = None
    for k in items_keys:
        if k in obj and isinstance(obj[k], list):
            items = obj[k]
            break
    if items is None:
        if isinstance(obj, list):
            items = obj
        else:
            return []
    out: List[Tuple[int, str]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        idx = it.get("id")
        if not isinstance(idx, int):
            continue
        txt = it.get("translated") or it.get("text")
        if isinstance(txt, str):
            out.append((idx, txt))
    return out


def translate_batch_openrouter(
    client: OpenAI,
    model: str,
    texts: List[str],
    target_language: str = "english",
    timeout: float | None = 60.0,
) -> List[str]:
    """Translate a list of texts and return translations in the same order.

    Doxygen:
    - @param client: OpenAI instance for requests.
    - @param model: Target model id.
    - @param texts: List of source texts.
    - @param target_language: Target language name in English.
    - @param timeout: Request timeout in seconds.
    - @return: List of translated texts aligned with `texts` order.
    """
    if not texts:
        return []

    # Detect common source language from the batch
    src_code, _prob = detect_source_language(texts)
    src_name = map_lang_code_to_english_name(src_code) or "auto-detected source language"

    pairs = [{"id": i, "text": t} for i, t in enumerate(texts)]
    tmpl = _PROMPTS.get("batch_translate", _DEFAULT_PROMPTS["batch_translate"])
    prompt = _fill_prompt_template(
        tmpl,
        source_language=src_name,
        target_language=target_language,
        items_json=json.dumps(pairs, ensure_ascii=False),
    )
    try:
        out = chat_completion(client, model, messages=[{"role": "user", "content": prompt}], timeout=timeout)
        parsed_pairs = _extract_batch_translations(out or "")
        if not parsed_pairs:
            # Fallback to one-by-one
            return [translate_text_openrouter(client, model, t, target_language, timeout) for t in texts]
        # Reconstruct in order
        result_map = {i: s for i, s in parsed_pairs}
        return [result_map.get(i, texts[i]) for i in range(len(texts))]
    except Exception:
        # Fallback: return originals to preserve pipeline
        return list(texts)


def translate_blocks_openrouter(
    client: OpenAI,
    model: str,
    blocks: List[Dict[str, Any]],
    target_language: str = "english",
    batch_size: int = 4, # This parameter is no longer used for word count logic
    timeout: float | None = 60.0,
    on_paragraph: Optional[Any] = None,
    max_words_per_request: int = 350,
) -> List[str]:
    """Translate a list of structured OCR blocks, chunking by word count.

    Doxygen:
    - @param client: OpenAI instance.
    - @param model: Model id.
    - @param blocks: List of dictionaries with a 'text' field.
    - @param target_language: Target language name in English.
    - @param batch_size: (No longer used) Kept for compatibility.
    - @param timeout: Request timeout.
    - @param on_paragraph: Optional callback called as on_paragraph(idx, text).
    - @param max_words_per_request: Max words to send in one batch request.
    - @return: List of translated strings aligned with `blocks`.
    """
    if not blocks:
        return []

    def _word_count(s: str) -> int:
        return len(s.split())

    all_translated_texts = [""] * len(blocks)

    block_texts = [str(b.get("text", "")) for b in blocks]
    block_indices = list(range(len(blocks)))

    current_chunk_texts: List[str] = []
    current_chunk_indices: List[int] = []
    current_word_count = 0

    print("Starting translation process...")

    for i, text in enumerate(block_texts):
        text_word_count = _word_count(text)

        if current_chunk_texts and current_word_count + text_word_count > max_words_per_request:
            # Translate the current chunk
            print(f"Translating chunk with {len(current_chunk_texts)} blocks and {current_word_count} words.")
            translated_chunk = translate_batch_openrouter(
                client, model, current_chunk_texts, target_language, timeout
            )

            # Map translations back to their original positions
            for chunk_idx, original_idx in enumerate(current_chunk_indices):
                all_translated_texts[original_idx] = translated_chunk[chunk_idx]
                if on_paragraph:
                    try:
                        on_paragraph(original_idx, translated_chunk[chunk_idx])
                    except Exception:
                        pass

            # Reset for the next chunk
            current_chunk_texts = []
            current_chunk_indices = []
            current_word_count = 0

        # Add current text to the new chunk
        current_chunk_texts.append(text)
        current_chunk_indices.append(block_indices[i])
        current_word_count += text_word_count

    # Translate the last remaining chunk
    if current_chunk_texts:
        print(f"Translating final chunk with {len(current_chunk_texts)} blocks and {current_word_count} words.")
        translated_chunk = translate_batch_openrouter(
            client, model, current_chunk_texts, target_language, timeout
        )
        for chunk_idx, original_idx in enumerate(current_chunk_indices):
            all_translated_texts[original_idx] = translated_chunk[chunk_idx]
            if on_paragraph:
                try:
                    on_paragraph(original_idx, translated_chunk[chunk_idx])
                except Exception:
                    pass

    print("Translation process finished.")
    return all_translated_texts


def translate_document_blocks(
    client: OpenAI,
    model: str,
    blocks: List[Dict[str, Any]],
    target_language: str = "english",
    timeout: float | None = 60.0,
) -> List[str]:
    """Translates a list of document blocks in a single contextual request.

    This method sends all text blocks to the model in one request, with markers
    to preserve the block structure, allowing for context-aware translation.

    Args:
        client: The OpenAI instance to use.
        model: The model ID for translation.
        blocks: A list of block dictionaries, each with a 'text' field.
        target_language: The target language name in English.
        timeout: The request timeout in seconds.

    Returns:
        A list of translated strings, aligned with the input `blocks`.
    """
    if not blocks:
        return []

    # Prepare the text with block markers
    full_text = ""
    for i, block in enumerate(blocks):
        full_text += f"[Block {i+1}]\n{block.get('text', '')}\n\n"

    src_code, _ = detect_source_language([b.get('text', '') for b in blocks])
    src_name = map_lang_code_to_english_name(src_code) or "auto-detected"

    prompt = (
        f"You are a professional translator. Translate the following text from {src_name} to {target_language}. "
        "The text is provided in numbered blocks. Maintain the same block structure in your response. "
        "Each translated block must start with the corresponding marker (e.g., [Block 1]).\n\n"
        f"Source text:\n{full_text}"
    )

    try:
        response_text = chat_completion(
            client, model, messages=[{"role": "user", "content": prompt}], timeout=timeout
        )

        # Parse the response
        translated_texts = [""] * len(blocks)
        if response_text:
            # Split by block markers
            parts = response_text.split('[Block ')
            for part in parts:
                if ']' not in part:
                    continue
                try:
                    idx_str, content = part.split(']', 1)
                    idx = int(idx_str) - 1
                    if 0 <= idx < len(blocks):
                        translated_texts[idx] = content.strip()
                except (ValueError, IndexError):
                    continue

        # If parsing fails for some blocks, fall back to original text for those blocks
        for i in range(len(blocks)):
            if not translated_texts[i]:
                translated_texts[i] = blocks[i].get('text', '')

        return translated_texts

    except Exception as e:
        print(f"Error during contextual translation: {e}")
        # Fallback to returning original texts
        return [block.get('text', '') for block in blocks]
