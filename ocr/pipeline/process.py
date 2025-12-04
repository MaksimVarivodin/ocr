"""High-level pipeline: OCR → translation via OpenRouter → render.

This module orchestrates the full flow and provides a single entry point
`process_image_translate` suitable for scripts and notebooks.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pytesseract

from ocr.llm import (
    get_picked_model,
    get_openrouter_client,
    test_model_health,
    translate_document_blocks,
)
from ocr.llm.language_detector import detect_source_language, map_lang_code_to_english_name
from ocr.ocr import (
    preprocess_image_for_ocr,
    ocr_image_boxes,
    choose_blocks_strategy,
)
from ocr.image import remove_text_from_image, remove_text_words_from_image
from ocr.render import add_translated_text


def print_progress_bar(translated_words: int, total_words: int, done_pars: int, total_pars: int, width: int = 10) -> None:
    """Render a colored one-line progress bar (10 fixed segments).

    Doxygen:
    - @param translated_words: Number of words already translated.
    - @param total_words: Total words to translate.
    - @param done_pars: Number of paragraphs translated.
    - @param total_pars: Total paragraphs.
    - @param width: Number of bar segments (default 10).
    """
    total_words = max(1, total_words)
    translated = max(0, min(translated_words, total_words))
    progress = translated / total_words
    segments = max(1, int(width))
    filled = int(progress * segments)
    if translated >= total_words:
        filled = segments
    pending = max(0, segments - filled)
    GREEN = "\x1b[32m"
    RED = "\x1b[31m"
    RESET = "\x1b[0m"
    bar = f"{GREEN}{'█'*filled}{RESET}{RED}{'█'*pending}{RESET} [{done_pars}/{total_pars}]"
    print(f"\r{bar}", end="", flush=True)


def process_image_translate(
    image_path: str,
    target_language: str = 'english',
    merge_strategy: str = 'paragraph',
    output_path: str = 'translated_image.jpg',
    lang: str = 'rus+eng',
    conf_threshold: int = 30,
    ocr_mode: str = 'auto',
    request_timeout: float | None = 60.0,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Run full OCR→Translate→Render pipeline and save output image and texts.

    Doxygen:
    - @param image_path: Path to input image file.
    - @param target_language: Language to translate to (e.g., 'english').
    - @param merge_strategy: One of {'line', 'paragraph', 'auto'}.
    - @param output_path: Output image file name (basename used inside folder).
    - @param lang: Tesseract languages, e.g., 'rus+eng'.
    - @param conf_threshold: Minimum OCR confidence to keep words.
    - @param ocr_mode: 'auto' for preprocessing, 'raw' for direct OCR.
    - @param request_timeout: Timeout per LLM request.
    - @param batch_size: Number of paragraphs per translation request.
    - @return: Dict with keys {'output_path', 'blocks', 'translations'}.
    """
    try:
        _ = pytesseract.get_tesseract_version()
        print("Tesseract found")
    except Exception:
        pass

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    # Use raw or preprocessed image based on ocr_mode
    if ocr_mode == 'raw':
        ocr_input_img = img_bgr
        print("OCR mode: 'raw'. Skipping image preprocessing.")
    else:
        ocr_input_img = preprocess_image_for_ocr(img_bgr)
        print("OCR mode: 'auto'. Applying image preprocessing.")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pre_rgb = cv2.cvtColor(ocr_input_img, cv2.COLOR_BGR2RGB)

    boxes = ocr_image_boxes(ocr_input_img, lang=lang, conf_threshold=conf_threshold)
    print("File found and OCR completed")

    final_blocks = choose_blocks_strategy(boxes, pre_rgb, merge_strategy=merge_strategy)

    # Detect source language from OCR text
    try:
        texts_for_detection = [str(b.get("text", "")) for b in final_blocks]
        src_code, src_prob = detect_source_language(texts_for_detection)
        src_name = map_lang_code_to_english_name(src_code) if src_code else None
        if src_name:
            print(f"Detected source language: {src_name} (code={src_code}, prob={src_prob:.3f})")
        else:
            print("Warning: failed to confidently detect source language; proceeding without explicit source.")
    except Exception as e:
        src_name = None
        print(f"Warning: language detection failed: {e}")

    model, api_key = get_picked_model()
    client = get_openrouter_client(api_key)
    test_model_health(client, model, timeout=min(10.0, request_timeout) if request_timeout else 10.0)
    print("Model connectivity check succeeded")

    # Contextual translation of all blocks at once
    translated_texts = translate_document_blocks(
        client,
        model,
        final_blocks,
        target_language=target_language,
        timeout=request_timeout,
    )

    img_current = remove_text_words_from_image(img_rgb, final_blocks)
    
    # Render all translated blocks onto the cleaned image
    img_final = add_translated_text(img_current, final_blocks, translated_texts)
    
    print("Image filled with translated text")

    base_filename = os.path.basename(image_path)
    base_name_no_ext = os.path.splitext(base_filename)[0] or base_filename
    out_dir = os.path.join(os.path.dirname(image_path), base_name_no_ext)
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception as e:
        print(f"Warning: failed to create output directory '{out_dir}': {e}")
        out_dir = os.path.dirname(image_path)

    out_image_name = os.path.basename(output_path) if output_path else f"translated_{base_filename}"
    output_image_path = os.path.join(out_dir, out_image_name)

    cv2.imwrite(output_image_path, cv2.cvtColor(img_final, cv2.COLOR_RGB2BGR))

    try:
        original_full_text = "\n\n".join([str(b.get('text', '')).strip() for b in final_blocks if str(b.get('text', '')).strip()])
    except Exception:
        original_full_text = "\n\n".join([str(b.get('text', '')) for b in final_blocks])

    try:
        translated_full_text = "\n\n".join([str(t).strip() for t in translated_texts if str(t).strip()])
    except Exception:
        translated_full_text = "\n\n".join([str(t) for t in translated_texts])

    try:
        with open(os.path.join(out_dir, "original.txt"), "w", encoding="utf-8") as f:
            f.write(original_full_text)
    except Exception as e:
        print(f"Warning: failed to write original text: {e}")

    try:
        with open(os.path.join(out_dir, "translated.txt"), "w", encoding="utf-8") as f:
            f.write(translated_full_text)
    except Exception as e:
        print(f"Warning: failed to write translated text: {e}")

    return {
        'output_path': output_image_path,
        'blocks': final_blocks,
        'translations': translated_texts,
    }
