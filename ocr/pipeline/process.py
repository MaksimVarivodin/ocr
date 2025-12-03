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
    translate_blocks_openrouter,
)
from ocr.ocr import (
    preprocess_image_for_ocr,
    ocr_image_boxes,
    choose_blocks_strategy,
)
from ocr.image import remove_text_from_image
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
    target_language: str = 'английский',
    merge_strategy: str = 'paragraph',
    output_path: str = 'translated_image.jpg',
    tesseract_cmd: Optional[str] = None,
    lang: str = 'rus+eng',
    conf_threshold: int = 30,
    request_timeout: float | None = 60.0,
    batch_size: int = 4,
) -> Dict[str, Any]:
    """Run full OCR→Translate→Render pipeline and save output image and texts.

    Doxygen:
    - @param image_path: Path to input image file.
    - @param target_language: Language to translate to (may be Russian label).
    - @param merge_strategy: One of {'line', 'paragraph', 'auto'}.
    - @param output_path: Output image file name (basename used inside folder).
    - @param tesseract_cmd: Optional absolute path to tesseract.exe.
    - @param lang: Tesseract languages, e.g., 'rus+eng'.
    - @param conf_threshold: Minimum OCR confidence to keep words.
    - @param request_timeout: Timeout per LLM request.
    - @param batch_size: Number of paragraphs per translation request.
    - @return: Dict with keys {'output_path', 'blocks', 'translations'}.
    """
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
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

    pre_bgr = preprocess_image_for_ocr(img_bgr)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pre_rgb = cv2.cvtColor(pre_bgr, cv2.COLOR_BGR2RGB)

    boxes = ocr_image_boxes(pre_bgr, lang=lang, conf_threshold=conf_threshold)
    print("File found and OCR completed")

    final_blocks = choose_blocks_strategy(boxes, pre_rgb, merge_strategy=merge_strategy)

    model, api_key = get_picked_model()
    client = get_openrouter_client(api_key)
    test_model_health(client, model, timeout=min(10.0, request_timeout) if request_timeout else 10.0)
    print("Model connectivity check succeeded")

    img_current = remove_text_from_image(img_rgb, final_blocks)

    rendered_texts: List[str] = [""] * len(final_blocks)

    def _on_par(idx: int, txt: str) -> None:
        nonlocal img_current
        rendered_texts[idx] = txt
        img_current = add_translated_text(img_current, [final_blocks[idx]], [txt])

    translated_texts = translate_blocks_openrouter(
        client,
        model,
        final_blocks,
        target_language,
        batch_size=batch_size,
        timeout=request_timeout,
        on_paragraph=_on_par,
    )

    try:
        for i, t in enumerate(translated_texts):
            preview = (str(t) or "").replace("\n", " ")[:120]
            print(f"[preview {i}] {preview}")
    except Exception:
        pass

    img_final = img_current
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
