"""
Entry point and compatibility facade for the OCR → LLM → Render pipeline.

This module exposes a stable API and a CLI suitable for PyInstaller builds.

Packages:
- ocr.llm: OpenRouter/OpenAI client helpers and translation functions
- ocr.ocr: Image preprocessing and OCR box/paragraph grouping
- ocr.image: Image cleanup (remove text)
- ocr.render: Text drawing into image
- ocr.pipeline: High-level orchestration (`process_image_translate`)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from openai import OpenAI

# LLM client and translation helpers
from ocr.llm.client import (
    CONFIG_PATH as CONFIG_PATH,
    get_picked_model,
    get_openrouter_client,
    chat_completion,
    test_model_health,
)
from ocr.llm.translate import (
    translate_text_openrouter,
    translate_batch_openrouter,
    translate_blocks_openrouter,
)

# OCR utilities
from ocr.ocr.reader import (
    build_dataframe_from_tesseract as _build_dataframe_from_tesseract,
    group_words_to_lines as _group_words_to_lines,
    horizontal_overlap as _horizontal_overlap,
    split_lines_into_columns as _split_lines_into_columns,
    finalize_paragraph as _finalize_paragraph,
    merge_lines_to_paragraphs as _merge_lines_to_paragraphs,
    detect_headings as _detect_headings,
    preprocess_image_for_ocr as _preprocess_image_for_ocr,
    ocr_image_boxes as _ocr_image_boxes,
    choose_blocks_strategy as _choose_blocks_strategy,
)

# Image processing and rendering
from ocr.image.processing import (
    get_background_color as _get_background_color,
    remove_text_from_image,
)
from ocr.render.draw import add_translated_text

# High-level pipeline
from ocr.pipeline.process import (
    process_image_translate,
    print_progress_bar as _print_progress_bar,
)

# Document pipeline (TXT, DOCX, PDF)
from ocr.docs.pipeline import process_document

__all__ = [
    # config/client
    "CONFIG_PATH",
    "get_picked_model",
    "get_openrouter_client",
    "chat_completion",
    "test_model_health",
    # translation
    "translate_text_openrouter",
    "translate_batch_openrouter",
    "translate_blocks_openrouter",
    # ocr helpers (prefixed underscores preserved for back-compat)
    "_build_dataframe_from_tesseract",
    "_group_words_to_lines",
    "_horizontal_overlap",
    "_split_lines_into_columns",
    "_finalize_paragraph",
    "_merge_lines_to_paragraphs",
    "_detect_headings",
    "_preprocess_image_for_ocr",
    "_ocr_image_boxes",
    "_choose_blocks_strategy",
    # image ops / rendering
    "_get_background_color",
    "remove_text_from_image",
    "add_translated_text",
    # pipeline
    "process_image_translate",
    "_print_progress_bar",
    # documents
    "process_document",
]


def _cli() -> None:
    """CLI for image or document translation.

    Image mode:
    --image / -i: Path to input image
    --target / -t: Target language (default: английский)
    --merge / -m: Block merge strategy: word|line|paragraph|auto (default: paragraph)
    --out / -o: Output image filename (default: translated_image.jpg)
    --tesseract: Path to tesseract executable (optional)
    --lang: Tesseract languages (default: rus+eng)
    --conf: OCR confidence threshold (default: 30)
    --timeout: Per-request timeout seconds (<=0 means no timeout)
    --batch-size: Paragraphs per request (default: 4)

    Document mode (processed before image mode if provided):
    --file / -f: Path to input document (txt|docx|pdf)
    --out-format: Output format for documents (txt|docx|pdf), default: docx
    --debug-buffer: Keep buffer under config/buffer (default: False)
    """
    import argparse

    parser = argparse.ArgumentParser(description="OCR and translate text on an image using OpenRouter model from config.")
    # document mode
    parser.add_argument("--file", "-f", type=str, help="Path to input document (txt|docx|pdf)")
    parser.add_argument("--out-format", type=str, default="docx", choices=["txt", "docx", "pdf"], help="Output format for document mode (default: docx)")
    parser.add_argument("--debug-buffer", action="store_true", help="Keep buffer directory under config/buffer")
    # image mode
    parser.add_argument("--image", "-i", type=str, help="Path to input image to translate")
    parser.add_argument("--target", "-t", type=str, default="английский", help="Target language for translation (default: английский)")
    parser.add_argument("--merge", "-m", type=str, default="paragraph", choices=["word", "line", "paragraph", "auto"], help="Block merge strategy (default: paragraph)")
    parser.add_argument("--out", "-o", type=str, default="translated_image.jpg", help="Path to save translated image (default: translated_image.jpg)")
    parser.add_argument("--tesseract", type=str, default=os.getenv("TESSERACT_CMD"), help="Path to tesseract executable; if omitted, uses pytesseract defaults or TESSERACT_CMD env var")
    parser.add_argument("--lang", type=str, default="rus+eng", help="Tesseract languages (default: rus+eng)")
    parser.add_argument("--conf", type=int, default=30, help="Confidence threshold for OCR boxes (default: 30)")
    parser.add_argument("--timeout", type=float, default=60.0, help="Per-request timeout in seconds for OpenRouter API (default: 60.0; set 0 or negative for no timeout)")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of paragraphs to send per LLM request (default: 4)")

    args = parser.parse_args()

    # Document mode takes precedence if --file is provided
    if args.file:
        timeout_value = None if args.timeout is not None and args.timeout <= 0 else args.timeout
        result = process_document(
            file_path=args.file,
            target_language=args.target,
            out_format=args.out_format,
            debug_buffer=bool(args.debug_buffer),
            image_merge_strategy=args.merge,
            request_timeout=timeout_value,
            batch_size=args.batch_size,
        )
        # Print produced paths
        for k, v in result.items():
            print(f"{k}: {v}")
        return

    if not args.image:
        print("Please provide either --file (txt|docx|pdf) or --image path to translate.")
        print("Examples:\n  python main.py --file path\\to\\doc.docx --out-format pdf\n  python main.py --image path\\to\\image.png --target английский --merge paragraph")
        raise SystemExit(2)

    timeout_value = None if args.timeout is not None and args.timeout <= 0 else args.timeout
    result = process_image_translate(
        image_path=args.image,
        target_language=args.target,
        merge_strategy=args.merge,
        output_path=args.out,
        tesseract_cmd=args.tesseract,
        lang=args.lang,
        conf_threshold=args.conf,
        request_timeout=timeout_value,
        batch_size=args.batch_size,
    )
    print(f"Saved translated image to: {result['output_path']}")
    print(f"Blocks translated: {len(result['blocks'])}")


if __name__ == "__main__":
    _cli()
