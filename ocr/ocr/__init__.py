"""OCR (Optical Character Recognition) utilities.

This package includes preprocessing, text detection, and layout/blocks
grouping helpers built on top of pytesseract and OpenCV.
"""

from .reader import (
    build_dataframe_from_tesseract,
    group_words_to_lines,
    horizontal_overlap,
    split_lines_into_columns,
    finalize_paragraph,
    merge_lines_to_paragraphs,
    detect_headings,
    preprocess_image_for_ocr,
    ocr_image_boxes,
    choose_blocks_strategy,
)

__all__ = [
    "build_dataframe_from_tesseract",
    "group_words_to_lines",
    "horizontal_overlap",
    "split_lines_into_columns",
    "finalize_paragraph",
    "merge_lines_to_paragraphs",
    "detect_headings",
    "preprocess_image_for_ocr",
    "ocr_image_boxes",
    "choose_blocks_strategy",
]
