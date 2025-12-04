"""Image cleanup helpers: background estimation and text removal.

These utilities operate on numpy image arrays (RGB/BGR) using OpenCV.
"""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np


def get_background_color(img: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Estimate background color around a rectangle using border median.

    Doxygen:
    - @param img: Input image array (RGB or BGR uint8).
    - @param x: Left coordinate of the rectangle.
    - @param y: Top coordinate of the rectangle.
    - @param w: Width of the rectangle.
    - @param h: Height of the rectangle.
    - @return: Estimated background color as array of 3 integers.
    """
    margin = 10
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(img.shape[1], x + w + margin)
    y_end = min(img.shape[0], y + h + margin)
    region = img[y_start:y_end, x_start:x_end]
    edges = np.concatenate([
        region[0, :].reshape(-1, 3),
        region[-1, :].reshape(-1, 3),
        region[:, 0].reshape(-1, 3),
        region[:, -1].reshape(-1, 3)
    ])
    return np.median(edges, axis=0).astype(int)


def remove_text_from_image(img: np.ndarray, boxes: List[Dict[str, Any]]) -> np.ndarray:
    """Fill given text boxes with estimated background and light blur.

    Doxygen:
    - @param img: Input RGB image array.
    - @param boxes: List of dicts with keys: x, y, width, height.
    - @return: Image with text regions removed/blended.
    """
    img_cleaned = img.copy()
    for box in boxes:
        x, y, w, h = box['x'], box['y'], box['width'], box['height']
        bg_color = get_background_color(img, x, y, w, h)
        cv2.rectangle(img_cleaned, (x, y), (x + w, y + h), bg_color.tolist(), -1)
        roi = img_cleaned[y:y+h, x:x+w]
        if roi.size:
            roi_blurred = cv2.GaussianBlur(roi, (5, 5), 0)
            img_cleaned[y:y+h, x:x+w] = roi_blurred
    return img_cleaned


def remove_text_words_from_image(img: np.ndarray, blocks: List[Dict[str, Any]]) -> np.ndarray:
    """More precise text removal using word-level boxes when available.

    If a block contains a ``words`` list with per-word geometry, those
    rectangles (slightly padded) are used instead of the whole block
    rectangle. This helps avoid leftover glyph fragments and preserves
    more of the background.
    """
    img_cleaned = img.copy()
    height, width = img.shape[:2]
    for block in blocks:
        words = block.get("words")
        if not words:
            # Fallback: use whole block rectangle
            x, y, w, h = int(block["x"]), int(block["y"]), int(block["width"]), int(block["height"])
            bg_color = get_background_color(img, x, y, w, h)
            cv2.rectangle(img_cleaned, (x, y), (x + w, y + h), bg_color.tolist(), -1)
            continue
        for wbox in words:
            x = int(wbox.get("x", 0))
            y = int(wbox.get("y", 0))
            w = int(wbox.get("width", 0))
            h = int(wbox.get("height", 0))
            if w <= 0 or h <= 0:
                continue
            pad = max(1, int(0.12 * max(w, h)))
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(width, x + w + pad)
            y1 = min(height, y + h + pad)
            if x1 <= x0 or y1 <= y0:
                continue
            bg_color = get_background_color(img, x0, y0, x1 - x0, y1 - y0)
            cv2.rectangle(img_cleaned, (x0, y0), (x1, y1), bg_color.tolist(), -1)
            roi = img_cleaned[y0:y1, x0:x1]
            if roi.size:
                roi_blurred = cv2.GaussianBlur(roi, (3, 3), 0)
                img_cleaned[y0:y1, x0:x1] = roi_blurred
    return img_cleaned
