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
