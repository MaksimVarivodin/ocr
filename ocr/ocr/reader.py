"""OCR reader and layout utilities built on top of pytesseract and OpenCV.

This module provides:
- Building a cleaned DataFrame from pytesseract output.
- Grouping words to lines and paragraphs.
- Splitting text into columns and detecting headings.
- Preprocessing images for OCR and extracting text boxes.

All functions include Doxygen-style documentation tags.
"""

from __future__ import annotations

from typing import Any, Dict, List
import os
import json

import cv2
import numpy as np
import pandas as pd
import pytesseract
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


_TESSERACT_CONFIGURED = False


def _configure_tesseract():
    """Reads dependency paths from config/dependencies.json and configures pytesseract."""
    global _TESSERACT_CONFIGURED
    if _TESSERACT_CONFIGURED:
        return

    try:
        # The root of the project is three levels up from this file (ocr/ocr/reader.py)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        deps_path = os.path.join(project_root, "config", "dependencies.json")

        if os.path.exists(deps_path):
            with open(deps_path, "r", encoding="utf-8") as f:
                deps = json.load(f)
                tess_path_rel = deps.get("tesseract_path")
                if tess_path_rel:
                    # The path in JSON is relative to the project root
                    tesseract_executable = os.path.abspath(os.path.join(project_root, tess_path_rel))
                    if os.path.exists(tesseract_executable):
                        pytesseract.pytesseract.tesseract_cmd = tesseract_executable
                        _TESSERACT_CONFIGURED = True
                    else:
                        print(f"Warning: Tesseract path from config does not exist: {tesseract_executable}")
    except Exception as e:
        print(f"Warning: Could not configure Tesseract from dependencies.json: {e}")


def build_dataframe_from_tesseract(data: Dict[str, Any]) -> pd.DataFrame:
    """Create and clean a DataFrame from pytesseract.image_to_data output.

    Doxygen:
    - @param data: Dict returned by `pytesseract.image_to_data(..., output_type=Output.DICT)`.
    - @return: Filtered DataFrame with columns including text, confidence, and geometry.
    """
    df = pd.DataFrame(data)
    df['conf'] = pd.to_numeric(df['conf'], errors='coerce').fillna(-1)
    df = df[df['conf'] > 0]
    df['text'] = df['text'].fillna('').str.strip()
    df = df[df['text'] != '']
    df['font_size'] = df['height']
    return df


def group_words_to_lines(df: pd.DataFrame, min_words_in_line: int = 1) -> List[Dict[str, Any]]:
    """Group OCR words into line-level blocks with aggregate metrics.

    Doxygen:
    - @param df: DataFrame produced by `build_dataframe_from_tesseract`.
    - @param min_words_in_line: Minimum words required to keep a line.
    - @return: List of line dictionaries with text, bbox, confidence, and baseline info.
    """
    if df.empty:
        return []
    lines: List[Dict[str, Any]] = []
    group_cols = ['block_num', 'par_num', 'line_num']
    for (_, _, _), g in df.groupby(group_cols):
        g_sorted = g.sort_values('left')
        words = g_sorted['text'].tolist()
        if len(words) < min_words_in_line:
            continue
        text = ' '.join(words)
        x = int(g_sorted['left'].min())
        y = int(g_sorted['top'].min())
        w = int((g_sorted['left'] + g_sorted['width']).max() - x)
        h = int((g_sorted['top'] + g_sorted['height']).max() - y)
        avg_conf = float(g_sorted['conf'].mean())
        font_size = float(g_sorted['font_size'].median())

        def _is_word_alpha(s: str) -> bool:
            s = str(s)
            for ch in s:
                if ch.isalpha():
                    return True
            return False

        word_heights = g_sorted['height'].tolist()
        alpha_mask = [_is_word_alpha(t) for t in g_sorted['text'].tolist()]
        alpha_heights = [int(hh) for hh, m in zip(word_heights, alpha_mask) if m]
        baseline_h = int(min(alpha_heights)) if alpha_heights else int(font_size)
        words_detail = [
            {
                'text': str(t),
                'x': int(l),
                'y': int(tp),
                'width': int(wd),
                'height': int(ht)
            }
            for t, l, tp, wd, ht in zip(
                g_sorted['text'].tolist(),
                g_sorted['left'].tolist(),
                g_sorted['top'].tolist(),
                g_sorted['width'].tolist(),
                g_sorted['height'].tolist(),
            )
        ]
        lines.append({
            'text': text,
            'x': x,
            'y': y,
            'width': w,
            'height': h,
            'confidence': avg_conf,
            'font_size': font_size,
            'baseline_height': baseline_h,
            'words': words_detail,
        })
    return lines


def horizontal_overlap(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    """Compute horizontal overlap ratio of two boxes.

    Doxygen:
    - @param a: First bbox dict with x, y, width, height.
    - @param b: Second bbox dict with x, y, width, height.
    - @return: Overlap ratio in [0, 1].
    """
    ax1, ax2 = a['x'], a['x'] + a['width']
    bx1, bx2 = b['x'], b['x'] + b['width']
    inter = max(0, min(ax2, bx2) - max(ax1, bx1))
    denom = max(1, max(ax2, bx2) - min(ax1, bx1))
    return inter / denom


def split_lines_into_columns(lines: List[Dict[str, Any]], img_width: int, max_cols: int = 4) -> List[List[Dict[str, Any]]]:
    """Split line blocks into columns using KMeans clustering on x-coordinates.

    Doxygen:
    - @param lines: List of line dicts.
    - @param img_width: Image width used as a heuristic for column thresholds.
    - @param max_cols: Maximum number of columns to split into.
    - @return: List of columns, each a list of line dicts.
    """
    if not lines or len(lines) < 3:
        return [sorted(lines, key=lambda d: (d['y'], d['x']))]

    xs = np.array([ln['x'] + ln['width'] / 2 for ln in lines]).reshape(-1, 1)

    # Determine the optimal number of columns (k)
    k_max = min(max_cols, len(lines) // 2)
    if k_max < 2:
        return [sorted(lines, key=lambda d: (d['y'], d['x']))]

    # Find best k for KMeans
    inertias = []
    for k in range(1, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto').fit(xs)
        inertias.append(kmeans.inertia_)

    # If inertia decreases sharply, it indicates a good k
    # We look for an "elbow" in the inertia plot
    try:
        deltas = np.diff(inertias)
        if len(deltas) > 1:
            rel_deltas = np.abs(deltas[:-1] / (deltas[1:] + 1e-6))
            best_k = np.argmax(rel_deltas) + 2 if np.max(rel_deltas) > 1.5 else 1
        else:
            best_k = 1
    except (ValueError, IndexError):
        best_k = 1

    if best_k == 1:
        return [sorted(lines, key=lambda d: (d['y'], d['x']))]

    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto').fit(xs)
    labels = kmeans.labels_

    cols: List[List[Dict[str, Any]]] = [[] for _ in range(best_k)]
    for i, line in enumerate(lines):
        cols[labels[i]].append(line)

    # Sort columns by their average x-position
    avg_x_cols = [np.mean([line['x'] for line in col]) for col in cols]
    sorted_cols = [col for _, col in sorted(zip(avg_x_cols, cols))]

    # Sort lines within each column by y-position
    for i in range(len(sorted_cols)):
        sorted_cols[i] = sorted(sorted_cols[i], key=lambda d: d['y'])

    return sorted_cols


def finalize_paragraph(lines_group: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a group of line dicts into a single paragraph block.

    Doxygen:
    - @param lines_group: Consecutive line dicts forming a paragraph.
    - @return: Paragraph dict with merged text and geometry.
    """
    if not lines_group:
        return {}
    xs = [ln['x'] for ln in lines_group]
    ys = [ln['y'] for ln in lines_group]
    ws = [ln['width'] for ln in lines_group]
    hs = [ln['height'] for ln in lines_group]
    text = '\n'.join([ln['text'] for ln in lines_group])
    return {
        'text': text,
        'x': int(min(xs)),
        'y': int(min(ys)),
        'width': int(max([x + w for x, w in zip(xs, ws)]) - min(xs)),
        'height': int(max([y + h for y, h in zip(ys, hs)]) - min(ys)),
        'font_size': float(np.median([ln.get('font_size', 0) for ln in lines_group])),
    }


def merge_lines_to_paragraphs(
    lines: List[Dict[str, Any]],
    font_tolerance: float = 0.30,
    vertical_gap_ratio: float = 1.1,
    min_overlap: float = 0.45,
) -> List[Dict[str, Any]]:
    """Merge nearby lines into paragraphs based on size and spacing heuristics.

    Doxygen:
    - @param lines: List of line dicts.
    - @param font_tolerance: Allowed relative font size difference to merge.
    - @param vertical_gap_ratio: Max vertical gap relative to baseline to merge.
    - @param min_overlap: Min horizontal overlap to keep lines together.
    - @return: List of paragraph dicts.
    """
    if not lines:
        return []
    # Sort lines primarily by y-coordinate, then x, to process top-to-bottom, left-to-right
    lines_sorted = sorted(lines, key=lambda d: (d['y'], d['x']))
    paragraphs: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for ln in lines_sorted:
        if not current:
            current = [ln]
            continue
        last = current[-1]
        # Check for font size similarity
        same_font = abs(ln['font_size'] - last['font_size']) <= font_tolerance * max(1.0, last['font_size'])
        # Check for vertical proximity
        vgap = ln['y'] - (last['y'] + last['height'])
        small_gap = vgap <= vertical_gap_ratio * max(1, last.get('baseline_height', last['height']))
        # Check for horizontal overlap
        overlap = horizontal_overlap(ln, last) >= min_overlap

        # If lines are similar in font, close vertically, and overlap horizontally, merge them
        if same_font and small_gap and overlap:
            current.append(ln)
        else:
            # Otherwise, finalize the current paragraph and start a new one
            paragraphs.append(current)
            current = [ln]
    if current:
        paragraphs.append(current)

    # Finalize paragraph objects and sort them one last time to ensure final reading order
    final_paragraphs = [finalize_paragraph(grp) for grp in paragraphs]
    return sorted(final_paragraphs, key=lambda p: (p['y'], p['x']))


def detect_headings(paragraphs: List[Dict[str, Any]], heading_factor: float = 1.35) -> List[Dict[str, Any]]:
    """Mark paragraphs that look like headings by font size heuristic.

    Doxygen:
    - @param paragraphs: List of paragraph dicts with `font_size`.
    - @param heading_factor: Multiplier over median font size to mark headings.
    - @return: The same list with an added boolean field `is_heading`.
    """
    if not paragraphs:
        return paragraphs
    sizes = [p['font_size'] for p in paragraphs if p.get('font_size')]
    if not sizes:
        return paragraphs
    median_size = float(np.median(sizes))
    for p in paragraphs:
        p['is_heading'] = p['font_size'] >= heading_factor * median_size
    return paragraphs


def preprocess_image_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    """Preprocess a BGR image to improve OCR accuracy.

    Doxygen:
    - @param img_bgr: Input image in BGR format.
    - @return: Preprocessed BGR image.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 10)
    th = cv2.medianBlur(th, 3)
    # convert back to 3-channel BGR for consistency with downstream
    return cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)


def ocr_image_boxes(img: np.ndarray, lang: str = 'rus+eng', conf_threshold: int = 30) -> List[Dict[str, Any]]:
    """Run Tesseract OCR and return word-level boxes.

    Doxygen:
    - @param img: Input image (BGR or RGB) for OCR.
    - @param lang: Tesseract language(s), e.g. 'rus+eng'.
    - @param conf_threshold: Minimum confidence to keep words.
    - @return: List of word box dicts with text and geometry fields.
    """
    rgb = img if img.ndim == 2 or img.shape[2] == 3 and (img[..., ::-1] is None) else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(rgb, lang=lang, output_type=pytesseract.Output.DICT)
    df = build_dataframe_from_tesseract(data)
    boxes: List[Dict[str, Any]] = []
    for i in range(len(df)):
        row = df.iloc[i]
        boxes.append({
            'text': str(row['text']),
            'x': int(row['left']),
            'y': int(row['top']),
            'width': int(row['width']),
            'height': int(row['height']),
            'confidence': float(row['conf']),
        })
    return boxes


def choose_blocks_strategy(boxes: List[Dict[str, Any]], img_rgb: np.ndarray, merge_strategy: str = 'paragraph') -> List[Dict[str, Any]]:
    """Group OCR word boxes into higher-level blocks according to strategy.

    - boxes: word-level boxes from `ocr_image_boxes` (fallback only)
    - img_rgb: RGB image used to run a fresh `pytesseract.image_to_data` for
      robust line/paragraph grouping (ensures presence of 'block_num',
      'par_num', 'line_num' columns to avoid KeyError).
    - merge_strategy: 'word' | 'line' | 'paragraph' | 'auto'
    """
    if boxes is None:
        boxes = []

    # Preferred path: use pytesseract dataframe to have grouping columns
    line_blocks: List[Dict[str, Any]] = []
    paragraph_blocks: List[Dict[str, Any]] = []
    try:
        data = pytesseract.image_to_data(
            cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            lang='rus+eng',
            output_type=pytesseract.Output.DICT,
        )
        df_full = build_dataframe_from_tesseract(data)
        if not df_full.empty:
            line_blocks = group_words_to_lines(df_full, min_words_in_line=1)
            img_w = int(img_rgb.shape[1]) if img_rgb is not None else 0
            cols = split_lines_into_columns(line_blocks, img_w, max_cols=2)

            # Clear and rebuild paragraph_blocks from sorted columns
            paragraph_blocks = []
            for col in cols:
                paragraph_blocks.extend(merge_lines_to_paragraphs(col))

            # Final sort of all paragraphs from all columns to ensure reading order
            paragraph_blocks = sorted(paragraph_blocks, key=lambda p: (p['y'], p['x']))
            paragraph_blocks = detect_headings(paragraph_blocks)
    except Exception:
        # Fall back to using provided boxes only (no advanced grouping)
        line_blocks = []
        paragraph_blocks = []

    # If no line/paragraph info available, derive trivial blocks from boxes
    if not line_blocks and boxes:
        # Treat each word box as a line
        line_blocks = [
            {
                'text': str(b.get('text', '')),
                'x': int(b.get('x', 0)),
                'y': int(b.get('y', 0)),
                'width': int(b.get('width', 0)),
                'height': int(b.get('height', 0)),
                'confidence': float(b.get('confidence', 0)),
                'font_size': float(b.get('height', 0)),
                'baseline_height': int(b.get('height', 0)),
                'words': [
                    {
                        'text': str(b.get('text', '')),
                        'x': int(b.get('x', 0)),
                        'y': int(b.get('y', 0)),
                        'width': int(b.get('width', 0)),
                        'height': int(b.get('height', 0)),
                    }
                ],
            }
            for b in boxes
        ]
        paragraph_blocks = merge_lines_to_paragraphs(line_blocks)
        paragraph_blocks = detect_headings(paragraph_blocks)

    # Select output by strategy
    ms = (merge_strategy or 'paragraph').lower()
    if ms == 'word':
        return boxes
    if ms == 'line' and line_blocks:
        return line_blocks
    if ms == 'paragraph' and paragraph_blocks:
        return paragraph_blocks
    if ms == 'auto':
        if paragraph_blocks and len(paragraph_blocks) <= max(1, int(0.9 * len(line_blocks or boxes))):
            return paragraph_blocks
        if line_blocks:
            return line_blocks
        return boxes
    # Default fallback
    return paragraph_blocks or line_blocks or boxes
