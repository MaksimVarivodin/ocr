"""Rendering helpers to draw translated text onto an image.

Uses PIL to layout and render text, then converts back to numpy arrays.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Try to load candidate font names from config/font_list.txt if present
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_CONFIG_DIR = os.path.join(_ROOT_DIR, "config")
_CONFIG_FONTS_DIR = os.path.join(_CONFIG_DIR, "fonts")  # optional extra search dir
_FONT_LIST_PATH = os.path.join(_CONFIG_DIR, "font_list.txt")
# Fallback to old location if not found
_ROOT_FONT_LIST_PATH = os.path.join(_ROOT_DIR, "font_list.txt")


def add_translated_text(img: np.ndarray, boxes: List[Dict[str, Any]], translated_texts: List[str]) -> np.ndarray:
    """Draw translated text into corresponding boxes on the image.

    Doxygen:
    - @param img: Input RGB image array.
    - @param boxes: List of box dicts with fields x, y, width, height, and optional style.
    - @param translated_texts: List of translated strings aligned with `boxes`.
    - @return: New RGB image with text drawn.
    """
    img_with_translation = img.copy()
    img_pil = Image.fromarray(cv2.cvtColor(img_with_translation, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # Default candidates
    CANDIDATE_FONTS = [
        "segoeui.ttf",
        "arial.ttf",
        "calibri.ttf",
        "verdana.ttf",
        "tahoma.ttf",
        "times.ttf",
        "times new roman.ttf",
        "georgia.ttf",
        "DejaVuSans.ttf",
    ]
    SAFE_FONT_EXTS = (".ttf", ".otf", ".ttc")

    def _normalize_font_list(names: List[str]) -> List[str]:
        seen = set()
        scalable: List[str] = []
        rest: List[str] = []
        for name in names:
            key = name.strip()
            if not key:
                continue
            key_lower = key.lower()
            if key_lower in seen:
                continue
            seen.add(key_lower)
            if key_lower.endswith(SAFE_FONT_EXTS):
                scalable.append(key)
            else:
                rest.append(key)
        return scalable + rest

    # Ensure config/font_list.txt exists with defaults if possible
    try:
        if not os.path.isfile(_FONT_LIST_PATH):
            os.makedirs(_CONFIG_DIR, exist_ok=True)
            with open(_FONT_LIST_PATH, "w", encoding="utf-8") as f:
                f.write("\n".join(CANDIDATE_FONTS) + "\n")
    except Exception:
        pass
    # Override/extend from config/font_list.txt if available (fallback to project root font_list.txt)
    try:
        font_list_file = _FONT_LIST_PATH if os.path.isfile(_FONT_LIST_PATH) else (
            _ROOT_FONT_LIST_PATH if os.path.isfile(_ROOT_FONT_LIST_PATH) else None
        )
        if font_list_file:
            with open(font_list_file, "r", encoding="utf-8", errors="ignore") as f:
                names = []
                for line in f:
                    s = line.strip()
                    # take plausible font file names
                    if s and (s.lower().endswith((".ttf", ".otf", ".ttc")) or s.lower().endswith(".fon")):
                        names.append(s)
                if names:
                    CANDIDATE_FONTS = _normalize_font_list(names)
                    if not CANDIDATE_FONTS:
                        CANDIDATE_FONTS = names
    except Exception:
        pass

    def _find_font_path(name: str) -> str | None:
        try:
            if os.path.isabs(name) and os.path.exists(name):
                return name
        except Exception:
            pass
        env_paths = os.environ.get("FONT_PATH", "")
        for base in [p for p in env_paths.split(";") if p.strip()]:
            p = os.path.join(base, name)
            if os.path.exists(p):
                return p
        candidates_dirs = []
        windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot")
        if windir:
            candidates_dirs.append(os.path.join(windir, "Fonts"))
        candidates_dirs.append("C\\Windows\\Fonts")
        candidates_dirs.append(os.getcwd())
        # Also allow project config/fonts directory
        candidates_dirs.append(_CONFIG_FONTS_DIR)
        for d in candidates_dirs:
            try:
                if os.path.isdir(d):
                    try_name_lower = name.lower()
                    for fname in os.listdir(d):
                        if fname.lower() == try_name_lower:
                            full = os.path.join(d, fname)
                            if os.path.exists(full):
                                return full
                    for fname in os.listdir(d):
                        if fname.lower().startswith(os.path.splitext(try_name_lower)[0]):
                            full = os.path.join(d, fname)
                            if os.path.exists(full):
                                return full
            except Exception:
                continue
        return None

    def _load_font_by_name(name: str, size: int) -> ImageFont.FreeTypeFont:
        path = _find_font_path(name)
        try:
            if path:
                return ImageFont.truetype(path, size)
        except Exception:
            pass
        try:
            pil_fonts_dir = os.path.join(os.path.dirname(ImageFont.__file__), "fonts")
            dv_path = os.path.join(pil_fonts_dir, "DejaVuSans.ttf")
            if os.path.exists(dv_path):
                return ImageFont.truetype(dv_path, size)
        except Exception:
            pass
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            return ImageFont.load_default()

    def _measure(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int, int]:
        bbox = draw.textbbox((0, 0), text, font=font)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def _layout_lines(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        words = str(text).split()
        if not words:
            return []
        lines: List[str] = []
        cur = ""
        for w_ in words:
            t = (cur + " " + w_) if cur else w_
            tw, _ = _measure(t, font)
            if tw <= max_width:
                cur = t
            else:
                if cur:
                    lines.append(cur)
                cur = w_
        if cur:
            lines.append(cur)
        return lines

    def _fits_lines(lines: List[str], font: ImageFont.FreeTypeFont, max_height: int, line_spacing: int) -> bool:
        if not lines:
            return True
        line_h = (font.size if hasattr(font, 'size') else 12) + line_spacing
        return len(lines) * line_h <= max_height

    def _fits_text(text: str, font: ImageFont.FreeTypeFont, inner_w: int, inner_h: int, line_spacing: int) -> bool:
        return _fits_lines(_layout_lines(text, font, inner_w), font, inner_h, line_spacing)

    def _max_fitting_font_size(
        text: str,
        font_name: str,
        inner_w: int,
        inner_h: int,
        line_spacing: int,
        low: int,
        high: int,
    ) -> int:
        lo, hi = low, high
        while lo < hi:
            mid = (lo + hi + 1) // 2
            f = _load_font_by_name(font_name, mid)
            if _fits_text(text, f, inner_w, inner_h, line_spacing):
                lo = mid
            else:
                hi = mid - 1
        return lo

    def _draw_outline_text(xy: Tuple[int, int], text: str, font: ImageFont.FreeTypeFont, fill=(0, 0, 0)) -> None:
        x, y = xy
        outline = [(x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)]
        for ox, oy in outline:
            draw.text((ox, oy), text, font=font, fill=(255, 255, 255))
        draw.text((x, y), text, font=font, fill=fill)

    def _draw_justified_line(x: int, y: int, inner_w: int, line: str, font: ImageFont.FreeTypeFont, is_last: bool) -> None:
        words = line.split()
        if not words:
            return
        if is_last or len(words) == 1:
            _draw_outline_text((x, y), line, font)
            return
        words_w = sum(_measure(w, font)[0] for w in words)
        spaces = len(words) - 1
        space_w = max(1, (inner_w - words_w) // spaces)
        cur_x = x
        for idx, w in enumerate(words):
            _draw_outline_text((cur_x, y), w, font)
            if idx < spaces:
                cur_x += _measure(w, font)[0] + space_w

    for box, text in zip(boxes, translated_texts):
        x, y, w, h = int(box['x']), int(box['y']), int(box['width']), int(box['height'])
        padding = int(0.08 * max(w, h))
        inner_w = max(1, w - 2 * padding)
        inner_h = max(1, h - 2 * padding)
        font_name = str(box.get('font_name') or '')
        line_spacing = int(box.get('line_spacing', 2))
        alignment = str(box.get('alignment', 'left')).lower()
        if not font_name:
            for cand in CANDIDATE_FONTS:
                font_name = cand
                break
        best_size = _max_fitting_font_size(str(text), font_name, inner_w, inner_h, line_spacing, low=8, high=max(12, int(0.9 * h)))
        font = _load_font_by_name(font_name, best_size or 12)
        lines = _layout_lines(str(text), font, inner_w)
        line_h = (font.size if hasattr(font, 'size') else 12) + line_spacing
        start_y = y + (h - len(lines) * line_h) // 2
        for li, line in enumerate(lines):
            ty = start_y + li * line_h
            if alignment == 'center':
                tw, _ = _measure(line, font)
                tx = x + (w - tw) // 2
                _draw_outline_text((tx, ty), line, font)
            elif alignment == 'justify':
                _draw_justified_line(x + padding, ty, inner_w, line, font, is_last=(li == len(lines) - 1))
            else:
                _draw_outline_text((x + padding, ty), line, font)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
