from __future__ import annotations

import os
from typing import List

from PIL import ImageFont

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
CONFIG_DIR = os.path.join(ROOT_DIR, "config")
FONT_LIST_PATH = os.path.join(CONFIG_DIR, "font_list.txt")

SAMPLE_TEXT = "Привет, мир! Azure Machine Learning"


def font_supports_cyrillic(path: str) -> bool:
    """Return True if the font at `path` can render basic Cyrillic text.

    We attempt to load the font and call getlength on a Cyrillic sample; if
    anything fails or returns zero, we treat it as unsupported.
    """
    try:
        font = ImageFont.truetype(path, 20)
    except Exception:
        return False
    try:
        # getlength is available in modern Pillow; fall back to getsize
        if hasattr(font, "getlength"):
            length = font.getlength(SAMPLE_TEXT)
            return length > 0
        else:
            w, _ = font.getsize(SAMPLE_TEXT)
            return w > 0
    except Exception:
        return False


def main() -> None:
    if not os.path.isfile(FONT_LIST_PATH):
        print(f"font_list.txt not found at {FONT_LIST_PATH}")
        return
    with open(FONT_LIST_PATH, "r", encoding="utf-8", errors="ignore") as f:
        names = [line.strip() for line in f if line.strip()]

    # Only consider scalable fonts
    scalable = [n for n in names if n.lower().endswith((".ttf", ".otf", ".ttc"))]
    if not scalable:
        print("No scalable fonts (.ttf/.otf/.ttc) found in font_list.txt")
        return

    windir = os.environ.get("WINDIR") or os.environ.get("SystemRoot") or "C\\Windows"
    fonts_dir = os.path.join(windir, "Fonts")

    kept: List[str] = []
    for name in scalable:
        # Try absolute path first
        if os.path.isabs(name):
            candidates = [name]
        else:
            candidates = [os.path.join(fonts_dir, name)]
        chosen_path = None
        for p in candidates:
            if os.path.exists(p):
                chosen_path = p
                break
        if not chosen_path:
            continue
        if font_supports_cyrillic(chosen_path):
            kept.append(name)

    if not kept:
        print("Warning: no fonts with Cyrillic support detected; leaving file unchanged.")
        return

    kept_sorted = sorted(set(kept), key=str.lower)
    with open(FONT_LIST_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(kept_sorted) + "\n")
    print(f"Filtered font_list.txt; kept {len(kept_sorted)} fonts with Cyrillic support.")


if __name__ == "__main__":
    main()

