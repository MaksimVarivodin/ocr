from __future__ import annotations

import os
from typing import List

from .model import Document, Page, ImageItem
from .buffer import BufferManager


def _ensure_poppler_env() -> None:
    """Ensure POPPLER_PATH is set so pdf2image can find Poppler binaries.

    Priority:
    1) Respect existing POPPLER_PATH if it points to a valid directory.
    2) Try app_dir/poppler/Library/bin (bundled with app build).
    3) Try parent_of_app_dir/poppler/Library/bin (common dev layout: ..\\poppler).
    """
    import os as _os

    cur = _os.environ.get("POPPLER_PATH")
    if cur and _os.path.isdir(cur):
        return

    # app_dir = project root (three levels up from this file: ocr/docs/pdf_io.py)
    app_dir = _os.path.dirname(_os.path.dirname(_os.path.dirname(__file__)))

    candidates = [
        # Bundled layout (e.g., conda-forge style)
        _os.path.join(app_dir, "poppler", "Library", "bin"),
        _os.path.join(_os.path.dirname(app_dir), "poppler", "Library", "bin"),
        # Classic layout (oschwartz releases)
        _os.path.join(app_dir, "poppler", "bin"),
        _os.path.join(_os.path.dirname(app_dir), "poppler", "bin"),
    ]
    for c in candidates:
        if _os.path.isdir(c):
            _os.environ["POPPLER_PATH"] = c
            break


def read_pdf_scanned(path: str, buffer: BufferManager, dpi: int = 200) -> Document:
    """Render each page of a scanned PDF into an image and return as ImageItems.

    Requires pdf2image and a working Poppler installation in PATH or POPPLER_PATH.
    """
    _ensure_poppler_env()
    try:
        from pdf2image import convert_from_path
    except Exception as e:
        raise RuntimeError("pdf2image is required to process scanned PDFs") from e

    pages_imgs = convert_from_path(path, dpi=dpi)
    pages: List[Page] = []
    for pi, pil_img in enumerate(pages_imgs):
        page = Page(index=pi)
        fname = f"pdf-page-{pi:04d}.png"
        out_path = buffer.path(fname)
        try:
            pil_img.save(out_path)
        except Exception:
            # skip page on failure
            continue
        page.items.append(ImageItem(page_index=pi, order=0, src_path=out_path))
        pages.append(page)
    return Document(pages=pages)
