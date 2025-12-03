from __future__ import annotations

import os
from typing import List

from .model import Document, Page, TextRun, ImageItem


def _split_paragraphs(text: str) -> List[str]:
    parts: List[str] = []
    buf: List[str] = []
    for line in (text or "").splitlines():
        if line.strip() == "":
            if buf:
                parts.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        parts.append("\n".join(buf).strip())
    if not parts and text:
        parts = [text]
    return parts


def read_txt(path: str) -> Document:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    paras = _split_paragraphs(content)
    page = Page(index=0)
    for i, p in enumerate(paras):
        page.items.append(TextRun(page_index=0, order=i, text=p))
    return Document(pages=[page])


def write_txt(doc: Document, out_path: str) -> str:
    lines: List[str] = []
    for item in doc.iter_items():
        if isinstance(item, TextRun):
            lines.append((item.translated_text or item.text or "").strip())
        elif isinstance(item, ImageItem):
            # inline marker
            name = os.path.basename(item.translated_path or item.src_path)
            lines.append(f"[image: {name}]")
    txt = "\n\n".join([ln for ln in lines if ln is not None])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    return out_path
