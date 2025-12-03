"""Unified document processing layer (TXT, DOCX, PDF).

Exposes:
- Data model: Document, Page, TextRun, ImageItem
- Buffer manager: BufferManager (stores temporary images under config/buffer)
- Readers: txt, docx, pdf (phase 1: scanned PDF via pdf2image)
- Writers: txt, docx, pdf (pdf via DOCX→PDF conversion)
"""

from .model import Document, Page, TextRun, ImageItem
from .buffer import BufferManager

__all__ = [
    "Document",
    "Page",
    "TextRun",
    "ImageItem",
    "BufferManager",
]
