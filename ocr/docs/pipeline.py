from __future__ import annotations

import os
from typing import Dict, List

from ocr.llm import get_picked_model, get_openrouter_client
from ocr.llm.translate import translate_batch_openrouter
from ocr.pipeline.process import process_image_translate

from .buffer import BufferManager
from .model import Document, TextRun, ImageItem
from .txt import read_txt, write_txt
from .docx_io import read_docx, write_docx
from .pdf_io import read_pdf_scanned


def _detect_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt",):
        return "txt"
    if ext in (".docx",):
        return "docx"
    if ext in (".pdf",):
        return "pdf"
    return "unknown"


def process_document(
    file_path: str,
    target_language: str = "английский",
    out_format: str = "docx",
    debug_buffer: bool = False,
    image_merge_strategy: str = "paragraph",
    request_timeout: float | None = 60.0,
    batch_size: int = 6,
) -> Dict[str, str]:
    """High-level pipeline for documents: read → translate → write DOCX → (optional) PDF.

    - Always assembles a DOCX preserving item order (text, images), then optionally exports to PDF.
    - TXT output concatenates paragraphs and inserts inline image markers.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    doc_type = _detect_type(file_path)
    buffer = BufferManager(debug=debug_buffer)

    try:
        # 1) Read into unified model
        if doc_type == "txt":
            doc: Document = read_txt(file_path)
        elif doc_type == "docx":
            doc = read_docx(file_path, buffer)
        elif doc_type == "pdf":
            # Phase 1: treat as scanned PDF (pages as images)
            doc = read_pdf_scanned(file_path, buffer)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        # 2) Translate text runs using existing LLM helpers
        texts: List[str] = []
        text_indices: List[int] = []
        all_items = doc.iter_items()
        for idx, item in enumerate(all_items):
            if isinstance(item, TextRun):
                texts.append(item.text)
                text_indices.append(idx)

        if texts:
            model, api_key = get_picked_model()
            client = get_openrouter_client(api_key)
            out_texts = translate_batch_openrouter(
                client, model, texts, target_language=target_language, timeout=request_timeout
            )
            # write back
            j = 0
            for idx, item in enumerate(all_items):
                if isinstance(item, TextRun):
                    item.translated_text = out_texts[j] if j < len(out_texts) else item.text
                    j += 1

        # 3) Translate images via existing image pipeline (per image)
        for item in all_items:
            if isinstance(item, ImageItem):
                # process_image_translate returns dict with 'output_image_path'
                try:
                    result = process_image_translate(
                        image_path=item.src_path,
                        target_language=target_language,
                        merge_strategy=image_merge_strategy,
                        output_path=os.path.basename(item.src_path),
                        request_timeout=request_timeout,
                        batch_size=batch_size,
                    )
                    item.translated_path = result.get("output_image_path") or result.get("output_path")
                except Exception:
                    # Fallback: leave original image
                    item.translated_path = item.src_path

        # 4) Write output
        base_dir = os.path.dirname(file_path)
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        out: Dict[str, str] = {}
        if out_format.lower() == "txt":
            out_txt = os.path.join(base_dir, f"{base_name}.translated.txt")
            out["txt"] = write_txt(doc, out_txt)
            return out

        # Always produce DOCX first
        out_docx = os.path.join(base_dir, f"{base_name}.translated.docx")
        out["docx"] = write_docx(doc, out_docx)

        if out_format.lower() == "pdf":
            # Convert DOCX → PDF via docx2pdf if available
            try:
                from docx2pdf import convert
                out_pdf = os.path.join(base_dir, f"{base_name}.translated.pdf")
                convert(out_docx, out_pdf)
                out["pdf"] = out_pdf
            except Exception as e:
                # Fallback: keep DOCX only
                out["pdf_error"] = f"DOCX→PDF conversion failed: {e}"
            return out

        # If requested DOCX → done
        return out
    finally:
        # 5) Cleanup buffer depending on mode
        buffer.cleanup()
