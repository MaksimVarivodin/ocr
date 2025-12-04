import json
import os
from typing import Optional

import pytesseract


def _resolve_path(base: str, relative: str) -> str:
    return os.path.abspath(os.path.join(base, relative))


def configure_dependencies() -> Optional[str]:
    """Configure external dependencies (Tesseract, Poppler) from config/dependencies.json."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    deps_path = os.path.join(project_root, "config", "dependencies.json")

    poppler_abs: Optional[str] = None

    if not os.path.exists(deps_path):
        print(f"Warning: dependencies.json not found at {deps_path}")
        return poppler_abs

    try:
        with open(deps_path, "r", encoding="utf-8") as deps_file:
            deps = json.load(deps_file) or {}

        tess_rel = deps.get("tesseract_path")
        if tess_rel:
            tess_abs = _resolve_path(project_root, tess_rel)
            if os.path.exists(tess_abs):
                pytesseract.pytesseract.tesseract_cmd = tess_abs
            else:
                print(f"Warning: Tesseract path from config does not exist: {tess_abs}")

        poppler_rel = deps.get("poppler_path")
        if poppler_rel:
            candidate = _resolve_path(project_root, poppler_rel)
            if os.path.isdir(candidate):
                poppler_abs = candidate
            else:
                print(f"Warning: Poppler path from config does not exist or is not a directory: {candidate}")

    except Exception as exc:
        print(f"Warning: Could not load dependencies from config/dependencies.json: {exc}")

    return poppler_abs
