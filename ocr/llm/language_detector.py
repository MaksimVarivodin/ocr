from __future__ import annotations

from typing import Iterable, List, Tuple

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0


_LANG_CODE_TO_ENGLISH = {
    "en": "english",
    "ru": "russian",
    "uk": "ukrainian",
    "de": "german",
    "fr": "french",
    "es": "spanish",
    "it": "italian",
    "pt": "portuguese",
    "zh-cn": "chinese",
    "zh-tw": "chinese",
}


def _normalize_texts(texts: Iterable[str]) -> str:
    chunks: List[str] = []
    total_len = 0
    for t in texts:
        if not t:
            continue
        s = str(t).strip()
        if not s:
            continue
        chunks.append(s)
        total_len += len(s)
        if total_len >= 4000:
            break
    return "\n".join(chunks)


def detect_source_language(texts: Iterable[str]) -> Tuple[str | None, float | None]:
    sample = _normalize_texts(texts)
    if not sample:
        return None, None
    try:
        candidates = detect_langs(sample)
    except LangDetectException:
        return None, None
    if not candidates:
        return None, None
    best = max(candidates, key=lambda c: c.prob)
    lang = best.lang
    prob = float(best.prob)
    if lang in ("zh-cn", "zh-tw"):
        code = lang
    else:
        code = lang.split("-")[0]
    return code, prob


def map_lang_code_to_english_name(code: str | None) -> str | None:
    if not code:
        return None
    code = code.lower()
    if code in _LANG_CODE_TO_ENGLISH:
        return _LANG_CODE_TO_ENGLISH[code]
    base = code.split("-")[0]
    return _LANG_CODE_TO_ENGLISH.get(base)


_ALLOWED_TARGET_LANGUAGES = {
    "english",
    "russian",
    "ukrainian",
    "german",
    "french",
    "spanish",
    "italian",
    "portuguese",
    "chinese",
}


def normalize_and_validate_target_language(name: str) -> str:
    if not name:
        raise ValueError(
            "Target language name must be provided in English, e.g. 'english', 'german', 'spanish'."
        )
    norm = str(name).strip().lower()
    if norm not in _ALLOWED_TARGET_LANGUAGES:
        allowed = ", ".join(sorted(_ALLOWED_TARGET_LANGUAGES))
        raise ValueError(
            f"Target language must be specified in English. Got: '{name}'. "
            f"Allowed values: {allowed}."
        )
    return norm


__all__ = [
    "detect_source_language",
    "map_lang_code_to_english_name",
    "normalize_and_validate_target_language",
]

