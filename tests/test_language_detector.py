from ocr.llm.language_detector import (
    detect_source_language,
    map_lang_code_to_english_name,
    normalize_and_validate_target_language,
)


def test_detect_source_language_english_text():
    code, prob = detect_source_language(["This is a simple English sentence."])
    assert code in {"en", "en-us", "en-gb"}
    assert prob is None or 0.0 <= prob <= 1.0


def test_map_lang_code_to_english_name():
    assert map_lang_code_to_english_name("en") == "english"
    assert map_lang_code_to_english_name("ru") == "russian"
    assert map_lang_code_to_english_name("en-US") == "english"


def test_normalize_and_validate_target_language_ok():
    assert normalize_and_validate_target_language("english") == "english"
    assert normalize_and_validate_target_language(" English ") == "english"


def test_normalize_and_validate_target_language_reject_non_english_name():
    import pytest

    with pytest.raises(ValueError):
        normalize_and_validate_target_language("английский")

