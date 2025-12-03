import pandas as pd

from ocr.ocr.reader import (
    build_dataframe_from_tesseract,
    group_words_to_lines,
    merge_lines_to_paragraphs,
    split_lines_into_columns,
    finalize_paragraph,
)


def test_build_dataframe_from_tesseract_filters_empty_and_low_conf():
    data = {
        'level': [5, 5, 5],
        'page_num': [1, 1, 1],
        'block_num': [1, 1, 1],
        'par_num': [1, 1, 1],
        'line_num': [1, 1, 1],
        'word_num': [1, 2, 3],
        'left': [10, 30, 50],
        'top': [10, 10, 10],
        'width': [10, 10, 10],
        'height': [10, 10, 10],
        'conf': ['0', '85', '95'],
        'text': [' ', 'Hello', ''],
    }
    df = build_dataframe_from_tesseract(data)
    # Only one valid row should remain ('Hello')
    assert len(df) == 1
    assert df.iloc[0]['text'] == 'Hello'


def _make_df_for_lines():
    data = {
        'block_num': [1, 1, 1, 2, 2],
        'par_num': [1, 1, 1, 1, 1],
        'line_num': [1, 1, 1, 1, 1],
        'left': [10, 30, 50, 300, 330],
        'top': [10, 10, 10, 15, 15],
        'width': [10, 10, 10, 10, 10],
        'height': [12, 12, 12, 14, 14],
        'conf': [80, 90, 95, 85, 88],
        'text': ['A', 'B', 'C', 'D', 'E'],
        'font_size': [12, 12, 12, 14, 14],
    }
    return pd.DataFrame(data)


def test_group_words_to_lines_basic():
    df = _make_df_for_lines()
    lines = group_words_to_lines(df, min_words_in_line=1)
    assert isinstance(lines, list)
    assert len(lines) >= 2
    assert all('text' in l for l in lines)


def test_finalize_paragraph_and_merge():
    df = _make_df_for_lines()
    lines = group_words_to_lines(df)
    paragraphs = merge_lines_to_paragraphs(lines)
    assert isinstance(paragraphs, list)
    assert paragraphs
    par = paragraphs[0]
    assert 'text' in par and 'width' in par and 'height' in par


def test_split_lines_into_columns_two_columns_detected():
    # Create two columns far apart
    lines = [
        {'x': 10, 'y': 10, 'width': 80, 'height': 12, 'text': 'L1'},
        {'x': 15, 'y': 30, 'width': 80, 'height': 12, 'text': 'L2'},
        {'x': 400, 'y': 12, 'width': 80, 'height': 12, 'text': 'R1'},
        {'x': 410, 'y': 28, 'width': 80, 'height': 12, 'text': 'R2'},
    ]
    cols = split_lines_into_columns(lines, img_width=600, max_cols=2)
    assert isinstance(cols, list)
    assert len(cols) >= 2
