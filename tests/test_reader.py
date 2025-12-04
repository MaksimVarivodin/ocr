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


def test_split_lines_into_columns_single_column():
    lines = [
        {'x': 10, 'y': 10, 'width': 80, 'height': 12, 'text': 'Line 1'},
        {'x': 12, 'y': 30, 'width': 80, 'height': 12, 'text': 'Line 2'},
    ]
    cols = split_lines_into_columns(lines, img_width=200, max_cols=2)
    assert isinstance(cols, list)
    assert len(cols) == 1
    assert len(cols[0]) == 2


def test_split_lines_into_columns_two_columns_detected():
    # KMeans should be able to separate these two distinct clusters
    lines = [
        {'x': 10, 'y': 10, 'width': 80, 'height': 12, 'text': 'L1'},
        {'x': 15, 'y': 30, 'width': 80, 'height': 12, 'text': 'L2'},
        {'x': 12, 'y': 50, 'width': 80, 'height': 12, 'text': 'L3'},
        {'x': 400, 'y': 12, 'width': 80, 'height': 12, 'text': 'R1'},
        {'x': 410, 'y': 28, 'width': 80, 'height': 12, 'text': 'R2'},
        {'x': 405, 'y': 52, 'width': 80, 'height': 12, 'text': 'R3'},
    ]
    cols = split_lines_into_columns(lines, img_width=600, max_cols=2)
    assert isinstance(cols, list)
    assert len(cols) == 2
    # Check if columns are correctly sorted by x-position
    assert cols[0][0]['x'] < 200
    assert cols[1][0]['x'] > 200
    # Check if lines within columns are sorted by y-position
    assert cols[0][0]['y'] < cols[0][1]['y'] < cols[0][2]['y']
    assert cols[1][0]['y'] < cols[1][1]['y'] < cols[1][2]['y']


def test_merge_lines_to_paragraphs_merges_correctly():
    df = _make_df_for_lines()
    lines = group_words_to_lines(df)
    paragraphs = merge_lines_to_paragraphs(lines)
    assert isinstance(paragraphs, list)
    assert len(paragraphs) == 1
    par = paragraphs[0]
    assert 'text' in par
    assert par['text'] == 'ABC'  # Assuming A, B, C are merged
    assert 'width' in par
    assert 'height' in par
    assert par['width'] == 330  # 10 + 10 + 10 + 300 + 10 (assuming no gaps)
    assert par['height'] == 14  # Max height of the lines
