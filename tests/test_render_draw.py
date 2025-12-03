import numpy as np

from ocr.render.draw import add_translated_text


def test_add_translated_text_draws_non_empty_text():
    # White image
    img = np.full((100, 200, 3), 255, dtype=np.uint8)
    # Single box in the center
    box = {'x': 50, 'y': 30, 'width': 100, 'height': 40}
    text = "Hello World"
    out = add_translated_text(img, [box], [text])
    # Output must have same shape
    assert out.shape == img.shape
    # Pixels should change within the box area (due to drawn text)
    before_mean = img[30:70, 50:150].mean()
    after_mean = out[30:70, 50:150].mean()
    assert abs(after_mean - before_mean) > 0.1
