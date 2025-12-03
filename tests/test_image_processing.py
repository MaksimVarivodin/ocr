import numpy as np

from ocr.image.processing import get_background_color, remove_text_from_image


def test_get_background_color_shape_and_type():
    img = np.full((100, 200, 3), 240, dtype=np.uint8)
    # Put a dark rectangle (simulating text) inside
    img[40:60, 80:120] = 10
    bg = get_background_color(img, 80, 40, 40, 20)
    assert bg.shape == (3,)
    assert (bg >= 0).all() and (bg <= 255).all()


def test_remove_text_from_image_changes_region():
    img = np.full((60, 100, 3), 240, dtype=np.uint8)
    # Black text area
    img[20:30, 40:60] = 0
    out = remove_text_from_image(img, [{'x': 40, 'y': 20, 'width': 20, 'height': 10}])
    # The region should no longer be pure black
    patch = out[20:30, 40:60]
    assert patch.mean() > 0
    # Image overall shape preserved
    assert out.shape == img.shape
