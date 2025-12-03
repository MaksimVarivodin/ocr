"""Image-level processing utilities (background estimation, cleanup)."""

from .processing import (
    get_background_color,
    remove_text_from_image,
)

__all__ = [
    "get_background_color",
    "remove_text_from_image",
]
