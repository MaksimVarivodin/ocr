from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Union


@dataclass
class TextRun:
    page_index: int
    order: int
    text: str
    translated_text: Optional[str] = None


@dataclass
class ImageItem:
    page_index: int
    order: int
    src_path: str
    translated_path: Optional[str] = None


PageItem = Union[TextRun, ImageItem]


@dataclass
class Page:
    index: int
    items: List[PageItem] = field(default_factory=list)


@dataclass
class Document:
    pages: List[Page] = field(default_factory=list)

    def iter_items(self) -> List[PageItem]:
        out: List[PageItem] = []
        for p in self.pages:
            # ensure stable order by 'order'
            out.extend(sorted(p.items, key=lambda it: getattr(it, "order", 0)))
        return out
