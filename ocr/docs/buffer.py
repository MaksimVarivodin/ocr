from __future__ import annotations

import os
import shutil
import time
from typing import Optional


class BufferManager:
    """Session buffer under config/buffer/<timestamp> for temporary files.

    Debug mode keeps the buffer on disk; release mode removes it on cleanup().
    """

    def __init__(self, project_root: Optional[str] = None, debug: bool = False) -> None:
        self.debug = bool(debug)
        root = project_root or os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        base = os.path.join(root, "config", "buffer")
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.base_dir = os.path.join(base, ts)
        os.makedirs(self.base_dir, exist_ok=True)

    def path(self, *parts: str) -> str:
        p = os.path.join(self.base_dir, *parts)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        return p

    def cleanup(self) -> None:
        if not self.debug:
            try:
                shutil.rmtree(self.base_dir, ignore_errors=True)
            except Exception:
                pass
