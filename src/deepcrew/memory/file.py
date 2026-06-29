from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from .base import MemoryProvider


class FileMemoryProvider(MemoryProvider):
    """JSON-file-backed persistent memory store."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, str] = {}
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}
        self._loaded = True

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, prefix=".deepcrew_mem_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            Path(tmp_path).replace(self._path)
        except Exception:
            os.unlink(tmp_path)
            raise

    async def store(self, key: str, value: str) -> None:
        self._load()
        self._data[key] = value
        self._flush()

    async def retrieve(self, key: str) -> str | None:
        self._load()
        return self._data.get(key)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[str, str]]:
        self._load()
        q = query.lower()
        matches = [
            (k, v) for k, v in self._data.items()
            if q in k.lower() or q in v.lower()
        ]
        return sorted(matches, key=lambda kv: kv[0])[:top_k]

    async def clear(self) -> None:
        self._load()
        self._data.clear()
        self._flush()
