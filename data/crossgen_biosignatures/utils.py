"""Utility helpers shared across the cross-generator dataset pipeline."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_hash_u64(*parts: object) -> int:
    """Return a stable unsigned 64-bit integer for arbitrary input parts."""

    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest()[:8], "big", signed=False)


def derived_seed(master_seed: int, *parts: object) -> int:
    """Derive a reproducible RNG seed from a master seed and any extra parts."""

    return stable_hash_u64(master_seed, *parts)


def ensure_directory(path: Path) -> Path:
    """Create *path* if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_text(path: Path, payload: str) -> None:
    """Write text atomically to a destination path."""

    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(payload)
    temp_path.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON payload atomically."""

    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def chunked(items: list[Any], size: int) -> Iterable[list[Any]]:
    """Yield contiguous chunks of *items* of length *size*."""

    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    for start in range(0, len(items), size):
        yield items[start : start + size]
