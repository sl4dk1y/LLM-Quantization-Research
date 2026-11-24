"""Чтение только явно предоставленных локальных JSONL экспортов."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_sha256(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_jsonl_dataset(path: str | Path, *, expected_revision: str, expected_sha256: str, expected_split: str) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise ValueError(f"Локальный dataset export отсутствует: {target}")
    actual_hash = file_sha256(target)
    if actual_hash.lower() != expected_sha256.lower():
        raise ValueError("SHA-256 локального dataset export не совпадает с manifest")
    if not expected_revision or not expected_split:
        raise ValueError("Не заданы закреплённые revision и split датасета")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    with target.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL датасета повреждён, строка {line_number}") from exc
            if not isinstance(row, dict) or not row.get("id") or row.get("split") != expected_split:
                raise ValueError(f"Строка {line_number}: ожидаются объект, уникальный id и split={expected_split}")
            item_id = str(row["id"])
            if item_id in seen:
                raise ValueError(f"Повторяющийся item ID {item_id} в {target}")
            seen.add(item_id)
            rows.append(row)
    if not rows:
        raise ValueError("Локальный dataset export пуст")
    return rows
