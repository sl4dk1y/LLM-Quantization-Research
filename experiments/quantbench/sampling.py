"""Воспроизводимое ранжирование элементов выборки."""

from __future__ import annotations

import hashlib
from typing import Iterable, Mapping, Any


def select_items(items: Iterable[Mapping[str, Any]], *, seed: int, revision: str, task: str, split: str, count: int) -> list[Mapping[str, Any]]:
    ranked = []
    seen: set[str] = set()
    for item in items:
        if "id" not in item:
            raise ValueError("Элемент датасета должен иметь поле id")
        item_id = str(item["id"])
        if item_id in seen:
            raise ValueError(f"Повторяющийся item ID: {item_id}")
        seen.add(item_id)
        key = f"{seed}|{revision}|{task}|{split}|{item_id}".encode()
        ranked.append((hashlib.sha256(key).hexdigest(), item_id, item))
    if count > len(ranked):
        raise ValueError(f"Запрошено {count} элементов, доступно {len(ranked)}")
    return [entry[2] for entry in sorted(ranked)[:count]]


def validate_disjoint(left: Iterable[str], right: Iterable[str]) -> None:
    overlap = set(left) & set(right)
    if overlap:
        raise ValueError(f"Обнаружено пересечение split: {len(overlap)} item ID")
