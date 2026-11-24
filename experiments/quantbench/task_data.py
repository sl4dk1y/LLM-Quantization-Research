"""Подготовка локального dataset export без предположений о колонках исходника."""
from __future__ import annotations
from typing import Any
from .datasets import load_jsonl_dataset
from .sampling import select_items


def prepare_task_items(dataset: dict[str, Any], *, task: str, count: int, seed: int,
                       input_field: str, reference_field: str) -> list[dict[str, Any]]:
    """Проверяет hash/revision/split и делает закреплённую task-wise выборку.

    Имена исходных полей задаются проверенным dataset manifest, поскольку экспорты MERA/LIBRA
    могут иметь разные схемы. Метод не загружает данные и не меняет исходный файл.
    """
    if task not in dataset.get("tasks", []):
        raise ValueError(f"Задача {task} не зарегистрирована в dataset manifest")
    if not input_field or not reference_field or input_field == reference_field:
        raise ValueError("В dataset manifest нужны разные input_field и reference_field")
    rows = load_jsonl_dataset(dataset["path"], expected_revision=dataset["revision"],
                              expected_sha256=dataset["sha256"], expected_split=dataset["split"])
    task_field = dataset.get("task_field")
    if task_field:
        matching = [row for row in rows if row.get(task_field) == task]
    elif len(dataset.get("tasks", [])) == 1:
        matching = rows
    else:
        raise ValueError("Многозадачному export нужен task_field в закреплённом manifest")
    if not matching:
        raise ValueError(f"В закреплённом export нет строк задачи {task}")
    for row in matching:
        if not isinstance(row.get(input_field), str) or not isinstance(row.get(reference_field), str):
            raise ValueError(f"Строка {row.get('id')}: неверный формат полей из dataset manifest")
    selected = select_items(matching, seed=seed, revision=dataset["revision"], task=task,
                            split=dataset["split"], count=count)
    return [{"id": str(row["id"]), "task": task, "split": dataset["split"],
             "prompt": row[input_field], "reference": row[reference_field]} for row in selected]
