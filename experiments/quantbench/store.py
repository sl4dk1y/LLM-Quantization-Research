"""Append-only JSONL хранилище с восстановлением оборванного хвоста."""

from __future__ import annotations

import json
import os
try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows hosts
    fcntl = None
    import msvcrt
import errno
from pathlib import Path
from typing import Any, Iterable

from .identity import canonical_json


def append_record(path: str | Path, record: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = canonical_json(record) + "\n"
    lock_fd = os.open(str(target) + ".lock", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except BaseException:
        os.close(lock_fd)
        raise
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
        else:
            os.lseek(lock_fd, 0, os.SEEK_SET)
            msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
        data = line.encode("utf-8")
        offset = 0
        while offset < len(data):
            try:
                written = os.write(fd, data[offset:])
            except InterruptedError:
                continue
            if written <= 0:
                raise OSError(errno.EIO, "Не удалось полностью дописать JSONL запись")
            offset += written
        os.fsync(fd)
    finally:
        try:
            if fcntl is not None:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            else:
                os.lseek(lock_fd, 0, os.SEEK_SET)
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        os.close(lock_fd)
        os.close(fd)


def read_records(path: str | Path, *, repair_incomplete_tail: bool = False) -> list[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return []
    data = target.read_bytes()
    complete_end = data.rfind(b"\n") + 1
    tail = data[complete_end:]
    if tail and repair_incomplete_tail:
        # Repair is safe only if every newline-terminated row parses. Hold an
        # exclusive advisory lock across validation and truncation.
        fd = os.open(target, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            current = os.read(fd, os.fstat(fd).st_size)
            if current != data:
                raise ValueError("JSONL изменился во время восстановления; повторите проверку")
            _parse_complete_rows(data[:complete_end])
            os.ftruncate(fd, complete_end)
            os.fsync(fd)
            data = data[:complete_end]
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
    elif tail:
        raise ValueError("JSONL содержит незавершённую последнюю запись; требуется явная проверка/восстановление")
    return _parse_complete_rows(data)


def _parse_complete_rows(data: bytes) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(data.splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Повреждённая JSONL запись на строке {line_number}") from exc
        if not isinstance(value, dict):
            raise ValueError(f"JSONL строка {line_number} должна содержать объект")
        records.append(value)
    return records


def completed_observations(path: str | Path) -> set[str]:
    return {record["observation_id"] for record in read_records(path) if record.get("status") == "complete" and record.get("observation_id")}


def assert_new_ids(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    existing = {r.get("attempt_id") for r in read_records(path)}
    new_ids = [r.get("attempt_id") for r in records]
    if any(not value for value in new_ids) or len(new_ids) != len(set(new_ids)) or existing.intersection(new_ids):
        raise ValueError("attempt_id должны быть заданы и уникальны; исходные записи не перезаписываются")
