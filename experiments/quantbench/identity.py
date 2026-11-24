"""Стабильная идентификация конфигураций, наблюдений и попыток."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_json(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def config_id(config: Mapping[str, Any]) -> str:
    """ID всей конфигурации без её самоссылочных полей."""
    material = {k: v for k, v in config.items() if k not in {"config_id", "config_hash"}}
    return "cfg-" + digest(material)[:24]


def observation_key(fields: Mapping[str, Any]) -> str:
    required = {
        "protocol_version", "protocol_hash", "config_id", "model_revision", "gguf_sha256",
        "quantization", "kv_cache", "dataset_revision", "split", "task", "item_id",
        "context_length", "workload_id", "mode", "generation_seed", "performance_repeat",
    }
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError(f"Отсутствуют поля ключа наблюдения: {', '.join(missing)}")
    if fields["mode"] not in {"quality", "performance", "warmup", "pilot", "development", "held_out"}:
        raise ValueError("Неизвестный режим наблюдения")
    return "obs-" + digest({key: fields[key] for key in sorted(required)})


def attempt_id(observation_id: str, attempt_number: int, run_id: str) -> str:
    if not observation_id.startswith("obs-") or attempt_number < 1 or not run_id:
        raise ValueError("Некорректные observation_id, attempt_number или run_id")
    return "try-" + digest([observation_id, attempt_number, run_id])[:24]
