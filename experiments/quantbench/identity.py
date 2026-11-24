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


def hardware_manifest_hash(manifest: Mapping[str, Any]) -> str:
    """Хеш фактически собранного manifest; неизвестные поля остаются null, не угадываются."""
    return digest(manifest)


def platform_observation_id(logical_key: str, platform_id: str,
                            manifest_hash: str, run_id: str,
                            attempt: str) -> str:
    """Уникальный ID записи при общем межплатформенном logical_key."""
    if not logical_key.startswith("obs-") or not platform_id or not run_id or not attempt:
        raise ValueError("Для platform observation нужны logical_key, platform_id, run_id и attempt")
    if len(manifest_hash) != 64 or any(c not in "0123456789abcdef" for c in manifest_hash.lower()):
        raise ValueError("hardware_manifest_hash должен быть SHA-256")
    return "pobs-" + digest([logical_key, platform_id, manifest_hash, run_id, attempt])[:24]
