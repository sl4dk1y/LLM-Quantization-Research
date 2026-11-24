"""Загрузка и предварительная проверка JSON конфигураций."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


TOP_LEVEL = {
    "schema_version", "protocol", "mode", "seed", "models", "datasets", "runtime",
    "generation", "sampling", "contexts", "limits", "repeats", "output",
}


def load_config(path: str | Path) -> dict[str, Any]:
    try:
        config = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать JSON конфигурацию: {exc}") from exc
    validate_config(config)
    return config


def validate_config(config: Any) -> None:
    if not isinstance(config, dict):
        raise ValueError("Конфигурация должна быть JSON-объектом")
    if any(not isinstance(key, str) for key in config):
        raise ValueError("Имена полей конфигурации должны быть строками")
    extra = sorted(config.keys() - TOP_LEVEL)
    missing = sorted(TOP_LEVEL - config.keys())
    if extra or missing:
        parts = []
        if missing:
            parts.append("нет полей: " + ", ".join(missing))
        if extra:
            parts.append("неизвестные поля: " + ", ".join(extra))
        raise ValueError("; ".join(parts))
    if type(config["schema_version"]) is not int or config["schema_version"] != 1:
        raise ValueError("Поддерживается schema_version=1")
    if not isinstance(config["mode"], str) or config["mode"] not in {"development", "pilot", "held_out_quality", "performance", "dry_run"}:
        raise ValueError("Недопустимый mode")
    if not isinstance(config["protocol"], dict) or not all(config["protocol"].get(k) for k in ("version", "sha256")):
        raise ValueError("protocol.version и protocol.sha256 обязательны")
    if len(config["protocol"]["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in config["protocol"]["sha256"].lower()):
        raise ValueError("protocol.sha256 должен быть SHA-256 hex")
    if not isinstance(config["models"], list) or not config["models"]:
        raise ValueError("models должен быть непустым списком")
    model_ids = set()
    for index, model in enumerate(config["models"]):
        required = ("id", "revision", "gguf_path", "gguf_sha256", "quantization", "kv_cache")
        if not isinstance(model, dict) or not all(model.get(k) for k in required):
            raise ValueError(f"models[{index}]: обязательны {', '.join(required)}")
        if len(model["gguf_sha256"]) != 64:
            raise ValueError(f"models[{index}].gguf_sha256 должен содержать 64 hex символа")
        if any(c not in "0123456789abcdef" for c in model["gguf_sha256"].lower()):
            raise ValueError(f"models[{index}].gguf_sha256 должен быть hex")
        if model["id"] in model_ids:
            raise ValueError("Идентификаторы моделей должны быть уникальны")
        model_ids.add(model["id"])
    if not isinstance(config["datasets"], list) or not config["datasets"]:
        raise ValueError("datasets должен быть непустым списком")
    dataset_ids = set()
    for index, dataset in enumerate(config["datasets"]):
        if not isinstance(dataset, dict) or not all(dataset.get(k) for k in ("id", "revision", "sha256", "path", "split", "tasks")):
            raise ValueError(f"datasets[{index}]: нужны id, revision, sha256, path, split и tasks")
        if len(dataset["sha256"]) != 64 or any(c not in "0123456789abcdef" for c in dataset["sha256"].lower()):
            raise ValueError(f"datasets[{index}].sha256 должен содержать SHA-256")
        if dataset["id"] in dataset_ids:
            raise ValueError("Идентификаторы датасетов должны быть уникальны")
        if not isinstance(dataset["tasks"], list) or not dataset["tasks"] or any(not isinstance(t, str) or not t for t in dataset["tasks"]):
            raise ValueError(f"datasets[{index}].tasks должен быть непустым списком строк")
        dataset_ids.add(dataset["id"])
    for section in ("protocol", "runtime", "sampling", "generation", "limits", "repeats", "output"):
        if not isinstance(config[section], dict):
            raise ValueError(f"{section} должен быть объектом")
    positive = (("seed", config["seed"]), ("runtime.port", config["runtime"].get("port")),
                ("runtime.load_timeout_s", config["runtime"].get("load_timeout_s")),
                ("runtime.request_timeout_s", config["runtime"].get("request_timeout_s")),
                ("repeats.performance", config["repeats"].get("performance")))
    for name, value in positive:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} должен быть положительным целым числом")
    if config["runtime"].get("host") not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("Runtime разрешено привязывать только к loopback")
    if config["sampling"].get("strategy") != "sha256_rank" or type(config["sampling"].get("items_per_task")) is not int or config["sampling"]["items_per_task"] < 1:
        raise ValueError("sampling должен задавать sha256_rank и положительный items_per_task")
    if not isinstance(config["contexts"], list) or not config["contexts"] or any(type(n) is not int or n < 1 for n in config["contexts"]):
        raise ValueError("contexts должен содержать положительные целые длины")
    if len(config["contexts"]) != len(set(config["contexts"])):
        raise ValueError("contexts не должен содержать дубли")
    if not isinstance(config["runtime"].get("executable"), str) or not config["runtime"]["executable"]:
        raise ValueError("runtime.executable обязателен")
    runtime = config["runtime"]
    runtime_required = ("gpu_layers", "cpu_threads", "cpu_threads_batch", "batch_size", "ubatch_size",
                        "flash_attn", "reasoning_mode", "chat_template_source")
    missing_runtime = [key for key in runtime_required if key not in runtime or runtime[key] is None]
    if missing_runtime:
        raise ValueError("runtime: не заданы параметры: " + ", ".join(missing_runtime))
    for key in ("cpu_threads", "cpu_threads_batch", "batch_size", "ubatch_size"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise ValueError(f"runtime.{key} должен быть положительным целым числом")
    if not (type(runtime["gpu_layers"]) is int and runtime["gpu_layers"] >= 0 or runtime["gpu_layers"] in {"auto", "all"}):
        raise ValueError("runtime.gpu_layers должен быть неотрицательным числом, auto или all")
    if type(runtime["flash_attn"]) is not bool or runtime["reasoning_mode"] not in {"on", "off"}:
        raise ValueError("runtime.flash_attn должен быть boolean, reasoning_mode — on/off")
    if runtime["chat_template_source"] != "gguf_metadata":
        raise ValueError("runtime.chat_template_source должен быть явно задан как gguf_metadata")
    if not 1 <= config["runtime"]["port"] <= 65535:
        raise ValueError("runtime.port должен быть в диапазоне 1..65535")
    if config["runtime"]["load_timeout_s"] > 300 or config["runtime"]["request_timeout_s"] > 600:
        raise ValueError("Тайм-ауты runtime не могут превышать предварительные пределы протокола")
    for key in ("temperature", "top_p"):
        if key in config["generation"] and (type(config["generation"][key]) not in {int, float} or not 0 <= config["generation"][key] <= 1):
            raise ValueError(f"generation.{key} должен быть числом 0..1")
    if "max_new_tokens" in config["generation"] and (type(config["generation"]["max_new_tokens"]) is not int or config["generation"]["max_new_tokens"] < 1):
        raise ValueError("generation.max_new_tokens должен быть положительным целым числом")
    generation_required = ("temperature", "top_p", "top_k", "seed", "max_new_tokens")
    missing_generation = [key for key in generation_required if key not in config["generation"]]
    if missing_generation:
        raise ValueError("generation: не заданы параметры: " + ", ".join(missing_generation))
    if type(config["generation"]["top_k"]) is not int or config["generation"]["top_k"] < 0:
        raise ValueError("generation.top_k должен быть неотрицательным целым числом")
    if type(config["generation"]["seed"]) is not int:
        raise ValueError("generation.seed должен быть целым числом")
    if not config["generation"] or not config["limits"] or not config["output"]:
        raise ValueError("generation, limits и output должны быть заполнены")
