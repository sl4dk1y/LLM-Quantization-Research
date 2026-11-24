"""Компиляция утверждённых явных ячеек в детерминированные platform jobs."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .identity import config_id, digest, observation_key

HASH_FIELDS = ("protocol_hash", "gguf_sha256", "dataset_sha256")


def compile_jobs(plan: dict[str, Any]) -> dict[str, Any]:
    if plan.get("ready_for_execution") is not True:
        raise ValueError("Queue plan не утверждён для компиляции")
    protocol_version = plan.get("protocol_version")
    protocol_hash = plan.get("protocol_hash")
    campaign_id = plan.get("campaign_id")
    if not protocol_version or not campaign_id or not isinstance(plan.get("cells"), list) or not plan["cells"]:
        raise ValueError("Нужны protocol_version, campaign_id и явный список cells")
    jobs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, cell in enumerate(plan["cells"]):
        for name in HASH_FIELDS:
            value = cell.get(name, protocol_hash if name == "protocol_hash" else None)
            if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                raise ValueError(f"cells[{index}].{name} должен быть полным SHA-256")
        required = ("block_id", "priority", "model_id", "model_revision", "gguf_repo_id", "gguf_revision",
                    "gguf_sha256", "quantization", "kv_cache", "dataset_id", "dataset_revision",
                    "dataset_sha256", "split", "task", "context_length", "workload_id", "mode",
                    "generation_seed", "generation_config", "tokenizer_revision", "chat_template_hash",
                    "reasoning_mode", "llama_cpp_commit", "scorer_version", "comparison_series",
                    "item_ids", "platforms")
        missing = [name for name in required if cell.get(name) is None]
        if missing:
            raise ValueError(f"cells[{index}] отсутствуют поля: {', '.join(missing)}")
        if type(cell["priority"]) is not int or not isinstance(cell["item_ids"], list) or not cell["item_ids"]:
            raise ValueError(f"cells[{index}]: priority и item_ids некорректны")
        if not isinstance(cell["scorer_version"], str) or not cell["scorer_version"]:
            raise ValueError(f"cells[{index}].scorer_version обязателен")
        if cell["comparison_series"] not in {"controlled", "platform_optimized"}:
            raise ValueError(f"cells[{index}].comparison_series должен быть controlled или platform_optimized")
        if not isinstance(cell["generation_config"], dict):
            raise ValueError(f"cells[{index}].generation_config должен быть объектом")
        template_hash = cell["chat_template_hash"]
        if (not isinstance(template_hash, str) or len(template_hash) != 64 or
                any(c not in "0123456789abcdef" for c in template_hash.lower())):
            raise ValueError(f"cells[{index}].chat_template_hash должен быть полным SHA-256")
        if type(cell["context_length"]) is not int or cell["context_length"] < 1:
            raise ValueError(f"cells[{index}].context_length должен быть положительным целым")
        if cell["mode"] == "performance" and (type(cell.get("performance_repeat")) is not int or
                                                cell["performance_repeat"] < 1):
            raise ValueError(f"cells[{index}]: performance требует положительный performance_repeat")
        if type(plan.get("max_attempts", 2)) is not int or plan.get("max_attempts", 2) not in range(1, 6):
            raise ValueError("max_attempts должен быть от 1 до 5")
        if len(cell["item_ids"]) != len(set(map(str, cell["item_ids"]))):
            raise ValueError(f"cells[{index}]: повторяются item_ids")
        common = {"protocol_version": protocol_version, "protocol_hash": cell.get("protocol_hash", protocol_hash),
                  "model_id": cell["model_id"], "model_revision": cell["model_revision"],
                  "gguf_repo_id": cell["gguf_repo_id"], "gguf_revision": cell["gguf_revision"],
                  "gguf_sha256": cell["gguf_sha256"], "quantization": cell["quantization"],
                  "artifact_hash": cell["gguf_sha256"],
                  "kv_cache": cell["kv_cache"], "dataset_id": cell["dataset_id"],
                  "dataset_revision": cell["dataset_revision"], "dataset_hash": cell["dataset_sha256"],
                  "split": cell["split"], "task": cell["task"], "context_length": cell["context_length"],
                  "workload_id": cell["workload_id"], "mode": cell["mode"],
                  "generation_seed": cell["generation_seed"], "generation_config": cell["generation_config"],
                  "tokenizer_revision": cell["tokenizer_revision"], "chat_template_hash": cell["chat_template_hash"],
                  "reasoning_mode": cell["reasoning_mode"], "llama_cpp_commit": cell["llama_cpp_commit"],
                  "scorer_version": cell["scorer_version"], "comparison_series": cell["comparison_series"],
                  "campaign_id": campaign_id}
        scientific_config_id = config_id({key: value for key, value in common.items()
                                           if key != "campaign_id"})
        platforms = cell["platforms"]
        if not isinstance(platforms, list) or not platforms:
            raise ValueError(f"cells[{index}].platforms должен быть непустым списком")
        for item_id in cell["item_ids"]:
            identity = {"protocol_version": protocol_version, "protocol_hash": common["protocol_hash"],
                        "config_id": scientific_config_id, "model_revision": cell["model_revision"],
                        "gguf_sha256": cell["gguf_sha256"], "quantization": cell["quantization"],
                        "kv_cache": cell["kv_cache"], "dataset_revision": cell["dataset_revision"],
                        "split": cell["split"], "task": cell["task"], "item_id": str(item_id),
                        "context_length": cell["context_length"], "workload_id": cell["workload_id"],
                        "mode": cell["mode"], "generation_seed": cell["generation_seed"],
                        "performance_repeat": cell.get("performance_repeat")}
            logical_id = observation_key(identity)
            for platform in platforms:
                if not isinstance(platform, dict):
                    raise ValueError(f"cells[{index}].platforms должен содержать объекты")
                platform_id, manifest_hash = platform.get("platform_id"), platform.get("hardware_manifest_hash")
                runtime_version = platform.get("runtime_version")
                binary_hash = platform.get("runtime_binary_sha256")
                if (not platform_id or not isinstance(manifest_hash, str) or len(manifest_hash) != 64 or
                        any(c not in "0123456789abcdef" for c in manifest_hash.lower()) or
                        not runtime_version or not isinstance(binary_hash, str) or len(binary_hash) != 64 or
                        any(c not in "0123456789abcdef" for c in binary_hash.lower())):
                    raise ValueError(f"cells[{index}]: нужны platform_id, runtime_version и полные SHA-256 manifest/binary")
                pair_key = (logical_id, platform_id)
                if pair_key in seen:
                    raise ValueError(f"Дубликат логического observation/platform: {logical_id}/{platform_id}")
                seen.add(pair_key)
                jobs.append({**identity, **common, "observation_id": logical_id,
                             "platform_id": platform_id, "hardware_manifest_hash": manifest_hash,
                             "runtime_version": runtime_version, "runtime_binary_sha256": binary_hash,
                             "job_id": "job-" + digest([logical_id, platform_id, campaign_id])[:24],
                             "block_id": cell["block_id"], "priority": cell["priority"],
                             "item_id": str(item_id), "attempt_policy": plan.get("max_attempts", 2)})
    job_ids = [job["job_id"] for job in jobs]
    if len(job_ids) != len(set(job_ids)):
        raise ValueError("Скомпилированная очередь содержит дублированные job_id")
    return {"schema_version": 1, "campaign_id": campaign_id,
            "protocol_version": protocol_version, "protocol_hash": protocol_hash,
            "job_plan_hash": digest(jobs), "job_count": len(jobs), "jobs": jobs}


def write_job_manifest(plan: dict[str, Any], output_path: str | Path, *, platform_id: str) -> dict[str, Any]:
    result = compile_jobs(plan)
    selected = [job for job in result["jobs"] if job["platform_id"] == platform_id]
    if not selected:
        raise ValueError(f"В утверждённом плане нет platform_id={platform_id}")
    result["jobs"] = selected
    result["job_count"] = len(selected)
    result["job_plan_hash"] = digest(selected)
    result["local_platform_id"] = platform_id
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
    return result
