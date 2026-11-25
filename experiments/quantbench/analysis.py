"""Агрегация и аналитические срезы без изменения исходных JSONL."""
from __future__ import annotations
import hashlib
import json
import math
import random
from collections import Counter
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from .identity import canonical_json, digest
from .store import read_records

PAIR_FIELDS = ("protocol_version", "protocol_hash", "config_id", "model_id", "model_revision",
               "gguf_repo_id", "gguf_revision", "artifact_hash", "gguf_sha256", "quantization", "kv_cache", "dataset_id",
               "dataset_revision", "dataset_hash", "split", "task", "item_id", "context_length",
               "workload_id", "mode", "generation_seed", "performance_repeat", "generation_config",
               "tokenizer_revision", "chat_template_hash", "reasoning_mode", "llama_cpp_commit",
               "runtime_version", "scorer_version", "comparison_series", "block_id")


def _analysis_eligible(row: dict[str, Any], stage: str) -> bool:
    if row.get("status") != "complete" or row.get("run_kind") != "production" \
            or row.get("scientific_observation") is not True:
        return False
    block = str(row.get("block_id", "")).casefold()
    mode = str(row.get("mode", "")).casefold()
    if stage == "pilot":
        return block.startswith("p0") or "pilot" in mode
    if stage == "confirmatory":
        return not (block.startswith("p0") or "pilot" in mode or "development" in mode or "train" in mode)
    if stage == "descriptive":
        return True
    raise ValueError("stage должен быть pilot, confirmatory или descriptive")

def source_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def validate_and_pair_platforms(records_by_platform: dict[str, Iterable[dict[str, Any]]], *,
                                stage: str = "confirmatory") -> dict[str, Any]:
    """Проверяет совместимость записей и строит пары по общему observation_id."""
    if len(records_by_platform) < 2:
        raise ValueError("Для межплатформенного сопоставления нужны минимум две платформы")
    by_platform: dict[str, dict[str, dict[str, Any]]] = {}
    issues: list[str] = []
    excluded_by_platform: dict[str, int] = {}
    for platform_id, records in records_by_platform.items():
        index: dict[str, dict[str, Any]] = {}
        excluded_by_platform[platform_id] = 0
        for row in records:
            if not _analysis_eligible(row, stage):
                excluded_by_platform[platform_id] += 1
                continue
            logical_id = row.get("observation_id")
            if not logical_id:
                issues.append(f"{platform_id}: запись без observation_id")
                continue
            required_values = set(PAIR_FIELDS) - {"performance_repeat"}
            missing_fields = [field for field in required_values if row.get(field) is None or row.get(field) == ""]
            for hash_field in ("protocol_hash", "artifact_hash", "hardware_manifest_hash"):
                value = row.get(hash_field)
                if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value.lower()):
                    missing_fields.append(f"{hash_field}(неполный SHA-256)")
            if not row.get("platform_observation_id"):
                missing_fields.append("platform_observation_id/hardware_manifest_hash")
            if missing_fields:
                issues.append(f"{platform_id}/{logical_id}: отсутствуют поля совместимости {', '.join(missing_fields)}")
            if logical_id in index:
                issues.append(f"{platform_id}: дубликат observation_id {logical_id}")
                continue
            if row.get("platform_id") != platform_id:
                issues.append(f"{platform_id}: platform_id записи не совпадает")
            index[logical_id] = row
        by_platform[platform_id] = index
    common_ids = set.intersection(*(set(rows) for rows in by_platform.values())) if by_platform else set()
    if not common_ids:
        issues.append("Нет ни одного общего завершённого observation_id")
    pairs = []
    for logical_id in sorted(common_ids):
        rows = [by_platform[p][logical_id] for p in sorted(by_platform)]
        platform_ids = [row.get("platform_observation_id") for row in rows]
        if len(set(platform_ids)) != len(platform_ids):
            issues.append(f"{logical_id}: collision platform_observation_id между платформами")
        for field in PAIR_FIELDS:
            if len({canonical_json(row.get(field)) for row in rows}) != 1:
                issues.append(f"{logical_id}: несовпадает контролируемое поле {field}")
        pairs.append({"observation_id": logical_id,
                      "platforms": {platform: by_platform[platform][logical_id] for platform in sorted(by_platform)}})
    union_ids = set.union(*(set(rows) for rows in by_platform.values())) if by_platform else set()
    missing_by_platform = {p: sorted(union_ids - set(rows)) for p, rows in by_platform.items()}
    return {"compatible": not issues, "complete": all(not ids for ids in missing_by_platform.values()),
            "issues": sorted(set(issues)), "pairs": pairs,
            "missing_by_platform": missing_by_platform,
            "counts": {p: len(rows) for p, rows in by_platform.items()},
            "excluded_counts": excluded_by_platform}


def merge_result_files(sources: dict[str, str | Path], output_path: str | Path, *,
                       stage: str = "confirmatory") -> dict[str, Any]:
    """Собирает новый offline package только после успешной проверки совместимости."""
    rows = {platform: read_records(path) for platform, path in sources.items()}
    validation = validate_and_pair_platforms(rows, stage=stage)
    if not validation["compatible"] or not validation["complete"]:
        raise ValueError("Серии несовместимы или неполны: " + "; ".join(validation["issues"]))
    body = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "sources": {platform: {"file": Path(path).name, "sha256": source_sha256(path),
                                   "record_count": len(rows[platform])}
                        for platform, path in sources.items()},
            "validation": {key: value for key, value in validation.items() if key != "pairs"},
            "paired_observations": validation["pairs"]}
    body["merge_id"] = "merge-" + digest(body)[:24]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write((json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"))
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return body

def bootstrap_mean_interval(values: Iterable[float], *, seed: int = 20261124,
                            resamples: int = 5000, alpha: float = 0.05) -> dict[str, Any]:
    values = [float(value) for value in values]
    if not values or not 0 < alpha < 1 or resamples < 100:
        raise ValueError("Нужны значения, alpha в (0,1) и не менее 100 resamples")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Bootstrap не принимает NaN/inf")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(resamples))
    low = means[int((alpha / 2) * (resamples - 1))]
    high = means[int((1 - alpha / 2) * (resamples - 1))]
    return {"n": len(values), "mean": sum(values) / len(values), "ci_level": 1 - alpha,
            "ci_low": low, "ci_high": high, "method": "percentile bootstrap over supplied units",
            "seed": seed, "resamples": resamples}

def export_snapshot(*, records_path: str | Path, output_path: str | Path,
                    block_metadata: dict[str, Any] | None = None,
                    stage: str = "confirmatory") -> dict[str, Any]:
    """Создаёт новый snapshot; исходный JSONL не изменяется."""
    rows = read_records(records_path)
    block_metrics_path = Path(records_path).parent / "block_metrics.jsonl"
    block_metrics = read_records(block_metrics_path) if block_metrics_path.is_file() else []
    complete_source = [row for row in rows if row.get("status") == "complete"]
    complete = [row for row in rows if _analysis_eligible(row, stage)]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        key = "|".join(str(row.get(field, "unknown")) for field in
                       ("block_id", "platform_id", "model_id", "model_revision", "gguf_sha256",
                        "quantization", "kv_cache", "dataset_id", "dataset_revision", "dataset_hash",
                        "split", "task", "context_length", "mode", "generation_seed"))
        grouped[key].append(row)
    groups = []
    for key, values in sorted(grouped.items()):
        scores = [float((row.get("score") or {}).get("exact_match", (row.get("score") or {}).get("f1")))
                  for row in values if isinstance(row.get("score"), dict)
                  and isinstance((row.get("score") or {}).get("exact_match", (row.get("score") or {}).get("f1")), (int, float))]
        durations = [float((row.get("metrics") or {}).get("end_to_end_latency_s")) for row in values
                     if isinstance((row.get("metrics") or {}).get("end_to_end_latency_s"), (int, float))
                     and (row.get("metrics") or {}).get("end_to_end_latency_s") >= 0]
        item = {"group": key, "n_complete": len(values)}
        if scores:
            seed = int(digest(key)[:8], 16)
            item["primary_score_descriptive"] = bootstrap_mean_interval(scores, seed=seed)
        if durations:
            ordered = sorted(durations)
            item["duration_s"] = {"n": len(durations), "median": ordered[len(ordered) // 2],
                                  "p95": ordered[min(int(0.95 * len(ordered)), len(ordered) - 1)]}
        if values and values[0].get("task") == "ruOpenBookQA":
            from .dataset_pipeline import score_mera
            predictions, references = [], []
            for row in values:
                prediction = row.get("prediction")
                reference = row.get("reference")
                if isinstance(prediction, str) and isinstance(reference, str) and reference:
                    if (row.get("metrics") or {}).get("truncated"):
                        prediction = "__TRUNCATED_GENERATION__"
                    predictions.append(prediction)
                    references.append(reference)
            if len(predictions) == len(values):
                item["official_macro_f1"] = score_mera(
                    task="ruOpenBookQA", predictions=predictions, references=references)["metrics"]["f1"]
            else:
                item["official_macro_f1_status"] = "unavailable_missing_prediction_or_reference"
        groups.append(item)
    metadata_fields = ("platform_id", "hardware_manifest", "hardware_manifest_hash", "runtime_version",
                       "runtime_binary_sha256", "llama_cpp_commit",
                       "protocol_hash", "artifact_hash", "dataset_revision", "dataset_hash",
                       "scorer_version", "generation_config")
    metadata = {field: sorted({canonical_json(row[field]) for row in complete if field in row})
                for field in metadata_fields}
    body = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": Path(records_path).name, "source_sha256": source_sha256(records_path),
            "source_record_count": len(rows), "complete_source_record_count": len(complete_source),
            "complete_record_count": len(complete),
            "failed_attempt_count": sum(row.get("status") == "failed" for row in rows),
            "error_counts": dict(sorted(Counter(row.get("error_type", "unknown")
                                                  for row in rows if row.get("status") in {"failed", "blocked"}).items())),
            "blocked_event_count": sum(row.get("status") == "blocked" for row in rows),
            "analysis_stage": stage, "eligible_record_count": len(complete),
            "excluded_record_count": len(rows) - len(complete),
            "block_resource_summaries": block_metrics,
            "block_resource_source": ({"path": block_metrics_path.name,
                                        "sha256": source_sha256(block_metrics_path),
                                        "record_count": len(block_metrics)}
                                       if block_metrics_path.is_file() else None),
            "block_metadata": block_metadata or {}, "metadata_values": metadata, "groups": groups,
            "notes": ["Bootstrap здесь описательный; paired permutation, Holm и McNemar относятся к M6.",
                      "Ресурсные пики берутся из сопоставленного block_metrics.jsonl, если он предоставлен."]}
    body["snapshot_id"] = "slice-" + digest(body)[:24]
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(body, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with target.open("xb") as stream:
        stream.write(data)
        stream.flush()
        import os
        os.fsync(stream.fileno())
    return body
