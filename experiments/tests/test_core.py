import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from quantbench.config import load_config, validate_config
from quantbench.datasets import file_sha256, load_jsonl_dataset
from quantbench.identity import attempt_id, config_id, hardware_manifest_hash, observation_key, platform_observation_id
from quantbench.platform import Metric, MetricStatus, validate_snapshot
from quantbench.runtime import build_server_args
from quantbench.safety import SafetyGuard, SafetyState
from quantbench.sampling import select_items, validate_disjoint
from quantbench.store import append_record, completed_observations, read_records
import quantbench.store as store_module


def fixture_config():
    return {
        "schema_version": 1,
        "protocol": {"version": "0.1", "sha256": "a" * 64},
        "mode": "dry_run", "seed": 1,
        "models": [{"id": "m", "revision": "r", "gguf_path": "/m.gguf", "gguf_sha256": "b" * 64, "quantization": "Q4_K_M", "kv_cache": "f16"}],
        "datasets": [{"id": "d", "revision": "r", "sha256": "c" * 64, "path": "/d.jsonl", "split": "dev", "tasks": ["t"]}],
        "runtime": {"executable": "llama-server", "host": "127.0.0.1", "port": 1, "load_timeout_s": 1, "request_timeout_s": 1, "version": "v"},
        "generation": {"temperature": 0}, "sampling": {"strategy": "sha256_rank", "items_per_task": 1},
        "contexts": [128], "limits": {"max_system_used_gib": 12}, "repeats": {"performance": 1, "warmup": 1},
        "output": {"root": "results/raw", "append_only": True},
    }


def test_config_accepts_complete_config_and_rejects_non_loopback():
    config = fixture_config()
    validate_config(config)
    config["runtime"]["host"] = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        validate_config(config)


def test_config_loader_reads_example():
    config = load_config("experiments/config.example.json")
    assert config["mode"] == "dry_run"


def test_identifiers_are_deterministic_and_attempts_distinct():
    fields = dict.fromkeys([
        "protocol_version", "protocol_hash", "config_id", "model_revision", "gguf_sha256",
        "quantization", "kv_cache", "dataset_revision", "split", "task", "item_id",
        "context_length", "workload_id", "mode", "generation_seed", "performance_repeat",
    ], "x")
    fields["mode"] = "quality"
    obs = observation_key(fields)
    assert obs == observation_key(fields)
    assert attempt_id(obs, 1, "run-a") != attempt_id(obs, 2, "run-a")
    assert config_id(fixture_config()) == config_id(fixture_config())


def test_observation_key_requires_full_compound_identity():
    with pytest.raises(ValueError, match="Отсутствуют"):
        observation_key({})


def test_sampling_is_deterministic_and_split_overlap_is_detected():
    items = [{"id": str(i)} for i in range(20)]
    first = select_items(items, seed=7, revision="r1", task="t", split="train", count=5)
    assert first == select_items(reversed(items), seed=7, revision="r1", task="t", split="train", count=5)
    with pytest.raises(ValueError, match="пересечение"):
        validate_disjoint(["a", "b"], ["b"])


def test_jsonl_is_append_only_and_repairs_only_incomplete_tail(tmp_path):
    path = tmp_path / "requests.jsonl"
    append_record(path, {"attempt_id": "a", "observation_id": "obs-a", "status": "complete"})
    with path.open("ab") as stream:
        stream.write(b'{"incomplete":')
    with pytest.raises(ValueError, match="незавершённую"):
        read_records(path)
    records = read_records(path, repair_incomplete_tail=True)
    assert records == [{"attempt_id": "a", "observation_id": "obs-a", "status": "complete"}]
    assert completed_observations(path) == {"obs-a"}


def test_invalid_jsonl_record_fails_closed(tmp_path):
    path = tmp_path / "broken.jsonl"
    path.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="строке 1"):
        read_records(path)


def test_jsonl_writer_completes_partial_os_writes(tmp_path, monkeypatch):
    path = tmp_path / "partial.jsonl"
    real_write = store_module.os.write

    def short_write(fd, data):
        return real_write(fd, data[:max(1, min(7, len(data)))])

    monkeypatch.setattr(store_module.os, "write", short_write)
    append_record(path, {"attempt_id": "partial", "observation_id": "obs-partial", "status": "complete"})
    assert read_records(path)[0]["attempt_id"] == "partial"


def test_jsonl_writer_serializes_concurrent_appends(tmp_path):
    path = tmp_path / "concurrent.jsonl"
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda index: append_record(path, {"attempt_id": str(index),
                                                         "observation_id": f"obs-{index}",
                                                         "status": "complete"}), range(32)))
    records = read_records(path)
    assert len(records) == 32
    assert len({row["attempt_id"] for row in records}) == 32


def test_local_dataset_checks_hash_split_and_unique_ids(tmp_path):
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"id":"1","split":"dev","question":"synthetic"}\n', encoding="utf-8")
    rows = load_jsonl_dataset(path, expected_revision="r1", expected_sha256=file_sha256(path), expected_split="dev")
    assert rows[0]["id"] == "1"
    with pytest.raises(ValueError, match="SHA-256"):
        load_jsonl_dataset(path, expected_revision="r1", expected_sha256="0" * 64, expected_split="dev")
    with pytest.raises(ValueError, match="split"):
        load_jsonl_dataset(path, expected_revision="r1", expected_sha256=file_sha256(path), expected_split="test")


def test_runtime_args_use_loopback_and_validated_model():
    config, model = fixture_config(), fixture_config()["models"][0]
    assert "--host" in build_server_args(config, model)
    config["runtime"]["host"] = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        build_server_args(config, model)


def test_safety_guard_fails_closed_on_missing_metric_and_stops_at_threshold():
    guard = SafetyGuard()
    assert guard.evaluate({}) == SafetyState.BLOCKED
    metrics = {"system_used_bytes": int(12 * 1024**3), "swap_delta_bytes": 0,
               "memory_pressure": {"category": "green"}, "thermal_state": "nominal",
               "process_state": "running", "physical_footprint_bytes": 100, "monotonic_time": 0}
    assert guard.evaluate(metrics, now=0) == SafetyState.WARNING
    assert guard.evaluate(metrics, now=5) == SafetyState.EMERGENCY_STOP
    assert not guard.allows_inference


def test_safety_guard_ignores_transient_pressure_and_requires_recovery_window():
    guard = SafetyGuard(recovery_window_s=10)
    normal = {"system_used_bytes": 7 * 1024**3, "swap_delta_bytes": 0,
              "memory_pressure": {"category": "green"}, "thermal_state": "nominal",
              "process_state": "running", "physical_footprint_bytes": 100}
    assert guard.evaluate({**normal, "monotonic_time": 1}, now=1) == SafetyState.RECOVERY
    assert guard.evaluate({**normal, "monotonic_time": 11}, now=11) == SafetyState.SAFE
    high = {**normal, "system_used_bytes": 11 * 1024**3}
    assert guard.evaluate({**high, "monotonic_time": 12}, now=12) == SafetyState.WARNING
    assert guard.evaluate({**normal, "monotonic_time": 14}, now=14) == SafetyState.RECOVERY
    assert guard.evaluate({**normal, "monotonic_time": 24}, now=24) == SafetyState.SAFE


@pytest.mark.parametrize("override", [
    {"system_used_bytes": -1},
    {"physical_footprint_bytes": float("nan")},
    {"monotonic_time": -20},
])
def test_safety_guard_blocks_invalid_and_stale_metrics(override):
    guard = SafetyGuard()
    metrics = {"system_used_bytes": 1, "swap_delta_bytes": 0,
               "memory_pressure": {"category": "green"}, "thermal_state": "nominal",
               "process_state": "running", "physical_footprint_bytes": 100,
               "monotonic_time": 10}
    metrics.update(override)
    assert guard.evaluate(metrics, now=20) == SafetyState.BLOCKED


def test_platform_metric_status_and_unique_platform_observation_ids():
    metrics = {"gpu_utilization": Metric("gpu_utilization", None, "%", MetricStatus.NOT_APPLICABLE,
                                         "GPU отсутствует")}
    assert validate_snapshot(metrics, {"gpu_utilization"}) == []
    required_missing = {"thermal_state": Metric("thermal_state", None, None, MetricStatus.UNAVAILABLE, "нет источника")}
    assert validate_snapshot(required_missing, {"thermal_state"})
    manifest_hash = hardware_manifest_hash({"platform_id": "apple-metal", "gpu_model": None})
    logical = observation_key({k: ("quality" if k == "mode" else "x") for k in [
        "protocol_version", "protocol_hash", "config_id", "model_revision", "gguf_sha256", "quantization",
        "kv_cache", "dataset_revision", "split", "task", "item_id", "context_length", "workload_id",
        "mode", "generation_seed", "performance_repeat"]})
    a = platform_observation_id(logical, "apple-metal", manifest_hash, "run-a", "try-a")
    b = platform_observation_id(logical, "nvidia-cuda", manifest_hash, "run-b", "try-b")
    assert a != b


def test_config_rejects_bool_as_integer_and_bad_hash():
    config = fixture_config()
    config["runtime"]["port"] = True
    with pytest.raises(ValueError):
        validate_config(config)
    config = fixture_config()
    config["models"][0]["gguf_sha256"] = "z" * 64
    with pytest.raises(ValueError):
        validate_config(config)
