import json
import hashlib

import pytest

from quantbench.analysis import (bootstrap_mean_interval, export_snapshot, merge_result_files,
                                 validate_and_pair_platforms)
from quantbench.orchestrator import QueueLock, SafetyStop, order_jobs, run_queue
from quantbench.planning import estimate_remaining
from quantbench.scoring import exact_match, token_f1
from quantbench.task_data import prepare_task_items
from quantbench.safety import SafetyGuard, SafetyState
from quantbench.watchdog import SafetyWatchdog
from quantbench.queue_compile import compile_jobs, write_job_manifest


def jobs():
    return [
        {"job_id": "p1-a", "block_id": "P1", "priority": 1, "observation_id": "obs-a",
         "model_id": "Qwen3-4B", "quantization": "Q4_K_M", "context_length": 4096,
         "platform_id": "metal", "hardware_manifest_hash": "1" * 64},
        {"job_id": "p1-b", "block_id": "P1", "priority": 1, "observation_id": "obs-b",
         "model_id": "Qwen3-4B", "quantization": "Q4_K_M", "context_length": 4096,
         "platform_id": "metal", "hardware_manifest_hash": "2" * 64},
        {"job_id": "p2", "block_id": "P2", "priority": 2, "observation_id": "obs-c"},
    ]


def test_priority_queue_holds_later_blocks_until_current_priority_completes():
    assert [row["job_id"] for row in order_jobs(jobs())] == ["p1-a", "p1-b"]
    assert [row["job_id"] for row in order_jobs(jobs(), completed={"obs-a", "obs-b"})] == ["p2"]
    with pytest.raises(ValueError, match="Повторяющийся observation_id"):
        order_jobs([jobs()[0], {**jobs()[1], "observation_id": "obs-a"}])


def test_queue_resume_skips_completed_observations_and_caps_retries(tmp_path):
    count = {"obs-a": 0, "obs-b": 0}
    def execute(job):
        count[job["observation_id"]] += 1
        if job["observation_id"] == "obs-a" and count["obs-a"] == 1:
            raise TimeoutError("temporary")
        return {"duration_s": 2.0, "score_primary": 1.0}

    status = run_queue(jobs()[:2], run_dir=tmp_path, execute=execute, preflight=lambda: None,
                       platform_id="metal", max_attempts=2, owner_authorized=True,
                       clock=iter([1, 2, 3, 4, 5, 6]).__next__)
    assert status["completed_jobs"] == 2
    assert status["failed_attempts"] == 1
    assert count == {"obs-a": 2, "obs-b": 1}
    assert (tmp_path / "checkpoint.json").exists()
    resumed = run_queue(jobs()[:2], run_dir=tmp_path,
                        execute=lambda _job: pytest.fail("completed observations reran"),
                        preflight=lambda: None, platform_id="metal", owner_authorized=True)
    assert resumed["completed_jobs"] == 2


def test_resume_limits_new_observations_after_skipping_completed_ids(tmp_path):
    selected = jobs()[:2] * 3
    selected = [{**job, "job_id": f"job-{index}", "observation_id": f"obs-{index}"}
                for index, job in enumerate(selected)]
    first = run_queue(selected, run_dir=tmp_path, execute=lambda _job: {},
                      preflight=lambda: None, platform_id="metal", owner_authorized=True,
                      max_new_observations=1)
    assert first["completed_jobs"] == 1

    called = []
    second = run_queue(selected, run_dir=tmp_path,
                       execute=lambda job: called.append(job["observation_id"]) or {},
                       preflight=lambda: None, platform_id="metal", owner_authorized=True,
                       max_new_observations=2)
    assert called == ["obs-1", "obs-2"]
    assert second["completed_jobs"] == 3

    called.clear()
    third = run_queue(selected, run_dir=tmp_path,
                      execute=lambda job: called.append(job["observation_id"]) or {},
                      preflight=lambda: None, platform_id="metal", owner_authorized=True,
                      max_new_observations=2)
    assert called == ["obs-3", "obs-4"]
    assert third["completed_jobs"] == 5


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_run_queue_rejects_invalid_new_observation_limit(tmp_path, limit):
    with pytest.raises(ValueError, match="max_new_observations"):
        run_queue(jobs()[:1], run_dir=tmp_path, execute=lambda _job: {}, preflight=lambda: None,
                  platform_id="metal", owner_authorized=True, max_new_observations=limit)


def test_run_queue_stops_between_observations_and_lock_prevents_second_runner(tmp_path):
    events = []
    with QueueLock(tmp_path / "orchestrator.lock"):
        with pytest.raises(RuntimeError, match="уже выполняется"):
            with QueueLock(tmp_path / "orchestrator.lock"):
                pass
    status = run_queue(jobs()[:2], run_dir=tmp_path / "run",
                       execute=lambda job: events.append(job["job_id"]) or {},
                       preflight=lambda: None, platform_id="metal", owner_authorized=True,
                       stop_requested=lambda: len(events) == 1)
    assert events == ["p1-a"]
    assert status["completed_jobs"] == 1


def test_retries_do_not_reset_after_resume(tmp_path):
    def fail(job):
        raise TimeoutError("offline")
    first = run_queue(jobs()[:1], run_dir=tmp_path, execute=fail,
                      preflight=lambda: None, platform_id="metal", max_attempts=1, owner_authorized=True)
    assert first["failed_attempts"] == 1
    calls = []
    second = run_queue(jobs()[:1], run_dir=tmp_path,
                       execute=lambda job: calls.append(job) or {},
                       preflight=lambda: None, platform_id="metal", max_attempts=1, owner_authorized=True)
    assert calls == []
    assert second["remaining_jobs"] == 1


def test_interrupted_execution_is_not_retried_even_if_oserror_is_retryable(tmp_path):
    calls = []

    def interrupted(job):
        calls.append(job["job_id"])
        raise InterruptedError("Safety Guard stopped the runtime")

    status = run_queue(jobs()[:1], run_dir=tmp_path, execute=interrupted,
                       preflight=lambda: None, platform_id="metal", max_attempts=3,
                       owner_authorized=True,
                       retryable_error=lambda exc: isinstance(exc, OSError))

    assert calls == [jobs()[0]["job_id"]]
    assert status["failed_attempts"] == 1
    rows = [json.loads(line) for line in (tmp_path / "observations.jsonl").read_text().splitlines()]
    assert [row["error_type"] for row in rows] == ["InterruptedError"]


def test_execution_requires_explicit_owner_authorization(tmp_path):
    with pytest.raises(PermissionError, match="разрешения владельца"):
        run_queue(jobs()[:1], run_dir=tmp_path, execute=lambda _job: {}, preflight=lambda: None,
                  platform_id="metal")


def test_changed_job_plan_and_safety_stop_fail_closed(tmp_path):
    job = jobs()[:1]
    run_queue(job, run_dir=tmp_path, execute=lambda _job: {"duration_s": 1},
              preflight=lambda: None, platform_id="metal", owner_authorized=True)
    with pytest.raises(ValueError, match="Job plan изменился"):
        run_queue([{**job[0], "quantization": "F16"}], run_dir=tmp_path,
                  execute=lambda _job: {}, preflight=lambda: None, platform_id="metal", owner_authorized=True)
    other_dir = tmp_path / "blocked"
    with pytest.raises(SafetyStop):
        run_queue(job, run_dir=other_dir, execute=lambda _job: {}, preflight=lambda: None,
                  platform_id="metal", owner_authorized=True,
                  safety_check=lambda _job: (_ for _ in ()).throw(SafetyStop("metric stale")))


def test_platform_pairing_checks_controlled_fields_and_missing_ids():
    base = {"status": "complete", "observation_id": "obs-1", "platform_id": "metal",
            "run_kind": "production", "scientific_observation": True, "block_id": "P1-main",
            "protocol_version": "v1", "protocol_hash": "1" * 64, "config_id": "cfg-1",
            "model_id": "Qwen3-4B", "model_revision": "mrev", "gguf_repo_id": "repo",
            "gguf_revision": "grev", "artifact_hash": "2" * 64, "gguf_sha256": "2" * 64,
            "quantization": "Q4_K_M", "kv_cache": "f16", "dataset_id": "MERA",
            "dataset_revision": "d", "dataset_hash": "3" * 64,
            "split": "test", "task": "MathLogicQA", "item_id": "i1", "context_length": 4096,
            "workload_id": "quality", "mode": "held_out", "generation_seed": 42,
            "performance_repeat": None, "generation_config": {"temperature": 0},
            "tokenizer_revision": "tok", "chat_template_hash": "4" * 64,
            "reasoning_mode": "off", "llama_cpp_commit": "commit", "scorer_version": "s",
            "runtime_version": "llama.cpp v1", "comparison_series": "controlled",
            "platform_observation_id": "pobs-m",
            "hardware_manifest_hash": "3" * 64}
    left = dict(base)
    right = {**base, "platform_id": "cuda", "platform_observation_id": "pobs-c",
             "hardware_manifest_hash": "4" * 64}
    result = validate_and_pair_platforms({"metal": [left], "cuda": [right]})
    assert result["compatible"] is True
    assert len(result["pairs"]) == 1
    right["artifact_hash"] = "different"
    assert validate_and_pair_platforms({"metal": [left], "cuda": [right]})["compatible"] is False
    right["artifact_hash"] = base["artifact_hash"]
    right["observation_id"] = "obs-2"
    partial = validate_and_pair_platforms({"metal": [left], "cuda": [right]})
    assert partial["complete"] is False


def test_merge_rejects_incomplete_series_and_preserves_sources(tmp_path):
    left = tmp_path / "metal.jsonl"
    right = tmp_path / "cuda.jsonl"
    left.write_text('{"status":"complete","observation_id":"obs-1","platform_id":"metal"}\n', encoding="utf-8")
    right.write_text('{"status":"complete","observation_id":"obs-1","platform_id":"cuda"}\n', encoding="utf-8")
    original = left.read_bytes(), right.read_bytes()
    with pytest.raises(ValueError, match="несовместимы или неполны"):
        merge_result_files({"metal": left, "cuda": right}, tmp_path / "merged.json")
    assert (left.read_bytes(), right.read_bytes()) == original


def test_immutable_snapshot_and_measured_only_time_estimates(tmp_path):
    raw = tmp_path / "observations.jsonl"
    raw.write_text(json.dumps({"status": "complete", "observation_id": "obs-1", "block_id": "P1",
                               "platform_id": "metal", "model_id": "Qwen3-4B", "quantization": "Q4_K_M",
                               "dataset_id": "MERA", "task": "MathLogicQA", "run_kind": "production",
                               "scientific_observation": True, "mode": "held_out", "score": {"exact_match": 1.0},
                               "metrics": {"end_to_end_latency_s": 2.0}}) + "\n", encoding="utf-8")
    (tmp_path / "block_metrics.jsonl").write_text(json.dumps({"record_type": "block_resource_summary",
        "block_id": "P1", "observed_peaks": {"physical_footprint_bytes": 123}}) + "\n")
    before = raw.read_bytes()
    result = export_snapshot(records_path=raw, output_path=tmp_path / "slice.json")
    assert result["source_record_count"] == 1
    assert result["eligible_record_count"] == 1
    assert result["groups"][0]["primary_score_descriptive"]["mean"] == 1.0
    assert result["block_resource_summaries"][0]["observed_peaks"]["physical_footprint_bytes"] == 123
    assert result["block_resource_source"]["record_count"] == 1
    assert raw.read_bytes() == before
    estimate = estimate_remaining([
        {"observation_id": "obs-1", "model_id": "m", "quantization": "q", "context_length": 4, "platform_id": "p"},
        {"observation_id": "obs-2", "model_id": "m", "quantization": "q", "context_length": 4, "platform_id": "p"}],
        [{"status": "complete", "observation_id": "obs-1", "model_id": "m", "quantization": "q", "context_length": 4, "platform_id": "p", "duration_s": 10}])
    assert estimate["estimated_seconds_from_measured_strata"] == 10
    assert estimate["includes_load_time"] is False


def test_confirmatory_snapshot_excludes_fake_and_pilot_and_uses_nested_metrics(tmp_path):
    raw = tmp_path / "observations.jsonl"
    valid = {"status": "complete", "observation_id": "obs-valid", "block_id": "P1",
             "platform_id": "metal", "model_id": "Qwen3-4B", "quantization": "Q4_K_M",
             "dataset_id": "MERA", "task": "MathLogicQA", "run_kind": "production",
             "scientific_observation": True, "mode": "held_out", "score": {"exact_match": 0.5},
             "metrics": {"end_to_end_latency_s": 4.0}}
    fake = {**valid, "observation_id": "obs-fake", "run_kind": "synthetic_fake", "scientific_observation": False}
    pilot = {**valid, "observation_id": "obs-pilot", "block_id": "P0-pilot"}
    raw.write_text("".join(json.dumps(row) + "\n" for row in (valid, fake, pilot)))
    snapshot = export_snapshot(records_path=raw, output_path=tmp_path / "confirmatory.json")
    assert snapshot["analysis_stage"] == "confirmatory"
    assert snapshot["eligible_record_count"] == 1
    assert snapshot["excluded_record_count"] == 2
    assert snapshot["groups"][0]["duration_s"]["median"] == 4.0


def test_ruopenbookqa_macro_f1_is_computed_over_complete_imbalanced_group(tmp_path):
    raw = tmp_path / "observations.jsonl"
    records = []
    for index, (reference, prediction) in enumerate((("A", "A"), ("A", "A"), ("B", "A"), ("B", "A"))):
        records.append({"status": "complete", "observation_id": f"obs-{index}", "block_id": "P1-main",
                        "platform_id": "metal", "model_id": "Qwen3-4B", "quantization": "Q4_K_M",
                        "dataset_id": "MERA", "task": "ruOpenBookQA", "run_kind": "production",
                        "scientific_observation": True, "mode": "held_out", "split": "test",
                        "score": {"exact_match": float(reference == prediction)}, "reference": reference,
                        "prediction": prediction, "metrics": {"end_to_end_latency_s": 1.0}})
    raw.write_text("".join(json.dumps(row) + "\n" for row in records))
    snapshot = export_snapshot(records_path=raw, output_path=tmp_path / "macro.json")
    assert snapshot["groups"][0]["official_macro_f1"] == pytest.approx(1 / 3)
    assert snapshot["groups"][0]["primary_score_descriptive"]["mean"] == 0.5


def test_pairing_rejects_synthetic_and_pilot_records_from_confirmatory_stage():
    base = {"status": "complete", "observation_id": "obs-1", "platform_id": "metal",
            "run_kind": "production", "scientific_observation": True, "block_id": "P1-main",
            "mode": "held_out"}
    left = {**base, "platform_id": "metal"}
    right = {**base, "platform_id": "cuda"}
    # Records without the full provenance are rejected; changing their kind/stage is also explicit.
    for bad in ({**left, "run_kind": "synthetic_fake", "scientific_observation": False},
                {**left, "block_id": "P0-pilot"}):
        paired = validate_and_pair_platforms({"metal": [bad], "cuda": [right]})
        assert paired["compatible"] is False
        assert not paired["pairs"]


def test_basic_metrics_are_deterministic_and_bootstrap_is_reproducible():
    assert exact_match("  Ёж  ", "ёж") == 1.0
    assert token_f1("кот сидит", "кот спит") == pytest.approx(0.5)
    assert bootstrap_mean_interval([0, 1, 1], seed=7) == bootstrap_mean_interval([0, 1, 1], seed=7)


def test_task_dataset_preparation_pins_hash_split_and_sample(tmp_path):
    source = tmp_path / "export.jsonl"
    content = '{"id":"1","split":"test","task":"MathLogicQA","prompt":"Вопрос","answer":"42"}\n'
    source.write_text(content, encoding="utf-8")
    dataset = {"id": "MERA", "revision": "r1", "sha256": hashlib.sha256(content.encode()).hexdigest(),
               "path": str(source), "split": "test", "tasks": ["MathLogicQA"]}
    result = prepare_task_items(dataset, task="MathLogicQA", count=1, seed=7,
                                input_field="prompt", reference_field="answer")
    assert result == [{"id": "1", "task": "MathLogicQA", "split": "test",
                       "prompt": "Вопрос", "reference": "42"}]


def test_watchdog_fails_closed_and_locks_after_emergency():
    stopped = []
    now = [40.0]
    def metrics():
        return {"system_used_bytes": 1, "available_memory_estimate_bytes": 8 * 1024**3, "reclaimable_memory_estimate_bytes": 8 * 1024**3,
                "swap_delta_bytes": 0, "memory_pressure": {"category": "red"},
                "thermal_state": "nominal", "process_state": "running",
                "physical_footprint_bytes": 1, "pressure_notifications_available": True,
                "runtime_active": True, "safety_model": "apple_fully_reclaimable_estimate",
                "monotonic_time": now[0]}
    watchdog = SafetyWatchdog(sample=metrics, guard=SafetyGuard(), stop_runtime=lambda: stopped.append(True),
                              clock=lambda: now[0])
    assert watchdog.sample_once() == SafetyState.EMERGENCY_STOP
    assert watchdog.stop_event.is_set()
    assert stopped == [True]
    with pytest.raises(RuntimeError, match="safety event"):
        watchdog.start()


def test_watchdog_missing_required_metric_blocks_and_stops():
    stopped = []
    watchdog = SafetyWatchdog(sample=lambda: {}, guard=SafetyGuard(),
                              stop_runtime=lambda: stopped.append(True))
    assert watchdog.sample_once() == SafetyState.BLOCKED
    assert stopped == [True]


def test_queue_compiler_expands_only_explicit_cells_and_platform_pairs(tmp_path):
    plan = {"ready_for_execution": True, "protocol_version": "v1", "protocol_hash": "a" * 64,
            "campaign_id": "campaign-1", "cells": [{
                "block_id": "P1", "priority": 1, "model_id": "Qwen3-4B", "model_revision": "rev1",
                "gguf_repo_id": "repo/Qwen3-4B-GGUF", "gguf_revision": "ggufrev1",
                "gguf_sha256": "b" * 64, "quantization": "Q4_K_M", "kv_cache": "f16",
                "dataset_id": "MERA", "dataset_revision": "drev1", "dataset_sha256": "c" * 64,
                "split": "test", "task": "MathLogicQA", "context_length": 4096,
                "workload_id": "quality", "mode": "held_out", "generation_seed": 42,
                "generation_config": {"temperature": 0}, "tokenizer_revision": "tokrev",
                "chat_template_hash": "f" * 64, "reasoning_mode": "off", "llama_cpp_commit": "commit1",
                "scorer_version": "scorer-v1", "comparison_series": "controlled", "item_ids": ["i-1"],
                "platforms": [{"platform_id": "metal", "hardware_manifest_hash": "d" * 64,
                               "runtime_version": "b12675", "runtime_binary_sha256": "1" * 64},
                               {"platform_id": "cuda", "hardware_manifest_hash": "e" * 64,
                                "runtime_version": "b12675", "runtime_binary_sha256": "2" * 64}]}]}
    compiled = compile_jobs(plan)
    assert compiled["job_count"] == 2
    assert compiled["jobs"][0]["observation_id"] == compiled["jobs"][1]["observation_id"]
    assert compiled["jobs"][0]["platform_id"] != compiled["jobs"][1]["platform_id"]
    local = write_job_manifest(plan, tmp_path / "metal-jobs.json", platform_id="metal")
    assert local["job_count"] == 1
    assert local["local_platform_id"] == "metal"
    assert {row["platform_id"] for row in local["jobs"]} == {"metal"}
    with pytest.raises(FileExistsError):
        write_job_manifest(plan, tmp_path / "metal-jobs.json", platform_id="metal")
    with pytest.raises(ValueError, match="не утверждён"):
        compile_jobs({**plan, "ready_for_execution": False})
