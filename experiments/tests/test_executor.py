import hashlib
import json
import os
import copy
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from quantbench.dataset_pipeline import SCORER_VERSION
from quantbench.executor import (ExecutionBlocked, ExperimentExecutor, _reference_fields,
                                 verify_owner_authorization)
from quantbench.identity import digest, hardware_manifest_hash
from quantbench.platform import HardwareManifest, Metric, MetricStatus
from quantbench.runtime import LlamaServer
from quantbench.orchestrator import SafetyStop


class FakePlatform:
    platform_id = "test-platform"

    def __init__(self):
        self.manifest = HardwareManifest(platform_id=self.platform_id, os_name="test", os_version="1",
                                         architecture="test-cpu", ram_bytes=8 * 1024**3)

    def hardware_manifest(self):
        return self.manifest

    def preflight(self, _config):
        return []

    def resource_snapshot(self, _pid=None):
        return {"physical_footprint_bytes": Metric("physical_footprint_bytes", 1024, "bytes", MetricStatus.AVAILABLE),
                "rss_bytes": Metric("rss_bytes", 2048, "bytes", MetricStatus.AVAILABLE),
                "memory_pressure": Metric("memory_pressure", None, None, MetricStatus.UNAVAILABLE, "not sampled in fake")}


def _test_port():
    return 30000 + os.getpid() % 20000


def _fixture(tmp_path):
    model_path = tmp_path / "fixture.gguf"
    model_path.write_bytes(b"synthetic model placeholder; never executable")
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    source_path = tmp_path / "pinned-train.jsonl"
    source_path.write_text('{"fixture":"pinned synthetic source"}\n')
    source_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    item = {"id": "synthetic-1", "dataset": "MERA", "task": "MathLogicQA", "split": "heldout",
            "prompt": "Synthetic prompt. Answer with A.", "prompt_sha256": hashlib.sha256(b"Synthetic prompt. Answer with A.").hexdigest(),
            "prompt_template_id": "fixture-template-v1", "prompt_template_sha256": "a" * 64,
            "prompt_token_count": 7, "reference": "A", "references": ["A"],
            "token_counts_by_tokenizer": {"Qwen/Qwen3-4B@tokenizer-rev":
                {"prompt_tokens": 7, "chat_template_sha256": "b" * 64}}}
    export_path = tmp_path / "export.jsonl"
    export_path.write_text(json.dumps(item, ensure_ascii=False) + "\n")
    export_hash = hashlib.sha256(export_path.read_bytes()).hexdigest()
    sidecar = {"dataset": "MERA", "dataset_revision": "mera-pinned-rev", "task": "MathLogicQA",
               "split": "heldout", "prepared_sha256": export_hash, "source_sha256": source_hash,
               "sample_count": 1, "item_ids": [item["id"]],
               "source_provenance": {"revision": "mera-pinned-rev", "verified": True,
                                      "local_path": str(source_path), "sha256": source_hash}}
    export_path.with_suffix(".manifest.json").write_text(json.dumps(sidecar))
    server = tmp_path / "fake-llama-server"
    server.write_text("""#!/usr/bin/env python3
import argparse, json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
if '--version' in sys.argv: print('llama.cpp-fixture commit-fixture'); sys.exit(0)
p=argparse.ArgumentParser(); p.add_argument('--host'); p.add_argument('--port', type=int)
a,_=p.parse_known_args()
class H(BaseHTTPRequestHandler):
 def do_GET(self):
  self.send_response(200); self.end_headers(); self.wfile.write(b'{\"ok\":true}')
 def do_POST(self):
  n=int(self.headers.get('content-length','0')); request=json.loads(self.rfile.read(n))
  self.send_response(200); self.send_header('Content-Type','text/event-stream'); self.end_headers()
  content=request['messages'][0]['content']
  answer='1' if 'PARus fixture' in content else 'A'
  events=[{'choices':[{'delta':{'content':answer},'finish_reason':None}]}, {'timings':{'prompt_n':7,'cache_n':0,'predicted_n':1,'prompt_ms':2.5,'predicted_ms':1.5},'choices':[{'delta':{'content':''},'finish_reason':'stop'}]}]
  for e in events: self.wfile.write(('data: '+json.dumps(e)+'\\n\\n').encode()); self.wfile.flush()
  self.wfile.write(b'data: [DONE]\\n\\n'); self.wfile.flush()
 def log_message(self,*args): pass
ThreadingHTTPServer((a.host,a.port),H).serve_forever()
""")
    server.chmod(0o700)
    runtime_hash = hashlib.sha256(server.read_bytes()).hexdigest()
    adapter = FakePlatform()
    generation = {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": 42, "max_new_tokens": 8}
    config = {"mode": "held_out_quality", "contexts": [4096], "generation": generation,
              "repeats": {"warmup": 0},
              "models": [{"id": "Qwen/Qwen3-4B", "revision": "model-rev", "gguf_path": str(model_path),
                          "gguf_sha256": model_hash, "quantization": "Q4_K_M", "kv_cache": "f16"}],
              "datasets": [{"id": "MERA", "revision": "mera-pinned-rev", "sha256": export_hash,
                             "path": str(export_path), "split": "heldout", "tasks": ["MathLogicQA"]}],
              "runtime": {"executable": str(server), "host": "127.0.0.1", "port": _test_port(),
                          "load_timeout_s": 5, "request_timeout_s": 5,
                          "version": "llama.cpp-fixture commit-fixture",
                          "llama_cpp_commit": "commit-fixture", "gpu_layers": 0, "cpu_threads": 1,
                          "cpu_threads_batch": 1, "batch_size": 8, "ubatch_size": 4,
                          "flash_attn": False, "reasoning_mode": "off", "chat_template_source": "gguf_metadata",
                          "chat_template_sha256": "b" * 64},
              "generation_config": generation}
    job = {"job_id": "job-1", "observation_id": "obs-" + digest("synthetic-1"),
           "platform_id": adapter.platform_id,
           "hardware_manifest_hash": hardware_manifest_hash(adapter.manifest.canonical_fields()),
           "block_id": "P0", "priority": 0, "model_id": "Qwen/Qwen3-4B", "model_revision": "model-rev",
           "gguf_sha256": model_hash, "quantization": "Q4_K_M", "kv_cache": "f16", "dataset_id": "MERA",
           "dataset_revision": "mera-pinned-rev", "dataset_hash": export_hash, "split": "heldout",
           "task": "MathLogicQA", "item_id": item["id"], "context_length": 4096,
           "generation_config": generation, "scorer_version": SCORER_VERSION,
           "protocol_version": "M2.1", "protocol_hash": "c" * 64,
           "runtime_version": "llama.cpp-fixture commit-fixture", "runtime_binary_sha256": runtime_hash,
           "llama_cpp_commit": "commit-fixture", "tokenizer_revision": "tokenizer-rev",
           "chat_template_hash": "b" * 64, "reasoning_mode": "off", "attempt_policy": 2,
           "workload_id": "quality", "mode": "held_out", "generation_seed": 42,
           "performance_repeat": None}
    manifest = {"schema_version": 1, "job_count": 1, "jobs": [job]}
    manifest["job_plan_hash"] = digest(manifest["jobs"])
    manifest["local_platform_id"] = adapter.platform_id
    return config, manifest, adapter, item


def test_fake_http_e2e_scores_and_isolates_synthetic_observations(tmp_path):
    config, manifest, adapter, row = _fixture(tmp_path)
    source_path = Path(config["datasets"][0]["path"])
    parus_row = {**row, "id": "synthetic-parus", "task": "PARus", "prompt": "PARus fixture: answer 1",
                 "prompt_sha256": hashlib.sha256(b"PARus fixture: answer 1").hexdigest(), "reference": "1",
                 "references": ["1"]}
    parus_path = tmp_path / "parus.jsonl"
    parus_path.write_text(json.dumps(parus_row) + "\n")
    parus_hash = hashlib.sha256(parus_path.read_bytes()).hexdigest()
    source_provenance = json.loads(source_path.with_suffix(".manifest.json").read_text())["source_provenance"]
    parus_path.with_suffix(".manifest.json").write_text(json.dumps({
        "dataset": "MERA", "dataset_revision": "mera-pinned-rev", "task": "PARus", "split": "heldout",
        "prepared_sha256": parus_hash, "source_sha256": source_provenance["sha256"], "sample_count": 1,
        "item_ids": [parus_row["id"]], "source_provenance": source_provenance}))
    config["datasets"].append({"id": "MERA", "revision": "mera-pinned-rev", "sha256": parus_hash,
                                "path": str(parus_path), "split": "heldout", "tasks": ["PARus"]})
    second_job = copy.deepcopy(manifest["jobs"][0])
    second_job.update({"job_id": "job-parus", "observation_id": "obs-" + digest("synthetic-parus"),
                       "task": "PARus", "item_id": parus_row["id"], "dataset_hash": parus_hash})
    manifest["jobs"].append(second_job)
    manifest["job_count"] = len(manifest["jobs"])
    manifest["job_plan_hash"] = digest(manifest["jobs"])
    run_dir = tmp_path / "fake-runs" / "case-1"
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=run_dir, fake=True,
                                  runtime_factory=lambda job, model: LlamaServer(
                                      {**config, "contexts": [job["context_length"]]}, model,
                                      log_path=run_dir / "runtime.log"))
    status = executor.run_fake()
    records = [json.loads(line) for line in (run_dir / "observations.jsonl").read_text().splitlines()]
    assert status["completed_jobs"] == 2
    assert {record["task"] for record in records} == {"MathLogicQA", "PARus"}
    assert all(record["run_kind"] == "synthetic_fake" for record in records)
    assert all(record["scientific_observation"] is False for record in records)
    assert {record["raw_model_output"] for record in records} == {"A", "1"}
    assert all(record["score"]["exact_match"] == 1 for record in records)
    assert all(record["metrics"]["ttft_s"] is not None for record in records)
    assert all(record["metrics"]["prompt_tokens"] == 7 for record in records)
    assert all(record["metrics"]["prompt_eval_tokens"] == 7 for record in records)
    assert all(record["metrics"]["cached_prompt_tokens"] == 0 for record in records)
    assert all(record["metrics"]["generated_tokens"] == 1 for record in records)
    assert all(record["metrics"]["throughput_tokens_per_s"] is not None for record in records)
    assert all(record["metrics"]["resource_metrics"]["after"]["rss_bytes"]["value"] == 2048 for record in records)
    assert not (tmp_path / "results").exists()


def test_runtime_metrics_separate_prompt_eval_from_cached_prompt_tokens():
    from quantbench.executor import _runtime_metrics

    metrics = _runtime_metrics({"ttft_s": 0.4, "request_wall_s": 0.6,
                                "timings": {"prompt_n": 119, "cache_n": 157,
                                            "prompt_ms": 368.658, "predicted_n": 8,
                                            "predicted_ms": 207.723}},
                               {"prompt_token_count": 276})
    assert metrics["prompt_eval_tokens"] == 119
    assert metrics["cached_prompt_tokens"] == 157
    assert metrics["prompt_tokens"] == metrics["prompt_eval_tokens"] + metrics["cached_prompt_tokens"] == 276
    assert metrics["export_prompt_tokens"] == 276
    assert metrics["prefill_duration_s"] == pytest.approx(0.368658)
    assert metrics["generated_tokens"] == 8
    assert metrics["decode_duration_s"] == pytest.approx(0.207723)
    assert metrics["ttft_s"] == pytest.approx(0.4)


def test_runtime_metrics_do_not_assume_cache_count_when_missing():
    from quantbench.executor import _runtime_metrics

    metrics = _runtime_metrics({"timings": {"prompt_n": 7}}, {})
    assert metrics["prompt_eval_tokens"] == 7
    assert metrics["cached_prompt_tokens"] is None
    assert metrics["prompt_tokens"] is None


def test_fake_run_rejects_manifest_digest_change(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    manifest["jobs"][0]["item_id"] = "other"
    with pytest.raises(ValueError, match="hash"):
        ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                           run_dir=tmp_path / "fake", fake=True)


def test_fake_run_rejects_missing_model_and_export(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    config["models"][0]["gguf_path"] = str(tmp_path / "missing.gguf")
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "model-check")
    with pytest.raises(ValueError, match="GGUF"):
        executor.validate_inputs()


def test_fake_run_rejects_missing_dataset_artifact(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    config["datasets"][0]["path"] = str(tmp_path / "missing.jsonl")
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "fake-dataset", fake=True)
    with pytest.raises(ValueError, match="отсутствует"):
        executor.validate_inputs()


def test_fake_run_rejects_mismatched_tokenizer_and_chat_template_profile(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    row_path = Path(config["datasets"][0]["path"])
    row = json.loads(row_path.read_text())
    row["token_counts_by_tokenizer"]["Qwen/Qwen3-4B@tokenizer-rev"]["chat_template_sha256"] = "d" * 64
    row_path.write_text(json.dumps(row) + "\n")
    export_hash = hashlib.sha256(row_path.read_bytes()).hexdigest()
    config["datasets"][0]["sha256"] = export_hash
    manifest["jobs"][0]["dataset_hash"] = export_hash
    manifest["job_plan_hash"] = digest(manifest["jobs"])
    sidecar_path = row_path.with_suffix(".manifest.json")
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["prepared_sha256"] = export_hash
    sidecar_path.write_text(json.dumps(sidecar))
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "fake-template", fake=True)
    with pytest.raises(ValueError, match="Chat template hash"):
        executor.validate_inputs()


def test_executor_fails_closed_without_hardware_manifest(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    adapter.hardware_manifest = lambda: None
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "no-hardware", fake=True)
    with pytest.raises(ValueError, match="Аппаратный manifest"):
        executor.validate_inputs()


def test_fake_run_rejects_scorer_mismatch_without_scientific_record(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    manifest["jobs"][0]["scorer_version"] = "unknown-scorer"
    manifest["job_plan_hash"] = digest(manifest["jobs"])
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "fake", fake=True)
    with pytest.raises(ValueError, match="Scorer version"):
        executor.validate_inputs()
    assert not (tmp_path / "fake" / "observations.jsonl").exists()


def test_cli_fake_run_never_executes_configured_llama_server(monkeypatch, tmp_path):
    from quantbench import cli

    config, manifest, adapter, _ = _fixture(tmp_path)
    marker = tmp_path / "real-runtime-was-executed"
    executable = tmp_path / "runtime-trap"
    executable.write_text(f"#!/bin/sh\ntouch '{marker}'\nexit 1\n")
    executable.chmod(0o700)
    config["runtime"]["executable"] = str(executable)
    jobs_path = tmp_path / "jobs.json"
    jobs_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(cli, "load_config", lambda _path: config)
    monkeypatch.setattr("quantbench.platforms.macos.MacOSAdapter", lambda: adapter)
    output = tmp_path / "runs" / "cli-fake"
    assert cli.main(["fake-run", "--config", "ignored.json", "--jobs", str(jobs_path),
                     "--run-dir", str(output)]) == 0
    record = json.loads((output / "observations.jsonl").read_text().splitlines()[0])
    assert record["run_kind"] == "synthetic_fake"
    assert record["scientific_observation"] is False
    assert not marker.exists()


def test_compatible_jobs_reuse_one_runtime_and_truncated_output_fails_score(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    config["repeats"]["warmup"] = 2
    job = manifest["jobs"][0]
    job["generation_config"] = copy.deepcopy(job["generation_config"])
    second = copy.deepcopy(job)
    second.update({"job_id": "job-2", "observation_id": "obs-" + digest("second-item")})
    second["generation_config"] = {**second["generation_config"], "max_new_tokens":
                                   second["generation_config"]["max_new_tokens"] + 1}
    config["generation"]["max_new_tokens"] = second["generation_config"]["max_new_tokens"]
    manifest["jobs"].append(second)
    manifest["job_count"] = 2
    manifest["job_plan_hash"] = digest(manifest["jobs"])

    class CountingRuntime:
        def __init__(self):
            self.starts = self.stops = self.calls = 0
            self.process = None
            self.safety_watchdog = None

        def start(self):
            self.starts += 1

        def generate(self, *_args, **_kwargs):
            self.calls += 1
            return {"text": "A", "finish_reason": "length", "ttft_s": 0.1,
                    "timings": {"prompt_n": 7, "predicted_n": 1, "prompt_ms": 1, "predicted_ms": 2},
                    "request_wall_s": 0.2}

        def stop(self):
            self.stops += 1

    runtime = CountingRuntime()
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "reuse", fake=True,
                                  runtime_factory=lambda _job, _model: runtime)
    executor.validate_inputs()
    first = executor._execute_job(job, synthetic=True)
    second_row = executor._execute_job(second, synthetic=True)
    executor._close_runtime_session()
    assert (runtime.starts, runtime.stops) == (1, 1)
    assert runtime.calls == 4  # два прогрева на блок и два quality items
    assert all(row["raw_model_output"] == "A" and row["truncated"] is True for row in (first, second_row))
    assert all(row["finish_reason"] == "length" and row["score"]["exact_match"] == 0
               for row in (first, second_row))
    assert all(row["score"]["truncation_rejected"] is True for row in (first, second_row))


def test_validate_inputs_rejects_task_output_cap_above_configured_limit(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    manifest["jobs"][0]["generation_config"] = copy.deepcopy(manifest["jobs"][0]["generation_config"])
    manifest["jobs"][0]["generation_config"]["max_new_tokens"] += 1
    manifest["job_plan_hash"] = digest(manifest["jobs"])
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "too-many-tokens", fake=True)
    with pytest.raises(ValueError, match="Параметры генерации отличаются"):
        executor.validate_inputs()


def test_production_requires_owner_file_and_stays_closed_after_matching_authorization(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "production")
    with pytest.raises(PermissionError, match="файл разрешения"):
        executor.run_production()
    authorization = tmp_path / "owner-authorization.json"
    authorization.write_text(json.dumps({"authorized": True, "purpose": "real_inference",
                                         "plan_hash": manifest["job_plan_hash"],
                                         "platform_id": adapter.platform_id,
                                         "authorization_id": "owner-decision-1", "owner": "owner"}))
    with pytest.raises(ExecutionBlocked, match="M4"):
        executor.run_production(authorization_file=authorization)
    assert not (tmp_path / "production" / "observations.jsonl").exists()


def test_owner_authorization_is_bound_to_exact_plan_and_platform(tmp_path):
    path = tmp_path / "owner.json"
    path.write_text(json.dumps({"authorized": True, "purpose": "real_inference", "plan_hash": "f" * 64,
                                "platform_id": "other-platform", "authorization_id": "decision-2",
                                "owner": "owner"}))
    with pytest.raises(PermissionError, match="не подтверждает"):
        verify_owner_authorization(path, plan_hash="e" * 64, platform_id="test-platform")


def test_libra_reference_payload_preserves_all_acceptable_answers():
    assert _reference_fields({"reference": "сад", "references": ["сад", "саду", "в саду"]}) == {
        "reference": "сад", "references": ["сад", "саду", "в саду"]}
    assert _reference_fields({"positive_outputs": ["ответ 1", "вариант 2"]}) == {
        "reference": None, "references": ["ответ 1", "вариант 2"]}


def test_safety_stop_is_persisted_without_retry(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)

    class StoppedWatchdog:
        stop_event = threading.Event()
        last_error = "critical Dispatch event"
        last_state = SimpleNamespace(value="EMERGENCY_STOP")

    StoppedWatchdog.stop_event.set()

    class Runtime:
        process = None
        safety_watchdog = StoppedWatchdog

        def start(self):
            return None

        def generate(self, *_args, **_kwargs):
            raise RuntimeError("cancelled by safety event")

        def stop(self):
            return None

    executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                                  run_dir=tmp_path / "safety-stop", fake=True,
                                  runtime_factory=lambda _job, _model: Runtime())
    with pytest.raises(SafetyStop, match="critical Dispatch event"):
        executor.run_fake()
    records = [json.loads(line) for line in (tmp_path / "safety-stop" / "observations.jsonl").read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["status"] == "blocked"


def test_missing_runtime_parameter_evidence_is_unverified(monkeypatch, tmp_path):
    from urllib.error import URLError

    config, _, _, _ = _fixture(tmp_path)
    runtime = LlamaServer(config, config["models"][0], log_path=tmp_path / "runtime.log")
    monkeypatch.setattr("quantbench.runtime.urllib.request.urlopen",
                        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("not available")))
    result = runtime.verify_parameters({"gpu_layers": 0}, {"min_p": 0})
    assert result["status"] == "unverified"
    assert result["static"]["gpu_layers"] == "unverified"
    assert result["generation"]["min_p"] == "unverified"


def test_fake_results_cannot_be_written_under_results_directory(tmp_path):
    config, manifest, adapter, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="вне results"):
        ExperimentExecutor(config=config, manifest=manifest, platform_adapter=adapter,
                           run_dir=tmp_path / "results" / "raw", fake=True)

