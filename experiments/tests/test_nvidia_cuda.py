"""Тесты адаптера NVIDIA CUDA на fake-пробниках без реального GPU.

Тесты не используют ``tmp_path``, не запускают ``nvidia-smi``/инференс и не
создают аппаратную нагрузку: все пробники (nvidia-smi, память, процесс,
runtime, CPU, ОС) подменяются детерминированными fake-функциями.
"""

from __future__ import annotations

import pytest

from quantbench.platform import HardwareManifest, Metric, MetricStatus
from quantbench.platforms.nvidia_cuda import (
    CUDASafetyProfile,
    CUDASafetyState,
    NVIDIA_CUDAAdapter,
    REQUIRED_METRICS,
    evaluate_cuda_safety,
    offload_mode,
    parse_nvidia_smi_query,
    parse_nvidia_smi_summary,
)
from quantbench.runtime import build_server_args


def _gpu(**overrides):
    data = {
        "name": "NVIDIA GeForce GTX 1660",
        "driver_version": "560.94",
        "cuda_version": "12.6",
        "memory_total_bytes": 6 * 1024**3,
        "memory_used_bytes": 1024 * 1024**2,
        "memory_free_bytes": 5 * 1024**3,
        "temperature_celsius": 55,
        "utilization_percent": 10,
        "power_draw_watts": 26.5,
        "power_limit_watts": 120.0,
        "compute_capability": "7.5",
    }
    data.update(overrides)
    return data


def _memory(**overrides):
    data = {
        "total_phys_bytes": 16 * 1024**3,
        "avail_phys_bytes": 8 * 1024**3,
        "total_pagefile_bytes": 32 * 1024**3,
        "avail_pagefile_bytes": 24 * 1024**3,
        "memory_load_percent": 50,
    }
    data.update(overrides)
    return data


def _process(**overrides):
    data = {"rss_bytes": 100 * 1024**2, "state": "running"}
    data.update(overrides)
    return data


def _adapter(gpu=None, memory=None, process=None, runtime="llama.cpp v1"):
    return NVIDIA_CUDAAdapter(
        nvidia_smi_probe=lambda: gpu,
        memory_probe=lambda: memory,
        process_probe=lambda pid: process,
        runtime_probe=lambda executable: runtime,
        cpu_probe=lambda: "AMD Ryzen 5 2600 Six-Core Processor",
        os_probe=lambda: {"system": "Windows", "release": "10", "version": "10.0.19045", "machine": "AMD64"},
    )


def _config(**runtime_overrides):
    config = {
        "schema_version": 1,
        "protocol": {"version": "0.1", "sha256": "a" * 64},
        "mode": "dry_run",
        "seed": 1,
        "models": [{"id": "m", "revision": "r", "gguf_path": "m.gguf", "gguf_sha256": "b" * 64,
                    "quantization": "Q4_K_M", "kv_cache": "f16"}],
        "datasets": [{"id": "d", "revision": "r", "sha256": "c" * 64, "path": "d.jsonl",
                      "split": "dev", "tasks": ["t"]}],
        "runtime": {"executable": "llama-server", "host": "127.0.0.1", "port": 8080,
                    "load_timeout_s": 120, "request_timeout_s": 120, "gpu_layers": 0,
                    "cpu_threads": 4, "cpu_threads_batch": 4, "batch_size": 512, "ubatch_size": 128,
                    "flash_attn": False, "reasoning_mode": "off", "chat_template_source": "gguf_metadata"},
        "generation": {"temperature": 0, "top_p": 1, "top_k": 0, "seed": 42, "max_new_tokens": 8},
        "sampling": {"strategy": "sha256_rank", "items_per_task": 1},
        "contexts": [128],
        "limits": {"max_system_used_gib": 12},
        "repeats": {"performance": 1, "warmup": 1},
        "output": {"root": "results/raw", "append_only": True},
    }
    config["runtime"].update(runtime_overrides)
    return config


def _snapshot(**overrides):
    snapshot = {
        "system_ram_bytes": Metric("system_ram_bytes", 16 * 1024**3, "bytes", MetricStatus.AVAILABLE),
        "system_available_bytes": Metric("system_available_bytes", 8 * 1024**3, "bytes", MetricStatus.AVAILABLE),
        "vram_total_bytes": Metric("vram_total_bytes", 6 * 1024**3, "bytes", MetricStatus.AVAILABLE),
        "vram_used_bytes": Metric("vram_used_bytes", 1024**3, "bytes", MetricStatus.AVAILABLE),
        "vram_free_bytes": Metric("vram_free_bytes", 5 * 1024**3, "bytes", MetricStatus.AVAILABLE),
        "gpu_utilization_percent": Metric("gpu_utilization_percent", 10, "%", MetricStatus.AVAILABLE),
        "gpu_temperature_celsius": Metric("gpu_temperature_celsius", 55, "°C", MetricStatus.AVAILABLE),
    }
    snapshot.update(overrides)
    return snapshot


# --- nvidia-smi parsing ---

def test_parse_nvidia_smi_query_converts_bytes_and_units():
    fields = ("name", "driver_version", "memory.total", "memory.used", "memory.free",
              "temperature.gpu", "utilization.gpu", "power.draw", "power.limit", "compute_cap")
    row = "NVIDIA GeForce GTX 1660, 560.94, 6144, 2979, 2988, 55, 40, 26.53, 120.00, 7.5"
    parsed = parse_nvidia_smi_query(fields, row)
    assert parsed["name"] == "NVIDIA GeForce GTX 1660"
    assert parsed["driver_version"] == "560.94"
    assert parsed["memory_total_bytes"] == 6144 * 1024**2
    assert parsed["memory_used_bytes"] == 2979 * 1024**2
    assert parsed["memory_free_bytes"] == 2988 * 1024**2
    assert parsed["temperature_celsius"] == 55
    assert parsed["utilization_percent"] == 40
    assert parsed["power_draw_watts"] == 26.53
    assert parsed["compute_capability"] == "7.5"


def test_parse_nvidia_smi_query_turns_na_into_none_not_zero():
    fields = ("name", "driver_version", "memory.total", "memory.used", "memory.free",
              "temperature.gpu", "utilization.gpu", "power.draw", "power.limit", "compute_cap")
    row = "NVIDIA GeForce GTX 1660, 560.94, 6144, 2979, 2988, N/A, 40, N/A, 120.00, 7.5"
    parsed = parse_nvidia_smi_query(fields, row)
    assert parsed["temperature_celsius"] is None
    assert parsed["power_draw_watts"] is None
    assert parse_nvidia_smi_query(fields, "") is None


def test_parse_nvidia_smi_summary_extracts_versions():
    text = "| NVIDIA-SMI 560.94   Driver Version: 560.94   CUDA Version: 12.6 |"
    parsed = parse_nvidia_smi_summary(text)
    assert parsed["driver_version"] == "560.94"
    assert parsed["cuda_version"] == "12.6"
    assert parse_nvidia_smi_summary("no matches here") == {}


# --- offload modes ---

def test_offload_mode_classification():
    assert offload_mode(0) == "cpu_only"
    assert offload_mode(5) == "partial"
    assert offload_mode("all") == "full"
    assert offload_mode("auto") == "auto"
    assert offload_mode(None) == "unknown"


# --- manifest and capabilities ---

def test_hardware_manifest_fills_facts_without_fakes():
    adapter = _adapter(gpu=_gpu(), memory=_memory())
    manifest = adapter.hardware_manifest()
    assert isinstance(manifest, HardwareManifest)
    assert manifest.platform_id == "nvidia-cuda"
    assert manifest.os_name == "Windows"
    assert manifest.os_version == "10.0.19045"
    assert manifest.architecture == "AMD64"
    assert manifest.cpu_model == "AMD Ryzen 5 2600 Six-Core Processor"
    assert manifest.ram_bytes == 16 * 1024**3
    assert manifest.gpu_model == "NVIDIA GeForce GTX 1660"
    assert manifest.vram_bytes == 6 * 1024**3
    assert manifest.driver_version == "560.94"
    assert manifest.backend == "CUDA"


def test_manifest_keeps_unknown_gpu_fields_null():
    adapter = _adapter(gpu=None, memory=None)
    manifest = adapter.hardware_manifest()
    assert manifest.gpu_model is None
    assert manifest.vram_bytes is None
    assert manifest.driver_version is None
    assert manifest.ram_bytes is None


def test_capabilities_report_backend_and_metric_specs():
    adapter = _adapter(gpu=_gpu())
    caps = adapter.capabilities()
    assert caps["backend"] == "CUDA"
    assert caps["unified_memory"] is False
    assert caps["compute_capability"] == "7.5"
    assert caps["cuda_driver_api_version"] == "12.6"
    assert set(caps["offload_modes"]) == {"cpu_only", "partial", "full", "auto"}
    assert caps["wddm"]["process_vram_attribution"] == "unavailable"
    assert set(caps["metrics"]) == set(REQUIRED_METRICS) | {
        "process_rss_bytes", "process_state", "gpu_power_watts",
        "pagefile_total_bytes", "pagefile_used_bytes",
    }
    for name, spec in caps["metrics"].items():
        assert {"unit", "source", "semantics", "required", "limitations"} <= set(spec)


# --- runtime args ---

def test_runtime_args_do_not_change_common_args_and_add_main_gpu():
    adapter = _adapter(gpu=_gpu(), memory=_memory())
    model = _config()["models"][0]
    cfg_cpu = _config(gpu_layers=0)
    cfg_gpu = _config(gpu_layers=5)
    base_cpu = build_server_args(cfg_cpu, model)
    base_gpu = build_server_args(cfg_gpu, model)

    cpu_args = adapter.runtime_args(cfg_cpu, model)
    gpu_args = adapter.runtime_args(cfg_gpu, model)

    assert cpu_args == base_cpu  # CPU-only: без CUDA-специфичных флагов
    assert gpu_args == base_gpu + ["--main-gpu", "0"]  # GPU offload: только --main-gpu


def test_runtime_args_raise_on_non_loopback_like_common_builder():
    adapter = _adapter(gpu=_gpu(), memory=_memory())
    model = _config()["models"][0]
    config = _config(gpu_layers=0)
    config["runtime"]["host"] = "0.0.0.0"
    with pytest.raises(ValueError, match="loopback"):
        adapter.runtime_args(config, model)


# --- resource snapshot ---

def test_resource_snapshot_with_pid_and_gpu():
    adapter = _adapter(gpu=_gpu(), memory=_memory(), process=_process())
    snapshot = adapter.resource_snapshot(pid=123)
    assert snapshot["system_ram_bytes"].value == 16 * 1024**3
    assert snapshot["system_ram_bytes"].status == MetricStatus.AVAILABLE
    assert snapshot["system_available_bytes"].value == 8 * 1024**3
    assert snapshot["vram_total_bytes"].value == 6 * 1024**3
    assert snapshot["vram_used_bytes"].value == 1024 * 1024**2
    assert snapshot["vram_free_bytes"].value == 5 * 1024**3
    assert snapshot["gpu_utilization_percent"].value == 10
    assert snapshot["gpu_temperature_celsius"].value == 55
    assert snapshot["gpu_power_watts"].value == 26.5
    assert snapshot["pagefile_used_bytes"].value == 8 * 1024**3
    assert snapshot["process_rss_bytes"].value == 100 * 1024**2
    assert snapshot["process_state"].value == "running"


def test_resource_snapshot_without_pid_marks_process_not_applicable():
    adapter = _adapter(gpu=_gpu(), memory=_memory())
    snapshot = adapter.resource_snapshot(pid=None)
    assert snapshot["process_rss_bytes"].status == MetricStatus.NOT_APPLICABLE
    assert snapshot["process_state"].status == MetricStatus.NOT_APPLICABLE
    assert snapshot["process_rss_bytes"].value is None


def test_resource_snapshot_cuda_unavailable_marks_gpu_metrics_unavailable():
    adapter = _adapter(gpu=None, memory=_memory())
    snapshot = adapter.resource_snapshot()
    for name in ("vram_total_bytes", "vram_used_bytes", "vram_free_bytes",
                 "gpu_utilization_percent", "gpu_temperature_celsius", "gpu_power_watts"):
        assert snapshot[name].status == MetricStatus.UNAVAILABLE
        assert snapshot[name].value is None


# --- preflight ---

def test_preflight_passes_for_healthy_environment():
    adapter = _adapter(gpu=_gpu(), memory=_memory(), runtime="llama.cpp v1")
    assert adapter.preflight(_config()) == []


def test_preflight_rejects_wrong_platform():
    adapter = _adapter(gpu=_gpu(), memory=_memory())
    config = _config()
    config["platform_id"] = "apple-metal"
    blockers = adapter.preflight(config)
    assert any("NVIDIA CUDA" in b for b in blockers)


def test_preflight_blocks_when_cuda_unavailable():
    adapter = _adapter(gpu=None, memory=_memory(), runtime="llama.cpp v1")
    blockers = adapter.preflight(_config())
    assert any("CUDA недоступны" in b for b in blockers)


def test_preflight_blocks_on_insufficient_vram():
    gpu = _gpu(memory_free_bytes=100 * 1024**2)  # ниже аварийного порога 256 MiB
    adapter = _adapter(gpu=gpu, memory=_memory(), runtime="llama.cpp v1")
    blockers = adapter.preflight(_config())
    assert any("VRAM" in b for b in blockers)


def test_preflight_blocks_on_insufficient_ram():
    memory = _memory(avail_phys_bytes=100 * 1024**2)  # ниже аварийного порога 512 MiB
    adapter = _adapter(gpu=_gpu(), memory=memory, runtime="llama.cpp v1")
    blockers = adapter.preflight(_config())
    assert any("системная память" in b for b in blockers)


def test_preflight_blocks_on_unavailable_required_metric():
    gpu = _gpu(temperature_celsius=None)
    adapter = _adapter(gpu=gpu, memory=_memory(), runtime="llama.cpp v1")
    blockers = adapter.preflight(_config())
    assert any("gpu_temperature_celsius" in b for b in blockers)


def test_preflight_blocks_on_incompatible_runtime():
    adapter = _adapter(gpu=_gpu(), memory=_memory(), runtime=None)
    blockers = adapter.preflight(_config())
    assert any("runtime несовместим" in b for b in blockers)


# --- safety evaluation ---

def test_evaluate_cuda_safety_emergency_stop_on_critical_temperature():
    snapshot = _snapshot(gpu_temperature_celsius=Metric(
        "gpu_temperature_celsius", 95, "°C", MetricStatus.AVAILABLE))
    state, reasons = evaluate_cuda_safety(snapshot, CUDASafetyProfile())
    assert state == CUDASafetyState.EMERGENCY_STOP
    assert reasons


def test_evaluate_cuda_safety_blocked_on_missing_required_metric():
    snapshot = _snapshot()
    del snapshot["gpu_temperature_celsius"]
    state, reasons = evaluate_cuda_safety(snapshot, CUDASafetyProfile())
    assert state == CUDASafetyState.BLOCKED
    assert any("gpu_temperature_celsius" in r for r in reasons)


def test_evaluate_cuda_safety_safe_for_normal_metrics():
    state, reasons = evaluate_cuda_safety(_snapshot(), CUDASafetyProfile())
    assert state == CUDASafetyState.SAFE
    assert reasons == []


# --- contract compatibility ---

def test_adapter_satisfies_platform_contract():
    adapter = _adapter(gpu=_gpu(), memory=_memory())
    assert adapter.platform_id == "nvidia-cuda"
    for method in ("hardware_manifest", "capabilities", "runtime_args", "resource_snapshot", "preflight"):
        assert callable(getattr(adapter, method))
    assert isinstance(adapter.hardware_manifest(), HardwareManifest)
    assert isinstance(adapter.capabilities(), dict)
    assert isinstance(adapter.runtime_args(_config(gpu_layers=0), _config()["models"][0]), list)
    assert all(isinstance(metric, Metric) for metric in adapter.resource_snapshot().values())
    assert isinstance(adapter.preflight(_config()), list)
    assert adapter.cuda_available() is True
    assert adapter.runtime_compatible("llama-server") is True
