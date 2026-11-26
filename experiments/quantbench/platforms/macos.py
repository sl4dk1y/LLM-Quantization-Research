"""Native metrics for Apple platforms; unsupported categories remain unavailable."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
import threading
import platform as py_platform
from pathlib import Path
from typing import Any

from ..platform import HardwareManifest, Metric, MetricStatus, PlatformAdapter
from ..runtime import build_server_args


SOURCE = Path(__file__).with_name("macos_metrics.m")


def _binary() -> Path:
    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / f"quantbench-macos-metrics-{os.getuid()}-{digest}"
    binary = cache / "macos_metrics"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    cache.mkdir(mode=0o700, parents=True, exist_ok=True)
    staging = cache / "macos_metrics.tmp"
    env = os.environ.copy()
    subprocess.run(["/usr/bin/clang", "-fmodules-cache-path=" + str(cache / "clang-cache"),
                    str(SOURCE), "-framework", "Foundation", "-o", str(staging)],
                   check=True, capture_output=True, text=True, timeout=60, env=env)
    staging.chmod(0o700)
    staging.replace(binary)
    return binary


def process_metrics(pid: int | None) -> dict[str, Any]:
    """Возвращает physical footprint и thermal state либо диагностические null."""
    try:
        requested_pid = pid or 0
        result = subprocess.run([str(_binary()), str(requested_pid)], check=True, capture_output=True,
                                text=True, timeout=3)
        data = json.loads(result.stdout)
        if data.get("pid") != requested_pid:
            raise ValueError("PID в ответе нативного сборщика не совпадает")
        return {"physical_footprint_bytes": data.get("physical_footprint_bytes"),
                "thermal_state": data.get("thermal_state"), "swap_used_bytes": data.get("swap_used_bytes"),
                "physical_memory_bytes": data.get("physical_memory_bytes"),
                "native_metrics_error": None}
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        return {"physical_footprint_bytes": None, "thermal_state": None, "swap_used_bytes": None,
                "physical_memory_bytes": None,
                "native_metrics_error": f"{type(exc).__name__}: {exc}"}


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", name], check=True,
                                capture_output=True, text=True, timeout=2)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class MemoryPressureEvents:
    """Читает изменения состояния из публичного Dispatch memory-pressure source."""

    def __init__(self) -> None:
        self.process = subprocess.Popen([str(_binary()), "--pressure-watch"], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1)
        self.available = self.process.poll() is None
        self.state: str | None = None
        self.last_error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._read, name="quantbench-memory-pressure", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                if self._stop.is_set():
                    return
                try:
                    event = json.loads(line).get("event")
                except (json.JSONDecodeError, AttributeError):
                    self.last_error = "Некорректное событие Dispatch memory pressure"
                    self.available = False
                    return
                if event not in {"normal", "warning", "critical"}:
                    self.last_error = f"Неизвестное состояние Dispatch memory pressure: {event!r}"
                    self.available = False
                    return
                self.state = event
        except (OSError, ValueError) as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.available = False
        finally:
            if not self._stop.is_set():
                self.available = False

    def close(self) -> None:
        self._stop.set()
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=0.5)
        if self.process.stdout is not None:
            self.process.stdout.close()
        self._thread.join(timeout=0.5)


class MacOSAdapter:
    platform_id = "apple-metal"

    def hardware_manifest(self) -> HardwareManifest:
        memory = _sysctl("hw.memsize")
        if not memory or not memory.isdigit():
            from ..monitor import snapshot
            memory_value = snapshot().get("physical_memory_bytes")
        else:
            memory_value = int(memory)
        return HardwareManifest(
            platform_id=self.platform_id, os_name="macOS", os_version=py_platform.mac_ver()[0] or None,
            architecture=py_platform.machine(), cpu_model=_sysctl("machdep.cpu.brand_string"),
            ram_bytes=memory_value if type(memory_value) is int else None,
            gpu_model=None, vram_bytes=None, backend="Metal")

    def capabilities(self) -> dict[str, Any]:
        return {"backend": "Metal", "unified_memory": True,
                "metrics": {"physical_footprint_bytes": "process", "rss_bytes": "process",
                            "thermal_state": "system", "memory_pressure.category": "event-only; no initial snapshot",
                            "available_memory_estimate_bytes": "broad vm_stat diagnostic; includes speculative, active and inactive pages",
                            "reclaimable_memory_estimate_bytes": "vm_stat free+file-backed+purgeable; safety estimate"}}

    def start_pressure_monitor(self) -> MemoryPressureEvents:
        return MemoryPressureEvents()

    def safety_snapshot(self, *, pid: int | None, swap_baseline_bytes: int | None,
                        pressure_events: MemoryPressureEvents) -> dict[str, Any]:
        from ..monitor import snapshot
        data = snapshot(pid)
        data["swap_delta_bytes"] = (max(0, data["swap_used_bytes"] - swap_baseline_bytes)
                                     if data.get("swap_used_bytes") is not None and swap_baseline_bytes is not None
                                     else None)
        data["runtime_active"] = pid is not None
        data["platform"] = "macOS"
        data["safety_model"] = "apple_fully_reclaimable_estimate"
        data["pressure_notifications_available"] = pressure_events.available
        if pressure_events.state is not None:
            data["memory_pressure"] = {"category": pressure_events.state,
                                        "source": "Dispatch memory pressure event"}
        else:
            data["memory_pressure"] = {"category": None,
                                        "source": "initial category unavailable; Safety Guard uses vm_stat estimate"}
        if not pressure_events.available:
            data["missing"]["pressure_notifications"] = pressure_events.last_error or "Dispatch source unavailable"
        return data

    def attach_safety(self, runtime: Any, *, guard: Any = None, interval_s: float | None = None,
                      sample_log_path: str | Path | None = None) -> Any:
        """Подключает монитор до старта server и на всём сроке его жизни."""
        from ..monitor import snapshot
        from ..safety import SafetyGuard
        from ..watchdog import SafetyWatchdog

        baseline = snapshot().get("swap_used_bytes")
        self._swap_baseline_bytes = baseline
        events = self.start_pressure_monitor()

        def sample() -> dict[str, Any]:
            process = runtime.process
            pid = process.pid if process is not None and process.poll() is None else None
            return self.safety_snapshot(pid=pid, swap_baseline_bytes=baseline, pressure_events=events)

        watchdog = SafetyWatchdog(sample=sample, guard=guard or SafetyGuard(
                                      recovery_window_s=30.0, sampling_interval_s=1.0),
                                  stop_runtime=lambda: runtime.stop(grace_s=0.25), interval_s=interval_s,
                                  sample_log_path=sample_log_path)
        runtime.platform_adapter = self
        runtime.safety_watchdog = watchdog
        runtime._pressure_monitor = events
        return watchdog

    def runtime_args(self, config: dict[str, Any], model: dict[str, Any]) -> list[str]:
        return build_server_args(config, model)

    def resource_snapshot(self, pid: int | None = None) -> dict[str, Metric]:
        from ..monitor import snapshot
        raw = snapshot(pid)
        baseline_swap = getattr(self, "_swap_baseline_bytes", None)
        raw["swap_delta_bytes"] = (max(0, raw["swap_used_bytes"] - baseline_swap)
                                   if raw.get("swap_used_bytes") is not None and baseline_swap is not None else None)
        specs = {"physical_footprint_bytes": ("bytes", True), "rss_bytes": ("bytes", True),
                 "system_used_bytes": ("bytes", True), "available_memory_estimate_bytes": ("bytes", True),
                 "reclaimable_memory_estimate_bytes": ("bytes", True), "file_backed_pages_bytes": ("bytes", True),
                 "free_pages_bytes": ("bytes", True), "active_pages_bytes": ("bytes", True),
                 "inactive_pages_bytes": ("bytes", True), "wired_pages_bytes": ("bytes", True),
                 "compressor_physical_bytes": ("bytes", True), "purgeable_pages_bytes": ("bytes", True),
                 "swap_used_bytes": ("bytes", True), "swap_delta_bytes": ("bytes", True),
                 "thermal_state": (None, True),
                 "memory_pressure": (None, True),
                 "gpu_vram_bytes": ("bytes", False), "gpu_utilization_percent": ("%", False)}
        result = {}
        for name, (unit, applicable) in specs.items():
            value = raw.get(name)
            if name == "memory_pressure" and isinstance(value, dict) and value.get("category") is None:
                value = None
            reason = raw.get("missing", {}).get(name) or raw.get("native_metrics_error")
            status = (MetricStatus.NOT_APPLICABLE if not applicable else
                      MetricStatus.AVAILABLE if value is not None else MetricStatus.UNAVAILABLE)
            result[name] = Metric(name, value, unit, status, reason)
        return result

    def preflight(self, config: dict[str, Any]) -> list[str]:
        if config.get("platform_id") not in {None, self.platform_id}:
            return ["Профиль конфигурации не соответствует Apple Metal"]
        return []

