"""Native metrics for Apple platforms; unsupported categories remain unavailable."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
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
                "thermal_state": data.get("thermal_state"), "native_metrics_error": None}
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        return {"physical_footprint_bytes": None, "thermal_state": None,
                "native_metrics_error": f"{type(exc).__name__}: {exc}"}


def _sysctl(name: str) -> str | None:
    try:
        result = subprocess.run(["/usr/sbin/sysctl", "-n", name], check=True,
                                capture_output=True, text=True, timeout=2)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


class MacOSAdapter:
    platform_id = "apple-metal"

    def hardware_manifest(self) -> HardwareManifest:
        memory = _sysctl("hw.memsize")
        return HardwareManifest(
            platform_id=self.platform_id, os_name="macOS", os_version=py_platform.mac_ver()[0] or None,
            architecture=py_platform.machine(), cpu_model=_sysctl("machdep.cpu.brand_string"),
            ram_bytes=int(memory) if memory and memory.isdigit() else None,
            gpu_model=None, vram_bytes=None, backend="Metal")

    def capabilities(self) -> dict[str, Any]:
        return {"backend": "Metal", "unified_memory": True,
                "metrics": {"physical_footprint_bytes": "process", "rss_bytes": "process",
                            "thermal_state": "system", "memory_pressure.category": "unavailable"}}

    def runtime_args(self, config: dict[str, Any], model: dict[str, Any]) -> list[str]:
        return build_server_args(config, model)

    def resource_snapshot(self, pid: int | None = None) -> dict[str, Metric]:
        from ..monitor import snapshot
        raw = snapshot(pid)
        specs = {"physical_footprint_bytes": ("bytes", True), "rss_bytes": ("bytes", True),
                 "system_used_bytes": ("bytes", True), "swap_used_bytes": ("bytes", True),
                 "thermal_state": (None, True), "memory_pressure": (None, True),
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

