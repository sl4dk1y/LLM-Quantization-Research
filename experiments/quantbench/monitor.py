"""Read-only macOS resource snapshots; unsupported measures remain null."""

from __future__ import annotations

import datetime as dt
import os
import re
import subprocess
import time
from typing import Any


def _command(argv: list[str], timeout: float = 3.0) -> tuple[int, str]:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode, result.stdout + result.stderr
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def _proc_memory(pid: int) -> dict[str, Any]:
    # macOS `ps` provides RSS, not Apple's phys_footprint. Never substitute it.
    code, output = _command(["/bin/ps", "-o", "rss=", "-p", str(pid)])
    if code == 0 and output.strip().isdigit():
        return {"rss_bytes": int(output.strip()) * 1024, "physical_footprint_bytes": None,
                "physical_footprint_reason": "proc_pid_rusage/ri_phys_footprint binding is not implemented"}
    return {"rss_bytes": None, "physical_footprint_bytes": None, "physical_footprint_reason": output.strip() or "process metric unavailable"}


def snapshot(pid: int | None = None) -> dict[str, Any]:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    result: dict[str, Any] = {
        "timestamp": timestamp,
        "monotonic_time": time.monotonic(),
        "platform": "macOS" if os.uname().sysname == "Darwin" else os.uname().sysname,
        "process_state": "unknown", "rss_bytes": None, "physical_footprint_bytes": None,
        "system_used_bytes": None, "available_memory_estimate_bytes": None,
        "reclaimable_memory_estimate_bytes": None, "file_backed_pages_bytes": None,
        "memory_pressure": None, "swap_used_bytes": None,
        "thermal_state": None, "swap_delta_bytes": None, "missing": {},
        "physical_memory_bytes": None,
    }
    if pid is not None:
        mem = _proc_memory(pid)
        result.update(mem)
        code, status = _command(["/bin/ps", "-o", "stat=", "-p", str(pid)])
        result["process_state"] = status.strip() if code == 0 and status.strip() else "not_found"
        if code != 0:
            result["missing"]["process_state"] = status.strip()
    else:
        result["missing"]["process"] = "no target pid supplied"
    if os.uname().sysname == "Darwin":
        from .platforms.macos import process_metrics
        result.update(process_metrics(pid))
        if result.get("physical_footprint_bytes") is not None:
            result.pop("physical_footprint_reason", None)
        if pid is None:
            result["physical_footprint_bytes"] = None

    code, vm = _command(["/usr/bin/vm_stat"])
    if code == 0:
        page = re.search(r"page size of (\d+) bytes", vm)
        values = {k: int(v.replace(",", "")) for k, v in
                  re.findall(r"Pages ([^:]+):\s+([\d,]+)", vm)}
        file_backed = re.search(r"File-backed pages:\s*([\d,]+)", vm, re.IGNORECASE)
        if file_backed:
            values["file-backed"] = int(file_backed.group(1).replace(",", ""))
        physical_total = result.get("physical_memory_bytes")
        if page and "free" in values and type(physical_total) is int:
            page_bytes = int(page.group(1))
            free_bytes = values["free"] * page_bytes
            result["free_pages_bytes"] = free_bytes
            result["active_pages_bytes"] = values.get("active", 0) * page_bytes
            result["inactive_pages_bytes"] = values.get("inactive", 0) * page_bytes
            result["wired_pages_bytes"] = values.get("wired down", 0) * page_bytes
            result["compressor_physical_bytes"] = values.get("occupied by compressor", 0) * page_bytes
            result["purgeable_pages_bytes"] = values.get("purgeable", 0) * page_bytes
            result["file_backed_pages_bytes"] = values.get("file-backed", 0) * page_bytes
            result["speculative_pages_bytes"] = values.get("speculative", 0) * page_bytes
            result["vm_page_size_bytes"] = page_bytes
            result["system_used_bytes"] = max(0, physical_total - free_bytes)
            # vm_stat's displayed free count excludes speculative pages, while
            # XNU's free_count includes them. Match the broad XNU estimate but
            # keep it diagnostic: active/inactive pages are not assuredly freeable.
            result["available_memory_estimate_bytes"] = sum(
                values.get(queue, 0) for queue in ("free", "speculative", "active", "inactive")) * page_bytes
            result["reclaimable_memory_estimate_bytes"] = (
                values.get("free", 0) + values.get("file-backed", 0) + values.get("purgeable", 0)
            ) * page_bytes
            result["system_memory_source"] = (
                "vm_stat; displayed free pages exclude speculative pages although XNU free_count includes them; "
                "system_used_bytes is total minus displayed free pages; available_memory_estimate_bytes is the "
                "broad free+speculative+active+inactive VM estimate for diagnostics only; "
                "reclaimable_memory_estimate_bytes follows XNU fully-reclaimable pages (free+file-backed+purgeable); "
                "active/inactive queues are not added, and secluded-over-target is unavailable to this collector")
            result["vm_page_counts"] = values
        else:
            result["missing"]["system_used_bytes"] = "native hw.memsize, vm_stat page size, or free-page count unavailable"
            result["missing"]["available_memory_estimate_bytes"] = result["missing"]["system_used_bytes"]
            result["missing"]["reclaimable_memory_estimate_bytes"] = result["missing"]["system_used_bytes"]
    else:
        result["missing"]["system_used_bytes"] = vm.strip()

    code, pressure = _command(["/usr/bin/memory_pressure", "-Q"])
    free = re.search(r"System-wide memory free percentage:\s*(\d+)%", pressure)
    if code == 0 and free:
        # memory_pressure -Q outputs a percentage, not Apple's named pressure category.
        result["memory_pressure"] = {"free_percent": int(free.group(1)), "category": None}
        result["memory_pressure_source"] = "memory_pressure -Q; category unavailable"
    else:
        result["missing"]["memory_pressure"] = pressure.strip() or "command failed"

    if result.get("swap_used_bytes") is None:
        code, swap = _command(["/usr/sbin/sysctl", "vm.swapusage"])
        match = re.search(r"used\s*=\s*([\d.]+)([KMG])", swap, re.I)
        if code == 0 and match:
            factor = {"k": 1024, "m": 1024**2, "g": 1024**3}[match.group(2).lower()]
            result["swap_used_bytes"] = int(float(match.group(1)) * factor)
        else:
            result["missing"]["swap_used_bytes"] = swap.strip() or "sysctl unavailable/denied"
    if result.get("thermal_state") is None:
        result["missing"]["thermal_state"] = result.get("native_metrics_error", "native ProcessInfo collector unavailable")
    return result


def collect_loop(pid: int | None, interval_s: float, stop_event: Any, sink: Callable[[dict[str, Any]], None]) -> None:
    """Periodic sampler callback; callers own thread lifecycle and persistence."""
    if interval_s < 0.25:
        raise ValueError("Интервал мониторинга не должен быть меньше 250 мс")
    while not stop_event.is_set():
        sink(snapshot(pid))
        stop_event.wait(interval_s)


def diagnose(executable: str = "llama-server", artifacts: list[str] | None = None) -> dict[str, Any]:
    code, version = _command([executable, "--version"], timeout=5)
    import shutil
    import shutil as disk_shutil
    artifact_checks = []
    for artifact in artifacts or []:
        path = __import__("pathlib").Path(artifact)
        artifact_checks.append({"path": artifact, "exists": path.is_file(),
                                "size_bytes": path.stat().st_size if path.is_file() else None})
    usage = disk_shutil.disk_usage("/")
    return {"runtime_available": code == 0, "runtime_version": version.strip() or None,
            "version_error": None if code == 0 else version.strip(), "runtime_path": shutil.which(executable),
            "disk_free_bytes": usage.free, "artifacts": artifact_checks, "snapshot": snapshot()}
