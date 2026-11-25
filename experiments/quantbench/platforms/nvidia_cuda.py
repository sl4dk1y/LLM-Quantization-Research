"""Адаптер платформы NVIDIA CUDA для Windows.

Реализует общий контракт :class:`~quantbench.platform.PlatformAdapter` для
компьютера с NVIDIA GPU и драйвером CUDA. Метрики собираются только чтением:

* ``nvidia-smi`` — GPU/VRAM, утилизация, температура, power draw, driver,
  Compute Capability;
* Win32 API (``kernel32``/``psapi``) — системная RAM, доступная память,
  RSS процесса, состояние процесса, pagefile.

Никакие системные настройки не изменяются; посторонние процессы не
завершаются. Запуск subprocess выполняется без shell и с ограниченным
тайм-аутом. Недоступные метрики возвращаются как ``null`` со статусом и
причиной, а не подменяются фиктивными значениями.

Ограничение WDDM: драйвер NVIDIA в режиме WDDM не предоставляет привязку
VRAM к отдельным процессам через ``nvidia-smi``, поэтому используются
device-level показатели VRAM (общая/занятая/свободная видеопамять).
"""

from __future__ import annotations

import csv
import io
import os
import platform as py_platform
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from ..platform import HardwareManifest, Metric, MetricStatus, PlatformAdapter, validate_snapshot
from ..runtime import build_server_args

PLATFORM_ID = "nvidia-cuda"

# Обязательные для CUDA safety-метрики: их недоступность блокирует preflight
# (fail-closed). Это системные и GPU-показатели, не требующие PID процесса.
REQUIRED_METRICS = frozenset({
    "system_ram_bytes",
    "system_available_bytes",
    "vram_total_bytes",
    "vram_used_bytes",
    "vram_free_bytes",
    "gpu_utilization_percent",
    "gpu_temperature_celsius",
})

# Поля структурированного запроса nvidia-smi (--format=csv,noheader,nounits).
GPU_QUERY_FIELDS = (
    "name", "driver_version", "memory.total", "memory.used", "memory.free",
    "temperature.gpu", "utilization.gpu", "power.draw", "power.limit", "compute_cap",
)

# Документация каждой метрики: источник, единица, семантика, ограничения.
METRIC_SPECS: dict[str, dict[str, Any]] = {
    "system_ram_bytes": {
        "unit": "bytes",
        "source": "kernel32.GlobalMemoryStatusEx.ullTotalPhys",
        "semantics": "Общий объём физической оперативной памяти",
        "required": True,
        "limitations": None,
    },
    "system_available_bytes": {
        "unit": "bytes",
        "source": "kernel32.GlobalMemoryStatusEx.ullAvailPhys",
        "semantics": "Физическая память, доступная процессам (не зарезервирована ОС/приложениями)",
        "required": True,
        "limitations": "Значение меняется между снимками",
    },
    "process_rss_bytes": {
        "unit": "bytes",
        "source": "psapi.GetProcessMemoryInfo.WorkingSetSize",
        "semantics": "Working set (RSS) целевого процесса",
        "required": False,
        "limitations": "Требует PID; не равно выделенной VRAM",
    },
    "process_state": {
        "unit": None,
        "source": "kernel32.OpenProcess",
        "semantics": "Состояние целевого процесса: running / not_found / unknown",
        "required": False,
        "limitations": "Требует PID",
    },
    "vram_total_bytes": {
        "unit": "bytes",
        "source": "nvidia-smi memory.total",
        "semantics": "Общий объём видеопамяти устройства",
        "required": True,
        "limitations": "Device-level; WDDM не даёт per-process VRAM",
    },
    "vram_used_bytes": {
        "unit": "bytes",
        "source": "nvidia-smi memory.used",
        "semantics": "Занятая видеопамять устройства",
        "required": True,
        "limitations": "Включает потребление ОС/десктопа (WDDM); не привязана к процессу",
    },
    "vram_free_bytes": {
        "unit": "bytes",
        "source": "nvidia-smi memory.free",
        "semantics": "Свободная видеопамять устройства",
        "required": True,
        "limitations": "Может быть меньше реально доступной из-за резервирования WDDM",
    },
    "gpu_utilization_percent": {
        "unit": "%",
        "source": "nvidia-smi utilization.gpu",
        "semantics": "Процент занятости GPU за интервал опроса",
        "required": True,
        "limitations": "Мгновенное значение",
    },
    "gpu_temperature_celsius": {
        "unit": "°C",
        "source": "nvidia-smi temperature.gpu",
        "semantics": "Температура GPU",
        "required": True,
        "limitations": None,
    },
    "gpu_power_watts": {
        "unit": "W",
        "source": "nvidia-smi power.draw",
        "semantics": "Текущее потребление GPU",
        "required": False,
        "limitations": "Недоступно на части GPU/драйверов; не сравнивать между платформами без общей методики",
    },
    "pagefile_total_bytes": {
        "unit": "bytes",
        "source": "kernel32.GlobalMemoryStatusEx.ullTotalPageFile",
        "semantics": "Общий объём pagefile (физическая память + файл подкачки)",
        "required": False,
        "limitations": None,
    },
    "pagefile_used_bytes": {
        "unit": "bytes",
        "source": "kernel32.GlobalMemoryStatusEx (total − avail)",
        "semantics": "Занятый объём pagefile",
        "required": False,
        "limitations": "Приближение committed memory",
    },
}


def run_command(argv: list[str], timeout: float = 3.0) -> tuple[int, str]:
    """Читающий subprocess без shell и с ограниченным тайм-аутом."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode, (result.stdout or "") + (result.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, f"{type(exc).__name__}: {exc}"


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_gpu_query(data: dict[str, str]) -> dict[str, Any]:
    def memory_bytes(raw: str | None) -> int | None:
        value = _int_or_none(raw)
        return value * 1024 * 1024 if value is not None else None

    return {
        "name": data.get("name") or None,
        "driver_version": data.get("driver_version") or None,
        "memory_total_bytes": memory_bytes(data.get("memory.total")),
        "memory_used_bytes": memory_bytes(data.get("memory.used")),
        "memory_free_bytes": memory_bytes(data.get("memory.free")),
        "temperature_celsius": _int_or_none(data.get("temperature.gpu")),
        "utilization_percent": _int_or_none(data.get("utilization.gpu")),
        "power_draw_watts": _float_or_none(data.get("power.draw")),
        "power_limit_watts": _float_or_none(data.get("power.limit")),
        "compute_capability": data.get("compute_cap") or None,
    }


def parse_nvidia_smi_query(fields: tuple[str, ...], output: str) -> dict[str, Any] | None:
    """Разбирает CSV-строку ``nvidia-smi --query-gpu=... --format=csv,noheader,nounits``.

    Возвращает нормализованный словарь полей первого GPU (одиночная GPU-система)
    либо ``None``, если вывод пуст или не распознан. Значения ``N/A`` становятся
    ``None`` (не подменяются нулями).
    """
    lines = [line.strip() for line in (output or "").splitlines() if line.strip()]
    if not lines:
        return None
    reader = csv.reader(io.StringIO("\n".join(lines)))
    for row in reader:
        if len(row) != len(fields):
            continue
        return _normalize_gpu_query({field: cell.strip() for field, cell in zip(fields, row)})
    return None


def parse_nvidia_smi_summary(output: str) -> dict[str, str]:
    """Извлекает версию драйвера и поддерживаемую версию CUDA из заголовка ``nvidia-smi``."""
    result: dict[str, str] = {}
    driver = re.search(r"Driver Version:\s*(\S+)", output or "")
    if driver:
        result["driver_version"] = driver.group(1)
    cuda = re.search(r"CUDA Version:\s*(\S+)", output or "")
    if cuda:
        result["cuda_version"] = cuda.group(1)
    return result


def probe_nvidia_smi() -> dict[str, Any] | None:
    """Возвращает сведения GPU из ``nvidia-smi`` либо ``None`` при недоступности."""
    exe = shutil.which("nvidia-smi") or "nvidia-smi"
    code, query_out = run_command(
        [exe, "--query-gpu=" + ",".join(GPU_QUERY_FIELDS), "--format=csv,noheader,nounits"],
        timeout=5.0,
    )
    gpu = parse_nvidia_smi_query(GPU_QUERY_FIELDS, query_out) if code == 0 else None
    if gpu is None:
        return None
    summary_code, summary_out = run_command([exe], timeout=5.0)
    if summary_code == 0:
        gpu.update(parse_nvidia_smi_summary(summary_out))
    return gpu


def probe_windows_memory() -> dict[str, Any] | None:
    """Системная RAM и pagefile через Win32 ``GlobalMemoryStatusEx``."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", wintypes.DWORD),
                ("dwMemoryLoad", wintypes.DWORD),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return {
            "total_phys_bytes": int(status.ullTotalPhys),
            "avail_phys_bytes": int(status.ullAvailPhys),
            "total_pagefile_bytes": int(status.ullTotalPageFile),
            "avail_pagefile_bytes": int(status.ullAvailPageFile),
            "memory_load_percent": int(status.dwMemoryLoad),
        }
    except (OSError, AttributeError, ValueError):
        return None


def probe_process_memory(pid: int | None) -> dict[str, Any] | None:
    """Working set (RSS) и состояние процесса через Win32 ``GetProcessMemoryInfo``."""
    if os.name != "nt" or pid is None:
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        access = 0x1000 | 0x0010  # PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_VM_READ
        handle = ctypes.windll.kernel32.OpenProcess(access, False, int(pid))
        if not handle:
            error = ctypes.windll.kernel32.GetLastError()
            state = "not_found" if error == 87 else "unknown"  # 87 = ERROR_INVALID_PARAMETER
            return {"rss_bytes": None, "state": state}
        try:
            counters = PROCESS_MEMORY_COUNTERS_EX()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return {"rss_bytes": int(counters.WorkingSetSize), "state": "running"}
            return {"rss_bytes": None, "state": "running"}
        finally:
            ctypes.windll.kernel32.CloseHandle(handle)
    except (OSError, AttributeError, ValueError):
        return None


def probe_cpu_model() -> str | None:
    """Модель CPU из реестра Windows с fallback на переменные окружения."""
    if os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if value and str(value).strip():
                    return str(value).strip()
        except (OSError, ImportError):
            pass
    return os.environ.get("PROCESSOR_IDENTIFIER") or py_platform.processor() or None


def probe_os_info() -> dict[str, str]:
    """Сведения об ОС и архитектуре из :mod:`platform`."""
    return {
        "system": py_platform.system() or "",
        "release": py_platform.release() or "",
        "version": py_platform.version() or "",
        "machine": py_platform.machine() or "",
    }


def probe_runtime_version(executable: str) -> str | None:
    """Первая строка ``<executable> --version`` либо ``None`` при недоступности."""
    if not executable:
        return None
    code, output = run_command([executable, "--version"], timeout=5.0)
    if code != 0:
        return None
    lines = (output or "").strip().splitlines()
    return lines[0].strip() if lines else None


def offload_mode(gpu_layers: Any) -> str:
    """Классифицирует режим размещения слоёв: cpu_only / partial / full / auto / unknown."""
    if gpu_layers == 0:
        return "cpu_only"
    if gpu_layers == "all":
        return "full"
    if gpu_layers == "auto":
        return "auto"
    if isinstance(gpu_layers, int) and gpu_layers > 0:
        return "partial"
    return "unknown"


class CUDASafetyState(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class CUDASafetyProfile:
    """Предварительные пороги для GTX 1660 (6 GiB VRAM).

    Это проектные ограничения для последующей безопасной калибровки, а не
    подтверждённые параметры устройства (см. docs/platform-comparison.md).
    """

    vram_total_bytes: int = 6 * 1024**3
    vram_free_emergency_bytes: int = 256 * 1024**2
    vram_free_warning_bytes: int = 512 * 1024**2
    system_available_emergency_bytes: int = 512 * 1024**2
    system_available_warning_bytes: int = 1024**3
    gpu_temp_emergency_celsius: float = 90.0
    gpu_temp_warning_celsius: float = 80.0


def evaluate_cuda_safety(metrics: dict[str, Metric],
                         profile: CUDASafetyProfile,
                         required: frozenset[str] = REQUIRED_METRICS) -> tuple[CUDASafetyState, list[str]]:
    """Fail-closed оценка безопасности по CUDA-метрикам.

    Возвращает состояние и список причин. Отсутствие обязательной метрики
    даёт BLOCKED; аварийные пороги температуры/свободной VRAM/RAM — EMERGENCY_STOP.
    """
    problems = validate_snapshot(metrics, set(required))
    if problems:
        return CUDASafetyState.BLOCKED, problems

    temperature = metrics["gpu_temperature_celsius"].value
    free_vram = metrics["vram_free_bytes"].value
    free_ram = metrics["system_available_bytes"].value

    if temperature >= profile.gpu_temp_emergency_celsius:
        return CUDASafetyState.EMERGENCY_STOP, [
            f"Температура GPU {temperature}°C достигает аварийного порога {profile.gpu_temp_emergency_celsius}°C"
        ]
    if free_vram <= profile.vram_free_emergency_bytes:
        return CUDASafetyState.EMERGENCY_STOP, [
            f"Свободная VRAM {free_vram} байт ниже аварийного порога {profile.vram_free_emergency_bytes} байт"
        ]
    if free_ram <= profile.system_available_emergency_bytes:
        return CUDASafetyState.EMERGENCY_STOP, [
            f"Доступная системная память {free_ram} байт ниже аварийного порога {profile.system_available_emergency_bytes} байт"
        ]

    warnings: list[str] = []
    if temperature >= profile.gpu_temp_warning_celsius:
        warnings.append(f"Температура GPU {temperature}°C достигает предупредительного порога {profile.gpu_temp_warning_celsius}°C")
    if free_vram <= profile.vram_free_warning_bytes:
        warnings.append(f"Свободная VRAM {free_vram} байт ниже предупредительного порога {profile.vram_free_warning_bytes} байт")
    if free_ram <= profile.system_available_warning_bytes:
        warnings.append(f"Доступная системная память {free_ram} байт ниже предупредительного порога {profile.system_available_warning_bytes} байт")
    if warnings:
        return CUDASafetyState.WARNING, warnings
    return CUDASafetyState.SAFE, []


class NVIDIA_CUDAAdapter:
    """Платформенный адаптер NVIDIA CUDA; метрики и manifest собираются чтением."""

    platform_id = PLATFORM_ID

    def __init__(self, *, nvidia_smi_probe=None, memory_probe=None, process_probe=None,
                 cpu_probe=None, runtime_probe=None, os_probe=None,
                 safety_profile: CUDASafetyProfile | None = None):
        # Пробники инъектируются для mock/fake-тестов без реального GPU.
        self._nvidia_smi_probe = nvidia_smi_probe or probe_nvidia_smi
        self._memory_probe = memory_probe or probe_windows_memory
        self._process_probe = process_probe or probe_process_memory
        self._cpu_probe = cpu_probe or probe_cpu_model
        self._runtime_probe = runtime_probe or probe_runtime_version
        self._os_probe = os_probe or probe_os_info
        self.safety_profile = safety_profile or CUDASafetyProfile()

    def hardware_manifest(self) -> HardwareManifest:
        os_info = self._os_probe()
        gpu = self._nvidia_smi_probe()
        memory = self._memory_probe()
        return HardwareManifest(
            platform_id=self.platform_id,
            os_name=os_info.get("system") or "Windows",
            os_version=os_info.get("version") or None,
            architecture=os_info.get("machine") or "AMD64",
            cpu_model=self._cpu_probe(),
            ram_bytes=memory.get("total_phys_bytes") if memory else None,
            gpu_model=gpu.get("name") if gpu else None,
            vram_bytes=gpu.get("memory_total_bytes") if gpu else None,
            driver_version=gpu.get("driver_version") if gpu else None,
            runtime_version=None,
            backend="CUDA",
        )

    def capabilities(self) -> dict[str, Any]:
        gpu = self._nvidia_smi_probe()
        return {
            "backend": "CUDA",
            "unified_memory": False,
            "offload_modes": ["cpu_only", "partial", "full", "auto"],
            "compute_capability": gpu.get("compute_capability") if gpu else None,
            "driver_version": gpu.get("driver_version") if gpu else None,
            "cuda_driver_api_version": gpu.get("cuda_version") if gpu else None,
            "wddm": {
                "process_vram_attribution": "unavailable",
                "reason": "NVIDIA WDDM не предоставляет привязку VRAM к процессам через nvidia-smi",
            },
            "metrics": METRIC_SPECS,
            "safety_profile": asdict(self.safety_profile),
        }

    def cuda_available(self) -> bool:
        """Проверка доступности CUDA (наличие GPU и рабочего ``nvidia-smi``)."""
        return self._nvidia_smi_probe() is not None

    def runtime_compatible(self, executable: str | None) -> bool:
        """Проверка совместимости runtime: ``<executable> --version`` выполняется успешно."""
        if not executable:
            return False
        return self._runtime_probe(executable) is not None

    def runtime_args(self, config: dict[str, Any], model: dict[str, Any]) -> list[str]:
        args = build_server_args(config, model)
        gpu_layers = config["runtime"]["gpu_layers"]
        if offload_mode(gpu_layers) != "cpu_only":
            # При любом GPU offload явно закрепляем единственное устройство CUDA.
            args += ["--main-gpu", "0"]
        return args

    def resource_snapshot(self, pid: int | None = None) -> dict[str, Metric]:
        gpu = self._nvidia_smi_probe()
        memory = self._memory_probe()
        process = self._process_probe(pid)

        result: dict[str, Metric] = {}

        # --- Системная память ---
        if memory and memory.get("total_phys_bytes") is not None:
            result["system_ram_bytes"] = Metric("system_ram_bytes", memory["total_phys_bytes"], "bytes", MetricStatus.AVAILABLE)
        else:
            result["system_ram_bytes"] = Metric("system_ram_bytes", None, "bytes", MetricStatus.UNAVAILABLE, "GlobalMemoryStatusEx не вернул общий объём RAM")

        if memory and memory.get("avail_phys_bytes") is not None:
            result["system_available_bytes"] = Metric("system_available_bytes", memory["avail_phys_bytes"], "bytes", MetricStatus.AVAILABLE)
        else:
            result["system_available_bytes"] = Metric("system_available_bytes", None, "bytes", MetricStatus.UNAVAILABLE, "GlobalMemoryStatusEx не вернул доступную RAM")

        # --- Pagefile ---
        if memory and memory.get("total_pagefile_bytes") is not None and memory.get("avail_pagefile_bytes") is not None:
            total = memory["total_pagefile_bytes"]
            used = max(0, total - memory["avail_pagefile_bytes"])
            result["pagefile_total_bytes"] = Metric("pagefile_total_bytes", total, "bytes", MetricStatus.AVAILABLE)
            result["pagefile_used_bytes"] = Metric("pagefile_used_bytes", used, "bytes", MetricStatus.AVAILABLE)
        else:
            result["pagefile_total_bytes"] = Metric("pagefile_total_bytes", None, "bytes", MetricStatus.UNAVAILABLE, "pagefile недоступен")
            result["pagefile_used_bytes"] = Metric("pagefile_used_bytes", None, "bytes", MetricStatus.UNAVAILABLE, "pagefile недоступен")

        # --- Процесс ---
        if pid is None:
            result["process_rss_bytes"] = Metric("process_rss_bytes", None, "bytes", MetricStatus.NOT_APPLICABLE, "PID процесса не задан")
            result["process_state"] = Metric("process_state", None, None, MetricStatus.NOT_APPLICABLE, "PID процесса не задан")
        elif process is None:
            result["process_rss_bytes"] = Metric("process_rss_bytes", None, "bytes", MetricStatus.ERROR, "Не удалось прочитать память процесса")
            result["process_state"] = Metric("process_state", None, None, MetricStatus.ERROR, "Не удалось определить состояние процесса")
        else:
            rss = process.get("rss_bytes")
            state = process.get("state")
            result["process_rss_bytes"] = Metric("process_rss_bytes", rss, "bytes", MetricStatus.AVAILABLE if rss is not None else MetricStatus.UNAVAILABLE, None if rss is not None else "RSS процесса недоступен")
            result["process_state"] = Metric("process_state", state, None, MetricStatus.AVAILABLE if state else MetricStatus.UNAVAILABLE)

        # --- GPU / VRAM ---
        for name, field, unit in (
            ("vram_total_bytes", "memory_total_bytes", "bytes"),
            ("vram_used_bytes", "memory_used_bytes", "bytes"),
            ("vram_free_bytes", "memory_free_bytes", "bytes"),
            ("gpu_utilization_percent", "utilization_percent", "%"),
            ("gpu_temperature_celsius", "temperature_celsius", "°C"),
            ("gpu_power_watts", "power_draw_watts", "W"),
        ):
            if gpu is None:
                result[name] = Metric(name, None, unit, MetricStatus.UNAVAILABLE, "nvidia-smi недоступен")
            else:
                value = gpu.get(field)
                if value is None:
                    result[name] = Metric(name, None, unit, MetricStatus.UNAVAILABLE, f"nvidia-smi не сообщил {field}")
                else:
                    result[name] = Metric(name, value, unit, MetricStatus.AVAILABLE)

        return result

    def preflight(self, config: dict[str, Any]) -> list[str]:
        """Предстартовые блокеры: платформа, CUDA, runtime, обязательные метрики, безопасность."""
        if config.get("platform_id") not in (None, self.platform_id):
            return ["Профиль конфигурации не соответствует NVIDIA CUDA"]

        if not self.cuda_available():
            return ["NVIDIA GPU/CUDA недоступны: nvidia-smi не вернул данные"]

        blockers: list[str] = []
        executable = (config.get("runtime") or {}).get("executable")
        if executable and not self.runtime_compatible(executable):
            blockers.append("llama-server runtime несовместим или --version недоступен")

        snapshot = self.resource_snapshot(pid=None)
        blockers.extend(validate_snapshot(snapshot, set(REQUIRED_METRICS)))

        state, reasons = evaluate_cuda_safety(snapshot, self.safety_profile, REQUIRED_METRICS)
        if state in (CUDASafetyState.BLOCKED, CUDASafetyState.EMERGENCY_STOP):
            blockers.extend(reasons)

        return blockers
