"""Минимальный общий контракт платформенных адаптеров для M3.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Protocol


class MetricStatus(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    NOT_APPLICABLE = "not_applicable"
    ERROR = "error"


@dataclass(frozen=True)
class Metric:
    name: str
    value: int | float | str | dict[str, Any] | None
    unit: str | None
    status: MetricStatus
    reason: str | None = None


@dataclass(frozen=True)
class HardwareManifest:
    platform_id: str
    os_name: str
    os_version: str | None
    architecture: str
    cpu_model: str | None = None
    ram_bytes: int | None = None
    gpu_model: str | None = None
    vram_bytes: int | None = None
    driver_version: str | None = None
    runtime_version: str | None = None
    backend: str | None = None

    def canonical_fields(self) -> dict[str, Any]:
        return asdict(self)


class PlatformAdapter(Protocol):
    """Контракт адаптера; значения метрик всегда сопровождаются статусом и единицей."""

    platform_id: str

    def hardware_manifest(self) -> HardwareManifest: ...
    def capabilities(self) -> dict[str, Any]: ...
    def runtime_args(self, config: dict[str, Any], model: dict[str, Any]) -> list[str]: ...
    def resource_snapshot(self, pid: int | None = None) -> dict[str, Metric]: ...
    def preflight(self, config: dict[str, Any]) -> list[str]: ...


def metric(name: str, value: Any, unit: str | None, *, applicable: bool = True,
           reason: str | None = None) -> Metric:
    if not applicable:
        return Metric(name, None, unit, MetricStatus.NOT_APPLICABLE, reason or "Метрика неприменима")
    if value is None:
        return Metric(name, None, unit, MetricStatus.UNAVAILABLE, reason or "Источник метрики недоступен")
    return Metric(name, value, unit, MetricStatus.AVAILABLE)


def validate_snapshot(metrics: dict[str, Metric], required: set[str]) -> list[str]:
    """Fail-closed: обязательные отсутствующие/ошибочные метрики возвращаются как блокеры."""
    problems = []
    for name in sorted(required):
        value = metrics.get(name)
        if value is None:
            problems.append(f"{name}: нет определения метрики")
        elif value.status in {MetricStatus.UNAVAILABLE, MetricStatus.ERROR}:
            problems.append(f"{name}: {value.reason or value.status.value}")
    return problems

