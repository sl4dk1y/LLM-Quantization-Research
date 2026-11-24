"""Fail-closed state machine for resource and runtime safety."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class SafetyState(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RECOVERY = "RECOVERY"


@dataclass
class SafetyGuard:
    max_used_gib: float = 12.0
    warning_used_gib: float = 10.5
    warning_swap_mib: float = 256.0
    max_swap_mib: float = 1024.0
    recovery_used_gib: float = 8.0
    state: SafetyState = SafetyState.BLOCKED

    def evaluate(self, metrics: dict[str, Any], *, load_timed_out: bool = False, request_timed_out: bool = False) -> SafetyState:
        required = ("system_used_bytes", "swap_delta_bytes", "memory_pressure", "thermal_state", "process_state", "physical_footprint_bytes")
        if any(metrics.get(key) is None for key in required):
            return self._set(SafetyState.BLOCKED)
        pressure = metrics["memory_pressure"]
        category = pressure.get("category") if isinstance(pressure, dict) else pressure
        used_gib = metrics["system_used_bytes"] / 1024**3
        swap_mib = metrics["swap_delta_bytes"] / 1024**2
        thermal = metrics["thermal_state"]
        if category in {"red", "critical"} or used_gib >= self.max_used_gib or swap_mib >= self.max_swap_mib or thermal == "critical" or load_timed_out or request_timed_out:
            return self._set(SafetyState.EMERGENCY_STOP)
        if category == "yellow" or used_gib > self.warning_used_gib or swap_mib >= self.warning_swap_mib or thermal == "serious":
            return self._set(SafetyState.WARNING)
        if category not in {"green", "normal"} or thermal not in {"nominal", "fair", "good"}:
            return self._set(SafetyState.BLOCKED)
        if used_gib > self.recovery_used_gib:
            return self._set(SafetyState.RECOVERY)
        return self._set(SafetyState.SAFE)

    def _set(self, state: SafetyState) -> SafetyState:
        self.state = state
        return state

    @property
    def allows_inference(self) -> bool:
        return self.state == SafetyState.SAFE
