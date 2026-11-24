"""Fail-closed state machine for resource and runtime safety."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
import math
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
    sampling_interval_s: float = 2.0
    exceedance_window_s: float = 5.0
    max_metric_age_s: float = 6.0
    recovery_window_s: float = 30.0
    _clock: Any = time.monotonic
    state: SafetyState = SafetyState.BLOCKED
    _unsafe_since: float | None = None
    _recovery_since: float | None = None

    def evaluate(self, metrics: dict[str, Any], *, load_timed_out: bool = False,
                 request_timed_out: bool = False, now: float | None = None) -> SafetyState:
        now = self._clock() if now is None else now
        required = ("system_used_bytes", "swap_delta_bytes", "memory_pressure", "thermal_state", "process_state", "physical_footprint_bytes")
        if any(metrics.get(key) is None for key in required):
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        sample_time = metrics.get("monotonic_time")
        if sample_time is None or not isinstance(sample_time, (int, float)) or now - sample_time > self.max_metric_age_s or sample_time > now:
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        for key in ("system_used_bytes", "swap_delta_bytes", "physical_footprint_bytes"):
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                self._unsafe_since = None
                self._recovery_since = None
                return self._set(SafetyState.BLOCKED)
        pressure = metrics["memory_pressure"]
        category = pressure.get("category") if isinstance(pressure, dict) else pressure
        used_gib = metrics["system_used_bytes"] / 1024**3
        swap_mib = metrics["swap_delta_bytes"] / 1024**2
        thermal = metrics["thermal_state"]
        if category in {"red", "critical"} or thermal == "critical" or load_timed_out or request_timed_out:
            self._unsafe_since = now
            self._recovery_since = None
            return self._set(SafetyState.EMERGENCY_STOP)
        unsafe = (used_gib >= self.max_used_gib or swap_mib >= self.max_swap_mib or
                  category == "yellow" or used_gib > self.warning_used_gib or
                  swap_mib >= self.warning_swap_mib or thermal == "serious")
        if unsafe:
            self._recovery_since = None
            if self._unsafe_since is None:
                self._unsafe_since = now
            if now - self._unsafe_since >= self.exceedance_window_s:
                return self._set(SafetyState.EMERGENCY_STOP)
            return self._set(SafetyState.WARNING)
        if category not in {"green", "normal"} or thermal not in {"nominal", "fair", "good"}:
            self._unsafe_since = None
            return self._set(SafetyState.BLOCKED)
        if self.state in {SafetyState.EMERGENCY_STOP, SafetyState.RECOVERY, SafetyState.BLOCKED,
                          SafetyState.WARNING}:
            safely_recovered = used_gib <= self.recovery_used_gib and swap_mib < self.warning_swap_mib
            if not safely_recovered:
                self._recovery_since = None
                return self._set(SafetyState.RECOVERY)
            if self._recovery_since is None:
                self._recovery_since = now
            if now - self._recovery_since < self.recovery_window_s:
                return self._set(SafetyState.RECOVERY)
        self._unsafe_since = None
        self._recovery_since = None
        if used_gib > self.recovery_used_gib:
            return self._set(SafetyState.RECOVERY)
        return self._set(SafetyState.SAFE)

    def _set(self, state: SafetyState) -> SafetyState:
        self.state = state
        return state

    @property
    def allows_inference(self) -> bool:
        return self.state == SafetyState.SAFE
