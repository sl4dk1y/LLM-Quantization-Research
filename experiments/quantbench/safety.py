"""Fail-closed state machine for resource and runtime safety."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
import math
from typing import Any, ClassVar


class SafetyState(str, Enum):
    SAFE = "SAFE"
    WARNING = "WARNING"
    BLOCKED = "BLOCKED"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    RECOVERY = "RECOVERY"


@dataclass
class SafetyGuard:
    policy_version: ClassVar[str] = "macos-safety-v2.0"
    max_used_gib: float = 12.0
    warning_used_gib: float = 10.5
    warning_swap_mib: float = 256.0
    max_swap_mib: float = 1024.0
    recovery_used_gib: float = 8.0
    sampling_interval_s: float = 1.0
    exceedance_window_s: float = 5.0
    max_metric_age_s: float = 6.0
    recovery_window_s: float = 30.0
    _clock: Any = time.monotonic
    state: SafetyState = SafetyState.BLOCKED
    _unsafe_since: float | None = None
    _recovery_since: float | None = None
    _memory_warning_only: bool = False

    def evaluate(self, metrics: dict[str, Any], *, load_timed_out: bool = False,
                 request_timed_out: bool = False, now: float | None = None) -> SafetyState:
        previous_memory_warning_only = self._memory_warning_only
        self._memory_warning_only = False
        now = self._clock() if now is None else now
        required = ("system_used_bytes", "swap_delta_bytes", "thermal_state", "process_state")
        if any(metrics.get(key) is None for key in required):
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        sample_time = metrics.get("monotonic_time")
        if sample_time is None or not isinstance(sample_time, (int, float)) or now - sample_time > self.max_metric_age_s or sample_time > now:
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        for key in ("system_used_bytes", "swap_delta_bytes"):
            value = metrics.get(key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                self._unsafe_since = None
                self._recovery_since = None
                return self._set(SafetyState.BLOCKED)
        pressure = metrics.get("memory_pressure")
        category = pressure.get("category") if isinstance(pressure, dict) else pressure
        if category is not None and not isinstance(category, str):
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        apple_reclaimable_model = metrics.get("safety_model") == "apple_fully_reclaimable_estimate"
        if metrics.get("platform") == "macOS" and not apple_reclaimable_model:
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        if apple_reclaimable_model:
            reclaimable = metrics.get("reclaimable_memory_estimate_bytes")
            if (reclaimable is None or metrics.get("pressure_notifications_available") is not True
                    or isinstance(reclaimable, bool) or not isinstance(reclaimable, (int, float))
                    or not math.isfinite(reclaimable) or reclaimable < 0):
                self._unsafe_since = None
                self._recovery_since = None
                return self._set(SafetyState.BLOCKED)
            if category not in {None, "green", "normal", "yellow", "warn", "warning", "red", "critical"}:
                self._unsafe_since = None
                self._recovery_since = None
                return self._set(SafetyState.BLOCKED)
            safety_reclaimable_gib = reclaimable / 1024**3
        elif category not in {"green", "normal", "yellow", "red", "warning", "critical", "warn"}:
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        else:
            safety_reclaimable_gib = None
        used_gib = metrics["system_used_bytes"] / 1024**3
        swap_mib = metrics["swap_delta_bytes"] / 1024**2
        thermal = metrics["thermal_state"]
        runtime_active = metrics.get("runtime_active", False)
        if runtime_active and metrics.get("process_state") in {"unknown", "not_found"}:
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        if runtime_active and any(metrics.get(key) is None for key in ("physical_footprint_bytes",)):
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        footprint = metrics.get("physical_footprint_bytes")
        if footprint is not None and (isinstance(footprint, bool) or not isinstance(footprint, (int, float))
                                      or not math.isfinite(footprint) or footprint < 0):
            self._unsafe_since = None
            self._recovery_since = None
            return self._set(SafetyState.BLOCKED)
        memory_emergency = (safety_reclaimable_gib < 2.0 if apple_reclaimable_model else used_gib >= self.max_used_gib)
        if (category in {"red", "critical"} or thermal == "critical" or load_timed_out or request_timed_out
                or memory_emergency):
            self._unsafe_since = now
            self._recovery_since = None
            return self._set(SafetyState.EMERGENCY_STOP)
        memory_warning = ((runtime_active and safety_reclaimable_gib < 3.0)
                          if apple_reclaimable_model else used_gib > self.warning_used_gib)
        independent_warning = (swap_mib >= self.max_swap_mib or
                               category in {"yellow", "warn", "warning"} or
                               swap_mib >= self.warning_swap_mib or thermal == "serious")
        if independent_warning:
            self._recovery_since = None
            if self._unsafe_since is None:
                self._unsafe_since = now
            if now - self._unsafe_since >= self.exceedance_window_s:
                return self._set(SafetyState.EMERGENCY_STOP)
            return self._set(SafetyState.WARNING)
        if memory_warning:
            self._unsafe_since = None
            self._recovery_since = None
            self._memory_warning_only = apple_reclaimable_model
            return self._set(SafetyState.WARNING)
        if (category not in ({None, "green", "normal"} if apple_reclaimable_model else {"green", "normal"})
                or thermal not in {"nominal", "fair", "good"}):
            self._unsafe_since = None
            return self._set(SafetyState.BLOCKED)
        if self.state in {SafetyState.EMERGENCY_STOP, SafetyState.RECOVERY, SafetyState.BLOCKED,
                          SafetyState.WARNING}:
            runtime_recovered = (safety_reclaimable_gib >= 3.0 if runtime_active else
                                 safety_reclaimable_gib >= 4.0)
            memory_recovered = (runtime_recovered and metrics["swap_delta_bytes"] == 0
                                if apple_reclaimable_model else
                                used_gib <= self.recovery_used_gib and swap_mib < self.warning_swap_mib)
            safely_recovered = memory_recovered
            if (runtime_active and previous_memory_warning_only and safely_recovered):
                self._recovery_since = None
            elif not safely_recovered:
                self._recovery_since = None
                return self._set(SafetyState.RECOVERY)
            else:
                if self._recovery_since is None:
                    self._recovery_since = now
                if now - self._recovery_since < self.recovery_window_s:
                    return self._set(SafetyState.RECOVERY)
        self._unsafe_since = None
        self._recovery_since = None
        runtime_minimum = (safety_reclaimable_gib < 3.0 if runtime_active else
                           safety_reclaimable_gib < 4.0)
        if ((runtime_minimum if apple_reclaimable_model else used_gib > self.recovery_used_gib)
                or (apple_reclaimable_model and metrics["swap_delta_bytes"] != 0)):
            return self._set(SafetyState.RECOVERY)
        return self._set(SafetyState.SAFE)

    def _set(self, state: SafetyState) -> SafetyState:
        self.state = state
        return state

    @property
    def allows_inference(self) -> bool:
        return self.state == SafetyState.SAFE or (
            self.state == SafetyState.WARNING and self._memory_warning_only)
