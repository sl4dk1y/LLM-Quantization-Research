"""Периодический Safety Guard с аварийным завершением текущего local runtime."""
from __future__ import annotations
import threading
import time
from typing import Any, Callable
from .safety import SafetyGuard, SafetyState


class SafetyWatchdog:
    def __init__(self, *, sample: Callable[[], dict[str, Any]], guard: SafetyGuard,
                 stop_runtime: Callable[[], None], interval_s: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.sample = sample
        self.guard = guard
        self.stop_runtime = stop_runtime
        self.interval_s = interval_s or guard.sampling_interval_s
        if self.interval_s <= 0:
            raise ValueError("sampling interval должен быть положительным")
        self.clock = clock
        self.stop_event = threading.Event()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.last_state = SafetyState.BLOCKED
        self.last_error: str | None = None
        self.last_metrics: dict[str, Any] | None = None
        self._sample_times: list[float] = []
        self._peaks: dict[str, float] = {}
        self._thermal_states: set[str] = set()
        self._pressure_states: set[str] = set()

    def sample_once(self) -> SafetyState:
        try:
            metrics = self.sample()
            if "monotonic_time" not in metrics:
                metrics["monotonic_time"] = self.clock()
            self.last_metrics = dict(metrics)
            observed_at = float(metrics["monotonic_time"])
            self._sample_times.append(observed_at)
            for key, value in metrics.items():
                if key == "monotonic_time" or not key.endswith(("_bytes", "_utilization", "_percent")):
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self._peaks[key] = max(float(value), self._peaks.get(key, float("-inf")))
            thermal = metrics.get("thermal_state")
            if thermal is not None:
                self._thermal_states.add(str(thermal))
            pressure = metrics.get("memory_pressure", {}).get("category") if isinstance(metrics.get("memory_pressure"), dict) else None
            if pressure is not None:
                self._pressure_states.add(str(pressure))
            state = self.guard.evaluate(metrics, now=self.clock())
            self.last_error = None
        except Exception as exc:
            self.last_metrics = None
            state = self.guard.evaluate({}, now=self.clock())
            self.last_error = f"{type(exc).__name__}: {exc}"
        self.last_state = state
        if state in {SafetyState.BLOCKED, SafetyState.EMERGENCY_STOP}:
            self._trigger_stop()
        return state

    def assert_safe(self) -> None:
        if self.last_state != SafetyState.SAFE or self.stop_event.is_set():
            detail = self.last_error or self.last_state.value
            raise RuntimeError(f"Safety Guard не разрешает новое наблюдение: {detail}")

    def wait_until_safe(self, *, timeout_s: float) -> bool:
        deadline = self.clock() + timeout_s
        thread = self._thread
        if thread is not None and thread.is_alive():
            while self.clock() < deadline and not self._shutdown.is_set():
                if self.stop_event.is_set():
                    return False
                if self.last_state == SafetyState.SAFE:
                    return True
                self._shutdown.wait(min(self.interval_s, max(0.0, deadline - self.clock())))
            return self.last_state == SafetyState.SAFE and not self.stop_event.is_set()
        while self.clock() < deadline and not self._shutdown.is_set():
            state = self.sample_once()
            if state == SafetyState.SAFE:
                return True
            if self.stop_event.is_set():
                return False
            self._shutdown.wait(min(self.interval_s, max(0.0, deadline - self.clock())))
        return False

    def start(self) -> None:
        with self._lock:
            if self.stop_event.is_set():
                raise RuntimeError("Watchdog остановлен safety event; для продолжения нужен новый проверенный запуск")
            if self._thread is not None and self._thread.is_alive():
                return
            self._shutdown.clear()
            self._thread = threading.Thread(target=self._run, name="quantbench-safety-watchdog", daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._shutdown.is_set() and not self.stop_event.is_set():
            self.sample_once()
            self._shutdown.wait(self.interval_s)

    def _trigger_stop(self) -> None:
        first = not self.stop_event.is_set()
        self.stop_event.set()
        if first:
            try:
                self.stop_runtime()
            except Exception as exc:
                self.last_error = (self.last_error + "; " if self.last_error else "") + \
                                  f"runtime stop failed: {type(exc).__name__}: {exc}"

    def close(self, *, timeout_s: float = 2.0) -> None:
        self._shutdown.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=timeout_s)
        self._thread = None

    def summary(self) -> dict[str, Any]:
        times = self._sample_times
        intervals = [right - left for left, right in zip(times, times[1:])]
        return {
            "sample_count": len(times),
            "configured_sample_interval_s": self.interval_s,
            "observed_sampling_interval_s": ({"min": min(intervals), "mean": sum(intervals) / len(intervals),
                                               "max": max(intervals), "n": len(intervals)}
                                              if intervals else None),
            "observed_peaks": dict(self._peaks),
            "thermal_states_observed": sorted(self._thermal_states),
            "memory_pressure_states_observed": sorted(self._pressure_states),
            "last_safety_state": self.last_state.value,
            "last_safety_error": self.last_error,
            "peak_semantics": "максимум дискретных успешных измерений; не гарантирует непрерывный максимум",
        }
