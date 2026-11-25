"""Изолированный runtime для проверки очереди без запуска языковой модели."""
from __future__ import annotations

import hashlib
import time
from typing import Any


class SyntheticRuntime:
    """Детерминированный тестовый адаптер без subprocess, сети и чтения GGUF."""

    def __init__(self) -> None:
        self.started = False
        self.process = None
        self.safety_watchdog = None
        self._calls = 0

    def start(self) -> None:
        if self.started:
            raise RuntimeError("SyntheticRuntime уже запущен")
        self.started = True

    def generate(self, prompt: str, generation: dict[str, Any]) -> dict[str, Any]:
        if not self.started:
            raise RuntimeError("SyntheticRuntime не запущен")
        self._calls += 1
        time.sleep(0)
        answer = "1" if "PARus" in prompt else "A"
        return {"text": answer, "finish_reason": "stop", "ttft_s": 0.001,
                "timings": {"prompt_n": 7, "predicted_n": 1,
                            "prompt_ms": 2.5, "predicted_ms": 1.5},
                "request_wall_s": 0.004,
                "synthetic_runtime": {"calls": self._calls,
                                      "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                      "generation": dict(generation)}}

    def stop(self, **_kwargs: Any) -> None:
        self.started = False
