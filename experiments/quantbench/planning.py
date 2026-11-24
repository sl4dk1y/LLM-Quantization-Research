"""Расчёт прогресса и оценка времени только по фактическим замерам."""
from __future__ import annotations
from collections import defaultdict
from statistics import median
from typing import Any, Iterable


def estimate_remaining(jobs: Iterable[dict[str, Any]], completed: Iterable[dict[str, Any]]) -> dict[str, Any]:
    done = {row.get("observation_id") for row in completed if row.get("status") == "complete"}
    times: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in completed:
        if row.get("status") != "complete":
            continue
        seconds = row.get("duration_s")
        if isinstance(seconds, (int, float)) and seconds >= 0:
            key = tuple(row.get(k) for k in ("model_id", "quantization", "context_length", "platform_id"))
            times[key].append(float(seconds))
    remaining = [job for job in jobs if job.get("observation_id") not in done]
    missing: dict[str, int] = defaultdict(int)
    seconds = 0.0
    for job in remaining:
        key = tuple(job.get(k) for k in ("model_id", "quantization", "context_length", "platform_id"))
        if key not in times:
            missing[str(key)] += 1
        else:
            seconds += median(times[key])
    return {"completed": len(done), "remaining": len(remaining),
            "estimated_seconds_from_measured_strata": seconds if not missing else None,
            "estimated_hours_from_measured_strata": seconds / 3600 if not missing else None,
            "strata_without_measurements": dict(sorted(missing.items())),
            "includes_load_time": False,
            "warning": "Оценка основана только на завершённых наблюдениях и не включает загрузку/прогрев/паузы."}
