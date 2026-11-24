"""Планировщик и checkpoint-контур для последовательного выполнения наблюдений."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable

from .identity import attempt_id as make_attempt_id, digest, platform_observation_id
from .store import append_record, completed_observations, read_records

try:
    import fcntl
except ImportError:  # pragma: no cover - exercised on Windows hosts
    fcntl = None


class QueueLock:
    """Не допускает два процесса, исполняющих один run directory."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fd: int | None = None

    def __enter__(self) -> "QueueLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            else:
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_NBLCK, 1)
        except (OSError, BlockingIOError) as exc:
            os.close(self.fd)
            self.fd = None
            raise RuntimeError("Очередь уже выполняется другим процессом") from exc
        return self

    def __exit__(self, *_: object) -> None:
        if self.fd is not None:
            if fcntl is not None:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            else:
                import msvcrt
                os.lseek(self.fd, 0, os.SEEK_SET)
                msvcrt.locking(self.fd, msvcrt.LK_UNLCK, 1)
            os.close(self.fd)
            self.fd = None


class SafetyStop(RuntimeError):
    """Немедленно закрывает очередь после отказа обязательной проверки ресурсов."""


def order_jobs(jobs: Iterable[dict[str, Any]], *, start_block: str | None = None,
               completed: set[str] | None = None) -> list[dict[str, Any]]:
    """Детерминированный приоритетный порядок, без полного декартова расширения."""
    result = [dict(job) for job in jobs]
    required = {"job_id", "block_id", "priority", "observation_id"}
    ids: set[str] = set()
    observations: set[str] = set()
    for job in result:
        missing = required - job.keys()
        if missing:
            raise ValueError(f"В задании отсутствуют поля: {', '.join(sorted(missing))}")
        if job["job_id"] in ids:
            raise ValueError(f"Повторяющийся job_id: {job['job_id']}")
        ids.add(job["job_id"])
        if job["observation_id"] in observations:
            raise ValueError(f"Повторяющийся observation_id в очереди: {job['observation_id']}")
        observations.add(job["observation_id"])
    if start_block is not None:
        priorities = [job["priority"] for job in result if job["block_id"] == start_block]
        if not priorities:
            raise ValueError(f"Неизвестный блок: {start_block}")
        floor = min(priorities)
        result = [job for job in result if job["priority"] >= floor]
    done = completed or set()
    pending = [job for job in result if job["observation_id"] not in done]
    if not pending:
        return []
    active_priority = min(job["priority"] for job in pending)
    # Более поздние блоки остаются закрыты, пока предыдущий priority полностью не завершён.
    return sorted((job for job in pending if job["priority"] == active_priority),
                  key=lambda job: (job["block_id"], job["job_id"]))


def checkpoint_status(records_path: str | Path, *, total_jobs: int) -> dict[str, Any]:
    records = read_records(records_path)
    completed = completed_observations(records_path)
    failed = sum(row.get("status") == "failed" for row in records)
    return {"total_jobs": total_jobs, "completed_jobs": len(completed),
            "remaining_jobs": max(total_jobs - len(completed), 0),
            "failed_attempts": failed, "last_record_at": records[-1].get("recorded_at") if records else None}


def run_queue(jobs: Iterable[dict[str, Any]], *, run_dir: str | Path,
              execute: Callable[[dict[str, Any]], dict[str, Any]],
              preflight: Callable[[], None], platform_id: str,
              max_attempts: int = 2,
              start_block: str | None = None,
              owner_authorized: bool = False,
              safety_check: Callable[[dict[str, Any]], None] = lambda _job: None,
              max_runtime_s: float | None = None,
              retryable_error: Callable[[Exception], bool] = lambda exc: isinstance(
                  exc, (TimeoutError, ConnectionError, OSError)),
              stop_requested: Callable[[], bool] = lambda: False,
              on_progress: Callable[[dict[str, Any]], None] | None = None,
              clock: Callable[[], float] = time.time) -> dict[str, Any]:
    """Запускает переданный executor по одному заданию; backend hooks обязательны."""
    if type(max_attempts) is not int or max_attempts < 1 or max_attempts > 5:
        raise ValueError("max_attempts должен быть от 1 до 5")
    if owner_authorized is not True:
        raise PermissionError("Очередь исполнения закрыта без явного разрешения владельца")
    if max_runtime_s is not None and max_runtime_s <= 0:
        raise ValueError("max_runtime_s должен быть положительным")
    job_list = [dict(job) for job in jobs]
    if not platform_id or any(job.get("platform_id") != platform_id for job in job_list):
        raise ValueError("Job manifest должен состоять из одной совпадающей локальной platform_id")
    plan_hash = digest(job_list)
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    records_path = root / "observations.jsonl"
    run_id = str(uuid.uuid4())
    with QueueLock(root / "orchestrator.lock"):
        preflight()
        checkpoint_path = root / "checkpoint.json"
        if checkpoint_path.exists():
            try:
                previous_checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Checkpoint повреждён; автоматическое восстановление запрещено") from exc
            if previous_checkpoint.get("job_plan_hash") != plan_hash:
                raise ValueError("Job plan изменился относительно checkpoint; запуск заблокирован")
        existing_rows = read_records(records_path)
        done = completed_observations(records_path)
        failed_counts: dict[str, int] = {}
        for row in existing_rows:
            if row.get("status") == "failed" and row.get("observation_id"):
                failed_counts[row["observation_id"]] = failed_counts.get(row["observation_id"], 0) + 1
        ordered = order_jobs(job_list, start_block=start_block, completed=done)
        wall_start = time.monotonic()
        for index, job in enumerate(ordered):
            if stop_requested():
                break
            if max_runtime_s is not None and time.monotonic() - wall_start >= max_runtime_s:
                break
            try:
                safety_check(job)
            except SafetyStop as exc:
                append_record(root / "events.jsonl", {"event": "safety_block", "job_id": job["job_id"],
                                                       "observation_id": job["observation_id"],
                                                       "error": str(exc), "recorded_at_epoch": clock()})
                raise
            observation_id = job["observation_id"]
            previous_failures = failed_counts.get(observation_id, 0)
            first_attempt = previous_failures + 1
            for attempt_no in range(first_attempt, max_attempts + 1):
                start = clock()
                try:
                    result = execute(job)
                    if not isinstance(result, dict):
                        raise TypeError("executor должен вернуть объект результата")
                    record = {**job, **result, "status": "complete", "run_id": run_id,
                              "job_plan_hash": plan_hash,
                              "attempt_id": make_attempt_id(observation_id, attempt_no, run_id),
                              "attempt": attempt_no, "started_at_epoch": start,
                              "finished_at_epoch": clock()}
                    record["platform_observation_id"] = platform_observation_id(
                        observation_id, job["platform_id"], job["hardware_manifest_hash"],
                        run_id, record["attempt_id"])
                    append_record(records_path, record)
                    done.add(observation_id)
                    break
                except KeyboardInterrupt:
                    raise
                except SafetyStop as exc:
                    append_record(records_path, {**job, "status": "blocked", "error_type": type(exc).__name__,
                                                 "error": str(exc), "run_id": run_id,
                                                 "job_plan_hash": plan_hash,
                                                 "attempt_id": make_attempt_id(observation_id, attempt_no, run_id),
                                                 "attempt": attempt_no})
                    raise
                except Exception as exc:
                    record = {**job, "status": "failed", "error_type": type(exc).__name__,
                              "error": str(exc), "run_id": run_id,
                              "job_plan_hash": plan_hash,
                              "attempt_id": make_attempt_id(observation_id, attempt_no, run_id),
                              "attempt": attempt_no, "started_at_epoch": start,
                              "finished_at_epoch": clock()}
                    record["platform_observation_id"] = platform_observation_id(
                        observation_id, job["platform_id"], job["hardware_manifest_hash"],
                        run_id, record["attempt_id"])
                    append_record(records_path, record)
                    if attempt_no >= max_attempts or not retryable_error(exc):
                        break
            if on_progress is not None:
                on_progress({"finished_this_run": index + 1, "total_pending_at_start": len(ordered),
                             "completed_observations": len(done), "next_job_id":
                             ordered[index + 1]["job_id"] if index + 1 < len(ordered) else None})
            status = checkpoint_status(records_path, total_jobs=len(job_list))
            status["run_id"] = run_id
            checkpoint = root / "checkpoint.json"
            temporary = checkpoint.with_suffix(".json.tmp")
            status["job_plan_hash"] = plan_hash
            temporary.write_text(json.dumps(status, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                                  encoding="utf-8")
            os.replace(temporary, checkpoint)
        status = checkpoint_status(records_path, total_jobs=len(job_list))
        status["stopped_by_request"] = bool(stop_requested())
        return status


def load_job_file(path: str | Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось загрузить файл очереди: {exc}") from exc
    if not isinstance(value, dict) or not isinstance(value.get("jobs"), list):
        raise ValueError("Файл очереди должен содержать массив jobs")
    return value["jobs"]
