"""Минимальный безопасный CLI. Runtime execution gated until local validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .identity import config_id
from .monitor import diagnose
from .analysis import export_snapshot, merge_result_files, validate_and_pair_platforms
from .orchestrator import load_job_file, order_jobs
from .queue_compile import write_job_manifest
from .planning import estimate_remaining
from .store import completed_observations, read_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantbench", description="Проверка и планирование локальных запусков LLM")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "plan", "preflight", "run", "resume", "status", "integrity",
                    "queue", "compile-queue", "progress", "slice", "compare", "merge"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        item.add_argument("--run-dir")
        item.add_argument("--jobs")
        item.add_argument("--plan")
        item.add_argument("--start-block")
        item.add_argument("--platform-id")
        item.add_argument("--output")
        item.add_argument("--left")
        item.add_argument("--right")
        item.add_argument("--left-platform")
        item.add_argument("--right-platform")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "validate":
            print(f"Конфигурация корректна: {config_id(config)}")
            return 0
        if args.command == "plan":
            planned = {
                "config_id": config_id(config), "mode": config["mode"],
                "models": [m["id"] for m in config["models"]],
                "datasets": [{"id": d["id"], "split": d["split"], "tasks": d["tasks"]} for d in config["datasets"]],
                "contexts": config["contexts"], "inference": False,
            }
            print(json.dumps(planned, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"run", "resume"}:
            raise ValueError("Исполнение пока закрыто: отсутствует интегрированный platform safety monitor и утверждённый scorer/runtime preflight")
        if args.command == "preflight":
            report = diagnose(config["runtime"]["executable"], [m["gguf_path"] for m in config["models"]] + [d["path"] for d in config["datasets"]])
            snapshot = report["snapshot"]
            pressure_category = (snapshot.get("memory_pressure") or {}).get("category")
            missing_required = [name for name, available in (
                ("system_used_bytes", snapshot.get("system_used_bytes") is not None),
                ("memory_pressure.category", pressure_category is not None),
                ("swap_used_bytes", snapshot.get("swap_used_bytes") is not None),
                ("thermal_state", snapshot.get("thermal_state") is not None),
            ) if not available]
            report["config_valid"] = True
            report["loopback"] = config["runtime"]["host"] in {"127.0.0.1", "localhost", "::1"}
            report["checks"] = {
                "configuration": {"ready": True},
                "platform_adapter": {"ready": False, "reason": "профиль M3.2 ещё не подключён к CLI preflight"},
                "system_metrics": {"ready": not missing_required, "missing": missing_required},
                "runtime": {"ready": report["runtime_available"], "version": report["runtime_version"]},
                "model_and_dataset_artifacts": {"ready": all(a["exists"] for a in report["artifacts"])},
                "resource_snapshot": {"timestamp": snapshot.get("timestamp"), "system_used_bytes": snapshot.get("system_used_bytes"),
                                      "swap_used_bytes": snapshot.get("swap_used_bytes")},
                "owner_authorization": {"ready": False, "reason": "разрешение на запуск эксперимента не зарегистрировано"},
            }
            report["blocked_reasons"] = []
            if not report["runtime_available"]:
                report["blocked_reasons"].append("llama-server отсутствует или --version завершился ошибкой")
            report["blocked_reasons"].append("Платформенный адаптер ещё не интегрирован в preflight")
            if missing_required:
                report["blocked_reasons"].append("Недоступны обязательные safety-метрики: " + ", ".join(missing_required))
            if not all(a["exists"] for a in report["artifacts"]):
                report["blocked_reasons"].append("Не все локальные model/dataset artifacts доступны")
            report["blocked_reasons"].append("Разрешение владельца на эксперимент не зарегистрировано")
            report["inference_started"] = False
            report["allowed_for_inference"] = False
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if not report["blocked_reasons"] else 2
        if args.command == "queue":
            if not args.jobs:
                raise ValueError("queue требует --jobs")
            jobs = load_job_file(args.jobs)
            if args.platform_id:
                jobs = [job for job in jobs if job.get("platform_id") == args.platform_id]
                if not jobs:
                    raise ValueError(f"В job manifest нет заданий platform_id={args.platform_id}")
            done = completed_observations(Path(args.run_dir) / "observations.jsonl") if args.run_dir else set()
            pending = order_jobs(jobs, start_block=args.start_block, completed=done)
            active_priority = min((j["priority"] for j in pending), default=None)
            print(json.dumps({"config_id": config_id(config), "job_count": len(jobs),
                              "completed": len(done), "active_priority": active_priority,
                              "active_block": pending[0]["block_id"] if pending else None,
                              "active_jobs": pending, "inference_started": False,
                              "ready_for_execution": False}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "compile-queue":
            if not args.plan or not args.output or not args.platform_id:
                raise ValueError("compile-queue требует --plan, --output и --platform-id")
            try:
                plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Не удалось прочитать queue plan: {exc}") from exc
            result = write_job_manifest(plan, args.output, platform_id=args.platform_id)
            print(json.dumps({"job_plan_hash": result["job_plan_hash"],
                              "job_count": result["job_count"], "output": args.output,
                              "execution_started": False}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "progress":
            if not args.jobs or not args.run_dir:
                raise ValueError("progress требует --jobs и --run-dir")
            jobs = load_job_file(args.jobs)
            records_path = Path(args.run_dir) / "observations.jsonl"
            records = read_records(records_path)
            status = {"job_count": len(jobs), "completed_observation_count": len(completed_observations(records_path)),
                      "failed_attempts": sum(row.get("status") == "failed" for row in records),
                      "unplanned_completed_observations": sorted(
                          completed_observations(records_path) - {job["observation_id"] for job in jobs}),
                      "remaining_estimate": estimate_remaining(jobs, records),
                      "next_jobs": order_jobs(jobs, start_block=args.start_block,
                                              completed=completed_observations(records_path))[:20]}
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return 0
        if args.command == "slice":
            if not args.run_dir or not args.output:
                raise ValueError("slice требует --run-dir и --output")
            result = export_snapshot(records_path=Path(args.run_dir) / "observations.jsonl",
                                     output_path=args.output)
            print(json.dumps({"snapshot_id": result["snapshot_id"], "output": args.output,
                              "source_sha256": result["source_sha256"],
                              "complete_record_count": result["complete_record_count"]},
                             ensure_ascii=False, indent=2))
            return 0
        if args.command == "compare":
            required = (args.left, args.right, args.left_platform, args.right_platform)
            if not all(required):
                raise ValueError("compare требует --left/--right и соответствующие --left-platform/--right-platform")
            result = validate_and_pair_platforms({
                args.left_platform: read_records(args.left),
                args.right_platform: read_records(args.right)})
            print(json.dumps({"compatible": result["compatible"], "complete": result["complete"],
                              "issues": result["issues"],
                              "paired_count": len(result["pairs"]),
                              "missing_by_platform": result["missing_by_platform"],
                              "counts": result["counts"]}, ensure_ascii=False, indent=2))
            return 0 if result["compatible"] and result["complete"] else 2
        if args.command == "merge":
            required = (args.left, args.right, args.left_platform, args.right_platform, args.output)
            if not all(required):
                raise ValueError("merge требует --left/--right, platform IDs и --output")
            result = merge_result_files({args.left_platform: args.left,
                                         args.right_platform: args.right}, args.output)
            print(json.dumps({"merge_id": result["merge_id"], "output": args.output,
                              "paired_count": len(result["paired_observations"]),
                              "sources": result["sources"]}, ensure_ascii=False, indent=2))
            return 0
        if args.command in {"status", "integrity"}:
            if not args.run_dir:
                raise ValueError(f"{args.command} требует --run-dir")
            records_path = Path(args.run_dir) / "observations.jsonl"
            records = read_records(records_path)
            ids = [row.get("attempt_id") for row in records]
            platform_ids = [row.get("platform_observation_id") for row in records
                            if row.get("status") in {"complete", "failed"}]
            if args.command == "integrity" and (None in ids or len(ids) != len(set(ids)) or
                                                   None in platform_ids or len(platform_ids) != len(set(platform_ids))):
                raise ValueError("Найдены отсутствующие или дублированные attempt_id/platform_observation_id")
            print(json.dumps({"records": len(records), "complete": sum(r.get("status") == "complete" for r in records),
                              "failed_attempts": sum(r.get("status") == "failed" for r in records),
                              "integrity": "ok" if len(ids) == len(set(ids)) and None not in ids else "invalid"},
                             ensure_ascii=False, indent=2))
            return 0
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    return 2
