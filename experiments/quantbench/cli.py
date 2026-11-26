"""Минимальный безопасный CLI. Runtime execution gated until local validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .identity import config_id, digest
from .monitor import diagnose
from .analysis import export_snapshot, merge_result_files, validate_and_pair_platforms
from .orchestrator import load_job_file, order_jobs
from .executor import validate_job_manifest
from .queue_compile import write_job_manifest
from .planning import estimate_remaining
from .store import completed_observations, read_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantbench", description="Проверка и планирование локальных запусков LLM")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "plan", "dry-run", "preflight", "run", "resume", "fake-run", "status", "integrity",
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
        item.add_argument("--authorization-file")
        item.add_argument("--max-runtime", type=float)
        if command in {"run", "resume"}:
            item.add_argument("--max-new-observations", type=int,
                              help="ограничить число новых заданий за один запуск runtime")
        if command in {"slice", "compare", "merge"}:
            item.add_argument("--analysis-stage", choices=("pilot", "confirmatory", "descriptive"),
                              default="confirmatory")
    stop = sub.add_parser("stop", help="безопасно запросить остановку после текущего наблюдения")
    stop.add_argument("--run-dir", required=True)
    dataset = sub.add_parser("dataset-export", help="подготовить воспроизводимый срез локального исходного split")
    dataset.add_argument("--dataset", choices=("MERA", "LIBRA"), required=True)
    dataset.add_argument("--task", required=True)
    dataset.add_argument("--split", required=True)
    dataset.add_argument("--revision", required=True)
    dataset.add_argument("--source", required=True)
    dataset.add_argument("--provenance", required=True,
                         help="запись файла из source-download-manifest.json с repo/revision/path/SHA-256")
    dataset.add_argument("--output", required=True)
    dataset.add_argument("--manifest", required=True)
    dataset.add_argument("--count", type=int, required=True)
    dataset.add_argument("--seed", type=int, default=20261124)
    dataset.add_argument("--source-sha256")
    dataset.add_argument("--context-length", choices=("4K", "8K", "16K"))
    dataset.add_argument("--disjoint-source", help="JSON file containing IDs from the other split")
    dataset.add_argument("--tokenizer-path", help="локальный каталог tokenizer; сетевые загрузки отключены")
    ids = sub.add_parser("dataset-ids", help="создать ID-only manifest для проверки пересечений split")
    ids.add_argument("--dataset", choices=("MERA", "LIBRA"), required=True)
    ids.add_argument("--task", required=True)
    ids.add_argument("--split", required=True)
    ids.add_argument("--revision", required=True)
    ids.add_argument("--source", required=True)
    ids.add_argument("--output", required=True)
    research_data = sub.add_parser("prepare-research-data", help="проверить pinned источники и подготовить P0/P1 и выборки M2")
    research_data.add_argument("--source-root", default="datasets/source")
    research_data.add_argument("--output-root", default="results/raw/dataset-exports/m3.3.1")
    research_data.add_argument("--seed", type=int, default=20261124)
    download_sources = sub.add_parser("download-research-sources", help="загрузить только pinned dataset/evaluator/tokenizer files")
    download_sources.add_argument("--registry", default="experiments/quantbench/source_registry.json")
    download_sources.add_argument("--source-root", default="datasets/source")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "dataset-export":
            from .dataset_pipeline import prepare_export
            source_provenance = json.loads(Path(args.provenance).read_text(encoding="utf-8"))
            if isinstance(source_provenance, dict) and isinstance(source_provenance.get("files"), list):
                matches = [entry for entry in source_provenance["files"]
                           if Path(entry.get("local_path", "")).resolve() == Path(args.source).resolve()]
                if len(matches) != 1:
                    raise ValueError("В provenance manifest должна быть ровно одна запись для --source")
                source_provenance = matches[0]
            disjoint_ids = None
            if args.disjoint_source:
                disjoint_ids = json.loads(Path(args.disjoint_source).read_text(encoding="utf-8"))
                if isinstance(disjoint_ids, dict) and isinstance(disjoint_ids.get("item_ids"), list):
                    disjoint_ids = disjoint_ids["item_ids"]
                if not isinstance(disjoint_ids, list) or not all(isinstance(x, (str, int)) for x in disjoint_ids):
                    raise ValueError("--disjoint-source должен содержать JSON-массив item IDs либо dataset manifest")
            tokenizer = None
            if args.tokenizer_path:
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True,
                                                              trust_remote_code=False)
                except (ImportError, OSError, ValueError) as exc:
                    raise ValueError(f"Локальный tokenizer не загружен: {exc}") from exc
            manifest = prepare_export(source_path=args.source, output_path=args.output,
                                      manifest_path=args.manifest, dataset=args.dataset, task=args.task,
                                      split=args.split, revision=args.revision, count=args.count,
                                      seed=args.seed, expected_source_sha256=args.source_sha256,
                                      context_length=args.context_length,
                                      disjoint_ids=map(str, disjoint_ids) if disjoint_ids is not None else None,
                                      tokenizer=tokenizer, source_provenance=source_provenance)
            print(json.dumps({"dataset": args.dataset, "task": args.task, "split": args.split,
                              "sample_count": manifest["sample_count"], "source_sha256": manifest["source_sha256"],
                              "prepared_sha256": manifest["prepared_sha256"],
                              "item_ids_sha256": manifest["item_ids_sha256"],
                              "official_scorer_status": manifest["official_scorer_status"],
                              "output": args.output, "manifest": args.manifest}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "dataset-ids":
            from .dataset_pipeline import export_split_ids
            manifest = export_split_ids(source_path=args.source, output_path=args.output,
                                        dataset=args.dataset, task=args.task, split=args.split,
                                        revision=args.revision)
            print(json.dumps({"dataset": args.dataset, "task": args.task, "split": args.split,
                              "item_count": len(manifest["item_ids"]),
                              "source_sha256": manifest["source_sha256"],
                              "item_ids_sha256": manifest["item_ids_sha256"],
                              "output": args.output}, ensure_ascii=False, indent=2))
            return 0
        if args.command == "prepare-research-data":
            from .research_data import build_research_exports
            summary = build_research_exports(source_root=args.source_root,
                                             output_root=args.output_root,
                                             seed=args.seed)
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.command == "download-research-sources":
            from .source_artifacts import download_pinned_artifacts
            report = download_pinned_artifacts(registry_path=args.registry, source_root=args.source_root)
            statuses = {}
            for artifact in report["artifacts"]:
                statuses[artifact["status"]] = statuses.get(artifact["status"], 0) + 1
            print(json.dumps({"artifact_count": report["artifact_count"],
                              "verified_count": report["verified_count"],
                              "total_bytes": report["total_bytes"], "statuses": statuses},
                             ensure_ascii=False, indent=2))
            return 0
        if args.command == "stop":
            root = Path(args.run_dir).resolve()
            root.mkdir(parents=True, exist_ok=True)
            marker = root / "STOP"
            marker.touch(exist_ok=True)
            print(json.dumps({"stop_requested": True, "marker": str(marker)}, ensure_ascii=False))
            return 0
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
        if args.command == "dry-run":
            if not args.jobs:
                raise ValueError("dry-run требует --jobs")
            try:
                job_manifest = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(f"Не удалось прочитать job manifest: {exc}") from exc
            platform_id = job_manifest.get("local_platform_id") if isinstance(job_manifest, dict) else None
            if not platform_id and isinstance(job_manifest, dict) and job_manifest.get("jobs"):
                platform_id = job_manifest["jobs"][0].get("platform_id")
            if not platform_id:
                raise ValueError("dry-run не определил локальную platform_id")
            jobs = validate_job_manifest(job_manifest, platform_id=platform_id)
            planned = order_jobs(jobs, start_block=args.start_block)
            print(json.dumps({"job_count": len(jobs), "planned_count": len(planned),
                              "job_plan_hash": digest(jobs),
                              "inference_started": False, "owner_authorization": "not used"},
                             ensure_ascii=False, indent=2))
            return 0
        if args.command in {"run", "resume"}:
            if not args.jobs or not args.run_dir:
                raise ValueError(f"{args.command} требует --jobs и --run-dir")
            from .executor import ExperimentExecutor
            from .platforms.macos import MacOSAdapter
            manifest = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
            executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=MacOSAdapter(),
                                          run_dir=args.run_dir)
            result = executor.run_production(authorization_file=args.authorization_file,
                                             start_block=args.start_block, max_runtime_s=args.max_runtime,
                                             max_new_observations=args.max_new_observations)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "fake-run":
            if not args.jobs or not args.run_dir:
                raise ValueError("fake-run требует --jobs и --run-dir вне results/")
            from .executor import ExperimentExecutor
            from .fake_runtime import SyntheticRuntime
            from .platforms.macos import MacOSAdapter
            manifest = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
            executor = ExperimentExecutor(config=config, manifest=manifest, platform_adapter=MacOSAdapter(),
                                          run_dir=args.run_dir, fake=True,
                                          runtime_factory=lambda _job, _model: SyntheticRuntime())
            result = executor.run_fake(start_block=args.start_block, max_runtime_s=args.max_runtime)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "preflight":
            report = diagnose(config["runtime"]["executable"], [m["gguf_path"] for m in config["models"]] + [d["path"] for d in config["datasets"]])
            snapshot = report["snapshot"]
            from .platforms.macos import MacOSAdapter
            adapter = MacOSAdapter() if snapshot.get("platform") == "macOS" else None
            adapter_errors = adapter.preflight(config) if adapter is not None else [
                f"Для платформы {snapshot.get('platform')} адаптер не реализован"]
            manifest = adapter.hardware_manifest().canonical_fields() if adapter is not None else None
            pressure_events = None
            safety_snapshot = snapshot
            if adapter is not None:
                try:
                    pressure_events = adapter.start_pressure_monitor()
                    baseline_swap = snapshot.get("swap_used_bytes")
                    safety_snapshot = adapter.safety_snapshot(pid=None, swap_baseline_bytes=baseline_swap,
                                                              pressure_events=pressure_events)
                except (OSError, RuntimeError, ValueError) as exc:
                    adapter_errors.append(f"Не удалось запустить аппаратный сборщик: {type(exc).__name__}: {exc}")
            missing_required = [name for name, available in (
                ("system_used_bytes", safety_snapshot.get("system_used_bytes") is not None),
                ("reclaimable_memory_estimate_bytes", safety_snapshot.get("reclaimable_memory_estimate_bytes") is not None),
                ("swap_used_bytes", safety_snapshot.get("swap_used_bytes") is not None),
                ("thermal_state", safety_snapshot.get("thermal_state") is not None),
                ("pressure_notifications", safety_snapshot.get("pressure_notifications_available") is True),
            ) if not available]
            if pressure_events is not None:
                pressure_events.close()
            from .safety import SafetyGuard
            safety_probe = SafetyGuard()
            safety_probe.evaluate({**safety_snapshot, "swap_delta_bytes": 0,
                                   "process_state": "not_started", "runtime_active": False},
                                  now=safety_snapshot.get("monotonic_time"))
            report["config_valid"] = True
            report["loopback"] = config["runtime"]["host"] in {"127.0.0.1", "localhost", "::1"}
            report["checks"] = {
                "configuration": {"ready": True},
                "platform_adapter": {"ready": adapter is not None and not adapter_errors,
                                     "platform_id": adapter.platform_id if adapter else None,
                                     "manifest": manifest, "reasons": adapter_errors},
                "system_metrics": {"ready": not missing_required, "missing": missing_required},
                "safety_guard_initial_state": {"state": safety_probe.state.value,
                                               "ready_for_inference": False,
                                               "reason": "одного снимка недостаточно для временных критериев допуска и восстановления"},
                "runtime": {"ready": report["runtime_available"], "version": report["runtime_version"]},
                "model_and_dataset_artifacts": {"ready": all(a["exists"] for a in report["artifacts"])},
                "resource_snapshot": {key: safety_snapshot.get(key) for key in (
                    "timestamp", "system_used_bytes", "available_memory_estimate_bytes", "reclaimable_memory_estimate_bytes", "free_pages_bytes",
                    "active_pages_bytes", "inactive_pages_bytes", "wired_pages_bytes",
                    "compressor_physical_bytes", "purgeable_pages_bytes", "speculative_pages_bytes",
                    "vm_page_size_bytes", "vm_page_counts", "swap_used_bytes", "swap_delta_bytes",
                    "memory_pressure", "pressure_notifications_available", "thermal_state",
                    "physical_memory_bytes", "physical_footprint_bytes", "system_memory_source",
                    "memory_pressure_source")},
                "owner_authorization": {"ready": False, "reason": "разрешение на запуск эксперимента не зарегистрировано"},
            }
            report["blocked_reasons"] = []
            if not report["runtime_available"]:
                report["blocked_reasons"].append("llama-server отсутствует или --version завершился ошибкой")
            report["blocked_reasons"].extend(adapter_errors)
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
                                     output_path=args.output, stage=args.analysis_stage)
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
                args.right_platform: read_records(args.right)}, stage=args.analysis_stage)
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
                                         args.right_platform: args.right}, args.output,
                                        stage=args.analysis_stage)
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
