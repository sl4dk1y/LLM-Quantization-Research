"""Минимальный безопасный CLI. Runtime execution gated until local validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_config
from .identity import config_id
from .monitor import diagnose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="quantbench", description="Проверка и планирование локальных запусков LLM")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("validate", "plan", "preflight", "run", "resume", "status", "integrity"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        item.add_argument("--run-dir")
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
            raise ValueError("Выполнение заблокировано до проверки macOS safety collector, runtime и scorer на целевой машине")
        if args.command == "preflight":
            report = diagnose(config["runtime"]["executable"], [m["gguf_path"] for m in config["models"]] + [d["path"] for d in config["datasets"]])
            snapshot = report["snapshot"]
            missing_required = [name for name in ("system_used_bytes", "memory_pressure", "swap_used_bytes", "thermal_state") if snapshot.get(name) is None]
            missing_required.append("physical_footprint_bytes (not exposed by current Python monitor)")
            report["config_valid"] = True
            report["loopback"] = config["runtime"]["host"] in {"127.0.0.1", "localhost", "::1"}
            report["blocked_reasons"] = []
            if not report["runtime_available"]:
                report["blocked_reasons"].append("llama-server отсутствует или --version завершился ошибкой")
            if missing_required:
                report["blocked_reasons"].append("Недоступны обязательные safety-метрики: " + ", ".join(missing_required))
            if not all(a["exists"] for a in report["artifacts"]):
                report["blocked_reasons"].append("Не все локальные model/dataset artifacts доступны")
            report["inference_started"] = False
            report["allowed_for_inference"] = False
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if not report["blocked_reasons"] else 2
        if args.command in {"status", "integrity"}:
            if not args.run_dir:
                raise ValueError(f"{args.command} требует --run-dir")
            raise ValueError(f"{args.command}: формат run directory ещё не создан; чтение данных не изменяло файлы")
    except (ValueError, OSError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    return 2
