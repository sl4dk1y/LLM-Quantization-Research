"""Интеграция очереди, локальных экспортов, runtime и safety monitor."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from .datasets import file_sha256, load_jsonl_dataset
from .identity import digest, hardware_manifest_hash
from .orchestrator import SafetyStop, run_queue


class ExecutionBlocked(RuntimeError):
    """Запуск остановлен обязательной проверкой допуска."""


# Реальный запуск остаётся закрытым до отдельного аудита и решения M4.
REAL_EXECUTION_ENABLED = False
REQUIRED_JOB_FIELDS = (
    "job_id", "observation_id", "platform_id", "hardware_manifest_hash", "block_id", "priority",
    "model_id", "model_revision", "gguf_sha256", "quantization", "kv_cache", "dataset_id", "dataset_revision",
    "dataset_hash", "split", "task", "item_id", "context_length", "generation_config", "scorer_version",
    "protocol_version", "protocol_hash", "runtime_version", "runtime_binary_sha256", "llama_cpp_commit",
    "tokenizer_revision", "chat_template_hash", "reasoning_mode",
)


def validate_job_manifest(manifest: dict[str, Any], *, platform_id: str) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Поддерживается job manifest schema_version=1")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Job manifest должен содержать непустой jobs")
    if any(not isinstance(job, dict) for job in jobs):
        raise ValueError("Каждое jobs entry должно быть JSON-объектом")
    if manifest.get("job_count") != len(jobs):
        raise ValueError("job_count не совпадает с длиной jobs")
    if manifest.get("job_plan_hash") != digest(jobs):
        raise ValueError("Job manifest hash не совпадает с содержимым")
    if manifest.get("local_platform_id", platform_id) != platform_id:
        raise ValueError("Job manifest предназначен другой платформе")
    seen: set[tuple[str, str]] = set()
    seen_job_ids: set[str] = set()
    for index, job in enumerate(jobs):
        missing = [key for key in REQUIRED_JOB_FIELDS if job.get(key) is None]
        if missing:
            raise ValueError(f"jobs[{index}] отсутствуют обязательные поля: {', '.join(missing)}")
        if job["platform_id"] != platform_id:
            raise ValueError("Очередь должна содержать только задания локальной platform_id")
        if (not isinstance(job["job_id"], str) or not job["job_id"]
                or not isinstance(job["observation_id"], str) or not job["observation_id"].startswith("obs-")):
            raise ValueError(f"jobs[{index}]: некорректные job_id/observation_id")
        if job["job_id"] in seen_job_ids:
            raise ValueError("Повтор job_id в job manifest")
        seen_job_ids.add(job["job_id"])
        for field in ("protocol_hash", "gguf_sha256", "dataset_hash", "hardware_manifest_hash",
                      "runtime_binary_sha256", "chat_template_hash"):
            value = job[field]
            if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
                raise ValueError(f"jobs[{index}].{field} должен быть SHA-256")
        key = (str(job["observation_id"]), platform_id)
        if key in seen:
            raise ValueError("Повтор observation_id в пределах платформы")
        seen.add(key)
    return jobs


def verify_owner_authorization(path: str | Path | None, *, plan_hash: str, platform_id: str) -> dict[str, Any]:
    if path is None:
        raise PermissionError("Для реального запуска требуется отдельный файл разрешения владельца")
    try:
        record = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PermissionError("Файл разрешения владельца отсутствует или повреждён") from exc
    required = {"authorized": True, "purpose": "real_inference", "plan_hash": plan_hash,
                "platform_id": platform_id}
    if not isinstance(record, dict) or any(record.get(key) != value for key, value in required.items()):
        raise PermissionError("Разрешение владельца не подтверждает этот план и платформу")
    if not record.get("authorization_id") or not record.get("owner"):
        raise PermissionError("В разрешении владельца обязательны authorization_id и owner")
    return record


def _load_manifest_sidecar(dataset_path: Path, dataset_row: dict[str, Any]) -> dict[str, Any]:
    configured = dataset_row.get("manifest_path")
    sidecar = Path(configured) if configured else dataset_path.with_suffix(".manifest.json")
    if not sidecar.is_file():
        raise ValueError(f"Dataset export manifest отсутствует: {sidecar}")
    try:
        manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Dataset export manifest повреждён: {sidecar}") from exc
    if manifest.get("prepared_sha256") != file_sha256(dataset_path):
        raise ValueError("SHA-256 dataset не совпадает с его sidecar manifest")
    if (not manifest.get("source_sha256") or len(manifest["source_sha256"]) != 64
            or not manifest.get("dataset_revision")):
        raise ValueError("Dataset manifest не фиксирует исходную ревизию и SHA-256")
    provenance = manifest.get("source_provenance")
    if not isinstance(provenance, dict) or provenance.get("revision") != manifest["dataset_revision"]:
        raise ValueError("Source provenance отсутствует или указывает другую ревизию")
    if provenance.get("verified") is not True:
        raise ValueError("Source provenance не помечен как проверенный")
    if provenance.get("original_local_path"):
        original = Path(provenance["original_local_path"])
        if not original.is_file() or file_sha256(original) != provenance.get("original_sha256"):
            raise ValueError("Pinned source artifact LIBRA отсутствует или не совпадает по SHA-256")
        transformed = Path(provenance.get("local_path", ""))
        if (not transformed.is_file()
                or file_sha256(transformed) != provenance.get("transformed_source_sha256")):
            raise ValueError("Преобразованный исходный LIBRA corpus отсутствует или не совпадает по SHA-256")
    else:
        source_file = Path(provenance.get("local_path", ""))
        expected_source_hash = provenance.get("sha256")
        if not source_file.is_file() or not isinstance(expected_source_hash, str) or file_sha256(source_file) != expected_source_hash:
            raise ValueError("Исходный dataset artifact отсутствует или его SHA-256 не совпадает")
    return manifest


def _find_model(config: dict[str, Any], job: dict[str, Any], *, verify_artifact: bool) -> dict[str, Any]:
    matches = [model for model in config.get("models", [])
               if model.get("id") == job["model_id"] and model.get("revision") == job["model_revision"]]
    if len(matches) != 1:
        raise ValueError("Для задания должна существовать ровно одна модель с закреплённой ревизией")
    model = dict(matches[0])
    if model.get("gguf_sha256") != job["gguf_sha256"]:
        raise ValueError("GGUF SHA-256 в конфигурации не совпадает с job manifest")
    if model.get("quantization") != job["quantization"] or model.get("kv_cache") != job["kv_cache"]:
        raise ValueError("Quantization/KV cache не совпадает с job manifest")
    if verify_artifact and (not Path(model["gguf_path"]).is_file()
                            or file_sha256(model["gguf_path"]) != job["gguf_sha256"]):
        raise ValueError("GGUF файл отсутствует или его SHA-256 не совпадает")
    return model


def _score(job: dict[str, Any], row: dict[str, Any], raw_output: str) -> tuple[str, dict[str, Any]]:
    from .dataset_pipeline import SCORER_VERSION, MERA_TASKS, LIBRA_TASKS, LibraScorer, score_mera
    if job["scorer_version"] != SCORER_VERSION:
        raise ValueError("Scorer version задания не совпадает с закреплённым адаптером")
    if job["dataset_id"] == "MERA" or job["dataset_id"].startswith("MERA/"):
        if job["task"] not in MERA_TASKS:
            raise ValueError("MERA task не поддерживается scorer")
        reference = row.get("reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError("В MERA item отсутствует reference")
        output = score_mera(task=job["task"], predictions=[raw_output], references=[reference])
        return output["predictions_processed"][0], {"scorer": output["scorer_revision"], **output["per_item"][0]}
    if job["dataset_id"] == "LIBRA" or job["dataset_id"].startswith("LIBRA/"):
        if job["task"] not in LIBRA_TASKS:
            raise ValueError("LIBRA task не поддерживается scorer")
        refs = row.get("references") or row.get("positive_outputs")
        if not isinstance(refs, list) or not refs or not all(isinstance(x, str) and x for x in refs):
            raise ValueError("В LIBRA item отсутствуют multiple acceptable references")
        scorer = LibraScorer()
        score = scorer.score_item(raw_output, refs)
        return raw_output, {"scorer": scorer.version, "exact_match": score}
    raise ValueError(f"Неизвестный dataset_id: {job['dataset_id']}")


def _reference_fields(row: dict[str, Any]) -> dict[str, Any]:
    references = row.get("references") or row.get("positive_outputs")
    if not references and row.get("reference") is not None:
        references = [row["reference"]]
    return {"reference": row.get("reference"), "references": references}


def _validate_scorer(job: dict[str, Any]) -> None:
    from .dataset_pipeline import (LIBRA_TASKS, MERA_TASKS, SCORER_VERSION,
                                   OfficialScorerUnavailable)
    if job["scorer_version"] != SCORER_VERSION:
        raise ValueError("Scorer version задания не совпадает с закреплённым адаптером")
    dataset_id = job["dataset_id"]
    if dataset_id == "LIBRA" or dataset_id.startswith("LIBRA/"):
        if job["task"] not in LIBRA_TASKS:
            raise ValueError("LIBRA task не поддерживается scorer")
        try:
            import pymorphy3  # noqa: F401
        except ImportError as exc:
            raise OfficialScorerUnavailable("Официальный LIBRA scorer требует pymorphy3") from exc
    elif dataset_id == "MERA" or dataset_id.startswith("MERA/"):
        if job["task"] not in MERA_TASKS:
            raise ValueError("MERA task не поддерживается scorer")
        if job["task"] == "MultiQ":
            try:
                from transformers.data.metrics import squad_metrics  # noqa: F401
            except ImportError as exc:
                raise OfficialScorerUnavailable("Официальный MERA MultiQ scorer недоступен") from exc
        if job["task"] == "ruOpenBookQA":
            try:
                from sklearn.metrics import f1_score  # noqa: F401
            except ImportError as exc:
                raise OfficialScorerUnavailable("Официальный MERA ruOpenBookQA scorer недоступен") from exc
    else:
        raise ValueError(f"Неизвестный dataset_id: {dataset_id}")


def _runtime_metrics(response: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
    prompt_tokens = timings.get("prompt_n")
    generated_tokens = timings.get("predicted_n")
    prefill_ms = timings.get("prompt_ms")
    decode_ms = timings.get("predicted_ms")
    return {
        "ttft_s": response.get("ttft_s"),
        "prompt_tokens": prompt_tokens if type(prompt_tokens) is int else None,
        "export_prompt_tokens": row.get("prompt_token_count"),
        "prefill_duration_s": prefill_ms / 1000 if isinstance(prefill_ms, (int, float)) else None,
        "generated_tokens": generated_tokens if type(generated_tokens) is int else None,
        "decode_duration_s": decode_ms / 1000 if isinstance(decode_ms, (int, float)) else None,
        "server_timings": timings,
        "client_request_wall_s": response.get("request_wall_s"),
        "end_to_end_latency_s": response.get("request_wall_s"),
        "finish_reason": response.get("finish_reason"),
        "truncated": response.get("finish_reason") == "length",
    }


class ExperimentExecutor:
    """Объединяет существующие компоненты и оставляет адаптер платформы внедряемым."""
    def __init__(self, *, config: dict[str, Any], manifest: dict[str, Any], platform_adapter: Any,
                 run_dir: str | Path, runtime_factory: Callable[..., Any] | None = None,
                 fake: bool = False):
        self.config = config
        self.platform_adapter = platform_adapter
        self.platform_id = platform_adapter.platform_id
        self.jobs = validate_job_manifest(manifest, platform_id=self.platform_id)
        self.plan_hash = manifest["job_plan_hash"]
        self.run_dir = Path(run_dir).resolve()
        self.fake = fake
        self.runtime_factory = runtime_factory
        self._dataset_cache: dict[tuple[str, str, str, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        self._model_cache: dict[tuple[str, str, str], dict[str, Any]] = {}
        self._verified_runtime_binaries: set[tuple[str, str]] = set()
        self._runtime_session: dict[str, Any] | None = None
        self._libra_scorer: Any | None = None
        if fake and any(part == "results" for part in self.run_dir.parts):
            raise ValueError("Fake run должен храниться вне results/")

    def _claim_run_mode(self, mode: str) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        marker = self.run_dir / "execution-mode.json"
        if marker.exists():
            try:
                previous = json.loads(marker.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError("Execution mode marker повреждён") from exc
            if previous.get("mode") != mode or previous.get("job_plan_hash") != self.plan_hash:
                raise ValueError("Каталог запуска уже закреплён за другим режимом или планом")
        else:
            if (self.run_dir / "observations.jsonl").exists():
                raise ValueError("Нельзя повторно использовать каталог с наблюдениями без execution mode marker")
            try:
                with marker.open("x", encoding="utf-8") as stream:
                    json.dump({"mode": mode, "job_plan_hash": self.plan_hash}, stream, sort_keys=True)
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
            except FileExistsError as exc:
                raise ValueError("Каталог запуска одновременно захватывается другим процессом") from exc

    def validate_inputs(self) -> None:
        if not self.platform_adapter.hardware_manifest():
            raise ValueError("Аппаратный manifest недоступен")
        actual_hw_hash = hardware_manifest_hash(self.platform_adapter.hardware_manifest().canonical_fields())
        for job in self.jobs:
            if job["hardware_manifest_hash"] != actual_hw_hash:
                raise ValueError("Аппаратный manifest не совпадает с утверждённым заданием")
            self._load_model(job)
            self._load_dataset(job)
            _validate_scorer(job)
            if job["runtime_version"] != self.config["runtime"].get("version"):
                raise ValueError("Версия runtime не совпадает с заданием")
            if job["llama_cpp_commit"] != self.config["runtime"].get("llama_cpp_commit"):
                raise ValueError("Ревизия llama.cpp не совпадает с заданием")
            executable = self.config["runtime"].get("executable")
            if self.fake:
                # Synthetic execution does not inspect or execute the configured runtime binary.
                if self.config["runtime"].get("reasoning_mode") != job["reasoning_mode"]:
                    raise ValueError("Reasoning mode в runtime не совпадает с заданием")
                if self.config["runtime"].get("chat_template_sha256") != job["chat_template_hash"]:
                    raise ValueError("Chat template SHA-256 не подтверждён runtime конфигурацией")
                if job["context_length"] not in self.config.get("contexts", []):
                    raise ValueError("Context length задания не включён в конфигурацию")
                if self.config.get("generation") != job["generation_config"]:
                    raise ValueError("Параметры генерации отличаются от утверждённой конфигурации")
                continue
            executable_path = Path(executable)
            resolved = str(executable_path) if executable_path.is_file() else shutil.which(executable)
            binary_key = (resolved or "", job["runtime_binary_sha256"])
            if not resolved or (binary_key not in self._verified_runtime_binaries
                                and file_sha256(resolved) != job["runtime_binary_sha256"]):
                raise ValueError("llama-server отсутствует или SHA-256 бинарного файла не совпадает")
            if binary_key not in self._verified_runtime_binaries:
                try:
                    version_result = subprocess.run([resolved, "--version"], capture_output=True,
                                                    text=True, timeout=5, check=False)
                except (OSError, subprocess.TimeoutExpired) as exc:
                    raise ValueError(f"Не удалось проверить версию llama-server: {type(exc).__name__}") from exc
                actual_version = (version_result.stdout + version_result.stderr).strip()
                if (version_result.returncode != 0 or actual_version != job["runtime_version"]
                        or job["llama_cpp_commit"] not in actual_version):
                    raise ValueError("Фактическая версия/build llama-server не совпадает с job manifest")
            self._verified_runtime_binaries.add(binary_key)
            if self.config["runtime"].get("reasoning_mode") != job["reasoning_mode"]:
                raise ValueError("Reasoning mode в runtime не совпадает с заданием")
            if self.config["runtime"].get("chat_template_sha256") != job["chat_template_hash"]:
                raise ValueError("Chat template SHA-256 не подтверждён runtime конфигурацией")
            if job["context_length"] not in self.config.get("contexts", []):
                raise ValueError("Context length задания не включён в конфигурацию")
            if self.config.get("generation") != job["generation_config"]:
                raise ValueError("Параметры генерации отличаются от утверждённой конфигурации")

    def _load_dataset(self, job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        entries = [dataset for dataset in self.config.get("datasets", [])
                   if dataset.get("revision") == job["dataset_revision"]
                   and dataset.get("sha256") == job["dataset_hash"]
                   and dataset.get("split") == job["split"]
                   and (dataset.get("id") == job["dataset_id"]
                        or dataset.get("id") == f"{job['dataset_id']}/{job['task']}"
                        or job["task"] in dataset.get("tasks", []))]
        if len(entries) != 1:
            raise ValueError("Для задания не найдена однозначная запись dataset в конфигурации")
        key = (str(Path(entries[0]["path"]).resolve()), job["dataset_hash"],
               job["dataset_revision"], job["split"])
        if key not in self._dataset_cache:
            rows = load_jsonl_dataset(key[0], expected_revision=key[2], expected_sha256=key[1], expected_split=key[3])
            manifest = _load_manifest_sidecar(Path(key[0]), entries[0])
            item_ids = [str(row.get("id", "")) for row in rows]
            manifest_ids = manifest.get("item_ids")
            if (manifest.get("sample_count") != len(rows) or not isinstance(manifest_ids, list)
                    or set(map(str, manifest_ids)) != set(item_ids) or len(set(item_ids)) != len(item_ids)):
                raise ValueError("Sample count или item IDs в sidecar manifest не совпадают с экспортом")
            self._dataset_cache[key] = ({str(row.get("id")): row for row in rows}, manifest)
        row_map, manifest = self._dataset_cache[key]
        row = row_map.get(str(job["item_id"]))
        if row is None or row.get("task") != job["task"]:
            raise ValueError("item_id из задания отсутствует в экспортированном split/task")
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("В экспортированном item отсутствует полный prompt")
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        if row.get("prompt_sha256") != prompt_hash:
            raise ValueError("SHA-256 prompt не совпадает с export")
        if job["dataset_id"].startswith("MERA") and not row.get("reference"):
            raise ValueError("В MERA item отсутствует reference")
        if job["dataset_id"].startswith("LIBRA") and not (row.get("references") or row.get("positive_outputs")):
            raise ValueError("В LIBRA item отсутствуют multiple acceptable references")
        profiles = row.get("token_counts_by_tokenizer")
        profile_name = f"{job['model_id']}@{job['tokenizer_revision']}"
        profile = profiles.get(profile_name) if isinstance(profiles, dict) else None
        if not isinstance(profile, dict):
            raise ValueError("В export отсутствует закреплённый tokenizer profile для модели и ревизии")
        if profile.get("chat_template_sha256") != job["chat_template_hash"]:
            raise ValueError("Chat template hash задания не совпадает с tokenizer profile export")
        if row.get("prompt_token_count") != profile.get("prompt_tokens"):
            raise ValueError("Prompt token count не совпадает с закреплённым tokenizer profile")
        if manifest.get("dataset_revision") != job["dataset_revision"]:
            raise ValueError("Dataset revision в manifest не совпадает с заданием")
        if manifest.get("split") != job["split"] or manifest.get("task") != job["task"]:
            raise ValueError("Dataset task/split в manifest не совпадает с заданием")
        if manifest.get("dataset") != job["dataset_id"]:
            raise ValueError("Dataset ID в manifest не совпадает с заданием")
        return row, manifest

    def _load_model(self, job: dict[str, Any]) -> dict[str, Any]:
        key = (job["model_id"], job["model_revision"], job["gguf_sha256"])
        if key not in self._model_cache:
            self._model_cache[key] = _find_model(self.config, job, verify_artifact=not self.fake)
        return self._model_cache[key]

    def run_fake(self, *, start_block: str | None = None, max_runtime_s: float | None = None) -> dict[str, Any]:
        if not self.fake:
            raise PermissionError("Fake execution требует изолированного fake режима")
        self.validate_inputs()
        self._claim_run_mode("synthetic_fake")
        try:
            return run_queue(self.jobs, run_dir=self.run_dir, platform_id=self.platform_id,
                             owner_authorized=True, max_attempts=1, start_block=start_block,
                             max_runtime_s=max_runtime_s, preflight=lambda: self.platform_adapter.preflight(self.config),
                             execute=lambda job: self._execute_job(job, synthetic=True), safety_check=lambda _job: None,
                             retryable_error=lambda _exc: False,
                             stop_requested=lambda: (self.run_dir / "STOP").exists())
        finally:
            self._close_runtime_session()

    def _execute_job(self, job: dict[str, Any], *, synthetic: bool) -> dict[str, Any]:
        row, source_manifest = self._load_dataset(job)
        model = self._load_model(job)
        runtime_factory = self.runtime_factory
        if runtime_factory is None:
            if synthetic:
                from .fake_runtime import SyntheticRuntime
                runtime_factory = lambda _job, _model: SyntheticRuntime()
            else:
                from .runtime import LlamaServer
                job_config = dict(self.config)
                job_config["contexts"] = [job["context_length"]]
                runtime_factory = lambda _job, _model: LlamaServer(
                    job_config, _model, log_path=self.run_dir / "runtime.log", platform_adapter=self.platform_adapter)
        session_key = digest({key: job.get(key) for key in (
            "block_id", "platform_id", "model_id", "model_revision", "gguf_sha256", "quantization",
            "kv_cache", "context_length", "runtime_version", "runtime_binary_sha256", "llama_cpp_commit",
            "generation_config", "tokenizer_revision", "chat_template_hash", "reasoning_mode")})
        if self._runtime_session is not None and self._runtime_session["key"] != session_key:
            self._close_runtime_session()
        if self._runtime_session is None:
            runtime = runtime_factory(job, model)
            runtime_started_at = time.time()
            try:
                runtime.start()
            except BaseException:
                runtime.stop()
                raise
            self._runtime_session = {"key": session_key, "block_id": job["block_id"],
                                     "runtime": runtime, "warmup_cells_done": set(),
                                     "started_at": runtime_started_at, "job": dict(job)}
        runtime = self._runtime_session["runtime"]
        measurements: dict[str, Any] = {}
        raw = ""
        try:
            warmup_count = int(self.config.get("repeats", {}).get("warmup", 0))
            generation = dict(job["generation_config"])
            generation.setdefault("max_new_tokens", self.config.get("generation", {}).get("max_new_tokens"))
            runtime_verification = {"status": "synthetic", "static": {}, "generation": {},
                                    "actual_static": {}, "actual_generation": {}}
            if not synthetic:
                runtime_verification = runtime.verify_parameters(
                    {"model_path": model["gguf_path"], "context_length": job["context_length"],
                     "gpu_layers": self.config["runtime"]["gpu_layers"],
                     "cpu_threads": self.config["runtime"]["cpu_threads"],
                     "cpu_threads_batch": self.config["runtime"]["cpu_threads_batch"],
                     "batch_size": self.config["runtime"]["batch_size"],
                     "ubatch_size": self.config["runtime"]["ubatch_size"],
                     "kv_cache": model["kv_cache"], "flash_attention": self.config["runtime"]["flash_attn"],
                     "reasoning_mode": self.config["runtime"]["reasoning_mode"],
                     "chat_template_source": self.config["runtime"]["chat_template_source"]}, generation)
                if runtime_verification.get("status") != "verified":
                    raise ExecutionBlocked("Runtime не подтвердил фактически принятые обязательные параметры")
            is_performance = job.get("mode") == "performance" or job.get("workload_id") == "performance"
            warmup_cell = (("performance", job.get("workload_id"), job.get("task"), job.get("context_length"))
                           if is_performance else ("runtime-start",))
            warmup_executed = 0
            if warmup_cell not in self._runtime_session["warmup_cells_done"]:
                for _ in range(warmup_count):
                    runtime.generate(row["prompt"], generation)
                    warmup_executed += 1
                self._runtime_session["warmup_cells_done"].add(warmup_cell)
            before = self._resource_snapshot(runtime)
            response = runtime.generate(row["prompt"], generation)
            after = self._resource_snapshot(runtime)
            raw = response.get("text")
            if not isinstance(raw, str):
                raise ValueError("Runtime вернул ответ без текстового поля")
            measurements = _runtime_metrics(response, row)
            measurements["resource_metrics"] = {"before": before, "after": after}
            measurements["gpu_offload"] = {"requested_layers": self.config["runtime"].get("gpu_layers"),
                                            "actual_layers": runtime_verification["actual_static"].get("gpu_layers"),
                                            "status": "synthetic" if synthetic else runtime_verification["static"].get("gpu_layers", "unverified")}
            generated, decode_s = measurements.get("generated_tokens"), measurements.get("decode_duration_s")
            measurements["throughput_tokens_per_s"] = (
                generated / decode_s if type(generated) is int and isinstance(decode_s, (int, float)) and decode_s > 0
                else None)
        except BaseException as exc:
            watchdog = getattr(runtime, "safety_watchdog", None)
            if watchdog is not None and watchdog.stop_event.is_set():
                raise SafetyStop(f"Safety Watchdog остановил runtime: {watchdog.last_error or watchdog.last_state.value}") from exc
            self._close_runtime_session()
            raise
        prediction, score = self._score_item(job, row, raw)
        finish_reason = measurements.get("finish_reason")
        truncation = finish_reason == "length"
        if truncation:
            score = {**score, "scored_metrics_before_truncation_rejection": {
                         key: value for key, value in score.items()
                         if isinstance(value, (int, float)) and not isinstance(value, bool)},
                     "truncation_rejected": True, "failure_reason": "generation_length_limit"}
            score.update({key: 0.0 for key, value in score.items()
                          if isinstance(value, (int, float)) and not isinstance(value, bool)})
        return {
            "run_kind": "synthetic_fake" if synthetic else "production",
            "scientific_observation": not synthetic,
            "dataset_source_sha256": source_manifest["source_sha256"],
            "dataset_source_original_sha256": source_manifest.get("source_original_sha256"),
            "dataset_source_transformed_sha256": source_manifest.get("source_transformed_sha256"),
            "dataset_source_provenance": source_manifest.get("source_provenance"),
            "dataset_export_sha256": job["dataset_hash"], "item_id": str(row["id"]),
            "task": job["task"], "split": job["split"],
            "prompt": row.get("prompt"), "prompt_sha256": row.get("prompt_sha256"),
            "prompt_template_id": row.get("prompt_template_id"),
            "prompt_template_sha256": row.get("prompt_template_sha256"),
            "tokenizer_profile": row.get("token_counts_by_tokenizer", {}).get(
                f"{job['model_id']}@{job['tokenizer_revision']}"),
            "warmup_count": warmup_executed,
            "model_artifact_sha256_checked": not synthetic,
            "runtime_parameter_verification": runtime_verification,
            **_reference_fields(row),
            "raw_model_output": raw, "prediction": prediction, "score": score,
            "finish_reason": finish_reason, "truncated": truncation, "metrics": measurements,
        }

    def _score_item(self, job: dict[str, Any], row: dict[str, Any], raw: str) -> tuple[str, dict[str, Any]]:
        from .dataset_pipeline import LibraScorer
        if job["dataset_id"] == "LIBRA" or job["dataset_id"].startswith("LIBRA/"):
            if self._libra_scorer is None:
                self._libra_scorer = LibraScorer()
            refs = row.get("references") or row.get("positive_outputs")
            if not isinstance(refs, list) or not refs:
                raise ValueError("В LIBRA item отсутствуют multiple acceptable references")
            from .dataset_pipeline import SCORER_VERSION, LIBRA_TASKS
            if job["scorer_version"] != SCORER_VERSION or job["task"] not in LIBRA_TASKS:
                raise ValueError("LIBRA scorer/task не совпадает с закреплённой версией")
            return raw, {"scorer": self._libra_scorer.version,
                         "exact_match": self._libra_scorer.score_item(raw, refs)}
        return _score(job, row, raw)

    def _close_runtime_session(self) -> None:
        session, self._runtime_session = self._runtime_session, None
        if session is None:
            return
        runtime = session["runtime"]
        watchdog = getattr(runtime, "safety_watchdog", None)
        summary = watchdog.summary() if watchdog is not None and hasattr(watchdog, "summary") else None
        try:
            runtime.stop()
        finally:
            if summary is not None:
                from .store import append_record
                append_record(self.run_dir / "block_metrics.jsonl", {
                    "record_type": "block_resource_summary", "block_id": session["block_id"],
                    "started_at_epoch": session["started_at"], "ended_at_epoch": time.time(),
                    **{key: session["job"].get(key) for key in (
                        "platform_id", "hardware_manifest_hash", "model_id", "model_revision", "gguf_sha256",
                        "quantization", "kv_cache", "context_length", "runtime_version",
                        "runtime_binary_sha256", "llama_cpp_commit", "protocol_version", "protocol_hash")},
                    **summary})

    def _resource_snapshot(self, runtime: Any) -> dict[str, Any]:
        pid = runtime.process.pid if runtime.process is not None and runtime.process.poll() is None else None
        metrics = self.platform_adapter.resource_snapshot(pid)
        return {name: {"value": metric.value, "unit": metric.unit,
                       "status": metric.status.value, "reason": metric.reason}
                for name, metric in metrics.items()}

    def run_production(self, *, authorization_file: str | Path | None = None,
                       start_block: str | None = None, max_runtime_s: float | None = None) -> dict[str, Any]:
        verify_owner_authorization(authorization_file, plan_hash=self.plan_hash, platform_id=self.platform_id)
        self._production_preflight()
        if not REAL_EXECUTION_ENABLED:
            raise ExecutionBlocked("Реальный инференс заблокирован до этапа M4 и отдельного аудита допуска")
        configured_output_root = self.config.get("output", {}).get("root")
        output_root = Path(configured_output_root).resolve() if configured_output_root else None
        if output_root is None or output_root not in self.run_dir.parents:
            raise ExecutionBlocked("Production run directory должен находиться в output.root конфигурации")
        self._claim_run_mode("production")
        (self.run_dir / "STOP").unlink(missing_ok=True)
        attempt_limits = {job.get("attempt_policy", 2) for job in self.jobs}
        if len(attempt_limits) != 1:
            raise ValueError("Очередь должна задавать единый ограниченный лимит повторов")
        try:
            return run_queue(self.jobs, run_dir=self.run_dir, platform_id=self.platform_id,
                             owner_authorized=True, max_attempts=attempt_limits.pop(), start_block=start_block,
                             max_runtime_s=max_runtime_s,
                             preflight=lambda: self._production_preflight(),
                             execute=lambda job: self._execute_job(job, synthetic=False),
                             safety_check=lambda job: self._safety_precheck(job),
                             retryable_error=lambda exc: isinstance(exc, (TimeoutError, ConnectionError, OSError)),
                             stop_requested=lambda: (self.run_dir / "STOP").exists())
        finally:
            self._close_runtime_session()

    def _production_preflight(self) -> None:
        errors = self.platform_adapter.preflight(self.config)
        if errors:
            raise ExecutionBlocked("Platform preflight завершился отказом: " + "; ".join(errors))
        self.validate_inputs()

    def _safety_precheck(self, job: dict[str, Any]) -> None:
        if job["platform_id"] != self.platform_id:
            raise SafetyStop("Safety check получил задание для другой platform_id")
        if self._runtime_session is not None:
            watchdog = getattr(self._runtime_session["runtime"], "safety_watchdog", None)
            if watchdog is not None:
                try:
                    watchdog.assert_safe()
                except RuntimeError as exc:
                    raise SafetyStop(str(exc)) from exc

