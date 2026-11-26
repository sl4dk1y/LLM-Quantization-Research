"""Deterministic, fail-closed dataset preparation and task scorer adapters."""
from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

from .datasets import file_sha256
from .sampling import select_items, validate_disjoint

MERA_REVISION = "15c0e5b8e27551b50aa27f5497074810c8740661"
LIBRA_REVISION = "10d668ab44165bbab73ce806aa536294abedc1ba"
SEED = 20261124
SCORER_VERSION = "quantbench-official-adapters-mera-libra-1.0"
MERA_EVALUATOR_REVISION = "8698451dd7c281462c684190c9e38d2fae0f6d7a"
MERA_HARNESS_REVISION = "d85f735223f11699765e2f5490ee526a3fe1b1e5"
LIBRA_EVALUATOR_REVISION = "d6de81643928d653435c431bae19945d41d32520"
MERA_TASKS = ("MathLogicQA", "MultiQ", "ruOpenBookQA", "PARus")
LIBRA_TASKS = ("ru_babilong_qa3", "long_context_multiq", "matreshka_names")
LIBRA_LENGTHS = ("4K", "8K", "16K")
LIBRA_BABILONG_QA3_TEMPLATE = (
    "Я даю тебе контекст с фактами о местоположении и действиях разных людей. Тебе нужно ответить на вопрос, "
    "основываясь только на фактической информации. Если человек взял предмет в одном месте и отправился в другое "
    "место, этот предмет также находится во втором месте. Если человек оставил предмет в первом метсе и перешел "
    "во второе место, предмет остается в первом месте.\n{{context}}\nОтветь на вопрос как можно более кратко.\n"
    "Вопрос: {{input}}\nОтвет:"
)
LIBRA_TEMPLATES = {
    "ru_babilong_qa3": LIBRA_BABILONG_QA3_TEMPLATE,
    "long_context_multiq": (
        "Тебе предоставляется длинный текст, в котором нужно найти ответ на вопрос.\n{{context}}\n"
        "Найди ответ в тексте на следующий вопрос.\nВопрос:{{input}}\nОтвет:"
    ),
    "matreshka_names": (
        "Тебе предоставляются несколько диалогов. Запомни имена людей и темы, на которые они говорили.\n"
        "{{context}}\nВ ответе укажи только имя собеседника, который говорил на тему из следующего вопроса.\n"
        "Вопрос:{{input}}\nОтвет:"
    ),
}
MERA_SOURCE_PATHS = {
    "MathLogicQA": {"train": "data/mathlogicqa/train.jsonl", "test": "data/mathlogicqa/test.jsonl"},
    "MultiQ": {"train": "data/multiq/train.jsonl", "test": "data/multiq/test.jsonl"},
    "ruOpenBookQA": {"train": "data/ruopenbookqa/train.jsonl", "test": "data/ruopenbookqa/test.jsonl"},
    "PARus": {"train": "data/parus/train.jsonl", "dev": "data/parus/dev.jsonl", "test": "data/parus/test.jsonl"},
}
LIBRA_SOURCE_PATHS = {task: f"{task}/test-00000-of-00001.parquet" for task in LIBRA_TASKS}
MERA_SCORER_SOURCE_FILES = {
    "MathLogicQA": ["benchmark_tasks/custom_generate_task.yaml", "benchmark_tasks/letter_choice_gen_task.yaml",
                    "benchmark_tasks/mathlogicqa/mathlogicqa_gen.yaml", "benchmark_tasks/mathlogicqa/mathlogicqa.yaml"],
    "MultiQ": ["benchmark_tasks/custom_generate_task.yaml", "benchmark_tasks/tape/multiq.yaml",
               "benchmark_tasks/tape/utils.py"],
    "ruOpenBookQA": ["benchmark_tasks/custom_generate_task.yaml", "benchmark_tasks/letter_choice_gen_task.yaml",
                     "benchmark_tasks/tape/ruopenbookqa_gen.yaml", "benchmark_tasks/tape/ruopenbookqa.yaml",
                     "benchmark_tasks/custom_metrics.py"],
    "PARus": ["benchmark_tasks/custom_generate_task.yaml", "benchmark_tasks/digit_choice_gen_task.yaml",
              "benchmark_tasks/rsg/parus_gen.yaml", "benchmark_tasks/rsg/parus.yaml"],
}
LIBRA_SCORER_SOURCE_FILES = {task: [f"lm_eval/tasks/libra/{task}.yaml", "lm_eval/tasks/libra/utils.py"]
                             for task in LIBRA_TASKS}
PINNED_SOURCE_SHA256 = {
    ("MERA-evaluation/MERA", "data/mathlogicqa/test.jsonl"): "a4d8f86bc315d694e27b4ee6f15803830c265effa8c3bc16211cecd5ba638c94",
    ("MERA-evaluation/MERA", "data/mathlogicqa/train.jsonl"): "4c759341e99e3624f5b9ea8b991e7dfedf8b74d0039be028ead8979b2c180c83",
    ("MERA-evaluation/MERA", "data/multiq/test.jsonl"): "7df2fa31fe677aec5ff8cb8e0ba804e7d3b3d3a0f830c7147ae111410265f72d",
    ("MERA-evaluation/MERA", "data/multiq/train.jsonl"): "7e7b82b589fcc4c96535ee7bb36a211c8299efe2092a16fadc799e04f2536755",
    ("MERA-evaluation/MERA", "data/parus/dev.jsonl"): "eca486514f2adee015553c0f94f2ec1aace44ed8d4e607a18d08056d6d1fbe3e",
    ("MERA-evaluation/MERA", "data/parus/test.jsonl"): "ca07d97a6b6d3bf7b7e2397de88d701563e99b94bbf50a7753321df092cb1a47",
    ("MERA-evaluation/MERA", "data/parus/train.jsonl"): "487ddc4fb7bb5c124dcd8754bf14bf1f7ebf08882411bca1456b7ab81b742a9e",
    ("MERA-evaluation/MERA", "data/ruopenbookqa/test.jsonl"): "742c2a825830c6cd101f5ea609a0cba03d1336d2091aa6b0446c0662259aaebe",
    ("MERA-evaluation/MERA", "data/ruopenbookqa/train.jsonl"): "dbbcc7227ad7f368b58fd8a802459be8d6d100bf524f9ee5494684574992bb70",
    ("ai-forever/LIBRA", "long_context_multiq/test-00000-of-00001.parquet"): "4631e22cad4884f7e3a1ce864c788c3cf38802a9647787a5f9dc0d42edd18f59",
    ("ai-forever/LIBRA", "matreshka_names/test-00000-of-00001.parquet"): "76d2431b48abdae3b025846516e999e3edcfb628052a4bc5f6aef5cabf4df15d",
    ("ai-forever/LIBRA", "ru_babilong_qa3/test-00000-of-00001.parquet"): "4b94bec718b3ebba3189121cc55bbc5c836793917aff3e7b7da6ef75c15667fa",
}

class OfficialScorerUnavailable(RuntimeError):
    """The task's pinned official evaluator cannot be executed in this environment."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _item_id(row: dict[str, Any], *, dataset: str, row_index: int, task: str, split: str) -> str:
    meta = row.get("meta", {})
    value = meta.get("id") if dataset == "MERA" and isinstance(meta, dict) else row.get("id")
    if value is None and dataset == "LIBRA":
        canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        row_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        value = f"libra:{task}:{split}:{row_index:08d}:{row_hash}"
    _require(value is not None and str(value).strip() != "", "В исходной записи отсутствует уникальный item ID")
    return str(value)


def _verify_source_provenance(*, source_path: Path, source_hash: str, dataset: str,
                              revision: str, task: str, source_split: str,
                              provenance: dict[str, Any] | None, allow_test_fixture: bool) -> dict[str, Any]:
    if allow_test_fixture and provenance and provenance.get("fixture") is True:
        return {**provenance, "verified": False, "verification_scope": "synthetic unit fixture"}
    _require(isinstance(provenance, dict), "Для экспорта требуется source provenance manifest")
    repo_id = "MERA-evaluation/MERA" if dataset == "MERA" else "ai-forever/LIBRA"
    expected_path = (MERA_SOURCE_PATHS.get(task, {}).get(source_split) if dataset == "MERA"
                     else LIBRA_SOURCE_PATHS.get(task))
    _require(expected_path is not None, "Upstream path не зарегистрирован для задачи/split")
    _require(provenance.get("repo_id") == repo_id, "Source provenance repository ID не совпадает")
    _require(provenance.get("revision") == revision, "Source provenance revision не совпадает")
    _require(provenance.get("path") == expected_path, "Source provenance upstream path не совпадает")
    local_record_hash = provenance.get("transformed_source_sha256") or provenance.get("sha256")
    _require(local_record_hash == source_hash, "SHA-256 преобразованного локального источника не совпадает с provenance")
    original_hash = provenance.get("original_sha256") or provenance.get("sha256")
    _require(isinstance(original_hash, str) and len(original_hash) == 64,
             "В provenance отсутствует SHA-256 оригинального источника")
    pinned_hash = PINNED_SOURCE_SHA256.get((repo_id, expected_path))
    _require(pinned_hash is not None and original_hash == pinned_hash,
             "Оригинальный SHA-256 не совпадает с проверенным файлом на закреплённой ревизии")
    if dataset == "MERA":
        _require(source_hash == pinned_hash, "MERA source должен быть исходным pinned JSONL без преобразования")
    elif source_hash != original_hash:
        _require(provenance.get("transformation_version") == "libra-parquet-jsonl-row-id-v1",
                 "LIBRA source conversion version не зарегистрирована")
        raw_path = Path(str(provenance.get("original_local_path", "")))
        _require(raw_path.is_file() and file_sha256(raw_path) == original_hash,
                 "LIBRA original parquet отсутствует либо SHA-256 не совпадает")
        try:
            import pyarrow.parquet as parquet
        except ImportError as exc:
            raise OfficialScorerUnavailable("Проверка LIBRA conversion требует pyarrow") from exc
        transformed = hashlib.sha256()
        row_index = 0
        for batch in parquet.ParquetFile(raw_path).iter_batches(batch_size=8):
            for raw in batch.to_pylist():
                canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                row_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                row = {"id": f"libra:{task}:test:{row_index:08d}:{row_hash}",
                       "source_row_index": row_index, **raw}
                line = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
                transformed.update(line)
                row_index += 1
        _require(transformed.hexdigest() == source_hash,
                 "LIBRA transformed JSONL не воспроизводится из оригинального pinned parquet")
    _require(Path(str(provenance.get("local_path", source_path))).resolve() == source_path.resolve(),
             "Путь файла в provenance не совпадает с source")
    return {**provenance, "verified": True}


def _render_mera(row: dict[str, Any], task: str) -> tuple[str, Any, str]:
    _require(isinstance(row.get("instruction"), str), f"{task}: отсутствует строковое instruction")
    inputs = row.get("inputs")
    _require(isinstance(inputs, dict), f"{task}: отсутствует объект inputs")
    required = {
        "MathLogicQA": {"text", "option_a", "option_b", "option_c", "option_d"},
        "MultiQ": {"text", "support_text", "question"},
        "PARus": {"premise", "choice1", "choice2"},
        "ruOpenBookQA": {"question", "option_a", "option_b", "option_c", "option_d"},
    }.get(task, set())
    _require(required.issubset(inputs), f"{task}: отсутствуют обязательные inputs: {sorted(required - set(inputs))}")
    try:
        prompt = row["instruction"].format_map(inputs)
        if task == "PARus":
            prompt += "\nОтвет:"
        else:
            prompt = prompt.strip()
    except (KeyError, ValueError, IndexError) as exc:
        raise ValueError(f"{task}: не удалось отрисовать исходный шаблон instruction: {exc}") from exc
    label = row.get("outputs")
    _require(isinstance(label, str) and bool(label.strip()), f"{task}: отсутствует обязательный outputs label")
    return prompt, label, row["instruction"]


def _render_libra(row: dict[str, Any], task: str, expected_length: str) -> tuple[str, list[str], str]:
    _require(isinstance(row.get("context"), str) and isinstance(row.get("input"), str),
             f"{task}: ожидаются строковые context и input")
    answers = row.get("positive_outputs")
    _require(isinstance(answers, list) and answers and all(isinstance(x, str) and x.strip() for x in answers),
             f"{task}: positive_outputs должен быть непустым списком строк")
    raw_length = str(row.get("length", "")).strip().lower().replace(" ", "")
    wanted = expected_length.lower()
    _require(raw_length in {wanted, wanted.replace("k", "k"), str(int(wanted[:-1]) * 1024)},
             f"{task}: длина {row.get('length')!r} не соответствует ячейке {expected_length}")
    template = LIBRA_TEMPLATES[task]
    prompt = template.replace("{{context}}", row["context"]).replace("{{input}}", row["input"])
    return prompt, list(answers), template


def prepare_export(*, source_path: str | Path, output_path: str | Path, manifest_path: str | Path,
                   dataset: str, task: str, split: str, revision: str, count: int,
                   seed: int = SEED, expected_source_sha256: str | None = None,
                   context_length: str | None = None, disjoint_ids: Iterable[str] | None = None,
                   tokenizer: Any | None = None, tokenizers: dict[str, Any] | None = None,
                   source_provenance: dict[str, Any] | None = None, source_split: str | None = None,
                   exclude_ids: Iterable[str] = (), fewshot_example: dict[str, Any] | None = None,
                   allow_test_fixture: bool = False, cohort: str | None = None,
                   tokenizer_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Prepare an immutable sample after checking its pinned upstream provenance."""
    expected_revision = MERA_REVISION if dataset == "MERA" else LIBRA_REVISION if dataset == "LIBRA" else None
    _require(expected_revision is not None, "dataset должен быть MERA или LIBRA")
    _require(revision == expected_revision, f"Неподдерживаемая revision {revision}; ожидается M2 pin {expected_revision}")
    _require(task in (MERA_TASKS if dataset == "MERA" else LIBRA_TASKS), "Задача не зарегистрирована в M2")
    source_split = source_split or split
    _require(split in ({"train", "dev", "test", "public_test", "local_development", "local_heldout", "p0"}
                       if dataset == "MERA" else {"test"}), "Недопустимый logical split")
    _require(source_split in ({"train", "dev", "test", "public_test"} if dataset == "MERA" else {"test"}),
             "Недопустимый upstream source split")
    if dataset == "MERA" and source_split in {"test", "public_test"}:
        _require(disjoint_ids is not None,
                 "Для held-out MERA split передайте --disjoint-source с ID всех dev/train элементов этой задачи")
    if dataset == "LIBRA":
        _require(context_length in LIBRA_LENGTHS, "Для LIBRA укажите длину 4K, 8K или 16K")
    source_path = Path(source_path)
    source_hash = file_sha256(source_path)
    verified_provenance = _verify_source_provenance(
        source_path=source_path, source_hash=source_hash, dataset=dataset, revision=revision,
        task=task, source_split=source_split, provenance=source_provenance,
        allow_test_fixture=allow_test_fixture)
    if expected_source_sha256:
        _require(source_hash.lower() == expected_source_sha256.lower(), "SHA-256 исходного split не совпадает")
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    with source_path.open(encoding="utf-8") as stream:
        for line, raw in enumerate(stream, 1):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Некорректный JSONL, строка {line}") from exc
            _require(isinstance(row, dict), f"Строка {line}: ожидался объект")
            item_id = _item_id(row, dataset=dataset, row_index=line - 1, task=task, split=split)
            if row.get("split") is not None:
                _require(row["split"] == source_split, f"item {item_id}: split источника не совпадает с provenance")
            _require(item_id not in seen, f"Повторяющийся item ID: {item_id}")
            seen.add(item_id)
            if dataset == "LIBRA":
                actual_length = str(row.get("length", "")).strip().lower()
                if actual_length != (context_length or "").lower():
                    continue
            try:
                if dataset == "MERA":
                    prompt, references, prompt_template = _render_mera(row, task)
                    label = row["outputs"]
                    template_id = f"mera:{task}:instruction.format_map(inputs):v1"
                    context_class = None
                else:
                    prompt, references, prompt_template = _render_libra(row, task, context_length or "")
                    label = row["positive_outputs"]
                    template_id = f"libra:{task}:official-doc_to_text:source-template:{LIBRA_EVALUATOR_REVISION}"
                    context_class = context_length
            except ValueError as exc:
                raise ValueError(f"item {item_id}: {exc}") from exc
            shot_id = None
            shot_template_sha256 = None
            if dataset == "MERA" and fewshot_example is not None:
                shot_row = fewshot_example.get("row")
                shot_id = str(fewshot_example.get("id", ""))
                _require(isinstance(shot_row, dict) and shot_id, "Некорректная фиксированная few-shot запись")
                shot_query, shot_label, _ = _render_mera(shot_row, task)
                prompt = shot_query + " " + shot_label + "\n\n" + prompt
                shot_template_sha256 = hashlib.sha256(
                    json.dumps({"id": shot_id, "prompt": shot_query, "label": shot_label},
                               ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            records.append({"id": item_id, "task": task, "dataset": dataset, "split": split,
                            "prompt": prompt, "reference": references[0] if isinstance(references, list) else references,
                            "references": references if isinstance(references, list) else [references],
                            "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                            "prompt_template_sha256": hashlib.sha256(
                                json.dumps({"template": prompt_template, "fewshot_id": shot_id,
                                            "fewshot_sha256": shot_template_sha256},
                                           ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
                            "prompt_token_count": None, "context_token_count": None,
                            "token_counts_by_tokenizer": {},
                            "source_label": label, "source_length": row.get("length"),
                            "context_length_class": context_class, "prompt_template_id": template_id,
                            "fewshot_example_id": shot_id, "fewshot_example_sha256": shot_template_sha256,
                            "scorer_version": SCORER_VERSION, "source_record": row})
    excluded = {str(value) for value in exclude_ids}
    candidates = [row for row in records if row["id"] not in excluded]
    selected = select_items(candidates, seed=seed, revision=revision, task=task, split=split, count=count)
    if split in {"test", "public_test", "local_heldout"} and disjoint_ids is not None:
        validate_disjoint([row["id"] for row in selected], disjoint_ids)
    if fewshot_example is not None:
        validate_disjoint([row["id"] for row in selected], [str(fewshot_example["id"])])
    if dataset == "LIBRA":
        position = {row["id"]: index for index, row in enumerate(records)}
        selected = sorted(selected, key=lambda row: position[row["id"]])
    token_profiles = tokenizers or ({"default": tokenizer} if tokenizer is not None else {})
    for item in selected:
        token_counts: dict[str, Any] = {}
        for profile, selected_tokenizer in token_profiles.items():
            encode = getattr(selected_tokenizer, "encode", None)
            _require(callable(encode), "tokenizer должен предоставлять encode")
            profile_info = {
                "prompt_tokens": len(selected_tokenizer.apply_chat_template(
                    [{"role": "user", "content": item["prompt"]}], tokenize=True,
                    add_generation_prompt=True)),
                "chat_template_sha256": hashlib.sha256(
                    str(getattr(selected_tokenizer, "chat_template", "")).encode("utf-8")).hexdigest(),
            }
            if dataset == "LIBRA":
                profile_info["context_tokens"] = len(encode(item["source_record"]["context"],
                                                              add_special_tokens=False))
            token_counts[profile] = profile_info
        item["token_counts_by_tokenizer"] = token_counts
        first = next(iter(token_counts.values()), {})
        item["prompt_token_count"] = first.get("prompt_tokens")
        item["context_token_count"] = first.get("context_tokens")
    output_path, manifest_path = Path(output_path), Path(manifest_path)
    resolved = {path.resolve() for path in (source_path, output_path, manifest_path)}
    _require(len(resolved) == 3, "Пути source, output и manifest должны быть разными")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in selected)
    prepared_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    ids = [row["id"] for row in selected]
    ids_hash = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    manifest = {"manifest_version": 2, "dataset": dataset, "dataset_revision": revision,
                "cohort": cohort,
                "task": task, "split": split, "source_split": source_split, "seed": seed,
                "sample_count": len(selected), "requested_count": count,
                "source_path": str(source_path), "source_sha256": source_hash,
                "source_provenance": verified_provenance,
                "source_original_sha256": verified_provenance.get("original_sha256") or source_hash,
                "source_transformed_sha256": source_hash,
                "prepared_path": str(output_path), "prepared_sha256": prepared_hash,
                "item_ids_sha256": ids_hash, "item_ids": ids, "scorer_version": SCORER_VERSION,
                "scorer_revision": (MERA_EVALUATOR_REVISION if dataset == "MERA" else LIBRA_EVALUATOR_REVISION),
                "scorer_source_files": (MERA_SCORER_SOURCE_FILES[task] if dataset == "MERA"
                                         else LIBRA_SCORER_SOURCE_FILES[task]),
                "generation_profile": {
                    "official_task_config": ("generate_until; do_sample=false; stop at newline"
                                              if dataset == "MERA" else
                                              "do_sample=false; temperature=0; official task cap is recorded in source YAML"),
                    "protocol_max_new_tokens": (128 if task == "MultiQ" else 64),
                    "official_libra_max_gen_toks": ({"ru_babilong_qa3": 16, "long_context_multiq": 32,
                                                       "matreshka_names": 16}.get(task) if dataset == "LIBRA" else None),
                    "official_metrics": ({"MathLogicQA": ["exact_match"], "MultiQ": ["em", "f1"],
                                          "ruOpenBookQA": ["exact_match", "f1"], "PARus": ["exact_match"]}.get(task)
                                         if dataset == "MERA" else ["libra_score"]),
                },
                "fewshot_examples": ([{"id": fewshot_example["id"],
                                        "sha256": hashlib.sha256(json.dumps(fewshot_example["row"], ensure_ascii=False,
                                                                            sort_keys=True).encode()).hexdigest()}]
                                      if fewshot_example is not None else []),
                "prompt_template_ids": sorted({row["prompt_template_id"] for row in selected}),
                "prompt_template_sha256s": sorted({row["prompt_template_sha256"] for row in selected}),
                "context_length_class": context_length,
                "tokenizer_profiles": sorted(token_profiles),
                "tokenizer_metadata": tokenizer_metadata or {},
                "token_count_status": "measured separately for each listed tokenizer and chat template" if token_profiles else "not measured; tokenizer-specific counts remain required",
                "licence": "MIT; see pinned upstream dataset card; source lineage remains dataset-specific",
                "official_scorer_status": "implemented from pinned MERA generation evaluator" if dataset == "MERA" else "implemented from pinned LIBRA task templates and EM scorer"}
    manifest_payload = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    for path, contents in ((output_path, payload), (manifest_path, manifest_payload)):
        if path.exists() and path.read_text(encoding="utf-8") != contents:
            raise FileExistsError(f"Отказ от перезаписи существующего экспорта: {path}")
    if not output_path.exists():
        output_path.write_text(payload, encoding="utf-8")
    if not manifest_path.exists():
        manifest_path.write_text(manifest_payload, encoding="utf-8")
    return manifest


def export_split_ids(*, source_path: str | Path, output_path: str | Path, dataset: str,
                     task: str, split: str, revision: str) -> dict[str, Any]:
    """Create an ID-only split manifest for overlap checks without exposing labels in it."""
    expected = MERA_REVISION if dataset == "MERA" else LIBRA_REVISION if dataset == "LIBRA" else None
    _require(expected is not None and revision == expected, "dataset revision не совпадает с pin M2")
    _require(task in (MERA_TASKS if dataset == "MERA" else LIBRA_TASKS), "Задача не зарегистрирована в M2")
    _require(split in ({"train", "dev", "test", "public_test"} if dataset == "MERA" else {"test"}),
             "Недопустимый split для данного источника")
    path = Path(source_path)
    ids = []
    seen = set()
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            _require(isinstance(row, dict), f"Строка {index + 1}: ожидался JSON object")
            if row.get("split") is not None:
                _require(row["split"] == split, f"Строка {index + 1}: несовпадающий split")
            item_id = _item_id(row, dataset=dataset, row_index=index, task=task, split=split)
            _require(item_id not in seen, f"Повторяющийся item ID: {item_id}")
            seen.add(item_id)
            ids.append(item_id)
    _require(bool(ids), "Исходный split пуст")
    manifest = {"manifest_version": 1, "dataset": dataset, "task": task, "split": split,
                "dataset_revision": revision, "source_path": str(path), "source_sha256": file_sha256(path),
                "item_ids": ids, "item_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if target.exists() and target.read_text(encoding="utf-8") != contents:
        raise FileExistsError(f"Отказ от перезаписи существующего ID manifest: {target}")
    if not target.exists():
        target.write_text(contents, encoding="utf-8")
    return manifest


class LibraScorer:
    """LIBRA upstream EM procedure, including Russian lemmatization and multiple references."""
    version = "LIBRA lm-evaluation-harness utils.py: process_results/aggregate_results_em; pymorphy3 required"

    def __init__(self, analyzer: Any | None = None):
        if analyzer is None:
            try:
                import pymorphy3
            except ImportError as exc:
                raise OfficialScorerUnavailable("LIBRA official scorer requires pymorphy3; install it in the experiment environment") from exc
            analyzer = pymorphy3.MorphAnalyzer()
        self.analyzer = analyzer

    def normalize(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("LIBRA prediction/reference must be strings")
        words = []
        for word in text.split():
            token = re.sub(r"[^a-zа-яй0-9_]+", "", word.lower())
            words.append(self.analyzer.parse(token)[0].normal_form.lower())
        return " ".join(words)

    def score_item(self, prediction: str, references: list[str]) -> float:
        if not isinstance(prediction, str) or not references or not all(isinstance(x, str) and x for x in references):
            raise ValueError("LIBRA requires prediction and non-empty positive_outputs")
        pred = self.normalize(prediction)
        for reference in set(self.normalize(x) for x in references):
            if reference in pred:
                return 1.0
        return 0.0

    def aggregate(self, observations: Iterable[dict[str, Any]]) -> dict[str, float]:
        values: dict[str, list[float]] = defaultdict(list)
        for row in observations:
            length = row.get("context_length_class") or row.get("source_length")
            if not length:
                raise ValueError("LIBRA aggregation requires original context length")
            values[str(length)].append(self.score_item(row["prediction"], row["references"]))
        if not values:
            raise ValueError("Нельзя агрегировать пустые LIBRA наблюдения")
        return {key: sum(scores) / len(scores) for key, scores in sorted(values.items())}


def postprocess_mera_prediction(task: str, prediction: str) -> str:
    """Apply the pinned generation filter chain (RegexFilter/remove_whitespace, take_first)."""
    if not isinstance(prediction, str):
        prediction = ""
    if task in {"MathLogicQA", "ruOpenBookQA"}:
        matches = re.findall(r"(\b([ABCD])\b)", prediction)
        return matches[0][0].strip() if matches else "-1"
    if task == "PARus":
        matches = re.findall(r"(\b([0-9])\b)", prediction)
        return matches[0][0].strip() if matches else "-1"
    if task == "MultiQ":
        return prediction.strip()
    raise ValueError(f"Неизвестная задача MERA: {task}")


def score_mera(*, task: str, predictions: Iterable[str], references: Iterable[str]) -> dict[str, Any]:
    if task not in MERA_TASKS:
        raise ValueError(f"Неизвестная задача MERA: {task}")
    predictions, references = list(predictions), list(references)
    _require(len(predictions) == len(references) and bool(predictions),
             "Для MERA scorer требуются равные непустые predictions и references")
    if any(not isinstance(ref, str) or not ref for ref in references):
        raise ValueError("MERA scorer блокирует отсутствующие или пустые references")
    processed = [postprocess_mera_prediction(task, value) for value in predictions]
    if task == "MultiQ":
        try:
            from transformers.data.metrics import squad_metrics
        except ImportError as exc:
            raise OfficialScorerUnavailable(
                "Pinned MERA MultiQ scorer требует transformers.data.metrics.squad_metrics") from exc
        exact_scores, f1_scores = [], []
        for prediction, reference in zip(processed, references):
            golds = reference.split(";")
            exact_scores.append(max(squad_metrics.compute_exact(prediction, gold) for gold in golds))
            f1_scores.append(max(squad_metrics.compute_f1(prediction, gold) for gold in golds))
        metrics = {"em": sum(exact_scores) / len(exact_scores), "f1": sum(f1_scores) / len(f1_scores)}
        per_item = [{"em": em, "f1": f1} for em, f1 in zip(exact_scores, f1_scores)]
    else:
        correct = [int(pred == ref) for pred, ref in zip(processed, references)]
        metrics = {"exact_match": sum(correct) / len(correct)}
        per_item = [{"exact_match": value} for value in correct]
        if task in {"MathLogicQA", "PARus", "ruOpenBookQA"}:
            metrics["accuracy_equivalent"] = metrics["exact_match"]
        if task == "ruOpenBookQA":
            try:
                from sklearn.metrics import f1_score
            except ImportError as exc:
                raise OfficialScorerUnavailable(
                    "Pinned MERA ruOpenBookQA scorer requires scikit-learn") from exc
            metrics["f1"] = float(f1_score(references, processed, average="macro"))
    return {"task": task, "scorer_revision": MERA_EVALUATOR_REVISION,
            "scorer_version": SCORER_VERSION, "count": len(predictions),
            "predictions_processed": processed, "per_item": per_item, "metrics": metrics}


def extract_unambiguous_label(text: str, allowed: Iterable[str]) -> str | None:
    """Strict format diagnostic only; it is not a MERA official scoring substitute."""
    if not isinstance(text, str):
        return None
    labels = sorted({str(value) for value in allowed}, key=lambda x: (-len(x), x))
    if not labels:
        return None
    pattern = r"(?<![\w])(" + "|".join(re.escape(x) for x in labels) + r")(?![\w])"
    found = set(re.findall(pattern, text, flags=re.IGNORECASE))
    canonical = {x.casefold(): x for x in labels}
    found = {canonical[x.casefold()] for x in found}
    return next(iter(found)) if len(found) == 1 else None
