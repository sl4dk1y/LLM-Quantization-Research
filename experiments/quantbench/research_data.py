"""Prepare the pinned MERA and LIBRA study exports from verified local sources."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .dataset_pipeline import (
    LIBRA_LENGTHS, LIBRA_REVISION, LIBRA_SOURCE_PATHS, MERA_REVISION,
    MERA_SOURCE_PATHS, MERA_TASKS, SEED, _item_id, _render_mera, file_sha256,
    prepare_export,
)
from .sampling import select_items, validate_disjoint

MODEL_TOKENIZERS = {
    "Qwen/Qwen3-1.7B": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    "Qwen/Qwen3-4B": "1cfa9a7208912126459214e8b04321603b3df60c",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}


def _load_download_manifest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[tuple[str, str], dict[str, Any]] = {}
    for row in payload.get("files", []):
        local_path = Path(row["local_path"])
        if not local_path.is_file() or file_sha256(local_path) != row.get("sha256"):
            raise ValueError(f"Скачанный исходный файл не прошёл SHA-256 проверку: {local_path}")
        key = (row["repo_id"], row["path"])
        if key in entries:
            raise ValueError(f"Повторная provenance запись: {key}")
        entries[key] = row
    return entries


def _read_jsonl(path: Path, *, dataset: str, task: str, split: str) -> list[dict[str, Any]]:
    rows = []
    ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Ожидался объект JSONL: {path}:{index + 1}")
            item_id = _item_id(row, dataset=dataset, row_index=index, task=task, split=split)
            if item_id in ids:
                raise ValueError(f"Повторяющийся item ID {item_id} в {path}")
            ids.add(item_id)
            rows.append({"id": item_id, "source_row_index": index, **row})
    if not rows:
        raise ValueError(f"Пустой исходный split: {path}")
    return rows


def _fewshot(rows: list[dict[str, Any]], *, task: str, seed: int) -> dict[str, Any]:
    candidates = []
    for row in rows:
        _render_mera(row, task)
        candidates.append(row)
    chosen = select_items(candidates, seed=seed, revision=MERA_REVISION, task=task,
                          split="M2.1-fixed-fewshot", count=1)[0]
    return {"id": chosen["id"], "row": chosen}


def _write_jsonl_immutable(path: Path, rows: list[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
                     for row in rows)
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if path.exists() and path.read_text(encoding="utf-8") != content:
        raise FileExistsError(f"Отказ от перезаписи преобразованного источника: {path}")
    if not path.exists():
        path.write_text(content, encoding="utf-8")
    return digest


def _convert_libra(path: Path, *, task: str, provenance: dict[str, Any], output_root: Path) -> tuple[Path, dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("Для преобразования LIBRA parquet требуется pyarrow") from exc
    if provenance["revision"] != LIBRA_REVISION or provenance["path"] != LIBRA_SOURCE_PATHS[task]:
        raise ValueError(f"LIBRA upstream provenance не совпадает с pin/task: {task}")
    source_hash = file_sha256(path)
    if source_hash != provenance["sha256"]:
        raise ValueError(f"LIBRA parquet hash не совпадает с pinned source: {path}")
    target = output_root / "converted" / "LIBRA" / f"{task}.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".jsonl.tmp")
    digest = hashlib.sha256()
    row_index = 0
    pf = parquet.ParquetFile(path)
    required = {"length", "context", "input", "positive_outputs"}
    if not required.issubset(pf.schema_arrow.names):
        raise ValueError(f"LIBRA parquet schema missing fields: {sorted(required - set(pf.schema_arrow.names))}")
    with temporary.open("wb") as output:
        for batch in pf.iter_batches(batch_size=8):
            for raw in batch.to_pylist():
                if not isinstance(raw.get("positive_outputs"), list) or not raw["positive_outputs"]:
                    raise ValueError(f"LIBRA row {row_index} has no positive_outputs")
                canonical = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                row_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
                row = {"id": f"libra:{task}:test:{row_index:08d}:{row_hash}",
                       "source_row_index": row_index, **raw}
                data = (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
                output.write(data)
                digest.update(data)
                row_index += 1
    converted_hash = digest.hexdigest()
    if target.exists() and file_sha256(target) != converted_hash:
        temporary.unlink(missing_ok=True)
        raise FileExistsError(f"Отказ от перезаписи преобразованного источника: {target}")
    if not target.exists():
        temporary.replace(target)
    else:
        temporary.unlink(missing_ok=True)
    converted_provenance = {
        "repo_id": provenance["repo_id"], "revision": provenance["revision"],
        "path": provenance["path"], "local_path": str(target),
        "original_local_path": str(path),
        "original_sha256": source_hash, "transformed_source_sha256": converted_hash,
        "bytes": path.stat().st_size, "transformed_rows": row_index,
        "transformation_version": "libra-parquet-jsonl-row-id-v1",
        "transformation": "pyarrow row order, preserve all columns, add revision-scoped row ID",
        "verified": True,
    }
    return target, converted_provenance


def _load_tokenizers(source_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Для подсчёта tokenizer-specific token counts требуется transformers") from exc
    tokenizers = {}
    metadata = {}
    for model, revision in MODEL_TOKENIZERS.items():
        local = source_root / "tokenizers" / model
        tokenizer = AutoTokenizer.from_pretrained(local, local_files_only=True, trust_remote_code=False)
        profile = f"{model}@{revision}"
        tokenizers[profile] = tokenizer
        files = sorted((p for p in local.iterdir() if p.is_file()), key=lambda p: p.name)
        metadata[profile] = {
            "repo_id": model, "revision": revision,
            "files": {p.name: file_sha256(p) for p in files},
            "chat_template_sha256": hashlib.sha256(str(tokenizer.chat_template or "").encode("utf-8")).hexdigest(),
        }
    return tokenizers, metadata


def build_research_exports(*, source_root: str | Path, output_root: str | Path,
                           tokenizer_paths: dict[str, Any] | None = None,
                           seed: int = SEED) -> dict[str, Any]:
    """Prepare real P0/P1 and remaining M2 datasets without running inference."""
    source_root, output_root = Path(source_root), Path(output_root)
    manifest_path = source_root / "source-download-manifest.json"
    provenance_entries = _load_download_manifest(manifest_path)
    if tokenizer_paths is None:
        tokenizers, tokenizer_metadata = _load_tokenizers(source_root)
    else:
        tokenizers = tokenizer_paths
        tokenizer_metadata = {name: {"external_tokenizer": True} for name in tokenizers}
    exports: list[dict[str, Any]] = []
    p0_ids: dict[str, set[str]] = {}
    development_ids: dict[str, set[str]] = {}
    shots: dict[str, dict[str, Any]] = {}
    rows_by_task: dict[str, list[dict[str, Any]]] = {}
    source_by_task: dict[str, tuple[Path, dict[str, Any]]] = {}

    for task in MERA_TASKS:
        source_split = "train"
        upstream_path = MERA_SOURCE_PATHS[task][source_split]
        entry = provenance_entries[("MERA-evaluation/MERA", upstream_path)]
        if entry["revision"] != MERA_REVISION:
            raise ValueError("MERA train source revision does not match M2 pin")
        source_path = Path(entry["local_path"])
        if file_sha256(source_path) != entry["sha256"]:
            raise ValueError(f"MERA source hash mismatch: {source_path}")
        rows_by_task[task] = _read_jsonl(source_path, dataset="MERA", task=task, split=source_split)
        source_by_task[task] = (source_path, entry)
        if task != "MultiQ":
            shots[task] = _fewshot(rows_by_task[task], task=task, seed=seed)

    def emit(*, cohort: str, task: str, split: str, source_split: str, source_path: Path,
             provenance: dict[str, Any], count: int, excluded: set[str] | None = None,
             shot: dict[str, Any] | None = None, context_length: str | None = None,
             disjoint: set[str] | None = None) -> dict[str, Any]:
        path = output_root / cohort / "MERA" / task / f"{split}.jsonl" if provenance["repo_id"] == "MERA-evaluation/MERA" \
            else output_root / cohort / "LIBRA" / task / f"{context_length}.jsonl"
        manifest_out = path.with_suffix(".manifest.json")
        manifest = prepare_export(
            source_path=source_path, output_path=path, manifest_path=manifest_out,
            dataset="MERA" if provenance["repo_id"] == "MERA-evaluation/MERA" else "LIBRA",
            task=task, split=split, source_split=source_split,
            revision=MERA_REVISION if provenance["repo_id"] == "MERA-evaluation/MERA" else LIBRA_REVISION,
            count=count, seed=seed, expected_source_sha256=provenance.get("transformed_source_sha256") or provenance["sha256"],
            source_provenance=provenance, exclude_ids=excluded or (), disjoint_ids=disjoint,
            fewshot_example=shot, context_length=context_length, tokenizers=tokenizers,
            cohort=cohort, tokenizer_metadata=tokenizer_metadata)
        exports.append(manifest)
        return manifest

    for task in ("MathLogicQA", "ruOpenBookQA"):
        source_path, provenance = source_by_task[task]
        shot = shots[task]
        candidates = [r for r in rows_by_task[task] if r["id"] != shot["id"]]
        p0_selected = select_items(candidates, seed=seed, revision=MERA_REVISION,
                                   task=task, split="P0-training", count=20)
        p0_ids[task] = {r["id"] for r in p0_selected}
        p0 = emit(cohort="P0", task=task, split="p0", source_split="train", source_path=source_path,
                  provenance=provenance, count=20, excluded={shot["id"]}, shot=shot,
                  disjoint={shot["id"]})
        p0_ids[task] = set(p0["item_ids"])
        candidates = [r for r in rows_by_task[task] if r["id"] not in p0_ids[task] | {shot["id"]}]
        dev_selected = select_items(candidates, seed=seed, revision=MERA_REVISION,
                                    task=task, split="local_development", count=100)
        development_ids[task] = {r["id"] for r in dev_selected}
        emit(cohort="M2", task=task, split="local_development", source_split="train", source_path=source_path,
             provenance=provenance, count=100, excluded=p0_ids[task] | {shot["id"]}, shot=shot,
             disjoint=p0_ids[task] | {shot["id"]})
        heldout_excluded = p0_ids[task] | development_ids[task] | {shot["id"]}
        heldout_n = 200 if task == "MathLogicQA" else 200
        if task == "ruOpenBookQA":
            heldout_n = 200
        emit(cohort="P1" if task == "MathLogicQA" else "M2", task=task,
             split="local_heldout", source_split="train", source_path=source_path,
             provenance=provenance, count=heldout_n, excluded=heldout_excluded, shot=shot,
             disjoint=heldout_excluded)

    task = "MultiQ"
    source_path, provenance = source_by_task[task]
    emit(cohort="P1", task=task, split="local_heldout", source_split="train", source_path=source_path,
         provenance=provenance, count=200, disjoint=set())

    task = "PARus"
    train_path, train_provenance = source_by_task[task]
    shot = shots[task]
    dev_entry = provenance_entries[("MERA-evaluation/MERA", MERA_SOURCE_PATHS[task]["dev"])]
    dev_path = Path(dev_entry["local_path"])
    emit(cohort="M2", task=task, split="dev", source_split="dev", source_path=dev_path,
         provenance=dev_entry, count=100, shot=shot, disjoint={shot["id"]})

    for task in ("MathLogicQA", "ruOpenBookQA", "MultiQ", "PARus"):
        source_path, provenance = source_by_task[task]
        official_test = provenance_entries[("MERA-evaluation/MERA", MERA_SOURCE_PATHS[task]["test"])]
        test_rows = _read_jsonl(Path(official_test["local_path"]), dataset="MERA", task=task, split="test")
        labeled = sum(isinstance(row.get("outputs"), str) and bool(row["outputs"].strip()) for row in test_rows)
        if labeled:
            raise ValueError(f"Pinned MERA test labels unexpectedly available; protocol amendment review required: {task}")

    for task in ("ru_babilong_qa3", "long_context_multiq", "matreshka_names"):
        upstream_path = LIBRA_SOURCE_PATHS[task]
        entry = provenance_entries[("ai-forever/LIBRA", upstream_path)]
        raw_path = Path(entry["local_path"])
        converted_path, converted_provenance = _convert_libra(raw_path, task=task,
                                                              provenance=entry, output_root=output_root)
        for length in LIBRA_LENGTHS:
            emit(cohort="M2", task=task, split="test", source_split="test",
                 source_path=converted_path, provenance=converted_provenance, count=30,
                 context_length=length)

    all_ids = {f"{m['cohort']}:{m['task']}:{m['split']}": set(m["item_ids"]) for m in exports}
    overlap_checks = []
    for task in ("MathLogicQA", "ruOpenBookQA"):
        p0 = all_ids[f"P0:{task}:p0"]
        dev = all_ids[f"M2:{task}:local_development"]
        held = all_ids["P1:MathLogicQA:local_heldout"] if task == "MathLogicQA" else all_ids[f"M2:{task}:local_heldout"]
        for left_name, left, right_name, right in (("P0", p0, "development", dev),
                                                    ("P0", p0, "heldout", held),
                                                    ("development", dev, "heldout", held)):
            validate_disjoint(left, right)
            overlap_checks.append({"task": task, "left": left_name, "right": right_name, "overlap_count": 0})
        if task in shots:
            validate_disjoint({shots[task]["id"]}, p0 | dev | held)
            for split_name, split_ids in (("P0", p0), ("development", dev), ("heldout", held)):
                overlap_checks.append({"task": task, "left": "fewshot", "right": split_name,
                                       "overlap_count": 0})
    summary = {"protocol_version": "M2.1", "inference_started": False,
               "dataset_revisions": {"MERA": MERA_REVISION, "LIBRA": LIBRA_REVISION},
               "seed": seed, "exports": [{"cohort": row["cohort"], "task": row["task"],
                                           "split": row["split"], "count": row["sample_count"],
                                           "path": row["prepared_path"],
                                           "sha256": row["prepared_sha256"],
                                           "manifest": str(Path(row["prepared_path"]).with_suffix(".manifest.json"))}
                                          for row in exports],
               "overlap_checks": overlap_checks,
               "fewshot_ids": {task: shot["id"] for task, shot in shots.items()},
               "tokenizer_profiles": tokenizer_metadata}
    summary_path = output_root / "export-summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return summary
