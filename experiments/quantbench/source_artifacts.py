"""Fetch only revision-pinned dataset, evaluator, tokenizer and config files."""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def _url(row: dict[str, Any]) -> str:
    repo, revision, path, kind = row["repo_id"], row["revision"], row["path"], row["kind"]
    if kind == "hf_dataset":
        return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{path}?download=true"
    if kind == "hf_model":
        return f"https://huggingface.co/{repo}/resolve/{revision}/{path}?download=true"
    if kind == "github":
        return f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
    raise ValueError(f"Unknown pinned artifact kind: {kind}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_pinned_artifacts(*, registry_path: str | Path,
                              source_root: str | Path) -> dict[str, Any]:
    registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    root = Path(source_root).resolve()
    results = []
    manifests: dict[str, list[dict[str, Any]]] = {
        "source-download-manifest.json": [],
        "tokenizer-config-manifest.json": [],
        "official-source-manifest.json": [],
    }
    for item in registry.get("artifacts", []):
        relative = Path(item["local_path"])
        if "datasets/source/" not in relative.as_posix():
            raise ValueError(f"Pinned artifact destination is outside source root: {relative}")
        suffix = Path(*relative.as_posix().split("datasets/source/", 1)[1].split("/"))
        target = (root / suffix).resolve()
        if root not in target.parents:
            raise ValueError(f"Pinned artifact destination escapes source root: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = item["sha256"]
        if target.exists():
            if target.stat().st_size != item["bytes"] or _sha256(target) != expected:
                raise FileExistsError(f"Existing artifact failed pinned verification: {target}")
            status = "verified-existing"
        else:
            temporary = target.with_name(target.name + ".download.tmp")
            request = urllib.request.Request(_url(item), headers={"User-Agent": "quantbench-pinned-artifact-fetch/1.0"})
            digest = hashlib.sha256()
            size = 0
            try:
                with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as out:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk)
                        out.write(chunk)
                        size += len(chunk)
                if size != item["bytes"] or digest.hexdigest() != expected:
                    raise ValueError(f"Downloaded artifact failed pinned verification: {item['repo_id']}:{item['path']}")
                os.replace(temporary, target)
            except Exception:
                temporary.unlink(missing_ok=True)
                raise
            status = "downloaded-and-verified"
        portable_path = item["local_path"] if root == Path("datasets/source").resolve() else str(target)
        stored = {key: value for key, value in item.items() if key != "kind"}
        stored["local_path"] = portable_path
        results.append({"repo_id": item["repo_id"], "revision": item["revision"],
                        "path": item["path"], "bytes": item["bytes"], "sha256": expected,
                        "status": status})
        if item["kind"] == "hf_dataset":
            manifests["source-download-manifest.json"].append(stored)
        elif item["kind"] == "hf_model":
            manifests["tokenizer-config-manifest.json"].append(stored)
        else:
            manifests["official-source-manifest.json"].append(stored)
    for filename, rows in manifests.items():
        manifest_path = root / filename
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if filename == "tokenizer-config-manifest.json":
            by_model: dict[str, dict[str, Any]] = {}
            for row in rows:
                model = row["repo_id"]
                record = by_model.setdefault(model, {"revision": row["revision"], "files": []})
                record["files"].append(row["path"])
            payload = {"models": by_model, "files": rows}
        elif filename == "official-source-manifest.json":
            repos = {row["repo_id"]: row["revision"] for row in rows}
            payload = {"repos": repos, "files": rows}
        else:
            payload = {"files": rows}
        contents = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
        if manifest_path.exists():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if current != payload:
                raise FileExistsError(f"Existing source manifest differs from pinned registry: {manifest_path}")
        else:
            manifest_path.write_text(contents, encoding="utf-8")
    return {"artifact_count": len(results), "verified_count": len(results),
            "total_bytes": sum(item["bytes"] for item in results), "artifacts": results}
