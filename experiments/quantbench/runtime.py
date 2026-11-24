"""Контролируемый адаптер локального llama-server OpenAI-compatible API."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable


class RuntimeErrorDetail(RuntimeError):
    pass


def build_server_args(config: dict[str, Any], model: dict[str, Any]) -> list[str]:
    runtime = config["runtime"]
    host = runtime.get("host")
    if host not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("llama-server разрешено привязывать только к loopback")
    return [runtime["executable"], "--model", model["gguf_path"], "--host", host,
            "--port", str(runtime["port"]), "--ctx-size", str(max(config["contexts"])),
            "--parallel", "1"]


class LlamaServer:
    def __init__(self, config: dict[str, Any], model: dict[str, Any], *, log_path: str | Path,
                 popen: Callable[..., Any] = subprocess.Popen):
        self.config, self.model = config, model
        self.log_path = Path(log_path)
        self._popen = popen
        self.process: Any = None
        self._log = None
        self._request_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.config["runtime"]["host"] == "localhost" else self.config["runtime"]["host"]
        return f"http://{host}:{self.config['runtime']['port']}"

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            raise RuntimeErrorDetail("Экземпляр сервера уже запущен")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = self.log_path.open("ab", buffering=0)
        self.process = self._popen(build_server_args(self.config, self.model), stdin=subprocess.DEVNULL,
                                   stdout=self._log, stderr=subprocess.STDOUT, shell=False, close_fds=True)
        deadline = time.monotonic() + self.config["runtime"]["load_timeout_s"]
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeErrorDetail(f"llama-server завершился при старте; лог: {self.log_path}")
            if self._ready():
                return
            time.sleep(0.25)
        self.stop()
        raise TimeoutError("Истёк тайм-аут готовности llama-server")

    def _ready(self) -> bool:
        try:
            request = urllib.request.Request(self.base_url + "/health", method="GET")
            with urllib.request.urlopen(request, timeout=1) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def generate(self, prompt: str, generation: dict[str, Any], *, cancel: threading.Event | None = None) -> dict[str, Any]:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeErrorDetail("llama-server не запущен")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeErrorDetail("Разрешён только один активный запрос")
        start = time.monotonic()
        try:
            payload = {"model": "local", "prompt": prompt, "stream": True,
                       "temperature": generation.get("temperature", 0),
                       "top_p": generation.get("top_p", 1), "top_k": generation.get("top_k", 0),
                       "seed": generation.get("seed", 42), "max_tokens": generation["max_new_tokens"]}
            request = urllib.request.Request(self.base_url + "/v1/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            chunks: list[str] = []
            timings: dict[str, Any] = {}
            first_token_s = None
            with urllib.request.urlopen(request, timeout=self.config["runtime"]["request_timeout_s"]) as response:
                for raw_line in response:
                    if cancel is not None and cancel.is_set():
                        raise InterruptedError("Запрос прерван safety monitor")
                    line = raw_line.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise RuntimeErrorDetail("Некорректное SSE-событие от runtime") from exc
                    if event.get("timings"):
                        timings.update(event["timings"])
                    choices = event.get("choices") or []
                    if choices:
                        text = choices[0].get("text", "")
                        if text:
                            if first_token_s is None:
                                first_token_s = time.monotonic() - start
                            chunks.append(text)
            return {"text": "".join(chunks), "ttft_s": first_token_s,
                    "request_wall_s": time.monotonic() - start, "timings": timings,
                    "finish_reason": "stream_end"}
        except urllib.error.HTTPError as exc:
            body = exc.read(2048).decode("utf-8", "replace")
            raise RuntimeErrorDetail(f"llama-server HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeErrorDetail(f"Ошибка соединения с llama-server: {exc.reason}") from exc
        finally:
            self._request_lock.release()

    def stop(self, *, grace_s: float = 30) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self.process = None
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
