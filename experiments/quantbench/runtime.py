"""Контролируемый адаптер локального llama-server OpenAI-compatible API."""

from __future__ import annotations

import hashlib
import json
import http.client
import queue
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
    required = ("gpu_layers", "cpu_threads", "cpu_threads_batch", "batch_size", "ubatch_size",
                "flash_attn", "reasoning_mode", "chat_template_source")
    missing = [key for key in required if key not in runtime or runtime[key] is None]
    if missing:
        raise ValueError("Не заданы контролируемые параметры runtime: " + ", ".join(missing))
    if type(runtime["flash_attn"]) is not bool or runtime["reasoning_mode"] not in {"on", "off"}:
        raise ValueError("flash_attn должен быть boolean, reasoning_mode — on/off")
    if runtime["chat_template_source"] != "gguf_metadata":
        raise ValueError("Поддерживается только явно закреплённый chat_template_source=gguf_metadata")
    for key in ("cpu_threads", "cpu_threads_batch", "batch_size", "ubatch_size"):
        if type(runtime[key]) is not int or runtime[key] < 1:
            raise ValueError(f"runtime.{key} должен быть положительным целым числом")
    gpu_layers = runtime["gpu_layers"]
    if not (type(gpu_layers) is int and gpu_layers >= 0 or gpu_layers in {"auto", "all"}):
        raise ValueError("runtime.gpu_layers должен быть неотрицательным числом, auto или all")
    kv_cache = model.get("kv_cache")
    if kv_cache not in {"f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}:
        raise ValueError("model.kv_cache не поддерживается установленной версией llama.cpp")
    args = [runtime["executable"], "--model", model["gguf_path"], "--host", host,
            "--port", str(runtime["port"]), "--ctx-size", str(max(config["contexts"])),
            "--parallel", "1", "--gpu-layers", str(gpu_layers),
            "--threads", str(runtime["cpu_threads"]), "--threads-batch", str(runtime["cpu_threads_batch"]),
            "--batch-size", str(runtime["batch_size"]), "--ubatch-size", str(runtime["ubatch_size"]),
            "--cache-type-k", kv_cache, "--cache-type-v", kv_cache,
            "--flash-attn", "on" if runtime["flash_attn"] else "off",
            "--reasoning", runtime["reasoning_mode"], "--jinja"]
    return args


class LlamaServer:
    def __init__(self, config: dict[str, Any], model: dict[str, Any], *, log_path: str | Path,
                 popen: Callable[..., Any] = subprocess.Popen, platform_adapter: Any = None,
                 safety_watchdog: Any = None):
        self.config, self.model = config, model
        self.log_path = Path(log_path)
        self._popen = popen
        self.platform_adapter = platform_adapter
        self.safety_watchdog = safety_watchdog
        self.process: Any = None
        self._launch_command: list[str] | None = None
        self._log = None
        self._request_lock = threading.Lock()
        self._lifecycle_lock = threading.RLock()
        self._stopping = False
        self._safety_diagnostic_written = False
        if (not config.get("synthetic_fake", False) and self.platform_adapter is not None and safety_watchdog is None
                and getattr(self.platform_adapter, "platform_id", None) == "apple-metal"):
            interval_s = config.get("limits", {}).get("memory_sample_interval_s")
            self.platform_adapter.attach_safety(self, interval_s=interval_s)

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.config["runtime"]["host"] == "localhost" else self.config["runtime"]["host"]
        return f"http://{host}:{self.config['runtime']['port']}"

    def start(self) -> None:
        if self.safety_watchdog is not None:
            guard = self.safety_watchdog.guard
            safe_wait = (guard.recovery_window_s + guard.max_metric_age_s + self.safety_watchdog.interval_s)
            self.safety_watchdog.start()
            if not self.safety_watchdog.wait_until_safe(timeout_s=safe_wait):
                self.stop(grace_s=0.0)
                raise RuntimeErrorDetail("Safety Guard не подтвердил безопасный допуск до загрузки модели")
        with self._lifecycle_lock:
            if self._stopping:
                raise RuntimeErrorDetail("Остановка предыдущего runtime ещё выполняется")
            if self.process is not None and self.process.poll() is None:
                raise RuntimeErrorDetail("Экземпляр сервера уже запущен")
            if self.process is not None or self._log is not None:
                raise RuntimeErrorDetail("Предыдущий runtime не завершён штатно")
            if self.safety_watchdog is not None and self.safety_watchdog.stop_event.is_set():
                raise InterruptedError("Запуск llama-server остановлен Safety Guard")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = self.log_path.open("ab", buffering=0)
            try:
                command = (self.platform_adapter.runtime_args(self.config, self.model)
                           if self.platform_adapter is not None else build_server_args(self.config, self.model))
                self._launch_command = list(command)
                process = self._popen(command, stdin=subprocess.DEVNULL,
                                      stdout=self._log, stderr=subprocess.STDOUT, shell=False, close_fds=True)
                self.process = process
            except BaseException:
                self._log.close()
                self._log = None
                raise
        deadline = time.monotonic() + self.config["runtime"]["load_timeout_s"]
        try:
            while time.monotonic() < deadline:
                with self._lifecycle_lock:
                    if self.process is not process:
                        raise InterruptedError("Запуск llama-server отменён Safety Guard")
                if self.safety_watchdog is not None and self.safety_watchdog.stop_event.is_set():
                    raise InterruptedError("Запуск llama-server остановлен Safety Guard")
                if process.poll() is not None:
                    raise RuntimeErrorDetail(f"llama-server завершился при старте; лог: {self.log_path}")
                if self._ready(timeout_s=min(0.1, max(0.01, deadline - time.monotonic()))):
                    with self._lifecycle_lock:
                        if self.process is process and process.poll() is None:
                            return
                    raise InterruptedError("Запуск llama-server отменён Safety Guard")
                time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
            raise TimeoutError("Истёк тайм-аут готовности llama-server")
        except BaseException:
            self.stop(grace_s=0.25)
            raise

    def _ready(self, *, timeout_s: float = 0.2) -> bool:
        try:
            request = urllib.request.Request(self.base_url + "/health", method="GET")
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return 200 <= response.status < 300
        except (OSError, urllib.error.URLError, TimeoutError):
            return False

    def verify_parameters(self, expected_static: dict[str, Any],
                          expected_generation: dict[str, Any]) -> dict[str, Any]:
        """Проверяет статические параметры по /props и argv; per-request параметры фиксируются ответом generate."""
        url = self.base_url + "/props"
        try:
            request = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(request, timeout=2.0) as response:
                props = json.loads(response.read().decode("utf-8"))
        except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return {"status": "unverified", "reason": f"runtime properties недоступны: {type(exc).__name__}",
                    "static": {key: "unverified" for key in expected_static},
                    "generation": {key: "unverified" for key in expected_generation},
                    "actual_static": {}, "actual_generation": {}}
        if not isinstance(props, dict):
            props = {}
        actual_static: dict[str, Any] = {}
        launch = self._launch_command or []

        def option_value(option: str) -> str | None:
            try:
                index = launch.index(option)
                return launch[index + 1]
            except (ValueError, IndexError):
                return None

        defaults = props.get("default_generation_settings")
        defaults = defaults if isinstance(defaults, dict) else {}
        actual_static["model_path"] = props.get("model_path")
        actual_static["context_length"] = defaults.get("n_ctx")
        actual_static["gpu_layers"] = option_value("--gpu-layers")
        actual_static["cpu_threads"] = option_value("--threads")
        actual_static["cpu_threads_batch"] = option_value("--threads-batch")
        actual_static["batch_size"] = option_value("--batch-size")
        actual_static["ubatch_size"] = option_value("--ubatch-size")
        cache_k, cache_v = option_value("--cache-type-k"), option_value("--cache-type-v")
        actual_static["kv_cache"] = cache_k if cache_k == cache_v else None
        flash = option_value("--flash-attn")
        actual_static["flash_attention"] = (flash == "on") if flash in {"on", "off"} else None
        actual_static["reasoning_mode"] = option_value("--reasoning")
        template_source = ("gguf_metadata" if "--chat-template" not in launch
                          and "--chat-template-file" not in launch else "custom")
        actual_static["chat_template_source"] = template_source
        template = props.get("chat_template")
        actual_static["chat_template_sha256"] = (
            hashlib.sha256(template.encode("utf-8")).hexdigest() if isinstance(template, str) else None)

        def equivalent(actual: Any, expected: Any, key: str) -> bool:
            if actual is None:
                return False
            if key == "model_path":
                try:
                    return Path(str(actual)).resolve() == Path(str(expected)).resolve()
                except OSError:
                    return False
            if type(expected) is bool:
                return actual is expected
            if type(expected) is int:
                try:
                    return int(actual) == expected
                except (TypeError, ValueError):
                    return False
            return str(actual) == str(expected)

        static_status = {key: ("verified" if equivalent(actual_static.get(key), value, key)
                               else "unverified") for key, value in expected_static.items()}
        static_verified = bool(static_status) and all(value == "verified" for value in static_status.values())
        generation_status = {key: ("pending_request" if static_verified else "unverified")
                             for key in expected_generation}
        return {"status": "static_verified_generation_pending" if static_verified else "unverified",
                "static": static_status, "generation": generation_status,
                "actual_static": actual_static, "actual_generation": {},
                "source": "llama-server /props + exact launch argv",
                "generation_verification": "deferred until completed request response"}

    def generate(self, prompt: str, generation: dict[str, Any], *, cancel: threading.Event | None = None) -> dict[str, Any]:
        if self.safety_watchdog is not None:
            self.safety_watchdog.assert_safe()
        if self.process is None or self.process.poll() is not None:
            raise RuntimeErrorDetail("llama-server не запущен")
        if not self._request_lock.acquire(blocking=False):
            raise RuntimeErrorDetail("Разрешён только один активный запрос")
        start = time.monotonic()
        deadline = start + float(self.config["runtime"]["request_timeout_s"])
        try:
            required_generation = ("temperature", "top_p", "top_k", "min_p", "seed", "max_new_tokens")
            missing_generation = [key for key in required_generation if key not in generation]
            if missing_generation:
                raise RuntimeErrorDetail("Не заданы параметры генерации: " + ", ".join(missing_generation))
            payload = {"model": "local", "messages": [{"role": "user", "content": prompt}], "stream": True,
                       "temperature": generation["temperature"], "top_p": generation["top_p"],
                       "top_k": generation["top_k"], "min_p": generation["min_p"], "seed": generation["seed"],
                       "max_tokens": generation["max_new_tokens"]}
            if generation.get("stop") is not None:
                payload["stop"] = generation["stop"]
            request = urllib.request.Request(self.base_url + "/v1/chat/completions",
                                             data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"}, method="POST")
            chunks: list[str] = []
            timings: dict[str, Any] = {}
            first_token_s = None
            completed = False
            finish_reason = None
            events: queue.Queue[tuple[str, Any]] = queue.Queue()
            state_lock = threading.Lock()
            state: dict[str, Any] = {"response": None}

            def abort_response(response: Any) -> None:
                if response is None:
                    return
                try:
                    sock = response.fp.raw._sock
                    sock.shutdown(socket.SHUT_RDWR)
                    sock.close()
                except (AttributeError, OSError):
                    pass

            def read_response() -> None:
                response = None
                try:
                    response = urllib.request.urlopen(
                        request, timeout=max(0.05, deadline - time.monotonic()))
                    with state_lock:
                        state["response"] = response
                    events.put(("response", response))
                    for raw_line in response:
                        events.put(("line", raw_line))
                    events.put(("eof", None))
                except urllib.error.HTTPError as exc:
                    try:
                        detail = exc.read(2048).decode("utf-8", "replace")
                    except Exception:
                        detail = ""
                    events.put(("http_error", (exc.code, detail)))
                except BaseException as exc:
                    events.put(("error", exc))
                finally:
                    if response is not None:
                        response.close()

            threading.Thread(target=read_response, name="quantbench-http-reader", daemon=True).start()
            while True:
                safety_cancelled = (self.safety_watchdog is not None and
                                    self.safety_watchdog.stop_event.is_set())
                if safety_cancelled or (cancel is not None and cancel.is_set()):
                    with state_lock:
                        response = state["response"]
                    abort_response(response)
                    self.stop(grace_s=0.25)
                    raise RuntimeErrorDetail("Запрос прерван Safety Guard")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    with state_lock:
                        response = state["response"]
                    abort_response(response)
                    self.stop(grace_s=0.25)
                    raise RuntimeErrorDetail("Истёк общий deadline запроса к llama-server")
                try:
                    kind, value = events.get(timeout=min(0.05, remaining))
                except queue.Empty:
                    continue
                if kind == "response":
                    pass
                elif kind == "line":
                    line = value.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        completed = True
                        break
                    try:
                        event = json.loads(body)
                    except json.JSONDecodeError as exc:
                        raise RuntimeErrorDetail("Некорректное SSE-событие от runtime") from exc
                    if event.get("timings"):
                        timings.update(event["timings"])
                    choices = event.get("choices") or []
                    if choices:
                        finish_reason = choices[0].get("finish_reason") or finish_reason
                        delta = choices[0].get("delta") or {}
                        text = delta.get("content", "")
                        if text:
                            if first_token_s is None:
                                first_token_s = time.monotonic() - start
                            chunks.append(text)
                elif kind == "http_error":
                    code, body = value
                    raise RuntimeErrorDetail(f"llama-server HTTP {code}: {body}")
                elif kind == "error":
                    raise value
                elif kind == "eof":
                    break
            if not completed:
                raise RuntimeErrorDetail("Неполный streaming response от runtime: отсутствует [DONE]")
            return {"text": "".join(chunks), "ttft_s": first_token_s,
                    "request_wall_s": time.monotonic() - start, "timings": timings,
                    "generation_parameters_submitted": {
                        "temperature": payload["temperature"], "top_p": payload["top_p"],
                        "top_k": payload["top_k"], "min_p": payload["min_p"], "seed": payload["seed"],
                        "max_new_tokens": payload["max_tokens"], "stop": payload.get("stop")},
                    "finish_reason": finish_reason or "stop"}
        except urllib.error.URLError as exc:
            raise RuntimeErrorDetail(f"Ошибка соединения с llama-server: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            raise RuntimeErrorDetail(f"Запрос к llama-server не завершён штатно: {exc}") from exc
        except http.client.HTTPException as exc:
            raise RuntimeErrorDetail(f"Неполное HTTP/SSE-соединение с llama-server: {exc}") from exc
        finally:
            self._request_lock.release()

    def stop(self, *, grace_s: float = 2.0) -> None:
        with self._lifecycle_lock:
            if self._stopping:
                return
            self._stopping = True
            process = self.process
        error = None
        try:
            if (self.safety_watchdog is not None and self.safety_watchdog.stop_event.is_set()
                    and self._log is not None and not self._safety_diagnostic_written):
                diagnostic = {"event": "SAFETY_GUARD_STOP", "state": self.safety_watchdog.last_state.value,
                              "error": self.safety_watchdog.last_error,
                              "metrics": self.safety_watchdog.last_metrics}
                try:
                    self._log.write((json.dumps(diagnostic, ensure_ascii=False, sort_keys=True) + "\n").encode())
                    self._safety_diagnostic_written = True
                except OSError as exc:
                    error = exc
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=max(0.0, grace_s))
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=1.0)
        except Exception as exc:
            error = exc
        finally:
            with self._lifecycle_lock:
                if process is None or process.poll() is not None:
                    if self.process is process:
                        self.process = None
                    if self._log is not None:
                        self._log.close()
                        self._log = None
                self._stopping = False
            if self.safety_watchdog is not None:
                self.safety_watchdog.close(timeout_s=1.0)
            pressure_monitor = getattr(self, "_pressure_monitor", None)
            if pressure_monitor is not None:
                pressure_monitor.close()
                self._pressure_monitor = None
        if error is not None:
            raise RuntimeErrorDetail(f"Не удалось своевременно остановить принадлежащий runtime процесс: {error}") from error

    def __enter__(self) -> "LlamaServer":
        self.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()
