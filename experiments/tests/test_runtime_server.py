import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantbench.runtime import LlamaServer, RuntimeErrorDetail
from quantbench.safety import SafetyGuard, SafetyState
from quantbench.watchdog import SafetyWatchdog

GENERATION = {"temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": 42,
              "max_new_tokens": 8, "stop": ["\n"]}


class FakeProcess:
    def __init__(self):
        self.terminated = False

    def poll(self):
        return 0 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        self.terminated = True
        return 0


def test_runtime_passes_protocol_monitoring_interval_to_platform_adapter(tmp_path):
    captured = {}

    class Adapter:
        platform_id = "apple-metal"

        def attach_safety(self, runtime, *, interval_s=None):
            captured["runtime"] = runtime
            captured["interval_s"] = interval_s

    config = {"runtime": {"host": "127.0.0.1", "port": 10001}, "limits": {"memory_sample_interval_s": 1.25}}
    runtime = LlamaServer(config, {"gguf_path": "unused.gguf", "kv_cache": "f16"},
                          log_path=tmp_path / "runtime.log", platform_adapter=Adapter())
    assert captured == {"runtime": runtime, "interval_s": 1.25}


@pytest.fixture
def fake_runtime(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        mode = "success"
        health_requests = 0
        last_payload = None
        last_path = None
        health_started = threading.Event()
        stream_started = threading.Event()

        def log_message(self, *_args):
            pass

        def do_GET(self):
            type(self).health_requests += 1
            if self.mode == "hanging_readiness":
                type(self).health_started.set()
                time.sleep(1.0)
                return
            if self.mode == "delayed_readiness" and self.health_requests < 3:
                self.send_error(503, "loading")
                return
            self.send_response(200 if self.mode != "health_error" else 500)
            self.end_headers()

        def do_POST(self):
            raw_payload = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            type(self).last_payload = json.loads(raw_payload)
            type(self).last_path = self.path
            if self.mode == "http500":
                self.send_error(500, "fake failure")
                return
            if self.mode == "http400":
                self.send_error(400, "fake bad request")
                return
            if self.mode == "hang":
                time.sleep(0.3)
                return
            if self.mode == "hang_stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                type(self).stream_started.set()
                time.sleep(1.0)
                return
            if self.mode == "drip_stream":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                type(self).stream_started.set()
                for index in range(20):
                    self.wfile.write((f'data: {{"choices":[{{"delta":{{"content":"{index}"}}}}]}}\n\n').encode())
                    self.wfile.flush()
                    time.sleep(0.03)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            if self.mode == "disconnect":
                self.send_response(200)
                self.end_headers()
                self.connection.shutdown(2)
                self.connection.close()
                return
            if self.mode == "invalid_json":
                chunks = [b"data: {bad json}\n\n", b"data: [DONE]\n\n"]
            elif self.mode == "truncated":
                chunks = [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n']
            elif self.mode == "no_timings":
                chunks = [b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}\n\n', b"data: [DONE]\n\n"]
            else:
                chunks = [b'data: {"choices":[{"delta":{"content":"hello "},"finish_reason":null}]}\n\n',
                          b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}],"timings":{"predicted_per_second":12.5}}\n\n',
                          b"data: [DONE]\n\n"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    config = {"runtime": {"executable": "fake-llama-server", "host": "127.0.0.1",
                          "port": httpd.server_port, "request_timeout_s": 1,
                          "load_timeout_s": 2, "gpu_layers": 0, "cpu_threads": 4,
                          "cpu_threads_batch": 4, "batch_size": 512, "ubatch_size": 128,
                          "flash_attn": False, "reasoning_mode": "off",
                          "chat_template_source": "gguf_metadata"}, "contexts": [128]}
    runtime = LlamaServer(config, {"gguf_path": "fake-model.gguf", "kv_cache": "f16"},
                          log_path=tmp_path / "fake.log")
    runtime.process = FakeProcess()
    try:
        yield runtime, Handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_fake_server_streaming_success_and_timings(fake_runtime):
    runtime, handler = fake_runtime
    response = runtime.generate("prompt", GENERATION)
    assert response["text"] == "hello world"
    assert response["finish_reason"] == "stop"
    assert response["timings"]["predicted_per_second"] == 12.5
    assert response["ttft_s"] is not None
    assert handler.last_path == "/v1/chat/completions"
    assert handler.last_payload["messages"] == [{"role": "user", "content": "prompt"}]
    assert {key: handler.last_payload[key] for key in ("temperature", "top_p", "top_k", "min_p", "seed", "max_tokens", "stop")} == {
        "temperature": 0, "top_p": 1, "top_k": 0, "min_p": 0, "seed": 42, "max_tokens": 8,
        "stop": ["\n"]}
    assert response["generation_parameters_submitted"] == GENERATION


def test_b12359_runtime_properties_and_launch_argv_verify_template_and_static_settings(monkeypatch, tmp_path):
    import hashlib

    from quantbench.runtime import LlamaServer

    template = "{{ '<|im_start|>assistant\\n' }}"
    config = {"runtime": {"executable": "fake-llama-server", "host": "127.0.0.1", "port": 18080}}
    model = {"gguf_path": "/models/Qwen3-4B-Q4_K_M.gguf", "kv_cache": "f16"}
    runtime = LlamaServer(config, model, log_path=tmp_path / "runtime.log")
    runtime._launch_command = [
        "fake-llama-server", "--model", model["gguf_path"], "--ctx-size", "4096", "--gpu-layers", "all",
        "--threads", "4", "--threads-batch", "4", "--batch-size", "512", "--ubatch-size", "128",
        "--cache-type-k", "f16", "--cache-type-v", "f16", "--flash-attn", "off", "--reasoning", "off", "--jinja",
    ]
    props = {"model_path": model["gguf_path"], "default_generation_settings": {"n_ctx": 4096},
             "chat_template": template}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(props).encode()

    monkeypatch.setattr("quantbench.runtime.urllib.request.urlopen", lambda *_args, **_kwargs: Response())
    expected_static = {"model_path": model["gguf_path"], "context_length": 4096, "gpu_layers": "all",
                       "cpu_threads": 4, "cpu_threads_batch": 4, "batch_size": 512, "ubatch_size": 128,
                       "kv_cache": "f16", "flash_attention": False, "reasoning_mode": "off",
                       "chat_template_source": "gguf_metadata",
                       "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest()}
    result = runtime.verify_parameters(expected_static, GENERATION)
    assert result["status"] == "static_verified_generation_pending"
    assert all(status == "verified" for status in result["static"].values())
    assert all(status == "pending_request" for status in result["generation"].values())


def test_fake_server_without_timings_keeps_client_metrics(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "no_timings"
    response = runtime.generate("prompt", GENERATION)
    assert response["text"] == "ok"
    assert response["timings"] == {}
    assert response["ttft_s"] is not None


@pytest.mark.parametrize("mode, message", [
    ("invalid_json", "Некорректное SSE"),
    ("truncated", "Неполный streaming"),
    ("http500", "HTTP 500"),
    ("http400", "HTTP 400"),
    ("disconnect", "Неполный streaming"),
])
def test_fake_server_errors_are_reported(fake_runtime, mode, message):
    runtime, handler = fake_runtime
    handler.mode = mode
    with pytest.raises(RuntimeErrorDetail, match=message):
        runtime.generate("prompt", GENERATION)


def test_fake_server_delayed_readiness(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "delayed_readiness"
    runtime.process = None
    runtime._popen = lambda *_args, **_kwargs: FakeProcess()
    runtime.start()
    assert handler.health_requests >= 3
    runtime.stop()


def test_runtime_watchdog_normal_start_request_and_stop(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "delayed_readiness"
    runtime.process = None
    process = FakeProcess()
    runtime._popen = lambda *_args, **_kwargs: process
    guard = SafetyGuard(state=SafetyState.SAFE)

    def sample():
        active = runtime.process is not None and runtime.process.poll() is None
        return {"system_used_bytes": 1, "available_memory_estimate_bytes": 8 * 1024**3, "reclaimable_memory_estimate_bytes": 8 * 1024**3,
                  "swap_delta_bytes": 0, "memory_pressure": {"category": "normal"},
                "thermal_state": "nominal", "process_state": "running" if active else "not_started",
                "physical_footprint_bytes": 1 if active else None, "pressure_notifications_available": True,
                "runtime_active": active, "safety_model": "apple_fully_reclaimable_estimate",
                "monotonic_time": time.monotonic()}

    watchdog = SafetyWatchdog(sample=sample, guard=guard,
                              stop_runtime=lambda: runtime.stop(grace_s=0.1), interval_s=0.02)
    runtime.safety_watchdog = watchdog
    runtime.start()
    response = runtime.generate("prompt", GENERATION)
    runtime.stop()
    assert response["text"] == "hello world"
    assert process.terminated
    assert watchdog._thread is None
    launch_rows = [json.loads(line) for line in runtime.log_path.read_text().splitlines()
                   if line.startswith("{")]
    launch_record = next(row for row in launch_rows if row.get("event") == "QUANTBENCH_RUNTIME_LAUNCH")
    assert launch_record["argv"] == runtime._launch_command


def test_runtime_watchdog_stops_owned_process_during_startup(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "hanging_readiness"
    runtime.process = None
    runtime.config["runtime"]["load_timeout_s"] = 3
    process = FakeProcess()
    process_started = [False]

    def start_process(*_args, **_kwargs):
        process_started[0] = True
        return process

    runtime._popen = start_process
    calls = [0]

    def sample():
        calls[0] += 1
        return {"system_used_bytes": 1, "available_memory_estimate_bytes": 8 * 1024**3, "reclaimable_memory_estimate_bytes": 8 * 1024**3,
                "swap_delta_bytes": 0,
                "memory_pressure": {"category": "critical" if process_started[0] else "normal"},
                "thermal_state": "nominal", "process_state": "running",
                "physical_footprint_bytes": 1, "pressure_notifications_available": True,
                "runtime_active": process_started[0], "safety_model": "apple_fully_reclaimable_estimate",
                "monotonic_time": time.monotonic()}

    guard = SafetyGuard()
    guard.state = SafetyState.SAFE
    watchdog = SafetyWatchdog(sample=sample, guard=guard,
                              stop_runtime=lambda: runtime.stop(grace_s=0.1), interval_s=0.01)
    runtime.safety_watchdog = watchdog
    with pytest.raises(InterruptedError, match="Safety Guard"):
        runtime.start()
    assert process.terminated
    assert runtime.process is None
    assert "SAFETY_GUARD_STOP" in runtime.log_path.read_text(encoding="utf-8")
    watchdog.close()


def test_fake_server_start_failure_cleans_up_owned_process(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "health_error"
    runtime.process = None
    process = FakeProcess()
    runtime._popen = lambda *_args, **_kwargs: process
    runtime.config["runtime"]["load_timeout_s"] = 0.3
    with pytest.raises(TimeoutError, match="готовности"):
        runtime.start()
    assert process.terminated
    assert runtime.process is None
    assert runtime._log is None


def test_fake_server_hanging_request_obeys_timeout(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "hang"
    runtime.config["runtime"]["request_timeout_s"] = 0.05
    with pytest.raises(RuntimeErrorDetail, match="deadline|не завершён штатно"):
        runtime.generate("prompt", GENERATION)


def test_stop_is_not_blocked_by_model_readiness(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "hanging_readiness"
    runtime.process = None
    process = FakeProcess()
    runtime._popen = lambda *_args, **_kwargs: process
    runtime.config["runtime"]["load_timeout_s"] = 5
    outcome = []

    def start_server():
        try:
            runtime.start()
        except Exception as exc:
            outcome.append(exc)

    start_thread = threading.Thread(target=start_server, daemon=True)
    start_thread.start()
    assert handler.health_started.wait(1)
    stop_started = time.monotonic()
    runtime.stop(grace_s=0)
    assert time.monotonic() - stop_started < 0.2
    assert process.terminated
    start_thread.join(timeout=0.5)
    assert not start_thread.is_alive()


def test_cancel_aborts_hanging_stream_without_waiting_for_next_line(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "hang_stream"
    runtime.config["runtime"]["request_timeout_s"] = 5
    cancel = threading.Event()
    outcome = []

    def run_request():
        try:
            runtime.generate("prompt", GENERATION, cancel=cancel)
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=run_request, daemon=True)
    worker.start()
    assert handler.stream_started.wait(1)
    cancel.set()
    worker.join(timeout=0.5)
    assert not worker.is_alive()
    assert outcome and isinstance(outcome[0], RuntimeErrorDetail)


def test_safety_watchdog_stops_runtime_during_hanging_stream(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "hang_stream"
    runtime.config["runtime"]["request_timeout_s"] = 5
    process = runtime.process
    outcome = []

    calls = [0]

    def sample():
        calls[0] += 1
        return {"system_used_bytes": 1, "available_memory_estimate_bytes": 8 * 1024**3, "reclaimable_memory_estimate_bytes": 8 * 1024**3,
            "swap_delta_bytes": 0, "memory_pressure": {"category": "critical" if calls[0] > 1 else "normal"},
                "thermal_state": "nominal", "process_state": "running",
                "physical_footprint_bytes": 1, "pressure_notifications_available": True,
                "runtime_active": True, "safety_model": "apple_fully_reclaimable_estimate",
                "monotonic_time": time.monotonic()}

    guard = SafetyGuard()
    guard.state = SafetyState.SAFE
    watchdog = SafetyWatchdog(sample=sample, guard=guard,
                              stop_runtime=lambda: runtime.stop(grace_s=0.1), interval_s=0.01)
    runtime.safety_watchdog = watchdog

    def run_request():
        try:
            watchdog.start()
            runtime.generate("prompt", GENERATION)
        except Exception as exc:
            outcome.append(exc)

    worker = threading.Thread(target=run_request, daemon=True)
    worker.start()
    assert handler.stream_started.wait(1)
    watchdog.start()
    worker.join(timeout=0.5)
    watchdog.close()
    assert not worker.is_alive()
    assert process.terminated
    assert outcome and isinstance(outcome[0], RuntimeErrorDetail)


def test_request_deadline_is_absolute_across_stream_events(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "drip_stream"
    runtime.config["runtime"]["request_timeout_s"] = 0.12
    started = time.monotonic()
    with pytest.raises(RuntimeErrorDetail):
        runtime.generate("prompt", GENERATION)
    assert time.monotonic() - started < 0.5
