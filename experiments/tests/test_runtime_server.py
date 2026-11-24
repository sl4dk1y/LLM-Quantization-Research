import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantbench.runtime import LlamaServer, RuntimeErrorDetail
from quantbench.safety import SafetyGuard
from quantbench.watchdog import SafetyWatchdog

GENERATION = {"temperature": 0, "top_p": 1, "top_k": 0, "seed": 42, "max_new_tokens": 8}


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


@pytest.fixture
def fake_runtime(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        mode = "success"
        health_requests = 0
        last_payload = None
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
                    self.wfile.write((f'data: {{"choices":[{{"text":"{index}"}}]}}\n\n').encode())
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
                chunks = [b'data: {"choices":[{"text":"partial"}]}\n\n']
            elif self.mode == "no_timings":
                chunks = [b'data: {"choices":[{"text":"ok","finish_reason":"stop"}]}\n\n', b"data: [DONE]\n\n"]
            else:
                chunks = [b'data: {"choices":[{"text":"hello ","finish_reason":null}]}\n\n',
                          b'data: {"choices":[{"text":"world","finish_reason":"stop"}],"timings":{"predicted_per_second":12.5}}\n\n',
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
    assert {key: handler.last_payload[key] for key in ("temperature", "top_p", "top_k", "seed", "max_tokens")} == {
        "temperature": 0, "top_p": 1, "top_k": 0, "seed": 42, "max_tokens": 8}


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

    def run_request():
        try:
            runtime.generate("prompt", GENERATION, cancel=watchdog.stop_event)
        except Exception as exc:
            outcome.append(exc)

    watchdog = SafetyWatchdog(sample=lambda: {}, guard=SafetyGuard(),
                              stop_runtime=lambda: runtime.stop(grace_s=0.1), interval_s=0.01)
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
