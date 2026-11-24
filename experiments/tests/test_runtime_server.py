import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from quantbench.runtime import LlamaServer, RuntimeErrorDetail


class FakeProcess:
    def poll(self):
        return None

    def terminate(self):
        self.terminated = True

    def wait(self, timeout=None):
        return 0


@pytest.fixture
def fake_runtime(tmp_path):
    class Handler(BaseHTTPRequestHandler):
        mode = "success"
        health_requests = 0

        def log_message(self, *_args):
            pass

        def do_GET(self):
            type(self).health_requests += 1
            if self.mode == "delayed_readiness" and self.health_requests < 3:
                self.send_error(503, "loading")
                return
            self.send_response(200 if self.mode != "health_error" else 500)
            self.end_headers()

        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            if self.mode == "http500":
                self.send_error(500, "fake failure")
                return
            if self.mode == "http400":
                self.send_error(400, "fake bad request")
                return
            if self.mode == "hang":
                time.sleep(0.3)
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
                          "load_timeout_s": 2}, "contexts": [128]}
    runtime = LlamaServer(config, {"gguf_path": "fake-model.gguf"},
                          log_path=tmp_path / "fake.log")
    runtime.process = FakeProcess()
    try:
        yield runtime, Handler
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_fake_server_streaming_success_and_timings(fake_runtime):
    runtime, _ = fake_runtime
    response = runtime.generate("prompt", {"max_new_tokens": 8})
    assert response["text"] == "hello world"
    assert response["finish_reason"] == "stop"
    assert response["timings"]["predicted_per_second"] == 12.5
    assert response["ttft_s"] is not None


def test_fake_server_without_timings_keeps_client_metrics(fake_runtime):
    runtime, handler = fake_runtime
    handler.mode = "no_timings"
    response = runtime.generate("prompt", {"max_new_tokens": 8})
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
        runtime.generate("prompt", {"max_new_tokens": 8})


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
    with pytest.raises(RuntimeErrorDetail, match="не завершён штатно"):
        runtime.generate("prompt", {"max_new_tokens": 8})
