"""Offline tests for transient Gemini API retries."""
from __future__ import annotations

import json
import urllib.error

import pytest

from llm_poc_rewrite import api, config


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps({
            "modelVersion": config.AGENT_MODELS["analyst"],
            "candidates": [{
                "content": {"parts": [{"text": "ok"}]},
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 3,
                "candidatesTokenCount": 1,
                "thoughtsTokenCount": 0,
                "totalTokenCount": 4,
            },
        }).encode()


def test_generate_retries_5xx_with_exponential_backoff(monkeypatch):
    attempts = []
    delays = []

    def fake_urlopen(request, timeout):
        attempts.append(1)
        if len(attempts) <= config.API_MAX_RETRIES:
            raise urllib.error.HTTPError(request.full_url, 503, "Unavailable", {}, None)
        return FakeResponse()

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(api, "_last_request_at", 0.0)
    monkeypatch.setattr(config, "MIN_REQUEST_INTERVAL_SEC", 0)
    monkeypatch.setattr(api.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(api.time, "sleep", delays.append)

    result = api.generate("prompt", "system", config.AGENT_MODELS["analyst"])

    assert result.text == "ok"
    assert len(attempts) == config.API_MAX_RETRIES + 1
    assert delays == [config.API_RETRY_BASE_SEC * (2 ** i)
                      for i in range(config.API_MAX_RETRIES)]


def test_generate_stops_after_configured_5xx_retries(monkeypatch):
    attempts = []
    delays = []

    def fail_5xx(request, timeout):
        attempts.append(1)
        raise urllib.error.HTTPError(request.full_url, 500, "Internal Server Error", {}, None)

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(api, "_last_request_at", 0.0)
    monkeypatch.setattr(config, "MIN_REQUEST_INTERVAL_SEC", 0)
    monkeypatch.setattr(api.urllib.request, "urlopen", fail_5xx)
    monkeypatch.setattr(api.time, "sleep", delays.append)

    with pytest.raises(RuntimeError, match="HTTP 500"):
        api.generate("prompt", "system", config.AGENT_MODELS["analyst"])

    assert len(attempts) == config.API_MAX_RETRIES + 1
    assert delays == [config.API_RETRY_BASE_SEC * (2 ** i)
                      for i in range(config.API_MAX_RETRIES)]


def test_generate_does_not_retry_client_http_errors(monkeypatch):
    attempts = []
    delays = []

    def fail_400(request, timeout):
        attempts.append(1)
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, None)

    monkeypatch.setenv("GEMINI_API_KEY", "offline-test")
    monkeypatch.setattr(api, "_last_request_at", 0.0)
    monkeypatch.setattr(config, "MIN_REQUEST_INTERVAL_SEC", 0)
    monkeypatch.setattr(api.urllib.request, "urlopen", fail_400)
    monkeypatch.setattr(api.time, "sleep", delays.append)

    with pytest.raises(RuntimeError, match="HTTP 400"):
        api.generate("prompt", "system", config.AGENT_MODELS["analyst"])

    assert len(attempts) == 1
    assert delays == []
