import pytest
import requests

from privacy_worker.comfyui import ComfyUIClient
from privacy_worker.config import settings
from privacy_worker.errors import ComfyUIError


class FakeResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError("simulated HTTP error")

    def json(self):
        return self.body


@pytest.mark.parametrize(
    "body",
    [[], None, "invalid", 123],
)
def test_non_object_response_is_controlled(monkeypatch, body):
    monkeypatch.setattr(
        "privacy_worker.comfyui.requests.post",
        lambda *args, **kwargs: FakeResponse(body),
    )

    with pytest.raises(ComfyUIError) as caught:
        ComfyUIClient(settings).queue_prompt({}, "offline")

    assert caught.value.code == "COMFYUI_RUNTIME_ERROR"
    assert caught.value.details["response_type"] == type(body).__name__


def test_valid_response_preserved(monkeypatch):
    monkeypatch.setattr(
        "privacy_worker.comfyui.requests.post",
        lambda *args, **kwargs: FakeResponse({"prompt_id": "test-123"}),
    )

    assert ComfyUIClient(settings).queue_prompt({}, "offline") == "test-123"


def test_http_error_remains_controlled(monkeypatch):
    monkeypatch.setattr(
        "privacy_worker.comfyui.requests.post",
        lambda *args, **kwargs: FakeResponse({}, 400),
    )

    with pytest.raises(ComfyUIError) as caught:
        ComfyUIClient(settings).queue_prompt({}, "offline")

    assert caught.value.code == "COMFYUI_RUNTIME_ERROR"
