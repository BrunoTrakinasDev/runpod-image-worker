import importlib
import json
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "image_closeup.json"


class FakeProcessManager:
    def ensure_started(self, request_id: str) -> None:
        assert request_id == "req-image-001"


class FakeClient:
    def queue_prompt(self, prompt, request_id: str) -> str:
        assert prompt["6"]["inputs"]["image"].startswith("privacy_req-image-001_identity")
        return "prompt-001"

    def wait_for_history(self, prompt_id: str, request_id: str):
        assert prompt_id == "prompt-001"
        return {"outputs": {"17": {"images": [{"filename": "mock.png", "type": "output"}]}}}

    def download_output(self, *, record, output_nodes, destination: Path, request_id: str):
        assert output_nodes == ("17",)
        output = destination.with_suffix(".png")
        output.write_bytes(b"mock-png")
        return output


def test_handler_consumes_canonical_contract_without_gpu(monkeypatch, tmp_path):
    fake_runpod = types.ModuleType("runpod")
    fake_runpod.serverless = types.SimpleNamespace(start=lambda payload: payload)
    monkeypatch.setitem(sys.modules, "runpod", fake_runpod)
    monkeypatch.setenv("APP_ROOT", str(ROOT))
    monkeypatch.setenv("WORKFLOW_ROOT", str(ROOT / "workflows"))
    monkeypatch.setenv("RUNTIME_ROOT", str(tmp_path / "runtime"))
    monkeypatch.setenv("COMFYUI_ROOT", str(tmp_path / "ComfyUI"))
    monkeypatch.setenv("SKIP_MODEL_VALIDATION", "true")

    for name in ["handler", "privacy_worker.config"]:
        sys.modules.pop(name, None)
    module = importlib.import_module("handler")

    def fake_download_media(*, destination_dir: Path, stem: str, fallback_extension: str, **kwargs):
        path = destination_dir / f"{stem}{fallback_extension}"
        path.write_bytes(b"mock-input")
        return path

    monkeypatch.setattr(module, "download_media", fake_download_media)
    monkeypatch.setattr(module, "_process_manager", FakeProcessManager())
    monkeypatch.setattr(module, "_client", FakeClient())
    monkeypatch.setattr(
        module,
        "publish_output",
        lambda path, settings, request_id: {
            "image_base64": "bW9jaw==",
            "mime_type": "image/png",
            "extension": "png",
            "size_bytes": path.stat().st_size,
            "private_output_only": True,
            "qa_required": True,
        },
    )

    event = json.loads(FIXTURE.read_text(encoding="utf-8"))
    response = module.handler(event)
    assert response["contract_version"] == "privacy-production-spec-v1"
    assert response["engine"] == "flux-1-schnell"
    assert response["workflow_id"] == "flux-schnell-t2i-v1"
    assert response["selected_reference_tag"] == "nsfw_closeup_front"
    assert response["private_output_only"] is True
    assert response["qa_required"] is True


class SyntheticProcessManager:
    def ensure_started(self, request_id: str) -> None:
        assert request_id == "req-klein-t2i-001"


class SyntheticClient:
    def queue_prompt(self, prompt, request_id: str) -> str:
        assert request_id == "req-klein-t2i-001"
        assert "200" not in prompt
        assert prompt["62"]["inputs"]["width"] == 1024
        assert prompt["66"]["inputs"]["height"] == 1024
        assert prompt["63"]["inputs"]["positive"] == ["74", 0]
        return "prompt-klein-001"

    def wait_for_history(self, prompt_id: str, request_id: str):
        assert prompt_id == "prompt-klein-001"
        return {
            "outputs": {
                "201": {
                    "images": [
                        {
                            "filename": "mock-klein.png",
                            "type": "output",
                        }
                    ]
                }
            }
        }

    def download_output(
        self,
        *,
        record,
        output_nodes,
        destination: Path,
        request_id: str,
    ):
        assert output_nodes == ("201",)

        output = destination.with_suffix(".png")
        output.write_bytes(b"mock-klein-png")

        return output


def test_handler_klein_synthetic_t2i_skips_reference_download(
    monkeypatch,
    tmp_path,
):
    fake_runpod = types.ModuleType("runpod")

    fake_runpod.serverless = types.SimpleNamespace(
        start=lambda payload: payload
    )

    monkeypatch.setitem(
        sys.modules,
        "runpod",
        fake_runpod,
    )

    monkeypatch.setenv(
        "APP_ROOT",
        str(ROOT),
    )

    monkeypatch.setenv(
        "WORKFLOW_ROOT",
        str(ROOT / "workflows"),
    )

    monkeypatch.setenv(
        "RUNTIME_ROOT",
        str(tmp_path / "runtime"),
    )

    monkeypatch.setenv(
        "COMFYUI_ROOT",
        str(tmp_path / "ComfyUI"),
    )

    monkeypatch.setenv(
        "SKIP_MODEL_VALIDATION",
        "true",
    )

    for name in [
        "handler",
        "privacy_worker.config",
    ]:
        sys.modules.pop(name, None)

    module = importlib.import_module("handler")

    def forbidden_download_media(**kwargs):
        raise AssertionError(
            "Synthetic Klein T2I must not download identity reference."
        )

    monkeypatch.setattr(
        module,
        "download_media",
        forbidden_download_media,
    )

    monkeypatch.setattr(
        module,
        "_process_manager",
        SyntheticProcessManager(),
    )

    monkeypatch.setattr(
        module,
        "_client",
        SyntheticClient(),
    )

    monkeypatch.setattr(
        module,
        "publish_output",
        lambda path, settings, request_id: {
            "image_base64": "bW9jay1rbGVpbg==",
            "mime_type": "image/png",
            "extension": "png",
            "size_bytes": path.stat().st_size,
            "private_output_only": True,
            "qa_required": True,
        },
    )

    fixture = (
        ROOT
        / "tests"
        / "fixtures"
        / "image_klein_t2i_synthetic.json"
    )

    event = json.loads(
        fixture.read_text(encoding="utf-8")
    )

    response = module.handler(event)

    assert response["contract_version"] == "privacy-production-spec-v1"
    assert response["engine"] == "flux-2-klein"
    assert response["workflow_id"] == "flux2-klein-4b-t2i-distilled-v1"
    assert response["selected_reference_tag"] is None
    assert response["private_output_only"] is True
    assert response["qa_required"] is True
