import json
from dataclasses import replace
from pathlib import Path

from privacy_worker.config import Settings
from privacy_worker.contracts import parse_production_request
from privacy_worker.workflows import prepare_workflow

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def settings():
    return replace(Settings(), workflow_root=ROOT / "workflows")


def test_schnell_workflow_injects_reference_prompt_and_sampling():
    request = parse_production_request(load("image_closeup.json"))
    prepared = prepare_workflow(
        request=request,
        reference_image_filename="identity.png",
        output_prefix="privacy/test/flux",
        settings=settings(),
    )
    prompt = prepared.prompt
    assert prepared.workflow_id == "flux-schnell-t2i-v1"
    assert prompt["4"]["inputs"]["text"] == request.positive_prompt
    assert prompt["5"]["inputs"]["text"] == request.negative_prompt
    assert prompt["6"]["inputs"]["image"] == "identity.png"
    assert prompt["10"]["inputs"]["width"] == 1024
    assert prompt["10"]["inputs"]["height"] == 1360
    assert prompt["11"]["inputs"]["guidance"] == request.guidance_scale
    assert prompt["13"]["inputs"]["noise_seed"] == 42
    assert prompt["14"]["inputs"]["steps"] == 10
    assert prompt["17"]["inputs"]["filename_prefix"] == "privacy/test/flux"
    assert prepared.output_nodes == ("17",)


def test_dev_workflow_uses_dev_model_name():
    request = parse_production_request(load("image_dev.json"))
    prepared = prepare_workflow(
        request=request,
        reference_image_filename="body.png",
        output_prefix="privacy/test/dev",
        settings=settings(),
    )
    prompt = prepared.prompt
    assert prompt["1"]["inputs"]["unet_name"] == settings().flux_dev_model_name
    assert prompt["6"]["inputs"]["image"] == "body.png"


def test_klein_synthetic_t2i_workflow_has_zero_reference_graph():
    request = parse_production_request(
        load("image_klein_t2i_synthetic.json")
    )

    prepared = prepare_workflow(
        request=request,
        reference_image_filename=None,
        output_prefix="privacy/test/klein-t2i",
        settings=settings(),
    )

    prompt = prepared.prompt

    assert prepared.workflow_id == "flux2-klein-4b-t2i-distilled-v1"
    assert prepared.output_nodes == ("201",)

    assert prompt["70"]["inputs"]["unet_name"] == settings().flux2_klein_model_name
    assert prompt["71"]["inputs"]["clip_name"] == settings().flux2_text_encoder_name
    assert prompt["72"]["inputs"]["vae_name"] == settings().flux2_vae_name

    assert prompt["74"]["inputs"]["text"] == request.positive_prompt

    assert prompt["62"]["inputs"]["width"] == 1024
    assert prompt["62"]["inputs"]["height"] == 1024
    assert prompt["66"]["inputs"]["width"] == 1024
    assert prompt["66"]["inputs"]["height"] == 1024

    assert prompt["62"]["inputs"]["steps"] == 4
    assert prompt["63"]["inputs"]["cfg"] == 1.0
    assert prompt["73"]["inputs"]["noise_seed"] == 42

    assert prompt["63"]["inputs"]["positive"] == ["74", 0]
    assert prompt["63"]["inputs"]["negative"] == ["82", 0]

    assert prompt["201"]["inputs"]["filename_prefix"] == "privacy/test/klein-t2i"

    class_types = {
        str(node.get("class_type"))
        for node in prompt.values()
        if isinstance(node, dict)
    }

    assert "LoadImage" not in class_types
    assert "ReferenceLatent" not in class_types
    assert "VAEEncode" not in class_types
    assert "GetImageSize" not in class_types
    assert "ImageScaleToTotalPixels" not in class_types
