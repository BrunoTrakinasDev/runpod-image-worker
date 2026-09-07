import json
from pathlib import Path

import pytest

from privacy_worker.contracts import parse_production_request
from privacy_worker.errors import ContractError

FIXTURES = Path(__file__).parent / "fixtures"


def load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_closeup_contract_prioritizes_specialized_nsfw_reference_and_alias_workflow():
    request = parse_production_request(load("image_closeup.json"))
    assert request.engine == "flux-1-schnell"
    assert request.reference_image_url.endswith("kyc-close.png")
    assert request.selected_reference_tag == "nsfw_closeup_front"
    assert request.width == 1024
    assert request.height == 1360
    assert request.workflow_id == "flux-image-v1"
    assert request.steps == 10


def test_dev_contract_reads_actor_reference_and_dimensions():
    request = parse_production_request(load("image_dev.json"))
    assert request.engine == "flux-1-dev"
    assert request.reference_image_url.endswith("body-front.png")
    assert request.selected_reference_tag == "body_front"
    assert request.width == 1152
    assert request.height == 896
    assert request.workflow_id == "flux-dev-t2i-v1"


def test_contract_rejects_missing_safety_flags():
    payload = load("image_closeup.json")
    payload["input"]["safety"]["qa_required"] = False
    with pytest.raises(ContractError):
        parse_production_request(payload)


def test_contract_rejects_without_identity_references():
    payload = load("image_closeup.json")
    payload["input"]["identity"]["references"] = []
    with pytest.raises(ContractError):
        parse_production_request(payload)


def test_klein_synthetic_contract_allows_zero_reference():
    request = parse_production_request(
        load("image_klein_t2i_synthetic.json")
    )

    assert request.engine == "flux-2-klein"
    assert request.identity_mode == "synthetic_prompt_definition"
    assert request.reference_image_url is None
    assert request.selected_reference_tag is None
    assert request.references == ()
    assert request.workflow_id == "flux-ai-avatar-base-v1"
    assert request.width == 1024
    assert request.height == 1024
    assert request.steps == 4
    assert request.guidance_scale == 1.0


def test_klein_synthetic_contract_rejects_human_reference_gate():
    payload = load("image_klein_t2i_synthetic.json")

    payload["input"]["safety"][
        "require_approved_identity_references"
    ] = True

    with pytest.raises(ContractError):
        parse_production_request(payload)


def test_klein_synthetic_contract_rejects_non_klein_engine():
    payload = load("image_klein_t2i_synthetic.json")
    payload["input"]["engine"] = "flux-1-schnell"

    with pytest.raises(ContractError):
        parse_production_request(payload)


def test_klein_synthetic_contract_rejects_human_reference_payload():
    payload = load("image_klein_t2i_synthetic.json")

    payload["input"]["identity"]["references"] = [
        {
            "url": "https://private.invalid/human.png",
            "media_type": "image",
            "content_type": "image/png",
            "system_tag": "face_front",
        }
    ]

    with pytest.raises(ContractError):
        parse_production_request(payload)
