#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_ROOT = ROOT / "workflows"
EXPECTED = {
    "flux-schnell-t2i-v1.json": "flux-1-schnell",
    "flux-dev-t2i-v1.json": "flux-1-dev",
    "flux2-klein-4b-image-edit-distilled-v1.json": "flux-2-klein",
    "flux2-klein-4b-t2i-distilled-v1.json": "flux-2-klein",
}


def main() -> int:
    for filename, engine in EXPECTED.items():
        path = WORKFLOW_ROOT / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == "privacy-comfyui-workflow-v1"
        assert payload["engine"] == engine
        assert str(payload["workflow_version"]) == "1"
        assert isinstance(payload["prompt"], dict) and payload["prompt"]
        assert isinstance(payload["bindings"], dict) and payload["bindings"]
        assert isinstance(payload["output_nodes"], list) and payload["output_nodes"]
        for output_node in payload["output_nodes"]:
            assert str(output_node) in payload["prompt"]
    print(json.dumps({"status": "FLUX_WORKFLOWS_VALID", "workflows": sorted(EXPECTED)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
