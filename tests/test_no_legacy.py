from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = [
    ROOT / "handler.py",
    ROOT / "Dockerfile",
    ROOT / "requirements.txt",
    *sorted((ROOT / "privacy_worker").glob("*.py")),
]
FORBIDDEN = (
    "autopipelinefortext2image",
    "stablediffusionxlcontrolnetpipeline",
    "juggernaut-xl",
    "juggernaut_xl",
    "diffusers",
    "stabilityai/stable-diffusion-xl",
)


def test_runtime_has_no_legacy_sdxl_diffusers_dependencies_or_fallbacks():
    merged = "\n".join(path.read_text(encoding="utf-8").lower() for path in RUNTIME_FILES)
    for token in FORBIDDEN:
        assert token not in merged, f"legacy token still present: {token}"
