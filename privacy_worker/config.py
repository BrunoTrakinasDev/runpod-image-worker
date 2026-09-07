from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(int(os.getenv(name, str(default))), minimum)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    app_root: Path = Path(os.getenv("APP_ROOT", "/app"))
    comfyui_root: Path = Path(os.getenv("COMFYUI_ROOT", "/opt/ComfyUI"))
    workflow_root: Path = Path(os.getenv("WORKFLOW_ROOT", "/app/workflows"))
    runtime_root: Path = Path(os.getenv("RUNTIME_ROOT", "/runpod-volume/privacy-flux-runtime"))
    model_root: Path = Path(os.getenv("MODEL_ROOT", "/runpod-volume/models"))
    comfyui_host: str = os.getenv("COMFYUI_HOST", "127.0.0.1")
    comfyui_port: int = _int("COMFYUI_PORT", 8188)
    comfyui_start_local: bool = _bool("COMFYUI_START_LOCAL", True)
    comfyui_start_timeout_seconds: int = _int("COMFYUI_START_TIMEOUT_SECONDS", 180)
    comfyui_job_timeout_seconds: int = _int("COMFYUI_JOB_TIMEOUT_SECONDS", 2400)
    comfyui_poll_interval_seconds: int = _int("COMFYUI_POLL_INTERVAL_SECONDS", 3)
    download_timeout_seconds: int = _int("DOWNLOAD_TIMEOUT_SECONDS", 120)
    max_image_download_mb: int = _int("MAX_IMAGE_DOWNLOAD_MB", 80)
    max_output_mb: int = _int("MAX_OUTPUT_MB", 80)
    max_base64_return_mb: int = _int("MAX_BASE64_RETURN_MB", 20)
    allow_private_download_hosts: bool = _bool("ALLOW_PRIVATE_DOWNLOAD_HOSTS", False)
    skip_model_validation: bool = _bool("SKIP_MODEL_VALIDATION", False)
    media_allowed_hosts: tuple[str, ...] = tuple(
        item.strip().lower()
        for item in os.getenv("MEDIA_ALLOWED_HOSTS", "").split(",")
        if item.strip()
    )
    output_mode: str = os.getenv("OUTPUT_MODE", "auto").strip().lower()

    model_source_mode: str = os.getenv(
        "MODEL_SOURCE_MODE",
        "local_volume",
    ).strip().lower()

    model_manifest_root: Path = Path(
        os.getenv(
            "MODEL_MANIFEST_ROOT",
            "/app/model_manifests",
        )
    )

    model_registry_r2_endpoint_url: str = os.getenv(
        "MODEL_REGISTRY_R2_ENDPOINT_URL",
        "",
    ).strip()

    model_registry_r2_access_key_id: str = os.getenv(
        "MODEL_REGISTRY_R2_ACCESS_KEY_ID",
        "",
    ).strip()

    model_registry_r2_secret_access_key: str = os.getenv(
        "MODEL_REGISTRY_R2_SECRET_ACCESS_KEY",
        "",
    ).strip()

    model_registry_r2_bucket_name: str = os.getenv(
        "MODEL_REGISTRY_R2_BUCKET_NAME",
        "ia-adulta-model-registry",
    ).strip()

    model_registry_r2_prefix: str = os.getenv(
        "MODEL_REGISTRY_R2_PREFIX",
        "image/flux-2-klein-4b/official-distilled-4b-v1",
    ).strip().strip("/")

    r2_endpoint_url: str = os.getenv("R2_ENDPOINT_URL", "").strip()
    r2_access_key_id: str = os.getenv("R2_ACCESS_KEY_ID", "").strip()
    r2_secret_access_key: str = os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
    r2_bucket_name: str = os.getenv("R2_BUCKET_NAME", "").strip()
    r2_prefix: str = os.getenv("R2_PREFIX", "flux/private-tmp").strip().strip("/")
    r2_signed_url_ttl_seconds: int = _int("R2_SIGNED_URL_TTL_SECONDS", 900)
    flux_schnell_model_name: str = os.getenv("FLUX_SCHNELL_MODEL_NAME", "flux1-schnell.safetensors")
    flux_dev_model_name: str = os.getenv("FLUX_DEV_MODEL_NAME", "flux1-dev.safetensors")
    flux2_klein_model_name: str = os.getenv("FLUX2_KLEIN_MODEL_NAME", "flux-2-klein-4b-fp8.safetensors")
    flux2_text_encoder_name: str = os.getenv("FLUX2_TEXT_ENCODER_NAME", "qwen_3_4b.safetensors")
    flux2_vae_name: str = os.getenv("FLUX2_VAE_NAME", "flux2-vae.safetensors")
    clip_l_name: str = os.getenv("FLUX_CLIP_L_NAME", "clip_l.safetensors")
    t5xxl_name: str = os.getenv("FLUX_T5XXL_NAME", "t5xxl_fp16.safetensors")
    vae_name: str = os.getenv("FLUX_VAE_NAME", "ae.safetensors")
    clip_vision_name: str = os.getenv("FLUX_CLIP_VISION_NAME", "sigclip_vision_patch14_384.safetensors")
    pulid_model_name: str = os.getenv("FLUX_PULID_MODEL_NAME", "pulid_flux_v0.9.1.safetensors")
    pulid_strength: float = float(os.getenv("FLUX_PULID_STRENGTH", "0.82"))

    @property
    def comfyui_base_url(self) -> str:
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @property
    def input_dir(self) -> Path:
        return self.runtime_root / "input"

    @property
    def output_dir(self) -> Path:
        return self.runtime_root / "output"

    @property
    def temp_dir(self) -> Path:
        return self.runtime_root / "temp"

    @property
    def r2_configured(self) -> bool:
        return all([self.r2_endpoint_url, self.r2_access_key_id, self.r2_secret_access_key, self.r2_bucket_name])

    def ensure_runtime_dirs(self) -> None:
        for path in (self.runtime_root, self.input_dir, self.output_dir, self.temp_dir):
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
