from __future__ import annotations

import base64
import uuid
from pathlib import Path

import boto3

from .config import Settings
from .errors import OutputError
from .telemetry import log_event


def _mime_from_suffix(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".jpg" or suffix == ".jpeg":
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _upload_private_r2(path: Path, settings: Settings, request_id: str) -> dict:
    client = boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint_url,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        region_name="auto",
    )
    key = f"{settings.r2_prefix}/{request_id}/{uuid.uuid4().hex}{path.suffix.lower() or '.png'}"
    content_type = _mime_from_suffix(path)
    try:
        with path.open("rb") as body:
            client.put_object(
                Bucket=settings.r2_bucket_name,
                Key=key,
                Body=body,
                ContentType=content_type,
                CacheControl="private, no-store",
                Metadata={
                    "private": "true",
                    "qa_required": "true",
                    "contract": "privacy-production-spec-v1",
                },
            )
        signed_url = client.generate_presigned_url(
            "get_object",
            Params={"Bucket": settings.r2_bucket_name, "Key": key},
            ExpiresIn=settings.r2_signed_url_ttl_seconds,
        )
    except Exception as error:
        raise OutputError("Falha ao enviar saída privada ao R2.") from error
    log_event("private_output_uploaded", request_id=request_id, r2_bucket=settings.r2_bucket_name, r2_key=key, size_bytes=path.stat().st_size)
    return {"image_url": signed_url, "url": signed_url, "r2_bucket": settings.r2_bucket_name, "r2_key": key, "storage_private": True}


def publish_output(path: Path, settings: Settings, request_id: str) -> dict:
    size_bytes = path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)
    mode = settings.output_mode
    if mode not in {"auto", "base64", "private_r2"}:
        raise OutputError("OUTPUT_MODE inválido; use auto, base64 ou private_r2.")

    if mode == "private_r2" or (mode == "auto" and settings.r2_configured and size_mb > settings.max_base64_return_mb):
        if not settings.r2_configured:
            raise OutputError("OUTPUT_MODE=private_r2 exige credenciais R2 privadas.")
        payload = _upload_private_r2(path, settings, request_id)
    else:
        if size_mb > settings.max_base64_return_mb:
            raise OutputError(f"Imagem tem {size_mb:.2f} MB. Configure R2 privado ou aumente MAX_BASE64_RETURN_MB.")
        payload = {"image_base64": base64.b64encode(path.read_bytes()).decode("ascii")}

    return {
        **payload,
        "mime_type": _mime_from_suffix(path),
        "extension": (path.suffix.lower().lstrip(".") or "png"),
        "size_bytes": size_bytes,
        "private_output_only": True,
        "qa_required": True,
    }
