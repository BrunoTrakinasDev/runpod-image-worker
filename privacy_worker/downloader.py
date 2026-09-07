from __future__ import annotations

import ipaddress
import socket
from pathlib import Path
from urllib.parse import urlparse

import requests

from .config import Settings
from .errors import DownloadError
from .telemetry import log_event


def _validate_host(url: str, settings: Settings) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        raise DownloadError("A URL de mídia deve usar http ou https.")
    host = (parsed.hostname or "").strip().lower()
    if not host:
        raise DownloadError("A URL de mídia não possui host válido.")

    if settings.media_allowed_hosts and host not in settings.media_allowed_hosts:
        raise DownloadError("Host não autorizado para download de mídia.", details={"host": host})

    if settings.allow_private_download_hosts:
        return

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as error:
        raise DownloadError("Falha ao resolver o host da mídia.", details={"host": host}) from error

    for family, _, _, _, sockaddr in infos:
        if family not in {socket.AF_INET, socket.AF_INET6}:
            continue
        ip = ipaddress.ip_address(sockaddr[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
            raise DownloadError("Host privado não permitido para download de mídia.", details={"host": host, "ip": str(ip)})


def download_media(
    *,
    url: str,
    destination_dir: Path,
    stem: str,
    fallback_extension: str,
    settings: Settings,
    request_id: str,
) -> Path:
    _validate_host(url, settings)
    destination_dir.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix or fallback_extension
    if not suffix.startswith("."):
        suffix = fallback_extension
    target = destination_dir / f"{stem}{suffix}"

    max_bytes = settings.max_image_download_mb * 1024 * 1024
    downloaded = 0
    try:
        with requests.get(url, stream=True, timeout=(15, settings.download_timeout_seconds), allow_redirects=False) as response:
            response.raise_for_status()
            content_type = str(response.headers.get("Content-Type") or "").lower()
            if content_type and not content_type.startswith("image/"):
                raise DownloadError("A referência biométrica precisa ser uma imagem.", details={"content_type": content_type})
            with target.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise DownloadError(
                            f"Download excedeu MAX_IMAGE_DOWNLOAD_MB={settings.max_image_download_mb}."
                        )
                    handle.write(chunk)
    except DownloadError:
        raise
    except Exception as error:
        raise DownloadError("Falha ao baixar a referência biométrica.") from error

    if downloaded == 0:
        raise DownloadError("A referência biométrica foi baixada vazia.")

    log_event("media_downloaded", request_id=request_id, size_bytes=downloaded, destination=str(target))
    return target
