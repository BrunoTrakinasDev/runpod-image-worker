from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings


MODEL_REGISTRY_MANIFEST_SCHEMA_VERSION = (
    "privacy-model-registry-manifest-v1"
)

SUPPORTED_MODEL_SOURCE_MODES = frozenset(
    {
        "local_volume",
        "r2_registry",
    }
)

ENGINE_MANIFEST_FILES = {
    "flux-2-klein":
        "flux2-klein-4b-official-distilled-v1.json",
}


class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelArtifact:
    role: str
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelManifest:
    schema_version: str
    engine: str
    model_set: str
    source_contract: str
    component_count: int
    total_bytes: int
    artifacts: tuple[ModelArtifact, ...]


@dataclass(frozen=True)
class ModelMaterializationPlanItem:
    role: str
    relative_path: str
    destination: Path
    r2_key: str
    bytes: int
    sha256: str


def _safe_relative_path(raw: Any) -> str:
    value = str(raw or "").strip().replace("\\", "/")

    if not value:
        raise ModelRegistryError(
            "Manifest contém relative_path vazio."
        )

    path = Path(value)

    if (
        path.is_absolute()
        or value.startswith("/")
        or ".." in path.parts
    ):
        raise ModelRegistryError(
            f"relative_path inseguro no manifest: {value}"
        )

    return value


def _validate_sha256(raw: Any) -> str:
    value = str(raw or "").strip().lower()

    if (
        len(value) != 64
        or any(
            character not in "0123456789abcdef"
            for character in value
        )
    ):
        raise ModelRegistryError(
            f"SHA-256 inválido no manifest: {value!r}"
        )

    return value


def _positive_int(raw: Any, *, field: str) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as error:
        raise ModelRegistryError(
            f"{field} inválido no manifest."
        ) from error

    if value <= 0:
        raise ModelRegistryError(
            f"{field} deve ser maior que zero."
        )

    return value


def _manifest_path(
    *,
    engine: str,
    settings: Settings,
) -> Path:
    filename = ENGINE_MANIFEST_FILES.get(engine)

    if not filename:
        raise ModelRegistryError(
            f"Engine sem manifest de model registry: {engine}"
        )

    return settings.model_manifest_root / filename


def load_model_manifest(
    *,
    engine: str,
    settings: Settings,
) -> ModelManifest:
    path = _manifest_path(
        engine=engine,
        settings=settings,
    )

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ModelRegistryError(
            f"Não foi possível carregar manifest: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise ModelRegistryError(
            "Manifest de modelos precisa ser um objeto JSON."
        )

    schema_version = str(
        payload.get("schema_version") or ""
    )

    if (
        schema_version
        != MODEL_REGISTRY_MANIFEST_SCHEMA_VERSION
    ):
        raise ModelRegistryError(
            "schema_version incompatível no model manifest."
        )

    manifest_engine = str(
        payload.get("engine") or ""
    ).strip()

    if manifest_engine != engine:
        raise ModelRegistryError(
            f"Manifest pertence a {manifest_engine}, não {engine}."
        )

    raw_artifacts = payload.get("artifacts")

    if (
        not isinstance(raw_artifacts, list)
        or not raw_artifacts
    ):
        raise ModelRegistryError(
            "Manifest não contém artifacts válidos."
        )

    artifacts = []

    for raw_artifact in raw_artifacts:
        if not isinstance(raw_artifact, dict):
            raise ModelRegistryError(
                "Artifact inválido no manifest."
            )

        artifacts.append(
            ModelArtifact(
                role=str(
                    raw_artifact.get("role") or ""
                ).strip(),
                relative_path=_safe_relative_path(
                    raw_artifact.get("relative_path")
                ),
                bytes=_positive_int(
                    raw_artifact.get("bytes"),
                    field="bytes",
                ),
                sha256=_validate_sha256(
                    raw_artifact.get("sha256")
                ),
            )
        )

    component_count = _positive_int(
        payload.get("component_count"),
        field="component_count",
    )

    total_bytes = _positive_int(
        payload.get("total_bytes"),
        field="total_bytes",
    )

    if component_count != len(artifacts):
        raise ModelRegistryError(
            "component_count diverge de artifacts."
        )

    calculated_total = sum(
        artifact.bytes
        for artifact in artifacts
    )

    if calculated_total != total_bytes:
        raise ModelRegistryError(
            "total_bytes diverge da soma dos artifacts."
        )

    relative_paths = [
        artifact.relative_path
        for artifact in artifacts
    ]

    if len(relative_paths) != len(set(relative_paths)):
        raise ModelRegistryError(
            "Manifest contém relative_path duplicado."
        )

    return ModelManifest(
        schema_version=schema_version,
        engine=manifest_engine,
        model_set=str(
            payload.get("model_set") or ""
        ).strip(),
        source_contract=str(
            payload.get("source_contract") or ""
        ).strip(),
        component_count=component_count,
        total_bytes=total_bytes,
        artifacts=tuple(artifacts),
    )


def build_model_materialization_plan(
    *,
    engine: str,
    settings: Settings,
) -> tuple[ModelMaterializationPlanItem, ...]:
    manifest = load_model_manifest(
        engine=engine,
        settings=settings,
    )

    prefix = settings.model_registry_r2_prefix.strip(
        "/"
    )

    if not prefix:
        raise ModelRegistryError(
            "MODEL_REGISTRY_R2_PREFIX vazio."
        )

    plan = []

    for artifact in manifest.artifacts:
        relative_path = artifact.relative_path

        destination = (
            settings.model_root /
            Path(relative_path)
        )

        key = (
            f"{prefix}/{relative_path}"
            .replace("\\", "/")
        )

        plan.append(
            ModelMaterializationPlanItem(
                role=artifact.role,
                relative_path=relative_path,
                destination=destination,
                r2_key=key,
                bytes=artifact.bytes,
                sha256=artifact.sha256,
            )
        )

    return tuple(plan)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(8 * 1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def _file_matches(
    path: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
) -> bool:
    if not path.is_file():
        return False

    if path.stat().st_size != expected_bytes:
        return False

    return (
        _sha256_file(path)
        == expected_sha256
    )


def _validate_registry_credentials(
    settings: Settings,
) -> None:
    values = {
        "MODEL_REGISTRY_R2_ENDPOINT_URL":
            settings.model_registry_r2_endpoint_url,
        "MODEL_REGISTRY_R2_ACCESS_KEY_ID":
            settings.model_registry_r2_access_key_id,
        "MODEL_REGISTRY_R2_SECRET_ACCESS_KEY":
            settings.model_registry_r2_secret_access_key,
        "MODEL_REGISTRY_R2_BUCKET_NAME":
            settings.model_registry_r2_bucket_name,
    }

    missing = [
        name
        for name, value in values.items()
        if not str(value or "").strip()
    ]

    if missing:
        raise ModelRegistryError(
            "Credenciais/configuração do model registry "
            "incompletas: " +
            ",".join(sorted(missing))
        )


def _build_r2_client(settings: Settings):
    # Lazy import: local_volume and static tests do not
    # initialize boto3 or touch the network.
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=
            settings.model_registry_r2_endpoint_url,
        aws_access_key_id=
            settings.model_registry_r2_access_key_id,
        aws_secret_access_key=
            settings.model_registry_r2_secret_access_key,
        region_name="auto",
    )


def _materialize_one(
    *,
    client,
    settings: Settings,
    item: ModelMaterializationPlanItem,
) -> None:
    destination = item.destination

    if destination.exists():
        if _file_matches(
            destination,
            expected_bytes=item.bytes,
            expected_sha256=item.sha256,
        ):
            return

        raise ModelRegistryError(
            "Arquivo de modelo existente diverge do manifest: "
            f"{destination}"
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = destination.with_name(
        destination.name +
        f".part-{uuid.uuid4().hex}"
    )

    try:
        with temp_path.open("xb") as handle:
            client.download_fileobj(
                settings.model_registry_r2_bucket_name,
                item.r2_key,
                handle,
            )

        if temp_path.stat().st_size != item.bytes:
            raise ModelRegistryError(
                "Tamanho baixado diverge do manifest: "
                f"{item.relative_path}"
            )

        actual_sha = _sha256_file(temp_path)

        if actual_sha != item.sha256:
            raise ModelRegistryError(
                "SHA-256 baixado diverge do manifest: "
                f"{item.relative_path}"
            )

        os.replace(
            temp_path,
            destination,
        )

    finally:
        if temp_path.exists():
            temp_path.unlink()


def prepare_model_storage(
    *,
    engine: str,
    settings: Settings,
) -> tuple[Path, ...]:
    mode = settings.model_source_mode

    if mode not in SUPPORTED_MODEL_SOURCE_MODES:
        raise ModelRegistryError(
            f"MODEL_SOURCE_MODE inválido: {mode}"
        )

    plan = build_model_materialization_plan(
        engine=engine,
        settings=settings,
    )

    if mode == "local_volume":
        return tuple(
            item.destination
            for item in plan
        )

    _validate_registry_credentials(settings)

    client = _build_r2_client(settings)

    for item in plan:
        _materialize_one(
            client=client,
            settings=settings,
            item=item,
        )

    return tuple(
        item.destination
        for item in plan
    )