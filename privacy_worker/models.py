from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class IdentityReference:
    url: str
    system_tag: str
    media_type: str
    content_type: str
    source_scope: str
    slot_index: int = 1
    actor_profile_id: str | None = None
    is_primary: bool = False


@dataclass(frozen=True)
class ProductionRequest:
    request_id: str
    contract_version: str
    engine: str
    task: str
    identity_mode: str
    positive_prompt: str
    negative_prompt: str
    reference_image_url: str | None
    selected_reference_tag: str | None
    width: int
    height: int
    steps: int
    guidance_scale: float
    seed: int | None
    workflow_id: str
    workflow_version: str
    graph_override: dict[str, Any] | None
    metadata: dict[str, Any]
    references: tuple[IdentityReference, ...]


@dataclass(frozen=True)
class PreparedWorkflow:
    workflow_id: str
    workflow_version: str
    prompt: dict[str, Any]
    output_nodes: tuple[str, ...]
