from __future__ import annotations

from typing import Any

from .errors import ContractError
from .models import IdentityReference, ProductionRequest

CONTRACT_VERSION = "privacy-production-spec-v1"

ENGINE_DEFAULTS = {
    "flux-1-schnell": {
        "workflow_id": "flux-schnell-t2i-v1",
        "steps": 8,
        "guidance_scale": 1.0,
    },
    "flux-1-dev": {
        "workflow_id": "flux-dev-t2i-v1",
        "steps": 28,
        "guidance_scale": 3.5,
    },
    "flux-2-klein": {
        "workflow_id": "flux2-klein-4b-image-edit-distilled-v1",
        "steps": 4,
        "guidance_scale": 1.0,
    },
}

SUPPORTED_ENGINES = frozenset(ENGINE_DEFAULTS)
EXPECTED_TASK = "image.generate"
COMFYUI_ADAPTER_VERSION = "comfyui-graph-contract-v1"
APPROVED_ACTOR_IDENTITY_MODE = "approved_actor_identity"
SYNTHETIC_PROMPT_IDENTITY_MODE = "synthetic_prompt_definition"
SYNTHETIC_KLEIN_WORKFLOW_ALIAS = "flux-ai-avatar-base-v1"
SYNTHETIC_KLEIN_WORKFLOW_ID = "flux2-klein-4b-t2i-distilled-v1"

CLOSEUP_PRIORITY = (
    "nsfw_closeup_front",
    "face_front",
    "nsfw_closeup_back",
    "face_profile",
    "body_front",
    "body_back",
)
DEFAULT_PRIORITY = (
    "face_front",
    "face_profile",
    "nsfw_closeup_front",
    "nsfw_closeup_back",
    "body_front",
    "body_back",
)
CLOSEUP_HINTS = (
    "close-up",
    "close up",
    "closeup",
    "macro",
    "portrait",
    "headshot",
    "rosto",
    "busto",
    "primeiro plano",
)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _number(value: Any, default: int | float, *, minimum: float, maximum: float):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    parsed = min(max(parsed, minimum), maximum)
    if isinstance(default, int):
        return int(parsed)
    return parsed


def _multiple_of_16(value: Any, default: int) -> int:
    parsed = _number(value, default, minimum=256, maximum=4096)
    return max(256, (int(parsed) // 16) * 16)


def _unwrap_payload(event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("input", event)
    if not isinstance(payload, dict):
        raise ContractError("O input do job precisa ser um objeto JSON.")
    nested = payload.get("production_spec") or payload.get("productionSpec")
    if isinstance(nested, dict):
        return nested
    return payload


def _collect_references(spec: dict[str, Any]) -> tuple[IdentityReference, ...]:
    identity = spec.get("identity") or {}
    references: list[IdentityReference] = []

    flat_refs = identity.get("references") or []
    for item in flat_refs if isinstance(flat_refs, list) else []:
        url = _text(item.get("signed_url") or item.get("url"))
        media_type = _text(item.get("media_type") or "image").lower()
        content_type = _text(item.get("content_type") or "image/png").lower()
        if not url or media_type != "image" or (content_type and not content_type.startswith("image/")):
            continue
        references.append(
            IdentityReference(
                url=url,
                system_tag=_text(item.get("system_tag")),
                media_type=media_type,
                content_type=content_type,
                source_scope="identity.references",
                is_primary=False,
            )
        )

    actors = identity.get("actors") or []
    for actor in actors if isinstance(actors, list) else []:
        actor_refs = actor.get("references") or []
        primary = _text(actor.get("primary_reference_url"))
        for item in actor_refs if isinstance(actor_refs, list) else []:
            url = _text(item.get("signed_url") or item.get("url"))
            media_type = _text(item.get("media_type") or "image").lower()
            content_type = _text(item.get("content_type") or "image/png").lower()
            if not url or media_type != "image" or (content_type and not content_type.startswith("image/")):
                continue
            references.append(
                IdentityReference(
                    url=url,
                    system_tag=_text(item.get("system_tag")),
                    media_type=media_type,
                    content_type=content_type,
                    source_scope="identity.actors.references",
                    slot_index=int(actor.get("slot_index") or 1),
                    actor_profile_id=_text(actor.get("actor_profile_id")) or None,
                    is_primary=bool(primary and primary == url),
                )
            )


    seen: set[str] = set()
    unique: list[IdentityReference] = []
    for item in references:
        if item.url in seen:
            continue
        seen.add(item.url)
        unique.append(item)
    return tuple(unique)


def _is_closeup_context(spec: dict[str, Any]) -> bool:
    parts = []
    prompt = spec.get("prompt") or {}
    camera = spec.get("camera") or {}
    action = spec.get("action") or {}
    metadata = spec.get("metadata") or {}
    parts.extend([
        _text(prompt.get("positive")),
        _text(camera.get("shot")),
        _text(camera.get("angle")),
        _text(action.get("label")),
        _text(action.get("technical_prompt")),
        _text(metadata.get("framing")),
    ])
    merged = " ".join(parts).lower()
    return any(token in merged for token in CLOSEUP_HINTS)


def _pick_reference(refs: tuple[IdentityReference, ...], closeup_context: bool) -> IdentityReference:
    priority = {tag: idx for idx, tag in enumerate(CLOSEUP_PRIORITY if closeup_context else DEFAULT_PRIORITY)}
    ordered = sorted(
        refs,
        key=lambda item: (
            0 if item.is_primary else 1,
            int(item.slot_index or 1),
            priority.get(item.system_tag, 999),
        ),
    )
    return ordered[0]


def parse_production_request(event: dict[str, Any]) -> ProductionRequest:
    spec = _unwrap_payload(event)

    contract_version = _text(spec.get("contract_version"))
    if contract_version != CONTRACT_VERSION:
        raise ContractError(
            f"contract_version inv?lido: esperado {CONTRACT_VERSION}.",
            details={"received": contract_version or None},
        )

    engine = _text(spec.get("engine")).lower()
    engine_defaults = ENGINE_DEFAULTS.get(engine)
    if engine_defaults is None:
        raise ContractError(
            "Engine de imagem n?o suportado.",
            details={"engine": engine, "supported": sorted(SUPPORTED_ENGINES)},
        )

    task = _text(spec.get("task"))
    if task != EXPECTED_TASK:
        raise ContractError(
            "Task incompat?vel para worker de imagem.",
            details={"task": task, "expected": EXPECTED_TASK},
        )

    safety = spec.get("safety") or {}

    common_required_flags = (
        "licensed_or_consented_assets_only",
        "private_output_only",
        "public_url_forbidden",
        "qa_required",
    )

    missing_flags = [
        name
        for name in common_required_flags
        if safety.get(name) is not True
    ]

    if missing_flags:
        raise ContractError(
            "As flags obrigat?rias de seguran?a n?o foram atendidas.",
            details={"missing_flags": missing_flags},
        )

    prompt = spec.get("prompt") or {}
    positive_prompt = _text(prompt.get("positive"))

    if not positive_prompt:
        raise ContractError(
            "Prompt positivo obrigat?rio para gera??o Flux."
        )

    negative_prompt = _text(prompt.get("negative"))

    identity = spec.get("identity") or {}

    if identity.get("strict") is not True:
        raise ContractError(
            "O worker de imagem exige identity.strict=true."
        )

    identity_mode = (
        _text(identity.get("mode")).lower()
        or APPROVED_ACTOR_IDENTITY_MODE
    )

    synthetic_identity = (
        identity_mode == SYNTHETIC_PROMPT_IDENTITY_MODE
    )

    references = _collect_references(spec)
    selected = None

    if synthetic_identity:
        if engine != "flux-2-klein":
            raise ContractError(
                "Avatar IA sint?tico exige engine flux-2-klein.",
                details={"engine": engine},
            )

        if safety.get("require_approved_identity_references") is not False:
            raise ContractError(
                "Avatar IA sint?tico exige "
                "require_approved_identity_references=false."
            )

        if (
            safety.get(
                "require_validated_synthetic_identity_definition"
            )
            is not True
        ):
            raise ContractError(
                "Avatar IA sint?tico exige defini??o sint?tica validada."
            )

        identity_source = _text(identity.get("source")).lower()

        if identity_source != SYNTHETIC_PROMPT_IDENTITY_MODE:
            raise ContractError(
                "identity.source incompat?vel com Avatar IA sint?tico."
            )

        if references:
            raise ContractError(
                "Avatar IA sint?tico n?o pode carregar refer?ncias "
                "biom?tricas humanas."
            )

        definition_id = _text(
            identity.get("ai_avatar_definition_id")
            or identity.get("aiAvatarDefinitionId")
        )

        definition_key = _text(
            identity.get("definition_key")
            or identity.get("definitionKey")
        )

        appearance_snapshot = (
            identity.get("appearance_snapshot")
            or identity.get("appearanceSnapshot")
        )

        if not definition_id or not definition_key:
            raise ContractError(
                "Avatar IA sint?tico exige definition id/key expl?citos."
            )

        if (
            not isinstance(appearance_snapshot, dict)
            or not appearance_snapshot
        ):
            raise ContractError(
                "Avatar IA sint?tico exige appearance_snapshot validado."
            )

    else:
        if safety.get("require_approved_identity_references") is not True:
            raise ContractError(
                "Produ??o com identidade humana exige "
                "require_approved_identity_references=true."
            )

        if not references:
            raise ContractError(
                "O job de imagem exige refer?ncias biom?tricas "
                "aprovadas do Cofre do Ator."
            )

        selected = _pick_reference(
            references,
            _is_closeup_context(spec),
        )

    comfyui = spec.get("comfyui") or {}

    adapter = _text(
        comfyui.get("adapter")
        or COMFYUI_ADAPTER_VERSION
    )

    if adapter != COMFYUI_ADAPTER_VERSION:
        raise ContractError(
            "Adapter ComfyUI incompat?vel.",
            details={"adapter": adapter},
        )

    requested_workflow_id = _text(
        comfyui.get("workflow_id")
    )

    if synthetic_identity:
        workflow_id = (
            requested_workflow_id
            or SYNTHETIC_KLEIN_WORKFLOW_ALIAS
        )

        allowed_synthetic_workflows = {
            SYNTHETIC_KLEIN_WORKFLOW_ALIAS,
            SYNTHETIC_KLEIN_WORKFLOW_ID,
        }

        if workflow_id not in allowed_synthetic_workflows:
            raise ContractError(
                "Avatar IA sint?tico exige workflow Klein T2I expl?cito.",
                details={"workflow_id": workflow_id},
            )
    else:
        workflow_id = (
            requested_workflow_id
            or str(engine_defaults["workflow_id"])
        )

        if workflow_id in {
            SYNTHETIC_KLEIN_WORKFLOW_ALIAS,
            SYNTHETIC_KLEIN_WORKFLOW_ID,
        }:
            raise ContractError(
                "Workflow Klein T2I sint?tico n?o pode ser usado "
                "no caminho de identidade humana."
            )

    workflow_version = (
        _text(comfyui.get("workflow_version"))
        or "1"
    )

    graph_override = (
        comfyui.get("graph")
        if isinstance(comfyui.get("graph"), dict)
        else None
    )

    output = spec.get("output") or {}

    width = _multiple_of_16(
        output.get("width"),
        1024,
    )

    height = _multiple_of_16(
        output.get("height"),
        1024,
    )

    sampling = spec.get("sampling") or {}

    steps = _number(
        sampling.get("steps"),
        engine_defaults["steps"],
        minimum=1,
        maximum=150,
    )

    guidance_scale = _number(
        sampling.get("guidance_scale"),
        engine_defaults["guidance_scale"],
        minimum=0.0,
        maximum=30.0,
    )

    seed = sampling.get("seed")

    if seed is not None:
        try:
            seed = int(seed)
        except (TypeError, ValueError):
            seed = None

    request_id = (
        _text(spec.get("request_id"))
        or "flux-job"
    )

    metadata = (
        spec.get("metadata")
        if isinstance(spec.get("metadata"), dict)
        else {}
    )

    return ProductionRequest(
        request_id=request_id,
        contract_version=contract_version,
        engine=engine,
        task=task,
        identity_mode=identity_mode,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        reference_image_url=(
            selected.url
            if selected is not None
            else None
        ),
        selected_reference_tag=(
            selected.system_tag
            if selected is not None
            else None
        ),
        width=width,
        height=height,
        steps=int(steps),
        guidance_scale=float(guidance_scale),
        seed=seed,
        workflow_id=workflow_id,
        workflow_version=workflow_version,
        graph_override=graph_override,
        metadata=metadata,
        references=references,
    )
