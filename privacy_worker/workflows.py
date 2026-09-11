from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .errors import WorkflowError
from .models import PreparedWorkflow, ProductionRequest

WORKFLOW_SCHEMA_VERSION = "privacy-comfyui-workflow-v1"
SYNTHETIC_PROMPT_IDENTITY_MODE = "synthetic_prompt_definition"
SYNTHETIC_KLEIN_WORKFLOW_ID = "flux2-klein-4b-t2i-distilled-v1"
WORKFLOW_ALIASES = {
    "flux-ai-avatar-base-v1": "flux2-klein-4b-t2i-distilled-v1",
    "flux-image-v1": "flux-schnell-t2i-v1",
}


def _resolve_flux_model_name(*, engine: str, settings: Settings) -> str:
    if engine == "flux-1-schnell":
        return settings.flux_schnell_model_name
    if engine == "flux-1-dev":
        return settings.flux_dev_model_name
    if engine == "flux-2-klein":
        return settings.flux2_klein_model_name
    raise WorkflowError(f"Engine sem modelo Flux explicitamente configurado: {engine}.")


def _set_single_node_input(prompt: dict[str, Any], binding: dict[str, Any], value: Any, logical_name: str) -> None:
    node_id = str(binding.get("node_id") or binding.get("node") or "")
    input_name = str(binding.get("input") or "")
    required = binding.get("required", True)
    if not node_id or not input_name:
        if required:
            raise WorkflowError(f"Binding inválido para {logical_name}.")
        return
    node = prompt.get(node_id)
    if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
        if required:
            raise WorkflowError(f"Nó {node_id} do binding {logical_name} não existe no workflow.")
        return
    node["inputs"][input_name] = value


def _set_node_input(prompt: dict[str, Any], binding: dict[str, Any] | list[dict[str, Any]], value: Any, logical_name: str) -> None:
    targets = binding if isinstance(binding, list) else [binding]
    for target in targets:
        if not isinstance(target, dict):
            raise WorkflowError(f"Binding inválido para {logical_name}.")
        _set_single_node_input(prompt, target, value, logical_name)


def _load_envelope(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise WorkflowError(f"Não foi possível carregar o workflow {path.name}.") from error
    if payload.get("schema_version") != WORKFLOW_SCHEMA_VERSION:
        raise WorkflowError(f"Workflow {path.name} usa schema_version incompatível.")
    if not isinstance(payload.get("prompt"), dict) or not payload["prompt"]:
        raise WorkflowError(f"Workflow {path.name} não contém um prompt API do ComfyUI.")
    if not isinstance(payload.get("bindings"), dict):
        raise WorkflowError(f"Workflow {path.name} não contém bindings.")
    return payload


def _override_envelope(local: dict[str, Any], graph_override: dict[str, Any] | None) -> dict[str, Any]:
    if not graph_override:
        return local
    override = copy.deepcopy(graph_override)
    if "prompt" in override:
        prompt = override.get("prompt")
        if not isinstance(prompt, dict) or not prompt:
            raise WorkflowError("comfyui.graph.prompt precisa ser um workflow em API format.")
        return {
            **local,
            **{key: value for key, value in override.items() if key != "prompt"},
            "prompt": prompt,
            "bindings": override.get("bindings") or local["bindings"],
            "output_nodes": override.get("output_nodes") or local.get("output_nodes", []),
        }
    if all(isinstance(value, dict) and "class_type" in value for value in override.values()):
        return {**local, "prompt": override}
    raise WorkflowError("comfyui.graph precisa ser um prompt API puro ou um envelope com prompt/bindings.")


def prepare_workflow(*, request: ProductionRequest, reference_image_filename: str | None, output_prefix: str, settings: Settings) -> PreparedWorkflow:
    workflow_name = WORKFLOW_ALIASES.get(request.workflow_id, request.workflow_id)
    prompt_only = request.identity_mode == "synthetic_t2i"
    if prompt_only and (request.references or request.reference_image_url or reference_image_filename is not None or request.graph_override is not None):
        raise WorkflowError("SYNTHETIC_T2I_IDENTITY_OR_GRAPH_REJECTED")
    synthetic_t2i = request.identity_mode in (SYNTHETIC_PROMPT_IDENTITY_MODE, "synthetic_t2i")
    if synthetic_t2i and workflow_name != SYNTHETIC_KLEIN_WORKFLOW_ID:
        raise WorkflowError("Avatar IA sint?tico exige workflow Klein T2I dedicado.")
    if not synthetic_t2i and workflow_name == SYNTHETIC_KLEIN_WORKFLOW_ID:
        raise WorkflowError("Workflow Klein T2I sint?tico n?o aceita identidade humana.")
    path = settings.workflow_root / f"{workflow_name}.json"
    if not path.exists():
        raise WorkflowError(f"Workflow versionado não encontrado: {request.workflow_id}.")
    envelope = _override_envelope(_load_envelope(path), request.graph_override)

    engine = str(envelope.get("engine") or "")
    if engine and engine != request.engine:
        raise WorkflowError(f"Workflow {workflow_name} pertence ao engine {engine}, não {request.engine}.")
    declared_version = str(envelope.get("workflow_version") or "")
    if declared_version and declared_version != request.workflow_version:
        raise WorkflowError(
            f"Versão do workflow incompatível: solicitado {request.workflow_version}, disponível {declared_version}."
        )

    prompt = copy.deepcopy(envelope["prompt"])
    model_name = _resolve_flux_model_name(engine=request.engine, settings=settings)
    values = {
        "positive_prompt": request.positive_prompt,
        "negative_prompt": request.negative_prompt,
        "reference_image": reference_image_filename,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg": request.guidance_scale,
        "seed": request.seed if request.seed is not None else 0,
        "filename_prefix": output_prefix,
        "flux_model_name": model_name,
        "flux2_text_encoder_name": settings.flux2_text_encoder_name,
        "flux2_vae_name": settings.flux2_vae_name,
        "clip_l_name": settings.clip_l_name,
        "t5xxl_name": settings.t5xxl_name,
        "vae_name": settings.vae_name,
        "clip_vision_name": settings.clip_vision_name,
        "pulid_model_name": settings.pulid_model_name,
        "pulid_strength": settings.pulid_strength,
    }
    for logical_name, binding in envelope["bindings"].items():
        if logical_name not in values:
            continue
        value = values[logical_name]
        if value is None and binding.get("required", True):
            raise WorkflowError(f"Valor obrigatório ausente para binding {logical_name}.")
        if value is None:
            continue
        _set_node_input(prompt, binding, value, logical_name)

    output_nodes = tuple(str(item) for item in envelope.get("output_nodes") or ())
    if not output_nodes:
        raise WorkflowError("O workflow não declara output_nodes.")

    return PreparedWorkflow(workflow_id=workflow_name, workflow_version=request.workflow_version, prompt=prompt, output_nodes=output_nodes)
