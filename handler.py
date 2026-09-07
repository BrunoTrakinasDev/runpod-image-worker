from __future__ import annotations

from pathlib import Path

import runpod

from privacy_worker.comfyui import ComfyUIClient, ComfyUIProcessManager
from privacy_worker.config import settings
from privacy_worker.contracts import parse_production_request
from privacy_worker.downloader import download_media
from privacy_worker.errors import WorkerError
from privacy_worker.output import publish_output
from privacy_worker.telemetry import log_event, now_ms
from privacy_worker.workflows import prepare_workflow

settings.ensure_runtime_dirs()
_process_manager = ComfyUIProcessManager(settings)
_client = ComfyUIClient(settings)


def _required_model_paths(engine: str) -> list[Path]:
    if engine == "flux-1-schnell":
        return [
            settings.model_root / "diffusion_models" / settings.flux_schnell_model_name,
            settings.model_root / "text_encoders" / settings.clip_l_name,
            settings.model_root / "text_encoders" / settings.t5xxl_name,
            settings.model_root / "vae" / settings.vae_name,
        ]

    if engine == "flux-1-dev":
        return [
            settings.model_root / "diffusion_models" / settings.flux_dev_model_name,
            settings.model_root / "text_encoders" / settings.clip_l_name,
            settings.model_root / "text_encoders" / settings.t5xxl_name,
            settings.model_root / "vae" / settings.vae_name,
        ]

    if engine == "flux-2-klein":
        return [
            settings.model_root / "diffusion_models" / settings.flux2_klein_model_name,
            settings.model_root / "text_encoders" / settings.flux2_text_encoder_name,
            settings.model_root / "vae" / settings.flux2_vae_name,
        ]

    raise RuntimeError(
        f"Engine sem contrato explícito de modelos: {engine}"
    )


def _validate_models(engine: str) -> None:
    if settings.skip_model_validation:
        return

    required = _required_model_paths(engine)
    missing = [str(path) for path in required if not path.exists()]

    if missing:
        raise RuntimeError(
            f"Modelos obrigatórios ausentes para engine {engine}: {missing}"
        )


def handler(event: dict) -> dict:
    started_at = now_ms()
    request = None
    phase = "contract_parse"

    try:
        request = parse_production_request(event)

        log_event(
            "flux_job_received",
            request_id=request.request_id,
            engine=request.engine,
            workflow_id=request.workflow_id,
            width=request.width,
            height=request.height,
            selected_reference_tag=request.selected_reference_tag,
        )

        phase = "model_validation"
        _validate_models(request.engine)

        reference_path = None

        if request.reference_image_url:
            phase = "reference_download"

            reference_path = download_media(
                url=request.reference_image_url,
                destination_dir=settings.input_dir,
                stem=f"privacy_{request.request_id}_identity",
                fallback_extension=".png",
                settings=settings,
                request_id=request.request_id,
            )

        phase = "workflow_prepare"

        prepared = prepare_workflow(
            request=request,
            reference_image_filename=(
                reference_path.name
                if reference_path is not None
                else None
            ),
            output_prefix=f"privacy/{request.request_id}",
            settings=settings,
        )

        phase = "comfyui_start"

        _process_manager.ensure_started(
            request.request_id
        )

        phase = "generation_submit"

        prompt_id = _client.queue_prompt(
            prepared.prompt,
            request.request_id,
        )

        phase = "generation_wait"

        record = _client.wait_for_history(
            prompt_id,
            request.request_id,
        )

        phase = "output_download"

        output_path = _client.download_output(
            record=record,
            output_nodes=prepared.output_nodes,
            destination=(
                settings.output_dir
                / f"{request.request_id}_result"
            ),
            request_id=request.request_id,
        )

        phase = "output_publish"

        published = publish_output(
            output_path,
            settings,
            request.request_id,
        )

        phase = "response_finalize"

        elapsed = now_ms() - started_at

        response = {
            **published,
            "contract_version": request.contract_version,
            "engine": request.engine,
            "task": request.task,
            "request_id": request.request_id,
            "workflow_id": prepared.workflow_id,
            "workflow_version": prepared.workflow_version,
            "selected_reference_tag": request.selected_reference_tag,
            "elapsed_ms": elapsed,
        }

        log_event(
            "flux_job_completed",
            request_id=request.request_id,
            engine=request.engine,
            elapsed_ms=elapsed,
            size_bytes=published["size_bytes"],
            output_mode=(
                "private_r2"
                if published.get("r2_key")
                else "base64"
            ),
        )

        return response

    except WorkerError as error:
        error.worker_phase = phase

        log_event(
            "flux_job_failed",
            request_id=(
                request.request_id
                if request is not None
                else None
            ),
            level="ERROR",
            error_code=error.code,
            retryable=error.retryable,
            worker_phase=phase,
            message=str(error),
            details=error.details,
            elapsed_ms=now_ms() - started_at,
        )

        raise

    except Exception as error:
        error.worker_phase = phase

        log_event(
            "flux_job_failed",
            request_id=(
                request.request_id
                if request is not None
                else None
            ),
            level="ERROR",
            error_code="UNEXPECTED_WORKER_ERROR",
            retryable=False,
            worker_phase=phase,
            message=str(error),
            elapsed_ms=now_ms() - started_at,
        )

        raise

    finally:
        if request is not None:
            for path in settings.input_dir.glob(
                f"privacy_{request.request_id}_identity*"
            ):
                path.unlink(missing_ok=True)

if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
