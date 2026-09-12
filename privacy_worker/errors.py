class WorkerError(Exception):
    code = "WORKER_ERROR"
    retryable = False

    def __init__(self, message: str, *, details: dict | None = None):
        super().__init__(message)
        self.details = details or {}


class ContractError(WorkerError):
    code = "INVALID_PRODUCTION_CONTRACT"


class DownloadError(WorkerError):
    code = "MEDIA_DOWNLOAD_FAILED"
    retryable = True


class WorkflowError(WorkerError):
    code = "WORKFLOW_ERROR"


class ComfyUIError(WorkerError):
    code = "COMFYUI_RUNTIME_ERROR"
    retryable = True


class OutputError(WorkerError):
    code = "OUTPUT_ERROR"


class ModelRegistryPreparationError(WorkerError):
    code = "MODEL_REGISTRY_PREPARATION_FAILED"
