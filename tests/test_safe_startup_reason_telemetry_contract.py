import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pod_server import _safe_worker_failure_payload
from privacy_worker.errors import ComfyUIError, ContractError

SOURCE = (ROOT / "pod_server.py").read_text(encoding="utf-8")

def require(value: bool, message: str) -> None:
    if not value:
        raise AssertionError(message)

def payload_for(error, phase: str):
    error.worker_phase = phase
    return _safe_worker_failure_payload(error)

require(SOURCE.count("def _safe_comfyui_start_reason_payload(error):") == 1, "startup reason helper missing or duplicated")
require(SOURCE.count("**_safe_comfyui_start_reason_payload(error),") == 1, "startup reason payload spread missing or duplicated")

cases = (
    ("ComfyUI não está acessível e COMFYUI_START_LOCAL=false.", "COMFYUI_START_LOCAL_DISABLED"),
    ("ComfyUI não encontrado em C:/SECRET/private/path.", "COMFYUI_ROOT_MISSING"),
    ("ComfyUI encerrou durante a inicialização.", "COMFYUI_PROCESS_EXITED"),
    ("Timeout aguardando a inicialização do ComfyUI.", "COMFYUI_START_TIMEOUT"),
    ("opaque secret-bearing startup failure https://secret.invalid/token", "COMFYUI_START_OTHER"),
)

for message, expected in cases:
    error = ComfyUIError(message)
    payload = payload_for(error, "comfyui_start")
    require(payload["message"] == expected, f"wrong mapping for {expected}")
    require(payload["workerCode"] == "COMFYUI_RUNTIME_ERROR", "workerCode changed")
    require(payload["workerType"] == "ComfyUIError", "workerType changed")
    require(payload["phase"] == "comfyui_start", "phase changed")
    require(payload["retryable"] is True, "retryable changed")
    serialized = json.dumps(payload)
    require("SECRET" not in serialized, "secret path leaked")
    require("secret.invalid" not in serialized, "secret URL leaked")
    require(message not in serialized, "raw startup message leaked")

queue_error = ComfyUIError(
    "ComfyUI rejeitou o workflow.",
    details={
        "diagnosticKind": "WORKFLOW_REJECTED",
        "responseType": "dict",
        "nodeErrorIds": ["1", "2", "2"],
    },
)
queue_payload = payload_for(queue_error, "generation_submit")
require("message" not in queue_payload, "non-startup message key introduced")
require(queue_payload.get("diagnosticKind") == "WORKFLOW_REJECTED", "existing diagnosticKind lost")
require(queue_payload.get("responseType") == "dict", "existing responseType lost")
require(queue_payload.get("nodeErrorIds") == ["1", "2"], "existing nodeErrorIds normalization lost")

contract_error = ContractError("sensitive raw contract message")
contract_payload = payload_for(contract_error, "contract_parse")
require("message" not in contract_payload, "legacy WorkerError payload changed")
require(contract_payload["workerCode"] == "INVALID_PRODUCTION_CONTRACT", "contract error code changed")
require(contract_payload["phase"] == "contract_parse", "contract phase changed")
require(contract_payload["retryable"] is False, "contract retryable changed")
require("sensitive raw contract message" not in json.dumps(contract_payload), "raw non-startup message leaked")

for forbidden in (
    '"message": str(error)',
    '"details": getattr(error',
    '"details": error.',
    '"stack":',
    '"traceback":',
):
    require(forbidden not in SOURCE, f"unsafe marker present: {forbidden}")

print("V7_COMFYUI_SAFE_START_REASON_TELEMETRY_V1_2_1_DYNAMIC_CONTRACT_READY")
