import hmac
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from privacy_worker.errors import WorkerError


MAX_REQUEST_BYTES = 4 * 1024 * 1024
DEFAULT_PORT = 8000

_GENERATION_LOCK = threading.Lock()
# R3C9K_W2_SAFE_FAILURE_BEGIN

SAFE_FAILURE_PHASES = frozenset(
    {
        "contract_parse",
        "model_validation",
        "reference_download",
        "workflow_prepare",
        "comfyui_start",
        "generation_submit",
        "generation_wait",
        "output_download",
        "output_publish",
        "response_finalize",
    }
)


def _safe_worker_phase(error):
    value = str(
        getattr(
            error,
            "worker_phase",
            "",
        )
        or ""
    ).strip()

    if value in SAFE_FAILURE_PHASES:
        return value

    return "unknown"


def _safe_worker_code(error):
    value = str(
        getattr(
            error,
            "code",
            "",
        )
        or ""
    ).strip().upper()

    allowed = (
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789_"
    )

    if (
        not value
        or len(value) > 80
        or any(
            character not in allowed
            for character in value
        )
    ):
        return "WORKER_ERROR"

    return value


def _safe_worker_type(error):
    value = type(error).__name__

    if (
        not value
        or len(value) > 80
        or any(
            not (
                character.isascii()
                and (
                    character.isalnum()
                    or character == "_"
                )
            )
            for character in value
        )
    ):
        return "WorkerError"

    return value


def _safe_worker_failure_payload(error):
    if isinstance(error, WorkerError):
        return {
            "ok": False,
            "error": "GENERATION_FAILED",
            "workerCode": _safe_worker_code(error),
            "workerType": _safe_worker_type(error),
            "phase": _safe_worker_phase(error),
            "retryable": bool(
                getattr(
                    error,
                    "retryable",
                    False,
                )
            ),
        }

    return {
        "ok": False,
        "error": "GENERATION_FAILED",
        "workerCode": "UNEXPECTED_WORKER_ERROR",
        "workerType": "UnexpectedWorkerError",
        "phase": _safe_worker_phase(error),
        "retryable": False,
    }


# R3C9K_W2_SAFE_FAILURE_END


def _shared_secret():
    return os.environ.get("NATIVE_POD_SHARED_SECRET", "")


def _json_bytes(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _invoke_production_handler(job):
    # Lazy import by design:
    # transport-only boot/tests do not load the generation stack.
    from handler import handler as production_handler

    return production_handler(job)


class NativePodRequestHandler(BaseHTTPRequestHandler):
    server_version = "PrivacyNativeImageWorker/1.0"
    sys_version = ""

    def log_message(self, format, *args):
        # Suppress default request logging.
        # Authorization headers/secrets must never be logged.
        return

    def _send_json(self, status_code, payload):
        body = _json_bytes(payload)

        self.send_response(status_code)
        self.send_header(
            "Content-Type",
            "application/json; charset=utf-8",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()

        self.wfile.write(body)

    def _authorized(self):
        expected = _shared_secret()

        if not expected:
            return False

        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "

        if not authorization.startswith(prefix):
            return False

        supplied = authorization[len(prefix):]

        if not supplied:
            return False

        return hmac.compare_digest(supplied, expected)

    def _require_auth(self):
        if not _shared_secret():
            self._send_json(
                503,
                {
                    "ok": False,
                    "error": "NATIVE_POD_SHARED_SECRET_NOT_CONFIGURED",
                },
            )
            return False

        if not self._authorized():
            self._send_json(
                401,
                {
                    "ok": False,
                    "error": "UNAUTHORIZED",
                },
            )
            return False

        return True

    def _read_json_body(self):
        raw_length = self.headers.get("Content-Length")

        if raw_length is None:
            self._send_json(
                411,
                {
                    "ok": False,
                    "error": "CONTENT_LENGTH_REQUIRED",
                },
            )
            return None

        try:
            content_length = int(raw_length)
        except (TypeError, ValueError):
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "INVALID_CONTENT_LENGTH",
                },
            )
            return None

        if content_length < 0:
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "INVALID_CONTENT_LENGTH",
                },
            )
            return None

        if content_length > MAX_REQUEST_BYTES:
            self._send_json(
                413,
                {
                    "ok": False,
                    "error": "PAYLOAD_TOO_LARGE",
                    "maxBytes": MAX_REQUEST_BYTES,
                },
            )
            return None

        raw_body = self.rfile.read(content_length)

        try:
            decoded = raw_body.decode("utf-8")
            payload = json.loads(decoded)
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "INVALID_JSON",
                },
            )
            return None

        if not isinstance(payload, dict):
            self._send_json(
                400,
                {
                    "ok": False,
                    "error": "JSON_OBJECT_REQUIRED",
                },
            )
            return None

        return payload

    def do_GET(self):
        if self.path == "/healthz":
            self._send_json(
                200,
                {
                    "ok": True,
                    "service": "native-image-worker",
                    "transport": "native_http",
                },
            )
            return

        if self.path == "/readyz":
            if not self._require_auth():
                return

            self._send_json(
                200,
                {
                    "ok": True,
                    "transportReady": True,
                    "generationConcurrency": 1,
                },
            )
            return

        self._send_json(
            404,
            {
                "ok": False,
                "error": "NOT_FOUND",
            },
        )

    def do_POST(self):
        if self.path != "/v1/generate":
            self._send_json(
                404,
                {
                    "ok": False,
                    "error": "NOT_FOUND",
                },
            )
            return

        if not self._require_auth():
            return

        payload = self._read_json_body()

        if payload is None:
            return

        acquired = _GENERATION_LOCK.acquire(blocking=False)

        if not acquired:
            self._send_json(
                409,
                {
                    "ok": False,
                    "error": "WORKER_BUSY",
                },
            )
            return

        try:
            if "input" in payload:
                job = payload
            else:
                job = {"input": payload}

            result = _invoke_production_handler(job)

            self._send_json(
                200,
                {
                    "ok": True,
                    "result": result,
                },
            )

        except Exception as error:
            self._send_json(
                500,
                _safe_worker_failure_payload(error),
            )
        finally:
            _GENERATION_LOCK.release()


def _resolve_port():
    raw_port = os.environ.get(
        "NATIVE_POD_PORT",
        str(DEFAULT_PORT),
    )

    try:
        port = int(raw_port)
    except (TypeError, ValueError):
        raise RuntimeError("INVALID_NATIVE_POD_PORT")

    if port < 1 or port > 65535:
        raise RuntimeError("INVALID_NATIVE_POD_PORT")

    return port


def main():
    port = _resolve_port()

    server = ThreadingHTTPServer(
        ("0.0.0.0", port),
        NativePodRequestHandler,
    )

    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()