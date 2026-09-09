import json

import pytest
import requests

from pod_server import _safe_worker_failure_payload
from privacy_worker.comfyui import ComfyUIClient
from privacy_worker.config import settings
from privacy_worker.errors import ComfyUIError


def response(body, status):
    result = requests.Response()
    result.status_code = status
    result._content = json.dumps(body).encode()
    return result


@pytest.mark.parametrize('status', [200, 400])
@pytest.mark.parametrize('body,expected', [([], 'list'), (None, 'NoneType'), ('PRIVATE', 'str'), (42, 'int'), (True, 'bool'), (1.5, 'float')])
def test_invalid_response_to_safe_payload(monkeypatch, status, body, expected):
    monkeypatch.setattr('privacy_worker.comfyui.requests.post', lambda *a, **k: response(body, status))
    with pytest.raises(ComfyUIError) as caught:
        ComfyUIClient(settings).queue_prompt({'private': 'PRIVATE'}, 'offline')
    caught.value.worker_phase = 'generation_submit'
    assert _safe_worker_failure_payload(caught.value) == {
        'ok': False, 'error': 'GENERATION_FAILED',
        'workerCode': 'COMFYUI_RUNTIME_ERROR', 'workerType': 'ComfyUIError',
        'phase': 'generation_submit', 'retryable': True,
        'diagnosticKind': 'INVALID_RESPONSE_TYPE', 'responseType': expected,
    }
    assert 'PRIVATE' not in json.dumps(caught.value.details)


@pytest.mark.parametrize('status', [200, 400])
def test_workflow_rejection_strips_raw_content(monkeypatch, status):
    body = {'error': 'PRIVATE token=PRIVATE https://private.invalid', 'node_errors': {
        '3': {'message': 'PRIVATE'}, 'https://private.invalid': {}, '12\n': {},
        '１２': {}, '12345678901': {}, **{str(i): {'prompt': 'PRIVATE'} for i in range(30)},
    }}
    monkeypatch.setattr('privacy_worker.comfyui.requests.post', lambda *a, **k: response(body, status))
    with pytest.raises(ComfyUIError) as caught:
        ComfyUIClient(settings).queue_prompt({}, 'offline')
    payload = _safe_worker_failure_payload(caught.value)
    assert payload['diagnosticKind'] == 'WORKFLOW_REJECTED'
    assert payload['responseType'] == 'dict'
    assert len(payload['nodeErrorIds']) == 20
    assert all(i.isascii() and i.isdigit() for i in payload['nodeErrorIds'])
    assert 'PRIVATE' not in json.dumps(caught.value.details)
    assert 'private.invalid' not in json.dumps(payload)


@pytest.mark.parametrize('body', [{}, {'prompt_id': 'must-not-succeed'}, {'node_errors': ['PRIVATE']}])
def test_http_400_never_succeeds(monkeypatch, body):
    monkeypatch.setattr('privacy_worker.comfyui.requests.post', lambda *a, **k: response(body, 400))
    with pytest.raises(ComfyUIError) as caught:
        ComfyUIClient(settings).queue_prompt({}, 'offline')
    assert caught.value.details['diagnosticKind'] == 'WORKFLOW_REJECTED'


@pytest.mark.parametrize('status', [400, 500])
def test_non_json_http_failure_remains_generic(monkeypatch, status):
    result = response(None, status)
    result._content = b'PRIVATE not json'
    monkeypatch.setattr('privacy_worker.comfyui.requests.post', lambda *a, **k: result)
    with pytest.raises(ComfyUIError) as caught:
        ComfyUIClient(settings).queue_prompt({}, 'offline')
    assert caught.value.details == {}
    assert 'PRIVATE' not in json.dumps(_safe_worker_failure_payload(caught.value))


@pytest.mark.parametrize('details', [None, [], {'diagnosticKind': []}, {'diagnosticKind': 'PRIVATE'}, {
    'diagnosticKind': 'WORKFLOW_REJECTED', 'responseType': {'token': 'PRIVATE'},
    'nodeErrorIds': ['1', '1', 2, '2\n', 'PRIVATE', '１２', '12345678901'], 'prompt': 'PRIVATE',
}])
def test_transport_revalidates_untrusted_details(details):
    error = ComfyUIError('PRIVATE')
    error.details = details
    payload = _safe_worker_failure_payload(error)
    assert 'PRIVATE' not in json.dumps(payload)
    if payload.get('diagnosticKind'):
        assert 'responseType' not in payload
        assert payload['nodeErrorIds'] == ['1']
    else:
        assert 'nodeErrorIds' not in payload


def test_unexpected_error_contract_preserved():
    assert _safe_worker_failure_payload(ValueError('PRIVATE')) == {
        'ok': False, 'error': 'GENERATION_FAILED', 'workerCode': 'UNEXPECTED_WORKER_ERROR',
        'workerType': 'UnexpectedWorkerError', 'phase': 'unknown', 'retryable': False,
    }
