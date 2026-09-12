import hashlib
import importlib
import json
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def harness(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'runpod', types.SimpleNamespace())
    from privacy_worker import config
    settings = replace(config.settings, model_source_mode='r2_registry',
        model_root=tmp_path / 'models', runtime_root=tmp_path / 'runtime',
        model_manifest_root=tmp_path / 'manifests', workflow_root=ROOT / 'workflows',
        skip_model_validation=False, model_registry_r2_endpoint_url='https://fixture.invalid',
        model_registry_r2_access_key_id='fixture', model_registry_r2_secret_access_key='fixture')
    settings.ensure_runtime_dirs()
    monkeypatch.setattr(config, 'settings', settings)
    sys.modules.pop('handler', None)
    module = importlib.import_module('handler')
    monkeypatch.setattr(module, 'settings', settings)
    from privacy_worker import model_registry as registry
    manifest = json.loads((ROOT / 'model_manifests/flux2-klein-4b-official-distilled-v1.json').read_text())
    blobs = {}
    for item in manifest['artifacts']:
        content = item['role'].encode()
        item['bytes'] = len(content)
        item['sha256'] = hashlib.sha256(content).hexdigest()
        blobs[item['relative_path']] = content
    manifest['total_bytes'] = sum(item['bytes'] for item in manifest['artifacts'])
    settings.model_manifest_root.mkdir()
    (settings.model_manifest_root / 'flux2-klein-4b-official-distilled-v1.json').write_text(json.dumps(manifest))
    calls = []
    class Client:
        def download_fileobj(self, bucket, key, handle):
            calls.append(key)
            handle.write(blobs[key.removeprefix(settings.model_registry_r2_prefix + '/')])
    monkeypatch.setattr(registry, '_build_r2_client', lambda settings: Client())
    events = []
    original_validate = module._validate_models
    def validate(engine):
        assert all(p.is_file() for p in module._required_model_paths(engine))
        events.append('validated')
        original_validate(engine)
    monkeypatch.setattr(module, '_validate_models', validate)
    def start(request_id):
        assert events[-1] == 'validated'
        events.append('start')
    monkeypatch.setattr(module, '_process_manager', types.SimpleNamespace(ensure_started=start))
    output = settings.output_dir / 'fixture.png'
    output.write_bytes(b'fixture')
    monkeypatch.setattr(module, '_client', types.SimpleNamespace(
        queue_prompt=lambda *a: 'fixture', wait_for_history=lambda *a: {}, download_output=lambda **k: output))
    monkeypatch.setattr(module, 'download_media', lambda **k: pytest.fail('Identity download forbidden'))
    monkeypatch.setattr(module, 'publish_output', lambda *a: {'size_bytes': 7, 'private_output_only': True})
    event = json.loads((ROOT / 'tests/fixtures/image_klein_t2i_synthetic.json').read_text())
    yield module, registry, settings, event, blobs, calls, events
    sys.modules.pop('handler', None)


def test_cold_hydration_then_validation_and_cache(harness):
    module, registry, settings, event, blobs, calls, events = harness
    assert not settings.model_root.exists()
    request = module.parse_production_request(event)
    assert request.references == () and request.reference_image_url is None
    result = module.handler(event)
    assert result['private_output_only'] is True
    assert len(calls) == 3
    assert events == ['validated', 'start']
    module.handler(event)
    assert len(calls) == 3  # Existing bytes and SHA are revalidated, not downloaded.
    assert events == ['validated', 'start', 'validated', 'start']
    assert set(blobs) == {str(p.relative_to(settings.model_root)).replace('\\', '/') for p in module._required_model_paths('flux-2-klein')}


@pytest.mark.parametrize('failure', ['size', 'sha', 'r2'])
def test_preparation_failure_is_safe_and_blocks_comfyui(harness, monkeypatch, capsys, failure):
    module, registry, settings, event, blobs, calls, events = harness
    if failure == 'r2':
        def fail(settings):
            raise RuntimeError('SECRET access_key https://signed.invalid/private')
        monkeypatch.setattr(registry, '_build_r2_client', fail)
    else:
        first = next(iter(blobs))
        blobs[first] = b'x' if failure == 'size' else b'x' * len(blobs[first])
    from privacy_worker.errors import ModelRegistryPreparationError
    from pod_server import _safe_worker_failure_payload
    with pytest.raises(ModelRegistryPreparationError) as caught:
        module.handler(event)
    assert events == []
    payload = _safe_worker_failure_payload(caught.value)
    assert payload['phase'] == 'model_validation'
    assert payload['workerCode'] == 'MODEL_REGISTRY_PREPARATION_FAILED'
    assert payload['retryable'] is False
    assert 'SECRET' not in json.dumps(payload) + capsys.readouterr().out
    assert not list(settings.model_root.rglob('*.safetensors'))
    assert not list(settings.model_root.rglob('*.part-*'))


def test_local_volume_keeps_existing_path_without_registry(harness, monkeypatch):
    module, registry, settings, event, blobs, calls, events = harness
    for relative, content in blobs.items():
        target = settings.model_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    monkeypatch.setattr(module, 'settings', replace(settings, model_source_mode='local_volume'))
    monkeypatch.setattr(module, 'prepare_model_storage', lambda **k: pytest.fail('Local volume must not require registry'))
    module.handler(event)
    assert calls == [] and events == ['validated', 'start']


def test_existing_corrupt_file_fails_closed_without_redownload(harness):
    module, registry, settings, event, blobs, calls, events = harness
    target = module._required_model_paths('flux-2-klein')[0]
    target.parent.mkdir(parents=True)
    target.write_bytes(b'corrupt')
    from privacy_worker.errors import ModelRegistryPreparationError
    with pytest.raises(ModelRegistryPreparationError):
        module.handler(event)
    assert calls == [] and events == []
