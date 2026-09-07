# Privacy IA — Flux ComfyUI Image Worker

Worker serverless do RunPod preparado para consumir o contrato `privacy-production-spec-v1` com engines `flux-1-schnell` e `flux-1-dev`.

## Características
- Fail-closed: sem referência biométrica aprovada, o job falha.
- ComfyUI headless com workflows versionados.
- Sem pipelines manuais Diffusers/Juggernaut/IP-Adapter legadas.
- Saída privada por base64 ou R2 privado com signed URL curta.

## Teste local estático
```bash
python -m pip install -r requirements-dev.txt
python scripts/validate_workflows.py
python -m pytest
```
