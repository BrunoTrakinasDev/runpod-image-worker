FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime@sha256:77f17f843507062875ce8be2a6f76aa6aa3df7f9ef1e31d9d7432f4b0f563dee

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ROOT=/app \
    COMFYUI_ROOT=/opt/ComfyUI \
    WORKFLOW_ROOT=/app/workflows \
    RUNTIME_ROOT=/runpod-volume/privacy-flux-runtime \
    MODEL_ROOT=/runpod-volume/models \
    COMFYUI_START_LOCAL=true \
    HF_HOME=/runpod-volume/huggingface \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface \
    HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements-dev.txt constraints-comfyui-runtime.txt /app/
RUN pip install -r /app/requirements.txt

RUN git init /opt/ComfyUI && cd /opt/ComfyUI && git remote add origin https://github.com/Comfy-Org/ComfyUI.git && git fetch --depth=1 origin 6e3c0bda5a756ec334df449cdc7d4a4685631e91 && git checkout --detach FETCH_HEAD && test "$(git rev-parse HEAD)" = "6e3c0bda5a756ec334df449cdc7d4a4685631e91"
RUN python -c "import torch, torchvision, torchaudio; assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__; assert torchvision.__version__.split('+')[0] == '0.21.0', torchvision.__version__; assert torchaudio.__version__.split('+')[0] == '2.6.0', torchaudio.__version__; assert torch.version.cuda == '12.4', torch.version.cuda; print('D4D_GPU_STACK_PRECHECK=PASS', torch.__version__, torchvision.__version__, torchaudio.__version__, torch.version.cuda)"
RUN pip install -c /app/constraints-comfyui-runtime.txt -r /opt/ComfyUI/requirements.txt
RUN python -c "import torch, torchvision, torchaudio, importlib.metadata as m; assert torch.__version__.split('+')[0] == '2.6.0', torch.__version__; assert torchvision.__version__.split('+')[0] == '0.21.0', torchvision.__version__; assert torchaudio.__version__.split('+')[0] == '2.6.0', torchaudio.__version__; assert torch.version.cuda == '12.4', torch.version.cuda; expected={'requests':'2.32.3','Pillow':'11.0.0','PyYAML':'6.0.2'}; actual={k:m.version(k) for k in expected}; assert actual == expected, actual; print('D4D_RUNTIME_POSTCHECK=PASS', torch.__version__, torchvision.__version__, torchaudio.__version__, torch.version.cuda, actual)"
RUN pip check

COPY . /app

RUN mkdir -p /runpod-volume/privacy-flux-runtime /runpod-volume/models /runpod-volume/huggingface \
    && python scripts/validate_workflows.py

CMD ["python", "-u", "/app/handler.py"]
