# AIR custom image (DCS Beta): Databricks base image + OTEL SDK.
# Base provides CUDA/NCCL/EFA and a uv-managed venv at /opt/venv.
# Keep it minimal — no torch needed for the smoke test (GPU check via nvidia-smi in command).
FROM databricksruntime/air:dcs-base-aws-runtime

RUN uv pip install --no-cache \
    'opentelemetry-sdk>=1.27' \
    'opentelemetry-exporter-otlp-proto-grpc>=1.27' \
    'requests>=2.31'

# WORKDIR is ignored at runtime — bake code at an absolute path and invoke as /app/train.py
COPY app/ /app/
