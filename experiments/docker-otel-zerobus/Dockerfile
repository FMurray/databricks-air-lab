# AIR custom image (DCS Beta): Databricks base image + vendored OTEL SDK.
# Base provides CUDA/NCCL/EFA and a uv-managed venv at /opt/venv (python 3.12, no pip/ensurepip).
#
# Cross-build note (Apple Silicon): RUN steps execute under qemu-user where the `uv` binary
# segfaults and the venv python lacks ensurepip. So we install NOTHING in the image — deps are
# vendored on the host into ./vendor via:
#   uv pip install --target vendor --python-platform x86_64-unknown-linux-gnu \
#     --python-version 3.12 --only-binary :all: \
#     'opentelemetry-sdk>=1.27' 'opentelemetry-exporter-otlp-proto-grpc>=1.27' 'requests>=2.31'
# On an x86_64 builder, a plain `RUN uv pip install` is the better choice.
FROM databricksruntime/air:dcs-base-aws-runtime

# WORKDIR is ignored at runtime — bake code at absolute paths and invoke as /app/train.py
COPY vendor/ /app/vendor/
COPY app/ /app/
ENV PYTHONPATH=/app/vendor
