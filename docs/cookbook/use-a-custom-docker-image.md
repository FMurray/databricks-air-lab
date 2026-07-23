# Use a custom Docker image

Goal: run a workload whose environment the managed Python envs can't express — native deps,
non-Python runtimes, pinned system libraries.

!!! note "Beta"
    Custom Docker (DCS) is Beta and CLI-only. Customers that can't adopt pre-GA features should
    use the [GA-surface alternatives](run-non-python-workloads.md) instead.

✅ **Verified end-to-end 2026-07-16** (e2-demo-field-eng, run 37776040541298): podman build →
Docker Hub push → `air register image` (~2 min) → `air run` → SUCCESS on A10.

## The flow

```bash
podman build --platform linux/amd64 -t docker.io/<you>/my-image:0.1 .
podman push docker.io/<you>/my-image:0.1
air register image docker.io/<you>/my-image:0.1 -p <profile>   # per user, per tag; 2–6 min
```

```yaml
environment:
  docker_image: {url: docker.io/<you>/my-image:0.1}   # excludes version/dependencies
```

Private repos: Docker Hub PAT via `docker login`, `--interactive-authenticate`, or a secret scope.

## Constraints (know before you build)

- **Docker Hub only** — no ECR/GCR/GHCR. (Flag early with security-sensitive customers.)
- Image < 20 GB.
- `WORKDIR` is **ignored** — use absolute paths everywhere.
- `docker_image` is mutually exclusive with `dependencies`/`version`.
- Base images `databricksruntime/air:dcs-base-aws-{runtime,devel}` ship CUDA/NCCL/EFA and a uv
  venv at `/opt/venv` — start there.

The same env vars are injected as on the snapshot path (`NUM_NODES`, `NODE_RANK`, `WORLD_SIZE`,
multi-node rendezvous vars) — ✅ confirmed in-container, run 37776040541298.

## Building on Apple Silicon

!!! warning "uv segfaults under qemu emulation"
    Cross-building linux/amd64 images on an M-series Mac: `RUN uv pip install ...` dies under
    qemu-user, and the base venv has no pip. **Vendor the wheels on the host** and COPY:

    ```bash
    uv pip install --target vendor \
      --python-platform x86_64-unknown-linux-gnu --python-version 3.12 \
      --only-binary :all: -r requirements.txt
    ```

    Then `COPY vendor /vendor` + `ENV PYTHONPATH=/vendor`. COPY-only builds are fine under
    emulation. (podman works without a Docker Desktop license; details in
    `experiments/docker-otel-zerobus/NOTES.md`.)

## Next

- [Run non-Python workloads](run-non-python-workloads.md) — when Beta Docker is off the table
- [Ship telemetry to Delta](ship-telemetry-to-delta.md) — the workload this flow was proven with
