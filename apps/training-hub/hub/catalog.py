"""The workload catalog IS the repo's workloads/ directory — no hypothetical entries.

Every `workloads/*.yaml` (live copies preferred over their `.example` twins) becomes a
runnable workload in the hub, owned by the team named in config `catalog_team` and mapped
to a use case by its experiment family. Ad-hoc workloads registered in the UI live in
SQLite alongside; the repo ones re-sync on every app start, so git stays the source of truth.
"""

from __future__ import annotations

from pathlib import Path

import subprocess

import yaml

WORKLOADS_DIR = Path(__file__).resolve().parents[3] / "workloads"

# experiment family -> use case (matches the UAT suite matrix in docs/06)
FAMILY_USE_CASE = {
    "node-acceptance": "node-acceptance",
    "probes": "env-diagnostics",
    "scheduling-isolation": "scheduling-isolation",
    "vendored-wheels": "dependencies",
    "env-flexibility": "dependencies",
    "docker-otel-zerobus": "telemetry",
    "foundation-models": "foundation-models",
    "tabicl": "classic-ml",
    "xgboost": "classic-ml",
    "multi-language": "multi-language",
    "multinode": "node-acceptance",
    "rdma-stress": "node-acceptance",
}


def _use_case_for(path: Path, raw: dict) -> str:
    hints = [path.stem]
    snap = (raw.get("code_source") or {}).get("snapshot") or {}
    hints += snap.get("include_paths", [])
    for hint in hints:
        for key, uc in FAMILY_USE_CASE.items():
            if key in str(hint):
                return uc
    return "env-diagnostics"


def _author(path: Path) -> str:
    """Last committer of the workload YAML — real provenance from git."""
    try:
        r = subprocess.run(
            ["git", "log", "-1", "--format=%ae", "--", str(path)],
            capture_output=True, text=True, cwd=path.parent, timeout=10)
        return r.stdout.strip()
    except Exception:
        return ""


def _header(path: Path) -> tuple[str, str]:
    """(title, description) from the YAML's leading comment block. Title = first sentence;
    description = the whole block minus command/override lines."""
    lines = []
    for line in path.read_text().splitlines():
        if not line.startswith("#"):
            break
        text = line.lstrip("#").strip()
        if text.startswith("--override") or text.startswith("air run"):
            continue
        lines.append(text)
    block = " ".join(l for l in lines if l)
    if not block:
        return "", ""
    title = block.split(". ")[0].strip().rstrip(".")
    return title[:120], block[:500]


def _nodes(raw: dict) -> int:
    comp = raw.get("compute") or {}
    n = int(comp.get("num_accelerators", 1))
    shape = comp.get("accelerator_type", "")
    per_node = 8 if "8x" in shape else 1
    return max(1, n // per_node)


def repo_workloads(team: str) -> list[dict]:
    """Parse the repo's workload YAMLs; live copies shadow their .example twins."""
    if not WORKLOADS_DIR.is_dir():
        return []
    files: dict[str, Path] = {}          # key -> file to RUN (live shadows example)
    doc_files: dict[str, Path] = {}      # key -> file to DOCUMENT from (example preferred)
    for p in sorted(WORKLOADS_DIR.glob("**/*.yaml")):
        stem = p.name.replace(".example.yaml", "").replace(".yaml", "")
        key = str(p.parent.relative_to(WORKLOADS_DIR)) + "/" + stem
        is_example = ".example" in p.name
        if key not in files or (".example" in files[key].name and not is_example):
            files[key] = p
        if key not in doc_files or is_example:
            doc_files[key] = p
    out = []
    for key, p in sorted(files.items()):
        try:
            raw = yaml.safe_load(p.read_text()) or {}
        except yaml.YAMLError:
            continue
        if "command" not in raw:
            continue
        comp = raw.get("compute") or {}
        cmd = str(raw.get("command", ""))
        title, description = _header(doc_files.get(key, p))
        out.append({
            "name": key.lstrip("./"),
            "title": title or key.lstrip("./"),
            "description": description,
            "author": _author(doc_files.get(key, p)),
            "experiment_name": str(raw.get("experiment_name", "")),
            "kind": "air_yaml",
            "ref": str(p.relative_to(WORKLOADS_DIR.parent)),
            "team": team,
            "use_case": _use_case_for(p, raw),
            "shape": comp.get("accelerator_type", "GPU_1xA10"),
            "nodes": _nodes(raw),
            "needs_torch": int("torchrun" in cmd or "torch" in cmd),
            "source": "repo",
        })
    return out
