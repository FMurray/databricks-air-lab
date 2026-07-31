"""Render the UAT results matrix from the registry, and police registry<->repo drift.

Usage (repo root):
    python -m utils.verification.results.build_results            # writes docs/uat-results.md + .csv
    python -m utils.verification.results.build_results --check    # drift check only (CI-able)

The registry (registry.py, same dir) is the single source of truth: prose + verdicts live
next to the tests. This script only renders it (markdown for the repo/doc site, CSV whose
columns match the shared results sheet for paste-over) and verifies two invariants:

  1. every registry `asset` path exists in the repo;
  2. every UAT-looking asset (workloads/*.yaml excluding probes/, uat checks) is claimed by
     exactly one registry entry — so adding a test without a results row fails the check.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

from .registry import COLUMNS, TESTS

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MD_OUT = os.path.join(REPO, "docs", "uat-results.md")
CSV_OUT = os.path.join(REPO, "docs", "uat-results.csv")

# Assets that are deliberately NOT results rows (diagnostics, templates, infra).
EXEMPT_GLOBS = [
    "workloads/probes/*.yaml",           # diagnostic probes back docs/06 receipts, not UAT rows
    "workloads/burn-debug.yaml",
    "workloads/snapshot-python-probe.yaml",
    "workloads/multinode-probe*.yaml",   # cheap plumbing probe (superseded by correctness row)
    "workloads/rdma-m*.yaml",            # rdma-stress family reports under allreduce-multi for now
    "workloads/docker-otel-zerobus-*.yaml",  # per-workspace copies of the registered example
    "workloads/vendored-wheels-*.yaml",  # deps recipes (documented in RUNNING-UAT), not UAT rows
    "workloads/tabicl-*.yaml",           # variants roll up into the tabular-fm row's asset
    "workloads/gpu-burn-nodeps.yaml",
    "workloads/java-docker.example.yaml",  # jvm row tracks djl-train; docker variant is fallback
    "workloads/xgboost-gpu.example.yaml",  # claimed
    "workloads/fsdp-multinode.example.yaml",  # claimed
]


import re


def _acceptance_files(asset_rel: str) -> list[str]:
    """Files that must carry the test's acceptance sentinels: the asset itself, plus any
    script a workload YAML's command references via $CODE_SOURCE_PATH/<path>."""
    files = [asset_rel]
    if asset_rel.endswith(".yaml"):
        try:
            text = open(os.path.join(REPO, asset_rel)).read()
            files += re.findall(r"\$CODE_SOURCE_PATH/(\S+\.(?:py|sh))", text)
        except OSError:
            pass
    return files


def check() -> int:
    errors = []
    warnings = []
    claimed = set()
    for t in TESTS:
        a = t.get("asset")
        if not a:
            continue
        claimed.add(a)
        if not os.path.exists(os.path.join(REPO, a)):
            errors.append(f"registry '{t['id']}' points at missing asset: {a}")
            continue
        # acceptance-criteria link: the workload's code must emit the declared sentinels —
        # a test whose code stops printing its acceptance line is drift, caught here.
        for sentinel in t.get("sentinels", []):
            found = False
            for rel in _acceptance_files(a):
                p = os.path.join(REPO, rel)
                if os.path.exists(p) and sentinel in open(p, errors="replace").read():
                    found = True
                    break
            if not found:
                msg = (f"'{t['id']}' declares acceptance sentinel '{sentinel}' but no "
                       f"linked file emits it ({', '.join(_acceptance_files(a))})")
                if t.get("sentinels_pending"):
                    warnings.append(f"{msg} — pending: {t['sentinels_pending']}")
                else:
                    errors.append(msg)

    exempt = set()
    for g in EXEMPT_GLOBS:
        exempt.update(os.path.relpath(p, REPO) for p in glob.glob(os.path.join(REPO, g)))
    for p in sorted(glob.glob(os.path.join(REPO, "workloads", "*.yaml"))):
        rel = os.path.relpath(p, REPO)
        # live copies of templates (repo convention: `x.yaml` beside `x.example.yaml`)
        # roll up into their example's registry row
        if not rel.endswith(".example.yaml") and os.path.exists(
                os.path.join(REPO, rel.replace(".yaml", ".example.yaml"))):
            continue
        if rel in claimed or rel in exempt:
            continue
        errors.append(f"unregistered UAT asset (add a registry row or an exemption): {rel}")

    ids = [t["id"] for t in TESTS]
    if len(ids) != len(set(ids)):
        errors.append("duplicate registry ids")
    for w in warnings:
        print("PENDING:", w)
    for e in errors:
        print("DRIFT:", e)
    linked = sum(1 for t in TESTS if t.get("sentinels"))
    print(f"drift check: {len(TESTS)} tests ({linked} sentinel-linked), "
          f"{len(errors)} problem(s), {len(warnings)} pending")
    return 1 if errors else 0


def render() -> None:
    headers = [h for _, h in COLUMNS]
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for t in TESTS:
            w.writerow([t.get(k, "") or "" for k, _ in COLUMNS])

    lines = [
        "# UAT results matrix",
        "",
        "**Generated from `utils/verification/results/registry.py` — edit the registry, not "
        "this file.** Rebuild: `python -m utils.verification.results.build_results`. The CSV "
        "(`uat-results.csv`) pastes over the shared results sheet (same columns).",
        "",
        "| " + " | ".join(headers) + " |",
        "|" + "---|" * len(headers),
    ]
    for t in TESTS:
        row = [(t.get(k, "") or "").replace("|", "\\|").replace("\n", " ") for k, _ in COLUMNS]
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "Verdict legend: PASS · PASS (1 gap) · BLOCKED (external precondition) · NOT RUN.",
        "Evidence naming run ids / MLflow experiments follows the receipts standard "
        "(`experiment-verification` skill).",
        "",
    ]
    with open(MD_OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {os.path.relpath(MD_OUT, REPO)} and {os.path.relpath(CSV_OUT, REPO)} "
          f"({len(TESTS)} rows)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="drift check only, non-zero on drift")
    args = ap.parse_args()
    rc = check()
    if args.check:
        sys.exit(rc)
    if rc:
        print("NOTE: drift problems above — rendering anyway; fix the registry.")
    render()
