"""Stdlib-only engine behind the `uat` CLI — no third-party imports, ever.

The customer's target workspace has no PyPI access (every workload pins `dependencies: []`),
so the pretty Typer+Rich front-end (uat_cli.py) may not import there. This module holds all
the actual work — repo detection, `air run` submission, run-state polling, the integrity
check, receipt formatting — using only the standard library, so the fallback front-end
(uat_min.py, pure argparse) is fully functional with zero installs:

    python3 -m utils.verification.uat_min run multinode --tier headline

Both front-ends call into here; they differ only in how they render. Keep this file
dependency-free.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from . import uat_prefs
from . import uat_suite
from .results.registry import TESTS

REG = {t["id"]: t for t in TESTS}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# We drive `air` itself (not `databricks jobs get-run`) for submit + status, so auth/workspace
# resolution always matches whatever `air run` used — no ambient-profile mismatch. `air --json`
# emits a structured envelope {"v":1,"ts":..,"data":{..}} on stdout; `data.status` is the run
# state, `data.dashboard_url` the run URL, `data.mlflow_url` the log/metrics link.
TERMINAL = {"SUCCESS", "FAILED", "TIMEDOUT", "CANCELED", "INTERNAL_ERROR", "SKIPPED"}


def find_repo(explicit: str | None = None) -> str:
    """Locate the repo checkout the suite launches from.

    The CLI operates on repo files (workloads/*.yaml, the registry), so it needs the checkout
    root even when installed into a uv venv/tool dir where __file__ is NOT inside the repo.
    Order: explicit (--repo), $UAT_REPO, walk up from CWD for the repo markers, else fall back
    to the package location (correct for an editable `uv run` from the repo)."""
    if explicit:
        return os.path.abspath(explicit)
    if os.environ.get("UAT_REPO"):
        return os.path.abspath(os.environ["UAT_REPO"])
    d = os.getcwd()
    while True:
        if os.path.isdir(os.path.join(d, "workloads")) and os.path.exists(os.path.join(d, "AGENTS.md")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def now() -> str:
    return datetime.now(timezone.utc).strftime("%H:%M:%S")


def profile_args(profile: str | None) -> list[str]:
    return ["-p", profile] if profile else []


def reg_line(registry_id: str | None) -> str:
    """One-line 'what this proves', pulled from the results registry (never duplicated here)."""
    if not registry_id:
        return "(no results-matrix row)"
    t = REG.get(registry_id)
    if not t:
        return f"(!! unknown registry_id: {registry_id})"
    crit = t.get("criteria", "")
    return f"[{registry_id}] {t['test']} — {crit}" if crit else f"[{registry_id}] {t['test']}"


# ── air run (multinode) ──────────────────────────────────────────────────────────────────

def _yaml_quote(s: str) -> str:
    """Single-quote a scalar for a YAML list item (wheel paths are absolute, no newlines)."""
    return "'" + s.replace("'", "''") + "'"


def _merge_dep_entries(existing: list[str], wheels: list[str]) -> list[str]:
    """Preserve existing deps order; append wheel paths that aren't already listed."""
    seen = set(existing)
    out = list(existing)
    for w in wheels:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out


def _parse_deps_block(lines: list[str], start: int) -> tuple[list[str], int]:
    """Parse a `dependencies:` value starting at `start` (the dependencies: line).

    Returns (entries, index_of_first_line_after_block).
    """
    line = lines[start]
    # Inline form: dependencies: []  or  dependencies: [a, b]
    m = re.match(r"^(\s*)dependencies:\s*(.*)$", line)
    if not m:
        return [], start + 1
    rest = m.group(2).strip()
    if rest.startswith("[") and rest.endswith("]"):
        inner = rest[1:-1].strip()
        if not inner:
            return [], start + 1
        parts = [p.strip().strip("'\"") for p in inner.split(",")]
        return [p for p in parts if p], start + 1
    # Block form: dependencies:\n  - foo\n  - bar
    entries: list[str] = []
    i = start + 1
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        item = re.match(r"^(\s+)-\s+(.*)$", ln)
        if not item:
            break
        entries.append(item.group(2).strip().strip("'\""))
        i += 1
    return entries, i


def inject_wheel_deps(src: Path, dest: Path, wheel_paths: list[str]) -> None:
    """Write dest YAML = src with wheel_paths merged into environment.dependencies.

    Stdlib-only line rewriter (uat_core stays PyYAML-free). Handles `dependencies: []`,
    inline lists, and block lists under `environment:`.
    """
    text = src.read_text()
    lines = text.splitlines(keepends=True)
    bare = [ln.rstrip("\n\r") for ln in lines]

    env_i = None
    for i, ln in enumerate(bare):
        if re.match(r"^environment:\s*$", ln) or re.match(r"^environment:\s+\S", ln):
            env_i = i
            break
    if env_i is None:
        # No environment block — insert a minimal one at the top after any leading comments.
        insert_at = 0
        while insert_at < len(bare) and (
            not bare[insert_at].strip() or bare[insert_at].lstrip().startswith("#")
        ):
            insert_at += 1
        block = ["environment:", "  dependencies:"] + [
            f"    - {_yaml_quote(w)}" for w in wheel_paths
        ] + [""]
        new_bare = bare[:insert_at] + block + bare[insert_at:]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(new_bare) + ("\n" if text.endswith("\n") else ""))
        return

    deps_i = None
    for i in range(env_i + 1, len(bare)):
        if re.match(r"^[^\s#]", bare[i]) and not bare[i].startswith(" "):
            break  # left the environment mapping
        if re.match(r"^\s+dependencies:\s*", bare[i]):
            deps_i = i
            break
    if deps_i is None:
        # Insert dependencies under environment:
        indent = "  "
        block = [f"{indent}dependencies:"] + [
            f"{indent}  - {_yaml_quote(w)}" for w in wheel_paths
        ]
        new_bare = bare[: env_i + 1] + block + bare[env_i + 1 :]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("\n".join(new_bare) + ("\n" if text.endswith("\n") else ""))
        return

    existing, end_i = _parse_deps_block(bare, deps_i)
    merged = _merge_dep_entries(existing, wheel_paths)
    indent_m = re.match(r"^(\s*)dependencies:", bare[deps_i])
    indent = indent_m.group(1) if indent_m else "  "
    block = [f"{indent}dependencies:"] + [
        f"{indent}  - {_yaml_quote(w)}" for w in merged
    ]
    new_bare = bare[:deps_i] + block + bare[end_i:]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(new_bare) + ("\n" if text.endswith("\n") else ""))


def composed_workload_file(item: dict, repo: str) -> str:
    """Repo-relative YAML path to submit: composed with wheel deps if configured, else base.

    Composed files land next to the base under workloads/ as `.composed-uat-<name>-<pid>.yaml`
    (gitignored). Returns a path relative to `repo` suitable for `air run --file`.
    """
    wheels = uat_prefs.resolve_dep_paths(item["name"])
    if not wheels:
        return item["file"]
    base = Path(repo) / item["file"]
    if not base.is_file():
        return item["file"]
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", item["name"])
    out_name = f".composed-uat-{safe}-{os.getpid()}.yaml"
    # Prefer writing beside the base file when it's under workloads/; else workloads/.
    parent = base.parent
    dest = parent / out_name
    inject_wheel_deps(base, dest, wheels)
    return str(dest.relative_to(repo))


def policy_override(name: str | None, policy_id: str | None) -> tuple[list[str], str | None]:
    """Turn --policy / --policy-id into air `--override` tokens. Returns (tokens, error).

    usage_policy_name and usage_policy_id are mutually exclusive (air rejects both); the run-as
    user must have access to the policy. Empty tokens when neither is given."""
    if name and policy_id:
        return [], ("pass only one of --policy / --policy-id — usage_policy_name and "
                    "usage_policy_id are mutually exclusive")
    if name:
        return [f"usage_policy_name={name}"], None
    if policy_id:
        return [f"usage_policy_id={policy_id}"], None
    return [], None


def air_cmd(item: dict, profile: str | None, dry_run: bool,
            repo: str | None = None, extra_overrides: list[str] | None = None) -> list[str]:
    file_arg = composed_workload_file(item, repo) if repo else item["file"]
    cmd = ["air", "run", "--file", file_arg, *profile_args(profile)]
    if dry_run:
        cmd.append("--dry-run")
    overrides = list(item["overrides"]) + list(extra_overrides or [])
    if overrides:
        cmd += ["--override", *overrides]
    return cmd


def _air_json(args: list[str], profile: str | None, cwd: str | None = None,
              timeout: int = 120) -> tuple[int, dict | None, str]:
    """Run `air --json [ -p PROFILE ] <args>` and return (returncode, data, raw_text).

    `data` is the parsed envelope's `data` object (None if no envelope was emitted — e.g. an
    early auth error). `--json` still prints an `[INFO] Using Profile:` preamble, so we scan
    lines for the one that parses as the {"v":..,"data":..} envelope rather than assuming it's
    the whole of stdout."""
    r = subprocess.run(["air", "--json", *profile_args(profile), *args],
                       capture_output=True, text=True, cwd=cwd, timeout=timeout)
    data = None
    for line in (r.stdout.splitlines() + r.stderr.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict) and "data" in obj:
                data = obj["data"]
                break
    return r.returncode, data, ANSI.sub("", r.stdout + r.stderr).strip()


def submit(item: dict, profile: str | None, dry_run: bool, repo: str,
           extra_overrides: list[str] | None = None) -> tuple[str | None, str, str]:
    """Submit one item; return (run_id, status, detail).

    Uses `air --json run …` so the returned envelope carries run_id/status directly. `air run`
    submits and returns (it does not block to completion), so we get the id and poll separately.
    A `--dry-run` validates + uploads the snapshot but skips the Jobs API: it returns
    status='DRY_RUN_OK' and NO run_id — that is success, not failure. run_id is also None on a
    real submission failure; the status disambiguates (empty/None status => nothing came back).

    When ~/.air-lab prefs set wheels_root + a wheel list for this item, submits a composed
    YAML with those absolute paths merged into environment.dependencies (variant C)."""
    file_arg = composed_workload_file(item, repo)
    args = ["run", "--file", file_arg]
    if dry_run:
        args.append("--dry-run")
    overrides = list(item["overrides"]) + list(extra_overrides or [])
    if overrides:
        args += ["--override", *overrides]
    _rc, data, raw = _air_json(args, profile, cwd=repo, timeout=600)
    data = data or {}
    return data.get("run_id"), data.get("status", ""), raw[-300:] or "no output"


def run_state(run_id: str, profile: str | None) -> dict:
    """Run details via `air --json get run <id>`: dict with status + dashboard_url + mlflow_url.

    Returns {'status': 'UNKNOWN', 'detail': …} if no envelope came back (auth/id problem)."""
    _rc, data, raw = _air_json(["get", "run", run_id], profile)
    if not data or "status" not in data:
        return {"status": "UNKNOWN", "detail": raw[-160:]}
    return data


def jobs_run_state(run_id: str, profile: str | None) -> dict:
    """State for the notebook DRIVER run — a plain Jobs API notebook job, NOT an AIR run, so
    `air get run` can't see it. Poll via `databricks jobs get-run --output json` (the explicit
    --output json avoids the text-table default that parses as UNKNOWN)."""
    r = subprocess.run(["databricks", "jobs", "get-run", run_id, "--output", "json",
                        *profile_args(profile)], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(r.stdout)
    except Exception:
        return {"status": "UNKNOWN", "detail": (r.stderr or r.stdout)[-160:]}
    s = d.get("state", {})
    return {"status": s.get("result_state") or s.get("life_cycle_state") or "UNKNOWN",
            "dashboard_url": d.get("run_page_url", ""), "detail": s.get("state_message", "")}


# ── check / receipts / notebook spec (pure; front-ends render) ─────────────────────────────

def check(repo: str) -> tuple[list[str], int, int]:
    """Integrity check. Returns (errors, n_items, n_tiers)."""
    errors: list[str] = []
    seen = set()
    for it in uat_suite.ITEMS:
        seen.add(it["name"])
        if not os.path.exists(os.path.join(repo, it["file"])):
            errors.append(f"{it['name']}: missing workload {it['file']}")
        rid = it["registry_id"]
        if rid and rid not in REG:
            errors.append(f"{it['name']}: registry_id '{rid}' not in results/registry.py")
        for hid in it["hardware"]:
            if hid not in uat_suite.HW:
                errors.append(f"{it['name']}: unknown hardware {hid}")
            elif not uat_suite.hw_supports_nodes(hid, it["nodes"]):
                errors.append(f"{it['name']}: {hid} is single-node-only but item is "
                              f"{it['nodes']} nodes (AIR would reject accelerator_count)")
        if it["default_hw"] not in it["hardware"]:
            errors.append(f"{it['name']}: default_hw {it['default_hw']} not in hardware list")
        for t in it["tiers"]:
            if t not in uat_suite.TIERS:
                errors.append(f"{it['name']}: unknown tier {t}")
    n_items = len(seen)
    return errors, n_items, len(uat_suite.all_tiers())


def receipt_lines(tier: str, runs: dict, final: dict, profile: str | None) -> list[str]:
    """The plain-text NOTES.md-pasteable receipt block (no styling — copies clean).

    `final[name]` is the status dict from run_state (status + dashboard_url + mlflow_url), or a
    synthetic {'status': 'DRY-OK'|'SUBMIT-FAIL', ...} for the non-polled cases."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = [f"UAT multinode/{tier} — {ts} (profile {profile or '<profile>'})"]
    for name, it in runs.items():
        info = final.get(name, {"status": "—"})
        st = info.get("status", "—")
        rid = it.get("run_id")
        out.append(f"  {name}: " + (f"run {rid} -> {st}" if rid else st))
        if info.get("dashboard_url"):
            out.append(f"      run:    {info['dashboard_url']}")
        if info.get("mlflow_url"):
            out.append(f"      mlflow: {info['mlflow_url']}")
        if st not in ("SUCCESS", "DRY-OK") and info.get("detail"):
            out.append(f"      note:   {info['detail'][:120]}")
    out.append("Verdicts come from MLflow receipts/sentinels, not run state alone — "
               "see results/registry.py.")
    return out


def notebook_spec(shapes: str, pool: str, mirror: str | None) -> dict:
    n = uat_suite.NOTEBOOK_SUITE
    mirror = mirror or n["default_mirror"]
    return {
        "run_name": f"uat-driver-{shapes}",
        "tasks": [{
            "task_key": "driver",
            "notebook_task": {
                "notebook_path": f"{mirror}/{n['driver_notebook']}",
                "base_parameters": {"shapes": shapes, "pool": pool},
            },
            "environment_key": "uat_env",
        }],
        "environments": [{
            "environment_key": "uat_env",
            "spec": {"environment_version": n["environment_version"], "dependencies": []},
        }],
    }


def select_items(tier: str, only: str | None) -> tuple[list[dict], str | None]:
    """Resolve tier + --only into an (unpinned) item list. Returns (items, error_message)."""
    try:
        items = uat_suite.items_for(tier)
    except KeyError:
        return [], f"unknown tier: {tier} (choose: {', '.join(uat_suite.all_tiers())}, all)"
    if only:
        want = {uat_suite._canonical(s.strip()) for s in only.split(",") if s.strip()}
        by_name = {it["name"]: it for it in items}
        # --only can name a row that isn't in this tier (e.g. allreduce under fabric);
        # look it up in the full suite so the picker/scripted path still works.
        all_by_name = {it["name"]: it for it in uat_suite.ITEMS}
        picked, unknown = [], []
        for name in want:
            it = by_name.get(name) or all_by_name.get(name)
            if it is None:
                unknown.append(name)
            else:
                picked.append(it)
        if unknown:
            return [], f"--only names not in the suite: {', '.join(sorted(unknown))}"
        items = picked
    if not items:
        return [], "nothing selected"
    return items, None


def select_launches(tier: str, only: str | None, hw: str | None) -> tuple[list[dict], str | None]:
    """Items × hardware → pinned launches. `hw=None` uses each row's default for the tier."""
    items, err = select_items(tier, only)
    if err:
        return [], err
    hw_ids, herr = uat_suite.parse_hw(hw)
    if herr:
        return [], herr
    launches = []
    skipped = []
    for it in items:
        targets = hw_ids if hw_ids else [uat_suite.default_hw_for(it, tier)]
        targets = [h for h in targets if h]
        pinned = 0
        for hid in targets:
            if hid not in it["hardware"]:
                continue
            launches.append(uat_suite.pin(it, hid))
            pinned += 1
        if not pinned:
            skipped.append(it["name"])
    if not launches:
        want = ",".join(hw_ids or ["<tier default>"])
        extra = f" (skipped, no cell on {want}: {', '.join(skipped)})" if skipped else ""
        return [], "nothing selected" + extra
    return launches, None


def logs_cmd(run_id: str, profile: str | None, node: int = 0) -> list[str]:
    """`air logs` argv — live stream for a running job, full history for a completed one."""
    return ["air", *profile_args(profile), "logs", str(run_id), "--node", str(node)]
