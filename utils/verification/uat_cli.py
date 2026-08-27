"""`uat` — pretty Typer + Rich front-end for the AIR UAT suites.

Two surfaces, one entrypoint (installed console script `uat`, or `uv run uat`):

    uat list                            # matrix; Enter submits the highlighted cell, then `air list runs`
                                        # `c` configures wheel deps for the highlighted row
    uat check                           # CI-able: manifest YAMLs + registry links exist
    uat config show|set|clear|wheels    # persisted wheels_root + per-workload .whl lists
    uat run multinode                   # TTY: pick cells, submit, then `air list runs`
    uat run multinode --hw 8xh100       # scripted: every row on 8×H100
    uat run notebook  [--shapes S]      # single-node check DRIVER as a one-time job
    uat status <run-id>...              # re-attach (TTY: `air list runs`)

This file is presentation only — every bit of real logic (repo detection, `air run`
submission, polling, the integrity check, receipt text) lives in uat_core, which is
stdlib-only. Prefs live in uat_prefs (~/.air-lab/config.json). The matrix picker lives
in uat_tui (Rich); after submit (and Enter on `uat list`) we hand the TTY to
`air list runs`. Where the environment can't install typer/rich (the customer's
no-PyPI workspace), use the identical-behavior fallback:

    python3 -m utils.verification.uat_min <same args>

The repo-root `./uat` wrapper picks this pretty front-end when typer+rich import, else the
fallback — so the command works either way.

Ground rule reminder: distributed multi-node is **air CLI only** (docs/RUNNING-UAT.md); any
H100 tier refuses to submit without --confirm-spend. No workspace/profile is hardcoded — pass
--profile or set UAT_PROFILE.
"""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from typing import List, Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from . import uat_core as core
from . import uat_prefs
from . import uat_suite
from . import uat_tui

_STATE_STYLE = {
    "SUCCESS": "bold green", "DRY-OK": "bold green", "FAILED": "bold red",
    "TIMEDOUT": "bold red", "INTERNAL_ERROR": "bold red", "DRY-FAIL": "bold red",
    "SUBMIT-FAIL": "bold red", "CANCELED": "yellow", "SKIPPED": "yellow",
}

console = Console()


def _state_text(st: str) -> Text:
    return Text(st, style=_STATE_STYLE.get(st, "cyan"))


def _id_cell(run_id: str, url: str | None) -> Text:
    """Run id as a terminal hyperlink to its dashboard URL when the terminal supports it."""
    return Text(run_id, style=f"link {url}" if url else "dim")


def _air_list_runs(profile: str | None) -> int:
    """Hand the TTY to `air list runs` — the CLI's own navigable run picker."""
    cmd = ["air", *core.profile_args(profile), "list", "runs"]
    console.print(f"[dim]$ {' '.join(cmd)}[/]")
    try:
        return subprocess.call(cmd)
    except FileNotFoundError:
        console.print("[red]air not on PATH[/] — install with: uv tool install databricks-air")
        return 127


def _refresh_states(runs: dict, profile: str | None) -> dict:
    """Re-read AIR run state after the user leaves `air list runs`."""
    out = {}
    for name, it in runs.items():
        rid = it.get("run_id")
        if rid:
            out[name] = core.run_state(rid, profile)
    return out


def _poll(runs: dict, profile: str | None, interval: int = 30, state_fn=core.run_state) -> dict:
    """Poll {name: {run_id,...}} to terminal with a live-updating Rich table. `state_fn` returns
    a status dict (core.run_state for AIR runs; core.jobs_run_state for the notebook DRIVER).
    Returns {name: info-dict}."""
    ids = {n: it["run_id"] for n, it in runs.items() if it.get("run_id")}
    if not ids:             # nothing to poll (dry-run, or all submits failed) — no empty Live
        return {}
    states = {n: {"status": "PENDING"} for n in ids}

    def table() -> Table:
        t = Table(show_edge=False, pad_edge=False, box=None)
        t.add_column("run", style="cyan", no_wrap=True)
        t.add_column("id", no_wrap=True)
        t.add_column("state")
        for name in ids:
            info = states[name]
            t.add_row(name, _id_cell(ids[name], info.get("dashboard_url")),
                      _state_text(info.get("status", "UNKNOWN")))
        return t

    with Live(table(), console=console, refresh_per_second=4) as live:
        while any(states[n].get("status") not in core.TERMINAL for n in ids):
            time.sleep(interval)
            for name in ids:
                if states[name].get("status") in core.TERMINAL:
                    continue
                states[name] = state_fn(ids[name], profile)
            live.update(table())
    return states


# ── Typer app ────────────────────────────────────────────────────────────────────────────

app = typer.Typer(no_args_is_help=True, add_completion=False,
                  help="Launch the AIR UAT suites.", rich_markup_mode="rich")
run_app = typer.Typer(no_args_is_help=True, help="Submit a suite (multinode | notebook).")
config_app = typer.Typer(no_args_is_help=True,
                         help="Persisted wheels prefs (~/.air-lab/config.json).")
app.add_typer(run_app, name="run")
app.add_typer(config_app, name="config")

_PROFILE_OPT = typer.Option(None, "--profile", "-p", help="databricks profile (or set UAT_PROFILE)")
_REPO_OPT = typer.Option(None, "--repo", help="repo checkout root (default: walk up from CWD / $UAT_REPO)")


@config_app.command("show")
def config_show() -> None:
    """Print wheels_root and per-workload wheel lists."""
    cfg = uat_prefs.load()
    path = uat_prefs.config_path()
    console.print(f"[dim]{path}[/]")
    root = uat_prefs.get_wheels_root(cfg)
    console.print(f"wheels_root: [cyan]{root or '(unset)'}[/]")
    wheels = cfg.get("workload_wheels") or {}
    if not wheels:
        console.print("workload_wheels: [dim](none)[/]")
        raise typer.Exit(0)
    console.print("workload_wheels:")
    for name in sorted(wheels):
        files = uat_prefs.get_workload_wheels(name, cfg)
        console.print(f"  [bold]{name}[/]: {', '.join(files) if files else '(empty)'}")
    raise typer.Exit(0)


@config_app.command("set")
def config_set(
    key: str = typer.Argument(..., help="wheels-root"),
    value: str = typer.Argument(..., help="absolute /Volumes/... or /Workspace/... path"),
) -> None:
    """Set a config key (currently: wheels-root)."""
    if key.replace("_", "-") != "wheels-root":
        console.print(f"[red]unknown key:[/] {key}  (supported: wheels-root)")
        raise typer.Exit(2)
    path = uat_prefs.set_wheels_root(value)
    console.print(f"wheels_root = [cyan]{uat_prefs.get_wheels_root()}[/]  [dim]→ {path}[/]")
    raise typer.Exit(0)


@config_app.command("clear")
def config_clear(
    key: str = typer.Argument(..., help="wheels-root"),
) -> None:
    """Clear a config key (currently: wheels-root)."""
    if key.replace("_", "-") != "wheels-root":
        console.print(f"[red]unknown key:[/] {key}  (supported: wheels-root)")
        raise typer.Exit(2)
    path = uat_prefs.set_wheels_root(None)
    console.print(f"wheels_root cleared  [dim]→ {path}[/]")
    raise typer.Exit(0)


@config_app.command("wheels")
def config_wheels(
    workload: str = typer.Argument(..., help="suite item name (e.g. allreduce)"),
    set_list: Optional[str] = typer.Option(None, "--set",
                                           help="comma-separated .whl filenames under wheels_root"),
    clear: bool = typer.Option(False, "--clear", help="remove this workload's wheel list"),
) -> None:
    """List or set wheel filenames attached to a suite item."""
    name = uat_suite.ALIASES.get(workload, workload)
    known = {it["name"] for it in uat_suite.ITEMS}
    if name not in known:
        console.print(f"[yellow]warning:[/] {name!r} is not a suite item "
                      f"(known: {', '.join(sorted(known))})")
    if clear and set_list is not None:
        console.print("[red]use either --set or --clear, not both[/]")
        raise typer.Exit(2)
    if clear:
        path = uat_prefs.set_workload_wheels(name, None)
        console.print(f"[cyan]{name}[/] wheels cleared  [dim]→ {path}[/]")
        raise typer.Exit(0)
    if set_list is not None:
        files = [p.strip() for p in set_list.split(",") if p.strip()]
        path = uat_prefs.set_workload_wheels(name, files)
        console.print(f"[cyan]{name}[/] = {files or '(empty)'}  [dim]→ {path}[/]")
        raise typer.Exit(0)
    files = uat_prefs.get_workload_wheels(name)
    root = uat_prefs.get_wheels_root()
    console.print(f"[cyan]{name}[/]: {', '.join(files) if files else '(none)'}")
    if files and root:
        console.print("[dim]resolved:[/]")
        for p in uat_prefs.resolve_dep_paths(name) or []:
            console.print(f"  {p}")
    elif files and not root:
        console.print("[yellow]wheels_root unset — submit will not inject deps until set[/]")
    raise typer.Exit(0)


@app.command("list")
def list_(
    tier: Optional[str] = typer.Option(None, help="pre-select this tier's default cells (headline | fabric | all)"),
    profile: Optional[str] = _PROFILE_OPT,
    repo: Optional[str] = _REPO_OPT,
) -> None:
    """Interactive matrix. Enter submits the highlighted cell, then `air list runs`. `l` lists without submitting. `c` configures wheel deps for the row."""
    profile = profile or os.environ.get("UAT_PROFILE")
    if console.is_terminal and sys.stdin.isatty():
        rows = list(uat_suite.ITEMS)
        # No auto-select: Enter runs the cell under the cursor, not a hidden set.
        seed = _preselect(tier, uat_suite.items_for(tier) if tier and tier != "all" else rows) if tier else set()
        try:
            result = uat_tui.pick(rows, seed, console, confirm_spend=False, profile=profile)
        except Exception as e:
            console.print(f"[red]matrix failed:[/] {e}")
            raise typer.Exit(2)
        if result == uat_tui.HANDOFF_AIR_LIST:
            raise typer.Exit(_air_list_runs(profile))
        if not result:
            raise typer.Exit(0)
        raise typer.Exit(_submit_launches(
            result, tier=tier or "pick", profile=profile,
            repo_root=core.find_repo(repo), tty=True,
        ))

    console.print(Rule("[bold]AIR UAT matrix[/]  ·  items × GPU_8xH100"))
    console.print("[bold]multinode[/] — distributed, [italic]air CLI only[/] "
                  "(docs/RUNNING-UAT.md). ◆ = default SKU for that row.\n")
    console.print(uat_tui.matrix_table(uat_suite.ITEMS))
    console.print(uat_tui._legend())
    console.print()
    for tname, spec in uat_suite.TIERS.items():
        badge = Text(" H100 · needs --confirm-spend ", style="black on yellow")
        console.print(Text.assemble((f"  {tname}", "bold"), "  ", badge,
                                    (f"  {spec['desc']}", "dim")))
    n = uat_suite.NOTEBOOK_SUITE
    console.print()
    console.print("[bold]notebook[/] — single-node checks via the DRIVER notebook "
                  f"[dim]({n['default_mirror']}/{n['driver_notebook']}, env v{n['environment_version']})[/]")
    console.print(Rule(style="dim"))
    console.print("run:  [cyan]uat list[/]                                           # TTY: Enter submits the highlighted cell")
    console.print("      [cyan]uat run multinode --hw 8xh100 --confirm-spend[/]      # every row on 8×H100")
    console.print("      [cyan]uat run notebook --shapes GPU_1xA10 --profile <p>[/]")
    console.print("      [cyan]uat config show[/]                                    # wheels_root + per-workload lists")


@app.command()
def check(repo: Optional[str] = _REPO_OPT) -> None:
    """Integrity check: every manifest YAML exists and every registry_id is real (CI-able)."""
    errors, n_items, n_tiers = core.check(core.find_repo(repo))
    for e in errors:
        console.print(f"[bold red]SUITE DRIFT[/] {e}")
    style = "green" if not errors else "red"
    console.print(f"[{style}]uat check: {n_items} items across {n_tiers} tiers, "
                  f"{len(errors)} problem(s)[/]")
    raise typer.Exit(1 if errors else 0)


def _submit_launches(
    launches: list[dict], *, tier: str, profile: str | None, repo_root: str,
    dry_run: bool = False, sequential: bool = False, no_poll: bool = False,
    no_watch: bool = False, tty: bool = False, extra_overrides: list[str] | None = None,
) -> int:
    """Submit pinned cells, then hand the TTY to `air list runs`. Returns process exit code."""
    flags = ("" if not dry_run else " [DRY-RUN]") + ("" if not sequential else " [sequential]")
    console.print(f"[dim]{core.now()}[/]  submitting {len(launches)} cell(s){flags}, "
                  f"profile={profile or '(ambient)'}")

    runs: dict = {}
    final: dict = {}
    for it in launches:
        name = it["key"]
        rid, status, detail = core.submit(it, profile, dry_run, repo_root, extra_overrides)
        runs[name] = {"item": it, "run_id": rid}
        if dry_run:
            ok = status == "DRY_RUN_OK"
            final[name] = {"status": "DRY-OK" if ok else "DRY-FAIL", "detail": "" if ok else detail}
            msg = "[green]DRY-OK[/] (config validated, no job submitted)" if ok \
                else f"[red]DRY-FAIL:[/] {detail[:80]}"
        elif rid:
            info = core.run_state(rid, profile)
            final[name] = info
            url = info.get("dashboard_url", "")
            link = f"  [link={url}]{url}[/link]" if url else ""
            msg = f"run {rid}  {info.get('status', '')}{link}"
        else:
            final[name] = {"status": "SUBMIT-FAIL", "detail": detail}
            msg = f"[red]SUBMIT-FAIL:[/] {detail[:80]}"
        console.print(f"[dim]{core.now()}[/]  [cyan]{name:32}[/] -> {msg}")
        if sequential and rid and not dry_run and not no_poll:
            final.update(_poll({name: runs[name]}, profile))

    if no_poll:
        ids = [r["run_id"] for r in runs.values() if r["run_id"]]
        again = "air list runs" + (f" -p {profile}" if profile else "")
        console.print(f"\nsubmitted; not polling. Re-attach: [cyan]{again}[/]")
        return 0

    if not sequential:
        use_air = tty and not dry_run and not no_watch and any(r.get("run_id") for r in runs.values())
        if use_air:
            _air_list_runs(profile)
            final.update(_refresh_states(runs, profile))
        else:
            final.update(_poll(runs, profile))

    _print_receipt(tier, runs, final, profile)
    bad = [n for n, info in final.items() if info.get("status") not in ("SUCCESS", "DRY-OK")]
    return 1 if bad else 0


def _print_receipt(tier: str, runs: dict, final: dict, profile: str | None) -> None:
    """Matrix (Rich) + a plain-text NOTES.md-pasteable block (the show-your-work receipt)."""
    console.print(Rule(f"[bold]multinode tier '{tier}' — results[/]"))
    t = Table(show_edge=True)
    t.add_column("item", style="bold cyan")
    t.add_column("shape")
    t.add_column("run id", style="dim")
    t.add_column("state")
    t.add_column("proves")
    for name, it in runs.items():
        info = final.get(name, {"status": "(not submitted)"})
        st = info.get("status", "—")
        reg = it["item"]["registry_id"]
        proves = "dry" if it["item"].get("dry") else (f"[{reg}]" if reg else "")
        t.add_row(name, it["item"].get("shape", ""), _id_cell(it.get("run_id") or "—", info.get("dashboard_url")),
                  _state_text(st), proves)
    console.print(t)
    console.print(Rule("paste into the matching experiments/*/NOTES.md", style="dim"))
    # plain print (no Rich markup/box) so the block copies clean
    for line in core.receipt_lines(tier, runs, final, profile):
        print(line)


def _preselect(tier: str, items: list[dict]) -> set[tuple[str, str]]:
    """Default cells for a tier: each row at that tier's default SKU."""
    out = set()
    for it in items:
        hid = uat_suite.default_hw_for(it, tier)
        if hid:
            out.add((it["name"], hid))
    return out


@run_app.command()
def multinode(
    tier: Optional[str] = typer.Option(None, help="headline | fabric | all  "
                                        "(rows / default cells; default headline, or all when --hw is set)"),
    only: Optional[str] = typer.Option(None, help="comma list of item names (aliases: allreduce-fabric, multinode-probe)"),
    hw: Optional[str] = typer.Option(None, "--hw", help="SKU column(s): 8xh100 / h100 / GPU_8xH100  "
                                     "(skip the picker; run every selected row on these columns)"),
    pick: bool = typer.Option(False, "--pick", help="force the interactive matrix even if --hw is set"),
    no_pick: bool = typer.Option(False, "--no-pick", help="never open the picker (scripted / CI)"),
    profile: Optional[str] = _PROFILE_OPT,
    override: Optional[List[str]] = typer.Option(None, "--override", "-o", metavar="KEY=VALUE",
                                                 help="raw KEY=VALUE passed straight to air --override on every run "
                                                      "(repeatable), e.g. -o usage_policy_name='my policy'"),
    confirm_spend: bool = typer.Option(False, "--confirm-spend", help="required for any H100 cell"),
    dry_run: bool = typer.Option(False, "--dry-run", help="air run --dry-run: validate, no GPU spend"),
    print_only: bool = typer.Option(False, "--print-only", help="print the air commands, submit nothing"),
    sequential: bool = typer.Option(False, "--sequential", help="submit+poll one at a time (avoid quota contention)"),
    no_poll: bool = typer.Option(False, "--no-poll", help="submit and exit without handing off to air list runs"),
    no_watch: bool = typer.Option(False, "--no-watch", help="poll a status table instead of handing off to air list runs"),
    repo: Optional[str] = _REPO_OPT,
) -> None:
    """Launch distributed items. TTY: pick cells, submit, then `air list runs`."""
    repo_root = core.find_repo(repo)
    extra_overrides = list(override or [])
    oerr = core.check_overrides(extra_overrides)
    if oerr:
        console.print(f"[red]{oerr}[/]")
        raise typer.Exit(2)
    tty = bool(console.is_terminal and sys.stdin.isatty())
    use_pick = (pick or (tty and hw is None)) and not no_pick
    # --hw names a column: unless the caller also named a tier, use every row that
    # has a cell in that column. No --hw → headline default.
    tier = tier or ("all" if hw else "headline")

    if use_pick:
        rows, err = core.select_items("all" if not only else tier, only)
        if err:
            console.print(f"[red]{err}[/]")
            raise typer.Exit(2)
        # Full matrix unless --only filtered it; --tier only drives which cells start selected.
        if not only:
            rows = list(uat_suite.ITEMS)
            seed_rows = uat_suite.items_for(tier) if tier != "all" else rows
        else:
            seed_rows = rows
        try:
            launches = uat_tui.pick(rows, _preselect(tier, seed_rows), console, confirm_spend,
                                    profile=profile or os.environ.get("UAT_PROFILE"))
        except Exception as e:
            console.print(f"[red]picker failed:[/] {e}")
            raise typer.Exit(2)
        if launches == uat_tui.HANDOFF_AIR_LIST:
            raise typer.Exit(_air_list_runs(profile or os.environ.get("UAT_PROFILE")))
        if launches is None:
            console.print("[dim]cancelled.[/]")
            raise typer.Exit(0)
        # picker already confirmed H100 spend interactively
        confirm_spend = True
    else:
        launches, err = core.select_launches(tier, only, hw)
        if err:
            console.print(f"[red]{err}[/]")
            raise typer.Exit(2)

    spendy = [it for it in launches if uat_suite.is_spendy(it)]
    if spendy and not (confirm_spend or dry_run):
        console.print("[bold red]Refusing:[/] these cells burn real H100 money — coordinate in the "
                      "team channel, then re-run with [cyan]--confirm-spend[/] (or [cyan]--dry-run[/] "
                      "to validate config with no GPU spend):")
        for it in spendy:
            console.print(f"  [cyan]{it['key']:32}[/] {it['shape']}")
        raise typer.Exit(3)

    profile = profile or os.environ.get("UAT_PROFILE")

    if print_only:
        for it in launches:
            tag = "  [dim][dry][/]" if dry_run else ""
            console.print(f"[dim]#[/] [bold cyan]{it['key']}[/]  ({it['shape']}){tag}")
            print("  " + shlex.join(core.air_cmd(it, profile, dry_run, repo_root, extra_overrides)))
        raise typer.Exit(0)

    raise typer.Exit(_submit_launches(
        launches, tier=tier, profile=profile, repo_root=repo_root,
        dry_run=dry_run, sequential=sequential, no_poll=no_poll,
        no_watch=no_watch, tty=tty, extra_overrides=extra_overrides,
    ))


@run_app.command()
def notebook(
    shapes: str = typer.Option("GPU_1xA10", help="GPU_1xA10 (default) | all | comma list"),
    pool: str = typer.Option("off", help="on = arm the 20-node pool sweep (coordinate first)"),
    profile: Optional[str] = _PROFILE_OPT,
    mirror: Optional[str] = typer.Option(None, help="workspace mirror dir holding DRIVER (default from manifest)"),
    confirm_spend: bool = typer.Option(False, "--confirm-spend", help="required for shapes=all (8xH100)"),
    print_only: bool = typer.Option(False, "--print-only"),
    no_poll: bool = typer.Option(False, "--no-poll"),
) -> None:
    """Submit the single-node check DRIVER as a one-time notebook job."""
    profile = profile or os.environ.get("UAT_PROFILE")
    spec = core.notebook_spec(shapes, pool, mirror)
    submit_cmd = ["databricks", "jobs", "submit", "--json", json.dumps(spec), *core.profile_args(profile)]
    if print_only:
        console.print("[dim]# submits the single-node check DRIVER as a one-time notebook job:[/]")
        console.print(f"[dim]#   databricks jobs submit --json <spec>"
                      f"{' -p ' + profile if profile else ''}[/]")
        console.print("[dim]# RunSubmit spec:[/]")
        print(json.dumps(spec, indent=2))
        raise typer.Exit(0)
    if shapes.lower() == "all" and not confirm_spend:
        console.print("[red]shapes=all includes 8xH100 (real money). Re-run with --confirm-spend.[/]")
        raise typer.Exit(3)

    console.print(f"[dim]{core.now()}[/]  submitting DRIVER (shapes={shapes} pool={pool}) "
                  f"profile={profile or '(ambient)'}")
    r = subprocess.run(submit_cmd, capture_output=True, text=True, timeout=120)
    try:
        rid = str(json.loads(r.stdout)["run_id"])
    except Exception:
        console.print(f"[red]submit failed:[/] {(r.stderr or r.stdout)[-300:]}")
        raise typer.Exit(1)
    info0 = core.jobs_run_state(rid, profile)
    url = info0.get("dashboard_url", "")
    console.print(f"[dim]{core.now()}[/]  DRIVER run [cyan]{rid}[/]"
                  + (f"  [link={url}]{url}[/link]" if url else ""))
    if no_poll:
        again = f"uat status {rid}" + (f" --profile {profile}" if profile else "")
        console.print(f"not polling. Re-attach: [cyan]{again}[/]")
        raise typer.Exit(0)
    final = _poll({"driver": {"run_id": rid}}, profile, state_fn=core.jobs_run_state)
    st = final.get("driver", {}).get("status", "UNKNOWN")
    console.print(Text.assemble((f"DRIVER {rid} -> ", ""), _state_text(st),
                                (f".  matrix is the notebook output (Jobs → Job runs → run {rid}).", "")))
    raise typer.Exit(0 if st == "SUCCESS" else 1)


@app.command()
def status(
    run_ids: List[str] = typer.Argument(..., help="run ids to poll to terminal"),
    profile: Optional[str] = _PROFILE_OPT,
) -> None:
    """Poll already-submitted run ids (TTY: `air list runs`)."""
    profile = profile or os.environ.get("UAT_PROFILE")
    runs = {rid: {"item": {"name": rid, "shape": "", "nodes": 2, "key": rid, "dry": False,
                           "registry_id": None}, "run_id": rid} for rid in run_ids}
    if console.is_terminal and sys.stdin.isatty():
        _air_list_runs(profile)
        final = _refresh_states(runs, profile)
    else:
        final = _poll(runs, profile)
    raise typer.Exit(0 if all(info.get("status") == "SUCCESS" for info in final.values()) else 1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
