"""`uat` — stdlib-only fallback front-end (no Typer, no Rich).

Same commands and behavior as the pretty front-end (uat_cli.py); plain text output. Use this
where the environment has no PyPI access to install typer/rich (e.g. the customer's hardened
workspace) — it needs only python3 + this repo checkout:

    python3 -m utils.verification.uat_min list
    python3 -m utils.verification.uat_min config show
    python3 -m utils.verification.uat_min run multinode --tier headline --profile <p>

The repo-root `./uat` wrapper falls back to this automatically when typer/rich are absent.
All real logic lives in uat_core (also stdlib-only); this file is just argparse + print.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time

from . import uat_core as core
from . import uat_prefs
from . import uat_suite


def _poll(runs: dict, profile: str | None, interval: int = 30, state_fn=core.run_state) -> dict:
    """Poll {name: {run_id,...}} to terminal, printing one line per cycle. `state_fn` returns a
    status dict (core.run_state for AIR runs; core.jobs_run_state for the notebook DRIVER)."""
    pending = {n: it["run_id"] for n, it in runs.items() if it.get("run_id")}
    final: dict = {}
    while pending:
        time.sleep(interval)
        for name, rid in list(pending.items()):
            info = state_fn(rid, profile)
            st = info.get("status", "UNKNOWN")
            if st in core.TERMINAL:
                final[name] = info
                del pending[name]
                print(f"{core.now()}  {name:24} {rid}  -> {st}")
            else:
                print(f"{core.now()}  {name:24} {rid}  ... {st}")
    return final


def cmd_list(_args) -> int:
    print("AIR UAT matrix  (items × GPU_8xH100)")
    print("  ◆ = default SKU for that row   · = n/a   (needs --confirm-spend)\n")
    cols = [h["short"] for h in uat_suite.HARDWARE]
    print(f"  {'item':24} {'nodes':>5}  " + "  ".join(f"{c:>8}" for c in cols))
    for it in uat_suite.ITEMS:
        cells = []
        for h in uat_suite.HARDWARE:
            if h["id"] not in it["hardware"]:
                cells.append(f"{'·':>8}")
            elif h["id"] == it["default_hw"]:
                cells.append(f"{'◆':>8}")
            else:
                cells.append(f"{'○':>8}")
        print(f"  {it['name']:24} {it['nodes']:5}  " + "  ".join(cells))
    print("\n  tiers:")
    for tier, spec in uat_suite.TIERS.items():
        gate = "  ! H100 — needs --confirm-spend"
        print(f"    {tier:12}{gate}")
        print(f"      {spec['desc']}")
    n = uat_suite.NOTEBOOK_SUITE
    print("\nnotebook — single-node checks via the DRIVER notebook")
    print(f"    mirror {n['default_mirror']}/{n['driver_notebook']}, env v{n['environment_version']}")
    print("\nrun:  uat run multinode --hw 8xh100 --confirm-spend --no-pick")
    print("      uat run multinode --tier headline --confirm-spend --no-pick")
    print("      uat run notebook --shapes GPU_1xA10 --profile <p>")
    print("  (pretty front-end: `uat run multinode` opens the matrix picker + log follow)")
    return 0


def cmd_check(args) -> int:
    repo = core.find_repo(args.repo)
    errors, n_items, n_tiers = core.check(repo)
    for e in errors:
        print("SUITE DRIFT:", e)
    print(f"uat check: {n_items} items across {n_tiers} tiers, {len(errors)} problem(s)")
    return 1 if errors else 0


def cmd_run_multinode(args) -> int:
    repo = core.find_repo(args.repo)
    tier = args.tier or ("all" if args.hw else "headline")
    items, err = core.select_launches(tier, args.only, args.hw)
    if err:
        print(err)
        return 2

    pol_overrides, perr = core.policy_override(args.policy, args.policy_id)
    if perr:
        print(perr)
        return 2

    spendy = [it for it in items if uat_suite.is_spendy(it)]
    if spendy and not (args.confirm_spend or args.dry_run):
        print("Refusing: these cells burn real H100 money — coordinate in the team channel, then")
        print("re-run with --confirm-spend (or --dry-run to validate config with no GPU spend):")
        for it in spendy:
            print(f"  {it['key']:32} {it['shape']}")
        return 3

    profile = args.profile or os.environ.get("UAT_PROFILE")

    if args.print_only:
        for it in items:
            print(f"# {it['key']}  ({it['shape']}){'  [dry]' if args.dry_run else ''}")
            print("  " + shlex.join(core.air_cmd(it, profile, args.dry_run, repo, pol_overrides)))
        return 0

    flags = ("" if not args.dry_run else " [DRY-RUN]") + ("" if not args.sequential else " [sequential]")
    print(f"{core.now()}  submitting {len(items)} cell(s){flags}, "
          f"profile={profile or '(ambient)'}")

    runs: dict = {}
    final: dict = {}
    for it in items:
        name = it["key"]
        rid, status, detail = core.submit(it, profile, args.dry_run, repo, pol_overrides)
        runs[name] = {"item": it, "run_id": rid}
        if args.dry_run:
            ok = status == "DRY_RUN_OK"
            final[name] = {"status": "DRY-OK" if ok else "DRY-FAIL", "detail": "" if ok else detail}
            print(f"{core.now()}  {name:32} -> "
                  f"{'DRY-OK (config validated, no job submitted)' if ok else 'DRY-FAIL: ' + detail[:80]}")
        elif rid:
            info = core.run_state(rid, profile)
            final[name] = info
            print(f"{core.now()}  {name:32} -> run {rid}  {info.get('status', '')}")
            if info.get("dashboard_url"):
                print(f"                              {info['dashboard_url']}")
            if args.sequential and not args.no_poll:
                final.update(_poll({name: runs[name]}, profile))
        else:
            final[name] = {"status": "SUBMIT-FAIL", "detail": detail}
            print(f"{core.now()}  {name:32} -> SUBMIT-FAIL: {detail[:80]}")

    if not args.no_poll and not args.sequential:
        final.update(_poll(runs, profile))

    if args.no_poll:
        ids = [r["run_id"] for r in runs.values() if r["run_id"]]
        print(f"\nsubmitted; not polling. Re-attach: uat status {' '.join(ids)}"
              + (f" --profile {profile}" if profile else ""))
        return 0

    print("\n" + "=" * 78)
    print(f"multinode tier '{tier}' — results")
    print(f"{'item':32} {'shape':14} {'run id':16} {'state':14} proves")
    for name, it in runs.items():
        st = final.get(name, {}).get("status", "(not submitted)")
        reg = it["item"]["registry_id"]
        proves = "dry" if it["item"].get("dry") else (f"[{reg}]" if reg else "")
        print(f"{name:32} {it['item'].get('shape',''):14} {(it.get('run_id') or '—'):16} {st:14} {proves}")
    print("\n--- paste into the matching experiments/*/NOTES.md ---")
    for line in core.receipt_lines(tier, runs, final, profile):
        print(line)
    print("=" * 78)
    return 1 if any(i.get("status") not in ("SUCCESS", "DRY-OK") for i in final.values()) else 0


def cmd_run_notebook(args) -> int:
    profile = args.profile or os.environ.get("UAT_PROFILE")
    spec = core.notebook_spec(args.shapes, args.pool, args.mirror)
    submit_cmd = ["databricks", "jobs", "submit", "--json", json.dumps(spec), *core.profile_args(profile)]
    if args.print_only:
        print("# submits the single-node check DRIVER as a one-time notebook job:")
        print(f"#   databricks jobs submit --json <spec>{' -p ' + profile if profile else ''}")
        print("# RunSubmit spec:")
        print(json.dumps(spec, indent=2))
        return 0
    if args.shapes.lower() == "all" and not args.confirm_spend:
        print("shapes=all includes 8xH100 (real money). Re-run with --confirm-spend.")
        return 3
    print(f"{core.now()}  submitting DRIVER (shapes={args.shapes} pool={args.pool}) "
          f"profile={profile or '(ambient)'}")
    r = subprocess.run(submit_cmd, capture_output=True, text=True, timeout=120)
    try:
        rid = str(json.loads(r.stdout)["run_id"])
    except Exception:
        print("submit failed:", (r.stderr or r.stdout)[-300:])
        return 1
    info0 = core.jobs_run_state(rid, profile)
    print(f"{core.now()}  DRIVER run {rid}" + (f"  {info0['dashboard_url']}" if info0.get("dashboard_url") else ""))
    if args.no_poll:
        print(f"not polling. Re-attach: uat status {rid}" + (f" --profile {profile}" if profile else ""))
        return 0
    final = _poll({"driver": {"run_id": rid}}, profile, state_fn=core.jobs_run_state)
    st = final.get("driver", {}).get("status", "UNKNOWN")
    print(f"\nDRIVER {rid} -> {st}. The aggregated matrix is the notebook output "
          f"(Jobs → Job runs → run {rid}).")
    return 0 if st == "SUCCESS" else 1


def cmd_status(args) -> int:
    profile = args.profile or os.environ.get("UAT_PROFILE")
    final = _poll({rid: {"run_id": rid} for rid in args.run_ids}, profile)
    print("\nstate:")
    for rid, info in final.items():
        print(f"  {rid}  {info.get('status', '—')}  {info.get('dashboard_url', '')}")
    return 0 if all(i.get("status") == "SUCCESS" for i in final.values()) else 1


def cmd_config_show(_args) -> int:
    cfg = uat_prefs.load()
    print(uat_prefs.config_path())
    root = uat_prefs.get_wheels_root(cfg)
    print(f"wheels_root: {root or '(unset)'}")
    wheels = cfg.get("workload_wheels") or {}
    if not wheels:
        print("workload_wheels: (none)")
        return 0
    print("workload_wheels:")
    for name in sorted(wheels):
        files = uat_prefs.get_workload_wheels(name, cfg)
        print(f"  {name}: {', '.join(files) if files else '(empty)'}")
    return 0


def cmd_config_set(args) -> int:
    key = args.key.replace("_", "-")
    if key != "wheels-root":
        print(f"unknown key: {args.key}  (supported: wheels-root)")
        return 2
    path = uat_prefs.set_wheels_root(args.value)
    print(f"wheels_root = {uat_prefs.get_wheels_root()}  → {path}")
    return 0


def cmd_config_clear(args) -> int:
    key = args.key.replace("_", "-")
    if key != "wheels-root":
        print(f"unknown key: {args.key}  (supported: wheels-root)")
        return 2
    path = uat_prefs.set_wheels_root(None)
    print(f"wheels_root cleared  → {path}")
    return 0


def cmd_config_wheels(args) -> int:
    name = uat_suite.ALIASES.get(args.workload, args.workload)
    known = {it["name"] for it in uat_suite.ITEMS}
    if name not in known:
        print(f"warning: {name!r} is not a suite item (known: {', '.join(sorted(known))})")
    if args.clear and args.set is not None:
        print("use either --set or --clear, not both")
        return 2
    if args.clear:
        path = uat_prefs.set_workload_wheels(name, None)
        print(f"{name} wheels cleared  → {path}")
        return 0
    if args.set is not None:
        files = [p.strip() for p in args.set.split(",") if p.strip()]
        path = uat_prefs.set_workload_wheels(name, files)
        print(f"{name} = {files or '(empty)'}  → {path}")
        return 0
    files = uat_prefs.get_workload_wheels(name)
    root = uat_prefs.get_wheels_root()
    print(f"{name}: {', '.join(files) if files else '(none)'}")
    if files and root:
        print("resolved:")
        for p in uat_prefs.resolve_dep_paths(name) or []:
            print(f"  {p}")
    elif files and not root:
        print("wheels_root unset — submit will not inject deps until set")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="uat", description="Launch the AIR UAT suites (stdlib fallback).")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="show suites, tiers, shapes, cost gates").set_defaults(fn=cmd_list)
    ck = sub.add_parser("check", help="integrity check (YAMLs + registry links exist)")
    ck.add_argument("--repo", help="repo checkout root (default: walk up from CWD / $UAT_REPO)")
    ck.set_defaults(fn=cmd_check)

    st = sub.add_parser("status", help="poll already-submitted run ids to terminal")
    st.add_argument("run_ids", nargs="+")
    st.add_argument("--profile", "-p")
    st.set_defaults(fn=cmd_status)

    cfg = sub.add_parser("config", help="persisted wheels prefs (~/.air-lab/config.json)")
    cfg_sub = cfg.add_subparsers(dest="config_cmd", required=True)
    cfg_sub.add_parser("show", help="print wheels_root + workload lists").set_defaults(fn=cmd_config_show)
    cs = cfg_sub.add_parser("set", help="set a key (wheels-root)")
    cs.add_argument("key")
    cs.add_argument("value")
    cs.set_defaults(fn=cmd_config_set)
    cc = cfg_sub.add_parser("clear", help="clear a key (wheels-root)")
    cc.add_argument("key")
    cc.set_defaults(fn=cmd_config_clear)
    cw = cfg_sub.add_parser("wheels", help="list/set wheel filenames for a suite item")
    cw.add_argument("workload")
    cw.add_argument("--set", help="comma-separated .whl filenames under wheels_root")
    cw.add_argument("--clear", action="store_true")
    cw.set_defaults(fn=cmd_config_wheels)

    run = sub.add_parser("run", help="submit a suite").add_subparsers(dest="suite", required=True)

    mn = run.add_parser("multinode", help="distributed items via the air CLI")
    mn.add_argument("--tier", default=None, help="headline | fabric | all (default: headline, or all when --hw is set)")
    mn.add_argument("--only", help="comma list of item names within the tier")
    mn.add_argument("--hw", help="SKU column(s): 8xh100 / h100 / GPU_8xH100")
    mn.add_argument("--profile", "-p", help="databricks profile (or set UAT_PROFILE)")
    mn.add_argument("--policy", help="usage_policy_name to assign to every submitted run")
    mn.add_argument("--policy-id", help="usage_policy_id (UUID); mutually exclusive with --policy")
    mn.add_argument("--confirm-spend", action="store_true", help="required for any H100 item")
    mn.add_argument("--dry-run", action="store_true", help="air run --dry-run: validate, no GPU spend")
    mn.add_argument("--print-only", action="store_true", help="print the air commands, submit nothing")
    mn.add_argument("--sequential", action="store_true", help="submit+poll one at a time")
    mn.add_argument("--no-poll", action="store_true", help="submit and exit without polling")
    mn.add_argument("--repo", help="repo checkout root (default: walk up from CWD / $UAT_REPO)")
    mn.set_defaults(fn=cmd_run_multinode)

    nb = run.add_parser("notebook", help="single-node check DRIVER as a one-time job")
    nb.add_argument("--shapes", default="GPU_1xA10", help="GPU_1xA10 (default) | all | comma list")
    nb.add_argument("--pool", default="off", help="on = arm the 20-node pool sweep")
    nb.add_argument("--profile", "-p")
    nb.add_argument("--mirror", help="workspace mirror dir holding DRIVER (default from manifest)")
    nb.add_argument("--confirm-spend", action="store_true", help="required for shapes=all (8xH100)")
    nb.add_argument("--print-only", action="store_true")
    nb.add_argument("--no-poll", action="store_true")
    nb.set_defaults(fn=cmd_run_notebook)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
