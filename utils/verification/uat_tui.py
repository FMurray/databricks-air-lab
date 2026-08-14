"""Rich TUI for the UAT matrix: pick cells with the arrow keys.

Pretty-front-end only (typer + rich). The stdlib fallback never imports this file.

  pick()   — interactive matrix; space toggles a cell. Enter in run mode returns
             launches; Enter in browse (`uat list`) returns HANDOFF_AIR_LIST so
             the caller can exec `air list runs`.
  watch()  — leftover split-pane log follower (run/status now hand off to air)

Keyboard is termios cbreak + a buffered `os.read` of whole CSI packets (no prompt toolkit).
"""
from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import termios
import threading
import time
import tty
from collections import deque

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich import box as rich_box
from rich.table import Table
from rich.text import Text

from . import uat_core as core
from . import uat_prefs
from . import uat_suite

# `uat list` Enter: leave the matrix and exec the air CLI's own run picker.
HANDOFF_AIR_LIST = "air-list-runs"

# ── keyboard ──────────────────────────────────────────────────────────────────
# Arrow keys arrive as CSI (`\x1b[A`) or SS3 (`\x1bOA`), usually in one 3-byte
# packet. Two things made them need a fast double-tap:
#   1. `select()` on Darwin often does NOT report the rest of a CSI sequence as
#      readable after you've already `read()` the ESC — so a second select
#      times out and the `[A` sits in the TTY until the next keystroke.
#   2. `sys.stdin.read(1)` is the TextIOWrapper, which does not mix with
#      `select()` on the raw fd.
# Fix: `os.read(fd, n)` the whole packet, parse a buffer. If we only got ESC,
# drain the rest with termios VMIN=0/VTIME (a blocking timed read) — never a
# second select().

_ARROW_FINAL = {"A": "up", "B": "down", "C": "right", "D": "left"}


def _decode_escape(seq: str) -> str:
    """Map the bytes *after* ESC to a key name. Unknown sequences are ignored ('')."""
    if not seq:
        return "esc"
    if seq[0] == "O" and len(seq) >= 2:
        return _ARROW_FINAL.get(seq[1], "")
    if seq[0] != "[":
        return ""
    final = seq[-1]
    if final in _ARROW_FINAL:
        return _ARROW_FINAL[final]
    if seq.startswith("[5") or seq == "[I":
        return "pageup"
    if seq.startswith("[6") or seq == "[G":
        return "pagedown"
    return ""


def _pull_key(buf: bytearray) -> str | None:
    """Pop one complete key from `buf`. None = empty or incomplete (buf unchanged)."""
    if not buf:
        return None
    if buf[0] != 0x1B:
        b = buf.pop(0)
        if b in (0x0D, 0x0A):
            return "enter"
        if b == 0x20:
            return "space"
        if b == 0x03:
            return "ctrl-c"
        if b == 0x7F:
            return "backspace"
        try:
            return bytes([b]).decode("ascii")
        except UnicodeDecodeError:
            return None
    if len(buf) < 2:
        return None
    if buf[1] == ord("O"):  # SS3: ESC O A
        if len(buf) < 3:
            return None
        seq = chr(buf[2])
        del buf[:3]
        return _ARROW_FINAL.get(seq) or None
    if buf[1] == ord("["):  # CSI: ESC [ ... final
        i = 2
        while i < len(buf):
            if 0x40 <= buf[i] <= 0x7E:
                seq = bytes(buf[1:i + 1]).decode("ascii", "replace")
                del buf[:i + 1]
                return _decode_escape(seq) or None
            i += 1
        return None
    del buf[0]  # ESC + unknown — drop ESC
    return None


class KeyReader:
    """cbreak stdin → named keys. One instance per TUI screen; always os.read."""

    def __init__(self, fd: int | None = None):
        self.fd = sys.stdin.fileno() if fd is None else fd
        self.buf = bytearray()
        self._saved = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def close(self) -> None:
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)

    def __enter__(self) -> "KeyReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _drain_vtime(self, tenths: int = 1) -> None:
        """Timed blocking read (VMIN=0, VTIME=tenths). Does not use select()."""
        old = termios.tcgetattr(self.fd)
        try:
            new = termios.tcgetattr(self.fd)
            new[6] = list(new[6])
            new[6][termios.VMIN] = 0
            new[6][termios.VTIME] = tenths
            termios.tcsetattr(self.fd, termios.TCSANOW, new)
            chunk = os.read(self.fd, 64)
            if chunk:
                self.buf.extend(chunk)
        except OSError:
            pass
        finally:
            termios.tcsetattr(self.fd, termios.TCSANOW, old)

    def poll(self, timeout: float = 0.05) -> str | None:
        key = _pull_key(self.buf)
        if key is not None:
            return key
        if self.buf[:1] == b"\x1b":
            self._drain_vtime(1)
            key = _pull_key(self.buf)
            if key is not None:
                return key
            if self.buf[:1] == b"\x1b":
                del self.buf[0]  # lone ESC after waiting — ignore
            return None
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        try:
            chunk = os.read(self.fd, 4096)
        except OSError:
            return None
        if not chunk:
            return None
        self.buf.extend(chunk)
        key = _pull_key(self.buf)
        if key is not None:
            return key
        if self.buf[:1] == b"\x1b":
            self._drain_vtime(1)
            return _pull_key(self.buf)
        return None



# ── matrix rendering (shared by `uat list` and the picker) ────────────────────

_COST_STYLE = {"cheap": "green", "H100": "yellow"}
# Cursor col 0 = nodes editor; cols 1.. = HARDWARE[col-1]. Pool is 20 nodes.
_NODES_COL = 0
_MAX_NODES = 20


def _n_cols() -> int:
    return 1 + len(uat_suite.HARDWARE)


def _hw_at(col: int) -> str | None:
    """Hardware id for a cursor column, or None when on the nodes editor."""
    if col <= _NODES_COL:
        return None
    return uat_suite.HARDWARE[col - 1]["id"]


def matrix_table(items: list[dict], selected: set[tuple[str, str]] | None = None,
                 cursor: tuple[int, int] | None = None,
                 node_n: dict[str, int] | None = None,
                 edit_buf: str | None = None) -> Table:
    """Items × (nodes editor + SKU) grid.

    `selected` is {(name, hw_id)}; `cursor` is (row, col) with col 0 = nodes,
    col >= 1 = HARDWARE[col-1]. `node_n` overrides per-row topology; `edit_buf`
    is the in-progress digit string when the nodes cell is being typed.
    """
    selected = selected or set()
    t = Table(show_header=True, header_style="bold", box=None, pad_edge=False, expand=True)
    t.add_column("", min_width=22, no_wrap=True)
    t.add_column("tier", style="dim", width=18, no_wrap=True)
    t.add_column("nodes", justify="right", style="bold", width=7, no_wrap=True)
    for h in uat_suite.HARDWARE:
        t.add_column(h["short"], justify="center", min_width=10,
                     header_style=_COST_STYLE[h["cost"]])

    for r, it in enumerate(items):
        name = it["name"]
        n_val = (node_n or {}).get(name, it["nodes"])
        on_nodes = cursor == (r, _NODES_COL)
        if on_nodes and edit_buf is not None:
            nodes_body = Text(f"{edit_buf}_", style="bold reverse cyan")
        elif on_nodes:
            nodes_body = Text(f"[{n_val}]", style="bold reverse cyan")
        elif node_n is not None and n_val != it["nodes"]:
            nodes_body = Text(str(n_val), style="bold yellow")
        else:
            nodes_body = Text(str(n_val), style="dim")

        cells = [
            Text(name, style="bold cyan"),
            Text("/".join(it["tiers"]), style="dim"),
            nodes_body,
        ]
        for c, h in enumerate(uat_suite.HARDWARE):
            hid = h["id"]
            on_cursor = cursor == (r, c + 1)
            if hid not in it["hardware"]:
                body = Text("·", style="dim")
            else:
                on = (name, hid) in selected
                mark = "●" if on else "○"
                style = "bold reverse cyan" if on_cursor else (
                    "bold green" if on and not h["spendy"] else
                    "bold yellow" if on else
                    "dim yellow" if h["spendy"] else "dim green"
                )
                # default-hw hint when nothing's selected yet (list view)
                if not selected and hid == it["default_hw"] and not on_cursor:
                    mark, style = "◆", _COST_STYLE[h["cost"]]
                body = Text(f" {mark} ", style=style)
            if on_cursor and hid not in it["hardware"]:
                body = Text(" · ", style="reverse dim")
            cells.append(body)
        t.add_row(*cells)
    return t


def _legend() -> Text:
    return Text.assemble(
        ("  ○", "dim"), (" open  ", ""),
        ("●", "bold yellow"), (" selected  ", ""),
        ("◆", "yellow"), (" default  ", ""),
        ("[N]", "bold cyan"), (" nodes (edit)  ", ""),
        ("·", "dim"), (" n/a   ", ""),
        ("yellow", "yellow"), (" = 8×H100 (needs --confirm-spend)", ""),
    )


# ── picker ────────────────────────────────────────────────────────────────────

def _next_cell(items, row, col, drow, dcol) -> tuple[int, int]:
    """Move cursor, wrapping. Col 0 = nodes editor; cols ≥1 = HARDWARE (skip n/a)."""
    n, m = len(items), _n_cols()
    if not n:
        return 0, 0
    row = (row + drow) % n
    col = (col + dcol) % m
    for _ in range(max(n, m)):
        hid = _hw_at(col)
        if hid is None or hid in items[row]["hardware"]:
            return row, col
        if dcol:
            col = (col + dcol) % m
        else:
            row = (row + drow) % n
    return row, col


def _clamp_nodes(n: int) -> int:
    return max(1, min(_MAX_NODES, int(n)))


def _configure_wheels(kbd: KeyReader, live: Live, workload: str,
                      profile: str | None) -> str:
    """Overlay: set wheels_root + toggle .whl files for `workload`. Returns status msg.

    Keyboard: type to edit root → Enter saves root & lists wheels; ↑↓/space toggle;
    Enter saves list; a = type a filename; q back without saving list (root already saved).
    """
    root = uat_prefs.get_wheels_root() or ""
    root_buf = root
    editing_root = True
    typing_name = False
    name_buf = ""
    inventory: list[str] = []
    selected = set(uat_prefs.get_workload_wheels(workload))
    idx = 0
    err = ""
    note = ""

    def refresh_inventory() -> None:
        nonlocal inventory, err, note, idx
        r = uat_prefs.get_wheels_root()
        if not r:
            inventory = []
            err = "wheels_root unset — type a /Volumes or /Workspace path, Enter to save"
            note = ""
            return
        files, ls_err = uat_prefs.list_wheels_via_fs(r, profile)
        # Keep previously selected names even if ls failed / they're not listed.
        inventory = sorted(set(files) | selected)
        err = ls_err or ""
        note = (f"listed {len(files)} .whl under {r}" if not ls_err
                else f"fs ls failed — toggle known names or press a to add a filename")
        if idx >= len(inventory):
            idx = max(0, len(inventory) - 1)

    def render_overlay() -> Group:
        lines: list[Text | str] = [
            Text.assemble(
                ("configure wheel deps", "bold white on dodger_blue4"),
                (f"  ·  {workload}", "bold"),
            ),
            Text(""),
            Text.assemble(
                ("  wheels_root: ", "dim"),
                (root_buf + ("▌" if editing_root and not typing_name else ""),
                 "cyan" if editing_root else ""),
            ),
        ]
        if typing_name:
            lines.append(Text.assemble(
                ("  add filename: ", "yellow"),
                (name_buf + "▌", "bold"),
            ))
        lines.append(Text(""))
        if inventory:
            for i, fn in enumerate(inventory):
                mark = "●" if fn in selected else "○"
                style = "bold yellow" if i == idx and not editing_root and not typing_name else ""
                prefix = "▶ " if i == idx and not editing_root and not typing_name else "  "
                lines.append(Text(f"{prefix}{mark} {fn}", style=style))
        else:
            lines.append(Text("  (no .whl names yet)", style="dim"))
        lines.append(Text(""))
        if editing_root and not typing_name:
            keys = ("  type path   enter save root & list wheels   "
                    "tab → wheel list   q back")
        elif typing_name:
            keys = "  type .whl filename   enter add   esc cancel"
        else:
            keys = ("  ↑↓ move   space toggle   enter save list   "
                    "e edit root   a add filename   q back")
        lines.append(Text(keys, style="dim"))
        if note:
            lines.append(Text(f"  {note}", style="dim"))
        if err:
            lines.append(Text(f"  {err}", style="bold red"))
        return Group(*lines)

    refresh_inventory()
    live.update(render_overlay())
    while True:
        key = kbd.poll()
        if key is None:
            live.update(render_overlay())
            continue
        if key in ("q", "Q", "ctrl-c"):
            return f"{workload}: left configure (list not saved this visit)"
        if key == "esc":
            if typing_name:
                typing_name = False
                name_buf = ""
                live.update(render_overlay())
                continue
            return f"{workload}: left configure"
        if typing_name:
            if key == "enter":
                name = name_buf.strip()
                if name:
                    if not name.endswith(".whl"):
                        name += ".whl"
                    selected.add(name)
                    inventory = sorted(set(inventory) | {name})
                    idx = inventory.index(name)
                typing_name = False
                name_buf = ""
            elif key == "backspace":
                name_buf = name_buf[:-1]
            elif key is not None and len(key) == 1 and key.isprintable() and key != " ":
                name_buf += key
            live.update(render_overlay())
            continue
        if editing_root:
            if key == "enter":
                path = root_buf.strip().rstrip("/")
                uat_prefs.set_wheels_root(path or None)
                root = uat_prefs.get_wheels_root() or ""
                root_buf = root
                editing_root = False
                refresh_inventory()
            elif key == "tab":
                editing_root = False
            elif key == "backspace":
                root_buf = root_buf[:-1]
            elif key is not None and len(key) == 1 and key.isprintable():
                root_buf += key
            live.update(render_overlay())
            continue
        # wheel list mode
        if key in ("e", "E"):
            editing_root = True
            root_buf = uat_prefs.get_wheels_root() or root_buf
        elif key in ("a", "A"):
            typing_name = True
            name_buf = ""
        elif key in ("up", "k"):
            if inventory:
                idx = (idx - 1) % len(inventory)
        elif key in ("down", "j"):
            if inventory:
                idx = (idx + 1) % len(inventory)
        elif key == "space":
            if inventory:
                fn = inventory[idx]
                if fn in selected:
                    selected.discard(fn)
                else:
                    selected.add(fn)
        elif key == "enter":
            uat_prefs.set_workload_wheels(workload, sorted(selected))
            n = len(selected)
            root_now = uat_prefs.get_wheels_root()
            if n and not root_now:
                return f"{workload}: saved {n} wheel(s) but wheels_root unset — set root to inject on submit"
            return f"{workload}: saved {n} wheel(s)" + (f" under {root_now}" if root_now else "")
        live.update(render_overlay())


def pick(items: list[dict], preselected: set[tuple[str, str]], console: Console,
         confirm_spend: bool, *, browse: bool = False,
         profile: str | None = None) -> list[dict] | str | None:
    """Interactive matrix.

    Cursor col 0 edits node count for the row; cols ≥1 are SKU cells. Enter on a
    SKU cell runs (plus any space-toggled set). Enter on nodes commits a typed
    value. `c` opens wheel-deps configure for the current row. `l` returns
    `HANDOFF_AIR_LIST`. None = quit.
    `browse` is unused (kept so callers don't break).
    """
    if not items:
        return []
    # land on the first preselected SKU cell, else first supported SKU
    row = col = 0
    if preselected:
        for r, it in enumerate(items):
            for c, h in enumerate(uat_suite.HARDWARE):
                if (it["name"], h["id"]) in preselected:
                    row, col = r, c + 1
                    break
    else:
        row, col = _next_cell(items, 0, 0, 0, 1)

    selected = set(preselected)
    node_n = {it["name"]: int(it["nodes"]) for it in items}
    edit_buf: str | None = None
    err = ""
    status = ""

    def commit_edit() -> None:
        nonlocal edit_buf, err
        if edit_buf is None:
            return
        name = items[row]["name"]
        if edit_buf == "":
            edit_buf = None
            return
        try:
            node_n[name] = _clamp_nodes(int(edit_buf))
            edit_buf = None
        except ValueError:
            err = f"nodes must be an integer 1–{_MAX_NODES}"
            edit_buf = None

    def launches_for(sel: set[tuple[str, str]]) -> list[dict]:
        out = []
        for (n, hid) in sel:
            for it in items:
                if it["name"] == n and hid in it["hardware"]:
                    out.append(uat_suite.pin(it, hid, nodes=node_n[n]))
        return out

    def render() -> Group:
        launches = launches_for(selected)
        n_h100 = sum(1 for ln in launches if ln["spendy"])
        banner = Text.assemble(
            ("AIR UAT matrix", "bold white on dodger_blue4"),
            (f"  {len(launches)} cell(s)", "bold"),
            (f"  {n_h100} H100", "yellow" if n_h100 else "dim"),
        )
        if n_h100 and not confirm_spend:
            banner = Text.assemble(banner, ("   H100 spend — Enter will ask you to confirm", "yellow"))
        if col == _NODES_COL:
            keys = Text(
                f"  ↑↓ row   ←→ sku   digits type nodes   +/− step   enter commit   "
                f"backspace   (1–{_MAX_NODES})   c configure deps   l air list runs   q quit",
                style="dim",
            )
        else:
            keys = Text(
                "  ↑↓←→ move   space toggle   h select-all   n edit-nodes   "
                "+/− nodes   enter run this cell   c configure deps   l air list runs   q quit",
                style="dim",
            )
        note = Text("  " + items[row]["note"], style="yellow") if items else Text("")
        shape_hint = Text(
            f"  {items[row]['name']} → {node_n[items[row]['name']]}×"
            f"{uat_suite.HARDWARE[0]['short'] if uat_suite.HARDWARE else '?'}",
            style="dim",
        )
        wheels = uat_prefs.get_workload_wheels(items[row]["name"])
        root = uat_prefs.get_wheels_root()
        if wheels:
            deps_hint = Text(
                f"  wheels ({len(wheels)}): {', '.join(wheels[:3])}"
                + ("…" if len(wheels) > 3 else "")
                + (f"  @ {root}" if root else "  [wheels_root unset]"),
                style="cyan" if root else "yellow",
            )
        else:
            deps_hint = Text(
                f"  wheels: (none)  root={root or 'unset'}  — press c to configure",
                style="dim",
            )
        body = [banner, Text(""), matrix_table(items, selected, (row, col), node_n, edit_buf),
                Text(""), note, shape_hint, deps_hint, _legend(), keys]
        if status:
            body.append(Text(f"  {status}", style="green"))
        if err:
            body.append(Text(f"  {err}", style="bold red"))
        return Group(*body)

    try:
        with KeyReader() as kbd, Live(render(), console=console, screen=True,
                                      refresh_per_second=12) as live:
            while True:
                key = kbd.poll()
                err = ""
                if key is None:
                    live.update(render())
                    continue
                # q / ctrl-c quit. Do NOT quit on 'esc': a split arrow key is ESC+[A,
                # and mapping that to cancel is what printed "cancelled."
                if key in ("q", "Q", "ctrl-c"):
                    return None
                if key == "up" or key == "k":
                    commit_edit()
                    row, col = _next_cell(items, row, col, -1, 0)
                elif key == "down" or key == "j":
                    commit_edit()
                    row, col = _next_cell(items, row, col, 1, 0)
                elif key == "left":
                    commit_edit()
                    row, col = _next_cell(items, row, col, 0, -1)
                elif key == "right":
                    commit_edit()
                    row, col = _next_cell(items, row, col, 0, 1)
                elif key == "n" and col != _NODES_COL:
                    commit_edit()
                    row, col = row, _NODES_COL
                elif key in ("+", "="):
                    commit_edit()
                    name = items[row]["name"]
                    node_n[name] = _clamp_nodes(node_n[name] + 1)
                elif key == "-":
                    commit_edit()
                    name = items[row]["name"]
                    node_n[name] = _clamp_nodes(node_n[name] - 1)
                elif key is not None and len(key) == 1 and key.isdigit():
                    # Digits always edit this row's node count (focus nodes if needed).
                    col = _NODES_COL
                    edit_buf = (edit_buf or "") + key
                    if len(edit_buf) > 2:  # 1–20 fits in 2 digits
                        edit_buf = edit_buf[-2:]
                elif key == "backspace" and col == _NODES_COL:
                    if edit_buf is None:
                        edit_buf = str(node_n[items[row]["name"]])
                    edit_buf = edit_buf[:-1]
                elif key == "space":
                    commit_edit()
                    hid = _hw_at(col)
                    if hid is None:
                        err = "move right onto a SKU cell to toggle"
                    else:
                        it = items[row]
                        if hid not in it["hardware"]:
                            err = f"{it['name']} doesn't run on {hid}"
                        else:
                            cell = (it["name"], hid)
                            selected.symmetric_difference_update({cell})
                elif key in ("h", "H"):
                    commit_edit()
                    hid = "GPU_8xH100"
                    for it in items:
                        if hid in it["hardware"]:
                            selected.add((it["name"], hid))
                elif key in ("c", "C"):
                    commit_edit()
                    status = _configure_wheels(kbd, live, items[row]["name"], profile)
                elif key in ("l", "L"):
                    commit_edit()
                    return HANDOFF_AIR_LIST
                elif key == "enter":
                    if col == _NODES_COL:
                        if edit_buf is not None:
                            commit_edit()
                        else:
                            err = "type a number or +/−, then → onto a SKU cell to run"
                        live.update(render())
                        continue
                    commit_edit()
                    # Move + Enter runs the highlighted cell (space is multi-select).
                    it, hid = items[row], _hw_at(col)
                    if hid and hid in it["hardware"]:
                        selected.add((it["name"], hid))
                    launches = launches_for(selected)
                    if not launches:
                        err = "nothing to run — Enter on a live cell, or space to toggle"
                        live.update(render())
                        continue
                    if any(ln["spendy"] for ln in launches) and not confirm_spend:
                        live.update(Group(
                            render(),
                            Text(""),
                            Text("  H100 cells selected. Type Y to spend, any other key to keep editing.",
                                 style="bold yellow"),
                        ))
                        live.refresh()
                        k = None
                        while k is None:
                            k = kbd.poll(0.2)
                        if k not in ("y", "Y"):
                            err = "spend not confirmed — deselect H100 cells, or pass --confirm-spend"
                            live.update(render())
                            continue
                    return launches
                live.update(render())
    except KeyboardInterrupt:
        return None


# ── log follow ────────────────────────────────────────────────────────────────

class _Tail:
    """Background `air logs` follower. One per (run_id, node). Restartable."""

    def __init__(self, run_id: str, profile: str | None, node: int = 0, maxlen: int = 2500):
        self.run_id = run_id
        self.profile = profile
        self.node = node
        self.lines: deque[str] = deque(maxlen=maxlen)
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self.alive = True
        self.thread = threading.Thread(target=self._loop, name=f"tail-{run_id}-{node}", daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            cmd = core.logs_cmd(self.run_id, self.profile, self.node)
            try:
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1,
                )
            except OSError as e:
                self.lines.append(f"(air logs failed to start: {e})")
                self.alive = False
                return
            self._proc = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                if self._stop.is_set():
                    break
                s = line.rstrip("\n")
                # drop the CLI's profile preamble — it isn't workload output
                if s.startswith("[INFO] Using Profile") or s.startswith("Using Profile"):
                    continue
                self.lines.append(s)
            rc = proc.wait()
            if self._stop.is_set():
                break
            # completed run: air logs dumps history and exits 0 — keep the buffer, stop.
            # running run whose stream dropped: retry. PENDING with "No logs": retry.
            last = self.lines[-1] if self.lines else ""
            if rc == 0 and self.lines and "No logs" not in last and "not available" not in last.lower():
                self.alive = False
                return
            self.lines.append(f"(log stream paused, retrying in 3s — air logs exit {rc})")
            self._stop.wait(3)
        self.alive = False

    def stop(self) -> None:
        self._stop.set()
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.send_signal(signal.SIGTERM)
            except OSError:
                pass


# ── watch screen ──────────────────────────────────────────────────────────────

_STATE_STYLE = {
    "SUCCESS": "bold green", "DRY-OK": "bold green", "FAILED": "bold red",
    "TIMEDOUT": "bold red", "INTERNAL_ERROR": "bold red", "DRY-FAIL": "bold red",
    "SUBMIT-FAIL": "bold red", "CANCELED": "yellow", "SKIPPED": "yellow",
    "RUNNING": "bold cyan", "PENDING": "cyan", "UNKNOWN": "dim",
}


def _state(st: str) -> Text:
    return Text(st, style=_STATE_STYLE.get(st, "cyan"))


def watch(runs: dict, profile: str | None, console: Console,
          poll_every: float = 8.0) -> dict:
    """Follow every run's logs; ↑↓ picks which pane is in front. Returns final status dict.

    `runs` is {key: {item, run_id}} — the same shape `uat run` already builds. Keys with
    no run_id (submit-fail / dry-run) show in the list but have no tail.
    """
    keys = list(runs)
    if not keys:
        return {}
    idx = 0
    node: dict[str, int] = {k: 0 for k in keys}
    tails: dict[tuple[str, int], _Tail] = {}
    final: dict[str, dict] = {k: {"status": "PENDING"} for k in keys}
    last_poll = 0.0
    log_scroll = 0  # 0 = follow (tail); >0 = lines up from the bottom
    quit = False

    def ensure_tail(key: str) -> _Tail | None:
        rid = runs[key].get("run_id")
        if not rid:
            return None
        n = node[key]
        slot = (key, n)
        if slot not in tails:
            tails[slot] = _Tail(rid, profile, n)
        return tails[slot]

    for k in keys:
        ensure_tail(k)

    def poll_statuses() -> None:
        for k in keys:
            rid = runs[k].get("run_id")
            if not rid:
                final[k] = {"status": "SUBMIT-FAIL", "detail": "no run id"}
                continue
            st = final[k].get("status", "")
            if st in core.TERMINAL:
                continue
            info = core.run_state(rid, profile)
            final[k] = info

    def selected_key() -> str:
        return keys[idx]

    def body(term_h: int) -> Layout:
        key = selected_key()
        it = runs[key]["item"]
        rid = runs[key].get("run_id") or "—"
        n = node[key]
        nodes = int(it.get("nodes") or 1)
        st = final[key].get("status", "PENDING")
        url = final[key].get("dashboard_url") or ""

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=1),
        )
        layout["main"].split_row(
            Layout(name="runs", minimum_size=34, ratio=1),
            Layout(name="logs", ratio=3),
        )

        n_run = sum(1 for v in final.values() if v.get("status") not in core.TERMINAL
                    and v.get("status") != "SUBMIT-FAIL")
        n_ok = sum(1 for v in final.values() if v.get("status") in ("SUCCESS", "DRY-OK"))
        n_bad = sum(1 for v in final.values()
                    if v.get("status") in core.TERMINAL - {"SUCCESS", "SKIPPED"}
                    or v.get("status") == "SUBMIT-FAIL")
        header = Text.assemble(
            ("  UAT watch  ", "bold white on dodger_blue4"),
            (f" {len(keys)} runs", "bold"),
            (f"  {n_run} live", "cyan"),
            (f"  {n_ok} ok", "green"),
            (f"  {n_bad} failed" if n_bad else "", "red"),
            (f"   profile {profile or 'ambient'}", "dim"),
        )
        layout["header"].update(Panel(header, box=rich_box.SIMPLE, padding=0))

        rt = Table(show_header=True, header_style="dim", box=None, pad_edge=False, expand=True)
        rt.add_column("", width=1)
        rt.add_column("run", style="cyan", no_wrap=True)
        rt.add_column("shape", style="magenta", no_wrap=True)
        rt.add_column("state")
        for i, k in enumerate(keys):
            mark = "▶" if i == idx else " "
            style = "bold reverse cyan" if i == idx else ""
            rt.add_row(
                Text(mark, style="bold cyan" if i == idx else "dim"),
                Text(runs[k]["item"]["name"], style=style or "cyan"),
                Text(runs[k]["item"].get("shape", ""), style="magenta"),
                _state(final[k].get("status", "PENDING")),
            )
        layout["runs"].update(Panel(rt, title="runs", title_align="left",
                                    border_style="cyan", padding=(0, 1)))

        tail = ensure_tail(key)
        lines = list(tail.lines) if tail else ["(no run id — nothing to follow)"]
        if not lines:
            lines = ["(waiting for logs — PENDING jobs have none yet)"]
        # follow vs scroll-back
        pane_h = max(6, term_h - 8)
        if log_scroll == 0:
            view = lines[-pane_h:]
            follow_tag = "follow"
        else:
            end = max(pane_h, len(lines) - log_scroll)
            start = max(0, end - pane_h)
            view = lines[start:end]
            follow_tag = f"back {log_scroll}"
        log_text = Text.from_ansi("\n".join(view)) if view else Text("(empty)", style="dim")
        title = (f"{it.get('key') or it['name']}  ·  run {rid}  ·  node {n}/{nodes - 1}  ·  "
                 f"{st}  ·  {follow_tag}")
        subtitle = url if url else "↑↓ run   ←→ node   pgup/pgdn scroll   f follow   q detach (jobs keep running)"
        layout["logs"].update(Panel(
            log_text, title=title, title_align="left", subtitle=subtitle,
            subtitle_align="left", border_style="green" if st == "RUNNING" else "dim",
            padding=(0, 1),
        ))
        layout["footer"].update(Text(
            "  ↑↓ switch run   ←→ node   pgup/pgdn (or b/n) scroll logs   f follow   "
            "q detach — jobs keep running",
            style="dim",
        ))
        return layout

    try:
        with KeyReader() as kbd, Live(body(console.size.height), console=console, screen=True,
                                      refresh_per_second=8) as live:
            poll_statuses()
            last_poll = time.monotonic()
            while not quit:
                now = time.monotonic()
                if now - last_poll >= poll_every:
                    poll_statuses()
                    last_poll = now
                keypress = kbd.poll(0.12)
                if keypress in ("q", "Q", "ctrl-c"):
                    quit = True
                    break
                if keypress == "enter" and all(
                    final[k].get("status") in core.TERMINAL or not runs[k].get("run_id")
                    for k in keys
                ):
                    quit = True
                    break
                if keypress in ("up", "k"):
                    idx = (idx - 1) % len(keys)
                    log_scroll = 0
                elif keypress in ("down", "j"):
                    idx = (idx + 1) % len(keys)
                    log_scroll = 0
                elif keypress in ("left", "h"):
                    k = selected_key()
                    nodes = max(1, int(runs[k]["item"].get("nodes") or 1))
                    node[k] = (node[k] - 1) % nodes
                    ensure_tail(k)
                    log_scroll = 0
                elif keypress in ("right", "l"):
                    k = selected_key()
                    nodes = max(1, int(runs[k]["item"].get("nodes") or 1))
                    node[k] = (node[k] + 1) % nodes
                    ensure_tail(k)
                    log_scroll = 0
                elif keypress in ("f", "F"):
                    log_scroll = 0
                elif keypress in ("pageup", "b"):
                    log_scroll += 20
                elif keypress in ("pagedown", "n"):
                    log_scroll = max(0, log_scroll - 20)
                live.update(body(console.size.height))
    except KeyboardInterrupt:
        pass
    finally:
        for t in tails.values():
            t.stop()
    # one last status pull so the receipt is current
    poll_statuses()
    return final
