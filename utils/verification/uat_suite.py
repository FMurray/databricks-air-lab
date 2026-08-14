"""Declarative multi-node UAT suite — launch recipes the `uat` CLI submits.

Owns ONE thing: *what to launch* and *which hardware it can run on*. The `uat` CLI
crosses items × hardware into a selectable matrix; `uat_core` turns a selected cell
into `air run --override compute.accelerator_type=… compute.num_accelerators=…`.

What/why/verdict/evidence lives once in `results/registry.py`. Items cross-link by
`registry_id`. Distributed multi-node is **air CLI only** (docs/RUNNING-UAT.md).

Hardware column (multinode acceptance shape only):
  GPU_8xH100  8 GPU/node. Multi-node = N × 8 accelerators.
  (A10 / 1×H100 are not multinode-UAT shapes — submit those via air run YAML overrides
  if you need a cheap dry-run outside this CLI.)

Tiers group *what's under test*, not node count. Every ≥2-node run already uses the
RDMA fabric (inter-node = EFA; intra-node = NVLink). An item's `nodes` is topology;
the SKU is the matrix column. `--hw` / the picker choose the column; `--tier` / `--only`
choose the rows.

The allreduce *is* the multinode probe (`allreduce_probe.py`).
"""
from __future__ import annotations

# ── Hardware columns (AIR accelerator_type) ───────────────────────────────────
# `id` is the exact `compute.accelerator_type` token. `gpus` is GPUs per node, so
# num_accelerators = nodes * gpus. `spendy` gates --confirm-spend (any H100).

HARDWARE = [
    {"id": "GPU_8xH100", "gpus": 8, "spendy": True, "short": "8×H100", "cost": "H100"},
]
HW = {h["id"]: h for h in HARDWARE}


def hw_supports_nodes(hw_id: str, nodes: int) -> bool:
    """False for SKUs AIR won't run at this node count.

    Kept for forward-compat if a `single_node_only` SKU is re-added; today every
    column supports any topology in the suite."""
    return not (HW[hw_id].get("single_node_only") and nodes > 1)

# Friendly --hw tokens → SKU id (case-insensitive, strip ×/x/-/_).
_HW_ALIASES = {
    "h100": "GPU_8xH100", "8xh100": "GPU_8xH100", "gpu8xh100": "GPU_8xH100",
}

# --only names that used to be separate items (same probe, different baked shape).
ALIASES = {
    "allreduce-fabric": "allreduce",
    "allreduce-dry": "allreduce",
    "multinode-probe": "allreduce",
}


def parse_hw(spec: str | None) -> tuple[list[str] | None, str | None]:
    """Parse `--hw a10,8xh100` into SKU ids. None spec → (None, None) meaning 'use defaults'."""
    if not spec or not spec.strip():
        return None, None
    out, unknown = [], []
    for raw in spec.split(","):
        token = raw.strip()
        if not token:
            continue
        key = token.lower().replace("×", "x").replace("-", "").replace("_", "")
        hid = _HW_ALIASES.get(key) or (token if token in HW else None)
        if hid is None:
            unknown.append(raw.strip())
        elif hid not in out:
            out.append(hid)
    if unknown:
        known = ", ".join(h["id"] + " (" + h["short"] + ")" for h in HARDWARE)
        return None, f"unknown --hw: {', '.join(unknown)}  (choose: {known}, or h100 / 8xh100)"
    return out, None


# ── Items ─────────────────────────────────────────────────────────────────────
# Fields:
#   name        CLI selector (`--only`) and matrix row label
#   file        repo-relative workload YAML
#   nodes       topology (AIR: num_accelerators = nodes * gpus_per_sku)
#   hardware    SKU ids this row can run on (matrix cells that aren't `·`)
#   default_hw  the acceptance/intended SKU
#   tiers       which --tier filters include this row
#   registry_id results/registry.py id (None = no results-matrix row)
#   dry         True = never an acceptance receipt (even on default_hw)
#   note        one-line "what to watch"
#   aliases     extra --only names


def _item(name, file, nodes, *, hardware=None, default_hw=None, tiers=(),
          registry_id=None, dry=False, note="", aliases=()):
    nodes = int(nodes)
    hardware = list(hardware or [h["id"] for h in HARDWARE])
    hardware = [h for h in hardware if hw_supports_nodes(h, nodes)]
    if not hardware:
        raise ValueError(f"{name}: no SKU supports {nodes} node(s)")
    if default_hw is None:
        default_hw = hardware[-1]
    if default_hw not in hardware:
        raise ValueError(f"{name}: default_hw {default_hw} unavailable at {nodes} node(s)")
    return {
        "name": name,
        "file": file,
        "nodes": nodes,
        "hardware": hardware,
        "default_hw": default_hw,
        "tiers": tuple(tiers),
        "registry_id": registry_id,
        "dry": dry,
        "note": note,
        "aliases": tuple(aliases),
    }


ITEMS = [
    _item(
        "allreduce",
        "workloads/multinode-probe.example.yaml",
        nodes=2,
        default_hw="GPU_8xH100",
        tiers=("headline",),
        registry_id="allreduce-multi",
        aliases=("multinode-probe", "allreduce-fabric", "allreduce-dry"),
        note="allreduce_probe.py via torchrun — MULTINODE_PROBE_OK + busbw (smoke-grade)",
    ),
    _item(
        "multinode-correctness",
        "workloads/multinode-correctness.example.yaml",
        nodes=2,
        default_hw="GPU_8xH100",
        tiers=("headline",),
        registry_id="dist-correctness",
        note="watch for rank-0 sentinel DISTRIBUTED_CORRECTNESS_OK",
    ),
    _item(
        "fsdp-multinode",
        "workloads/fsdp-multinode.example.yaml",
        nodes=2,
        default_hw="GPU_8xH100",
        tiers=("headline",),
        registry_id="fsdp",
        note="FSDP_BR4_COMPLETE (Proofs 1+2+3). Proof 4 needs a UC-volume ckpt dir.",
    ),
    _item(
        "rdma-m1-soak",
        "workloads/rdma-m1-soak.example.yaml",
        nodes=2,
        default_hw="GPU_8xH100",
        tiers=("fabric",),
        registry_id="allreduce-multi",
        note="10-min 1GB all-reduce soak; drift + RDMA hw-counter deltas per node",
    ),
    _item(
        "rdma-m2a-fabric-only",
        "workloads/rdma-m2a-fabric-only.example.yaml",
        nodes=4,
        default_hw="GPU_8xH100",
        tiers=("fabric",),
        registry_id="allreduce-multi",
        note="1 GPU/node communicator — every byte crosses EFA (no NVLink dilution)",
    ),
    _item(
        "rdma-m2b-p2p-ring",
        "workloads/rdma-m2b-p2p-ring.example.yaml",
        nodes=4,
        default_hw="GPU_8xH100",
        tiers=("fabric",),
        registry_id="allreduce-multi",
        note="directed send/recv ring — link-level stress",
    ),
    _item(
        "rdma-m4-nccl-tests",
        "workloads/rdma-m4-nccl-tests.example.yaml",
        nodes=1,
        default_hw="GPU_8xH100",
        tiers=("fabric",),
        registry_id="allreduce-multi",
        note="nccl-tests built in-run (CUDA 12.9); single-node NVLink "
             "(multi-node needs mpirun — expected MISSING)",
    ),
    _item(
        "rdma-m5-parambench",
        "workloads/rdma-m5-parambench.example.yaml",
        nodes=1,
        default_hw="GPU_8xH100",
        tiers=("fabric",),
        registry_id="allreduce-multi",
        dry=True,
        note="parambench-train-comms probe — proves the runner before multi-node wiring",
    ),
]

TIERS = {
    "headline": {
        "gated": True,
        "default_hw": "GPU_8xH100",
        "desc": "Acceptance — allreduce + distributed-correctness + FSDP BR-4 on 8×H100. "
                "Real money; announce in the team channel.",
    },
    "fabric": {
        "gated": True,
        "default_hw": "GPU_8xH100",
        "desc": "The interconnect is the object under test: soak, fabric-only + P2P-ring, "
                "nccl-tests, parambench. Heaviest spend; coordinate first.",
    },
}

# `uat run notebook` submits THIS notebook (the single-node check DRIVER) as a one-time job.
NOTEBOOK_SUITE = {
    "driver_notebook": "DRIVER",
    "default_mirror": "/Workspace/Shared/databricks-air-lab/uat",
    "environment_version": "5",
    "widgets": {
        "shapes": "GPU_1xA10",
        "pool": "off",
    },
}


def all_tiers() -> list[str]:
    return list(TIERS)


def _canonical(name: str) -> str:
    return ALIASES.get(name, name)


def items_for(tier: str) -> list[dict]:
    """Rows in a tier, or the whole suite for 'all' (headline→fabric order)."""
    if tier == "all":
        return list(ITEMS)
    if tier not in TIERS:
        raise KeyError(tier)
    return [it for it in ITEMS if tier in it["tiers"]]


def pin(item: dict, hw_id: str, nodes: int | None = None) -> dict:
    """One matrix cell → a concrete launch (overrides + shape + spend flag).

    `nodes` overrides the item's default topology (TUI editor / future --nodes)."""
    if hw_id not in HW:
        raise KeyError(hw_id)
    if hw_id not in item["hardware"]:
        raise ValueError(f"{item['name']} does not run on {hw_id}")
    n = int(item["nodes"] if nodes is None else nodes)
    if n < 1:
        raise ValueError(f"{item['name']}: nodes must be >= 1 (got {n})")
    spec = HW[hw_id]
    if not hw_supports_nodes(hw_id, n):
        raise ValueError(
            f"{hw_id} only supports single-node workloads, but {item['name']} is "
            f"{n} nodes (accelerator_count would resolve to {n * spec['gpus']})"
        )
    n_acc = n * spec["gpus"]
    launch = dict(item)
    launch["nodes"] = n
    launch["hw"] = hw_id
    launch["spendy"] = spec["spendy"]
    launch["dry"] = item["dry"] or hw_id != item["default_hw"]
    launch["shape"] = f"{n}×{spec['short']}"
    launch["overrides"] = [
        f"compute.accelerator_type={hw_id}",
        f"compute.num_accelerators={n_acc}",
    ]
    launch["key"] = f"{item['name']}@{launch['shape']}"
    return launch


def default_hw_for(item: dict, tier: str) -> str | None:
    """SKU to use when the caller didn't pass --hw — tier default, else item default."""
    thw = (TIERS.get(tier) or {}).get("default_hw")
    if thw and thw in item["hardware"]:
        return thw
    return item["default_hw"] if item["default_hw"] in item["hardware"] else None


def is_spendy(item: dict) -> bool:
    """True if this launch (pinned) or item-at-default burns H100 money."""
    if "spendy" in item and "hw" in item:
        return bool(item["spendy"])
    return HW[item["default_hw"]]["spendy"]


def supports(item: dict, hw_id: str) -> bool:
    return hw_id in item["hardware"]
