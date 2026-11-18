"""Build src/air_gpu_usage.lvdash.json from utils/billing/air_gpu_usage.sql.

    python3 bundles/air-usage-dashboard/build_dashboard.py           # rewrite the dashboard JSON
    python3 bundles/air-usage-dashboard/build_dashboard.py --check   # exit 1 if it's out of date

The SQL file is the single source of the dataset (it also runs on its own in the SQL editor); this script wraps it
with the dashboard's parameters, pages and widgets. Edit the SQL or the layout below, rebuild, and commit both.
The offline test fails if the committed JSON doesn't match a fresh build. Stdlib only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SQL = HERE.parents[1] / "utils" / "billing" / "air_gpu_usage.sql"
OUT = HERE / "src" / "air_gpu_usage.lvdash.json"
DS = "gpu_usage"

# keyword, display name, data type, default. A DATE default like now-90d/d is relative to the day the dashboard runs.
PARAMS = [
    ("since", "Since", "DATE", "now-90d/d"),
    ("reserved_accelerators", "Reserved accelerators", "STRING", "GPU_8xH100"),
    ("hold_basis", "Job hold basis", "STRING", "execution"),
    ("nb_idle_timeout_min", "Notebook idle timeout (min)", "INTEGER", "60"),
    ("nb_gpus", "GPUs per notebook session", "INTEGER", "8"),
    ("spec_lookback_days", "Job spec lookback (days)", "INTEGER", "180"),
]
# Parameters shown as filters on the dashboard; the rest keep their defaults (edit them in the dataset).
PARAM_WIDGETS = [("since", "filter-date-picker"), ("reserved_accelerators", "filter-text-entry")]
# Dataset columns offered as multi-select filters on every page.
FIELD_FILTERS = [("capacity", "Capacity"), ("accelerator", "Accelerator"), ("shape", "Shape"),
                 ("workload", "Workload"), ("policy_id", "Usage policy"), ("workspace_id", "Workspace")]

GPU_HOURS = ("sum(gpu_hours)", "SUM(`gpu_hours`)")


def field(name: str, expression: str | None = None) -> dict:
    return {"name": name, "expression": expression or f"`{name}`"}


def query(fields: list[dict], filters: list[str] | None = None, name: str = "main_query") -> dict:
    q: dict = {"datasetName": DS, "fields": fields, "disaggregated": False}
    if filters:
        q["filters"] = [{"expression": f} for f in filters]
    return {"name": name, "query": q}


def frame(title: str, description: str | None = None) -> dict:
    f: dict = {"showTitle": True, "title": title}
    if description:
        f.update(showDescription=True, description=description)
    return f


def counter(name: str, title: str, filters: list[str] | None = None, description: str | None = None) -> dict:
    return {
        "name": name,
        "queries": [query([field(*GPU_HOURS)], filters)],
        "spec": {"version": 2, "widgetType": "counter",
                 "encodings": {"value": {"fieldName": GPU_HOURS[0], "displayName": "GPU-hours"}},
                 "frame": frame(title, description)},
    }


def bar_over_time(name: str, title: str, color: str, color_label: str, description: str | None = None) -> dict:
    return {
        "name": name,
        "queries": [query([field("day"), field(color), field(*GPU_HOURS)])],
        "spec": {"version": 3, "widgetType": "bar",
                 "encodings": {
                     "x": {"fieldName": "day", "displayName": "Day", "scale": {"type": "temporal"}},
                     "y": {"fieldName": GPU_HOURS[0], "displayName": "GPU-hours", "scale": {"type": "quantitative"}},
                     "color": {"fieldName": color, "displayName": color_label, "scale": {"type": "categorical"}}},
                 "frame": frame(title, description)},
    }


def bar_by(name: str, title: str, x: str, x_label: str, color: str, color_label: str,
           description: str | None = None) -> dict:
    return {
        "name": name,
        "queries": [query([field(x), field(color), field(*GPU_HOURS)])],
        "spec": {"version": 3, "widgetType": "bar",
                 "encodings": {
                     "x": {"fieldName": x, "displayName": x_label,
                           "scale": {"type": "categorical", "sort": {"by": "y-reversed"}}},
                     "y": {"fieldName": GPU_HOURS[0], "displayName": "GPU-hours", "scale": {"type": "quantitative"}},
                     "color": {"fieldName": color, "displayName": color_label, "scale": {"type": "categorical"}}},
                 "frame": frame(title, description)},
    }


def pivot(name: str, title: str, rows: list[tuple[str, str]], columns: list[tuple[str, str]],
          description: str | None = None) -> dict:
    return {
        "name": name,
        "queries": [query([field(f) for f, _ in rows + columns] + [field(*GPU_HOURS)])],
        "spec": {"version": 3, "widgetType": "pivot",
                 "encodings": {
                     "rows": [{"fieldName": f, "displayName": d} for f, d in rows],
                     "columns": [{"fieldName": f, "displayName": d} for f, d in columns],
                     "cell": {"type": "multi-cell",
                              "fields": [{"fieldName": GPU_HOURS[0], "cellType": "text", "displayName": "GPU-hours"}]}},
                 "frame": frame(title, description)},
    }


def text(name: str, markdown: str) -> dict:
    return {"name": name, "multilineTextboxSpec": {"lines": [line + "\n" for line in markdown.strip().splitlines()]}}


def filter_widgets(page: str) -> list[tuple[dict, int, int]]:
    """(widget, width, height) for the parameter and field filters at the top of a page."""
    out = []
    for keyword, kind in PARAM_WIDGETS:
        title = next(d for k, d, _, _ in PARAMS if k == keyword)
        qname = f"{page}_param_{keyword}"
        out.append(({"name": f"{page}_param_{keyword}",
                     "queries": [{"name": qname, "query": {"datasetName": DS, "disaggregated": False,
                                                           "parameters": [{"name": keyword, "keyword": keyword}]}}],
                     "spec": {"version": 2, "widgetType": kind,
                              "encodings": {"fields": [{"parameterName": keyword, "queryName": qname}]},
                              "frame": frame(title)}}, 1, 1))
    for col, title in FIELD_FILTERS:
        qname = f"{page}_filter_{col}"
        out.append(({"name": f"{page}_filter_{col}",
                     "queries": [query([field(col), field(f"{col}_associativity",
                                                          "COUNT_IF(`associative_filter_predicate_group`)")],
                                       name=qname)],
                     "spec": {"version": 2, "widgetType": "filter-multi-select",
                              "encodings": {"fields": [{"fieldName": col, "displayName": title, "queryName": qname}]},
                              "frame": frame(title)}}, 1, 1))
    return out


def layout(page: str, rows: list[list[tuple[dict, int, int]]]) -> list[dict]:
    """Place rows of (widget, width, height) on the 6-column grid, top to bottom."""
    items, y = [], 0
    for row in [filter_widgets(page)] + rows:
        x, h = 0, 0
        for widget, w, wh in row:
            if x + w > 6:
                x, y = 0, y + h
                h = 0
            items.append({"widget": widget, "position": {"x": x, "y": y, "width": w, "height": wh}})
            x, h = x + w, max(h, wh)
        y += h
    return items


INTRO = """
## AIR GPU-hours by capacity, accelerator, shape and usage policy
GPU-hours are **GPU-hold time x GPUs** from the jobs and notebook audit events and the job task run timeline, not
billed DBUs: reserved billing rows are per node and name no job, notebook or policy. A workload counts as
**on-demand** if it has on-demand billing rows (or isn't on a reserved accelerator), **reserved** by elimination,
and **pending billing** if it ended less than 24 hours ago. Notebooks' accelerator isn't logged: reserved notebook
holds are counted on the first reserved accelerator, on-demand ones as `unknown (notebook)`, both with the
`nb_gpus` parameter's GPU count. The **Attribution quality** page shows how each number was found.
Source: `utils/billing/air_gpu_usage.sql`; method: `utils/billing/reserved_gpu_attribution.sql`.
"""

QUALITY = """
## How each GPU-hour was attributed
- **Policy source**: `billing` = the on-demand billing rows (what was billed); `jobs audit` = the latest
  policy-setting jobs event before the run; `notebook billing (other compute)` = the notebook's stored policy, seen
  on its other billed usage; `none found` = neither (look up the job's effective policy with
  `databricks jobs get <job_id>`).
- **Capacity basis**: reserved is always by elimination: no on-demand billing row matched the workload.
- **GPUs basis**: `job spec` = accelerator_count from the task; `assumed: one node` = no count found;
  notebooks always use the `nb_gpus` parameter.
"""


def build() -> dict:
    sql = SQL.read_text().rstrip().rstrip(";")
    dataset = {
        "name": DS,
        "displayName": "AIR GPU-hours per day",
        "queryLines": [line + "\n" for line in sql.splitlines()],
        "parameters": [{"displayName": display, "keyword": keyword, "dataType": dtype,
                        "defaultSelection": {"values": {"dataType": dtype, "values": [{"value": default}]}}}
                       for keyword, display, dtype, default in PARAMS],
    }
    usage = layout("usage", [
        [(text("usage_intro", INTRO), 6, 3)],
        [(counter("usage_total", "Total GPU-hours"), 2, 2),
         (counter("usage_reserved", "Reserved", ["`capacity` = 'reserved'"]), 2, 2),
         (counter("usage_on_demand", "On-demand", ["`capacity` = 'on-demand'"]), 2, 2)],
        [(bar_over_time("usage_by_capacity", "GPU-hours per day: reserved vs on-demand", "capacity", "Capacity"), 6, 6)],
        [(bar_over_time("usage_by_accelerator", "GPU-hours per day by accelerator", "accelerator", "Accelerator"), 3, 6),
         (bar_over_time("usage_by_shape", "GPU-hours per day by workload shape", "shape", "Shape",
                        "Nodes x GPUs per node, from the job's accelerator_count."), 3, 6)],
        [(bar_over_time("usage_by_workload", "GPU-hours per day: jobs vs notebooks", "workload", "Workload"), 3, 6),
         (bar_over_time("usage_by_policy_time", "GPU-hours per day by usage policy", "policy_id", "Usage policy"), 3, 6)],
        [(bar_by("usage_by_policy", "GPU-hours by usage policy", "policy_id", "Usage policy",
                 "capacity", "Capacity"), 6, 7)],
        [(pivot("usage_policy_pivot", "GPU-hours by usage policy, capacity, accelerator and shape",
                [("policy_id", "Usage policy"), ("accelerator", "Accelerator"), ("shape", "Shape"),
                 ("workload", "Workload")],
                [("capacity", "Capacity")]), 6, 10)],
    ])
    quality = layout("quality", [
        [(text("quality_intro", QUALITY), 6, 3)],
        [(bar_by("quality_policy_source", "GPU-hours by policy source", "policy_source", "Policy source",
                 "capacity", "Capacity"), 3, 6),
         (bar_by("quality_capacity_basis", "GPU-hours by capacity basis", "capacity_basis", "Capacity basis",
                 "workload", "Workload"), 3, 6)],
        [(bar_by("quality_gpus_basis", "GPU-hours by GPU-count basis", "gpus_basis", "GPUs basis",
                 "capacity", "Capacity"), 3, 6),
         (bar_over_time("quality_policy_source_time", "GPU-hours per day by policy source", "policy_source",
                        "Policy source"), 3, 6)],
        [(pivot("quality_pivot", "GPU-hours by workload, capacity basis, workspace and policy source",
                [("workload", "Workload"), ("capacity_basis", "Capacity basis"), ("workspace_id", "Workspace")],
                [("policy_source", "Policy source")]), 6, 7)],
    ])
    return {"datasets": [dataset],
            "pages": [{"name": "usage", "displayName": "Usage", "layout": usage},
                      {"name": "quality", "displayName": "Attribution quality", "layout": quality}]}


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


def main(argv: list[str]) -> int:
    new = render()
    if "--check" in argv:
        if not OUT.exists() or OUT.read_text() != new:
            print(f"{OUT.relative_to(HERE.parents[1])} is out of date: run {Path(__file__).name}", file=sys.stderr)
            return 1
        return 0
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(new)
    print(f"wrote {OUT.relative_to(HERE.parents[1])}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
