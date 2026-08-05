"""Training Hub — fleet visibility + template-based submission for a shared AIR pool."""

import tempfile

import streamlit as st

from hub import config, identity, queue as hubq, templates

st.set_page_config(page_title="Training Hub", page_icon="🛠", layout="wide")
st.title("Training Hub")

cfg = config.load()
user, user_source = identity.current_user()
my_teams = cfg.teams_of(user)
st.sidebar.write(f"**{user}**")
st.sidebar.caption(f"identity source: {user_source}")
st.sidebar.write("Teams: " + (", ".join(t.name for t in my_teams) or "none — read-only"))
fleet_tab, runs_tab, workloads_tab, submit_tab = st.tabs(["Fleet", "Runs", "Workloads", "Workload YAML builder"])


with fleet_tab:
    res = cfg.reservation
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reserved nodes", res.total_nodes)
    c2.metric("GPUs", res.total_nodes * res.gpus_per_node, res.accelerator_type)
    c3.metric("Quota allocated", f"{cfg.allocated_nodes} nodes")
    c4.metric("Unallocated", f"{res.total_nodes - cfg.allocated_nodes} nodes")

    st.subheader("Declared team quotas")
    st.dataframe(
        [
            {"team": t.name, "quota_nodes": t.quota_nodes, "members": len(t.members)}
            for t in cfg.teams
        ],
        width="stretch",
        hide_index=True,
    )

    st.subheader("GPU spend by team")
    days = st.slider("Lookback (days)", 7, 90, 30)
    try:
        from hub import usage

        st.caption(f"Billing source: {usage.workspace_host()}")
        by_team = usage.usage_by_team(cfg, days)
        if by_team.empty:
            st.info("No GPU-SKU usage found in the lookback window.")
        else:
            st.dataframe(by_team, width="stretch", hide_index=True)
            unmapped = by_team.loc[by_team["team"] == "unmapped", "dbus"].sum()
            if unmapped:
                st.warning(
                    f"{unmapped:,.0f} DBUs unattributed — principals missing from "
                    "teams.yaml, or aggregate reserved-pool billing records "
                    "(known product gap: per-workload tagging in pools)."
                )
    except Exception as e:  # surface setup problems in-app instead of a blank panel
        st.error(f"Usage query unavailable: {e}")

    st.subheader("Active runs")
    st.caption(
        "All active job runs for now — reliable AIR-run filtering is an open question."
    )
    try:
        from hub import jobs

        runs = jobs.active_runs()
        if runs.empty:
            st.info("No active runs.")
        else:
            st.dataframe(
                runs,
                width="stretch",
                hide_index=True,
                column_config={"run_page": st.column_config.LinkColumn("run_page")},
            )
    except Exception as e:
        st.error(f"Jobs API unavailable: {e}")


with submit_tab:
    left, right = st.columns(2)
    with left:
        team = st.selectbox("Team", [t.name for t in cfg.teams])
        template_name = st.selectbox(
            "Template",
            list(templates.TEMPLATES),
            format_func=lambda k: f"{k} — {templates.TEMPLATES[k]['description']}",
        )
        tmpl = templates.TEMPLATES[template_name]
        experiment_name = st.text_input("Experiment name", f"{team}-training")
        command = st.text_area("Command", tmpl["command"])
        dependencies = st.text_input("Dependencies (comma-separated)", "")
        accelerator_type = st.selectbox(
            "Accelerator",
            templates.ACCELERATOR_TYPES,
            index=templates.ACCELERATOR_TYPES.index(tmpl["accelerator_type"]),
        )
        num_accelerators = st.number_input(
            "Num accelerators", 1, 64, tmpl["num_accelerators"]
        )
        timeout_minutes = st.number_input("Timeout (minutes)", 10, 10080, 120)

    workload = templates.build_workload(
        experiment_name=experiment_name,
        template=template_name,
        command=command,
        dependencies=[d.strip() for d in dependencies.split(",") if d.strip()],
        num_accelerators=int(num_accelerators),
        accelerator_type=accelerator_type,
        timeout_minutes=int(timeout_minutes),
    )
    workload_yaml = templates.to_yaml(workload)

    with right:
        st.code(workload_yaml, language="yaml")
        st.download_button(
            "Download YAML", workload_yaml, file_name=f"{experiment_name}.yaml"
        )
        if st.button("Submit via air CLI"):
            with tempfile.NamedTemporaryFile(
                "w", suffix=".yaml", delete=False
            ) as f:
                f.write(workload_yaml)
            ok, output = templates.submit(f.name)
            (st.success if ok else st.error)(output or "submitted")


broker = hubq.Broker(cfg=cfg, ws=None)  # ws=None: dry-run; pass WorkspaceClient() for live

with workloads_tab:
    st.caption("A workload is a registered definition owned by a use case. Runs are created from workloads.")
    if my_teams:
        with st.form("register_workload"):
            c1, c2 = st.columns(2)
            team = c1.selectbox("Team", [t.name for t in my_teams])
            team_obj = next(t for t in my_teams if t.name == team)
            use_case = c2.selectbox("Use case", [u.name for u in team_obj.use_cases] or ["(none configured)"])
            name = st.text_input("Workload name")
            c3, c4, c5 = st.columns(3)
            kind = c3.selectbox("Kind", ["notebook", "air_yaml"])
            shape = c4.selectbox("Shape", [cfg.reservation.accelerator_type, "GPU_1xA10", "GPU_1xH100"])
            nodes = c5.number_input("Nodes", 1, 20, 1)
            ref = st.text_input("Notebook workspace path / workload YAML repo path")
            needs_torch = st.checkbox("Needs torch (adds the AI-environment interpreter setup)")
            if st.form_submit_button("Register"):
                try:
                    wid = broker.register_workload(user=user, team=team, use_case=use_case,
                                                   name=name.strip(), kind=kind, ref=ref.strip(),
                                                   shape=shape, nodes=int(nodes),
                                                   needs_torch=needs_torch)
                    st.success(f"registered workload {wid}: {name}")
                except hubq.GateError as e:
                    st.error(str(e))
    else:
        st.info("You are not in any team — workload registration is disabled. Browsing is open to everyone.")
    rows = broker.workloads()
    if rows:
        st.dataframe([dict(r) for r in rows], use_container_width=True)

with runs_tab:
    shapes = sorted({cfg.reservation.accelerator_type, "GPU_1xA10", "GPU_1xH100"})
    cols = st.columns(len(shapes))
    for col, shape in zip(cols, shapes):
        c = broker.capacity(shape)
        cap_str = c.platform_quota_nodes or c.reserved_nodes or 0
        col.metric(shape, f"{c.admittable} free", f"{c.in_flight} in flight of {cap_str or 'unknown'}")

    mine = broker.workloads([t.name for t in my_teams]) if my_teams else []
    if mine:
        with st.form("request_run"):
            labels = {f"[{w['team']}/{w['use_case']}] {w['name']} — {w['shape']}×{w['nodes']}": w["id"]
                      for w in mine}
            pick = st.selectbox("Workload", list(labels))
            if st.form_submit_button("Run"):
                try:
                    rid = broker.request_run(user=user, workload_id=labels[pick])
                    st.success(f"run {rid} queued")
                except hubq.GateError as e:
                    st.error(str(e))
    else:
        st.info("No workloads available to you. Register one under Workloads.")

    if st.button("Dispatch now"):
        for ev in broker.tick():
            st.write(ev)
    st.caption("Order: teams furthest under their quota share first, then arrival order.")

    all_runs = broker.runs()
    if all_runs:
        show = [{k: r[k] for k in ("id", "team", "use_case", "name", "shape", "nodes",
                                    "requested_by", "state", "run_id", "detail")} for r in all_runs]
        st.dataframe(show, use_container_width=True)
    else:
        st.caption("No runs yet.")
