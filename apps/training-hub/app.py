"""Training Hub — run GPU workloads (practitioner) and see the fleet (management)."""

import streamlit as st

from hub import config, identity, queue as hubq

st.set_page_config(page_title="Training Hub", page_icon="🛠", layout="wide")

cfg = config.load()
user, user_source = identity.current_user()
my_teams = cfg.teams_of(user)
broker = hubq.Broker(cfg=cfg, ws=None)  # ws=None: dry-run; pass WorkspaceClient() for live

st.sidebar.write(f"**{user}**")
st.sidebar.caption(f"identity source: {user_source}")
st.sidebar.write("Teams: " + (", ".join(t.name for t in my_teams) or "none — read-only"))
page = st.sidebar.radio("View", ["Run", "Overview"], index=0 if my_teams else 1)

# the queue advances on every page load — no button to press
dispatch_events = broker.tick()


def _shapes():
    return sorted({cfg.reservation.accelerator_type, "GPU_1xA10", "GPU_1xH100"})


if page == "Run":
    st.title("Run")
    if not my_teams:
        st.info("You are not in any team, so you can look but not run. "
                "Ask your platform admin to add you to a team.")
    else:
        mine = broker.workloads([t.name for t in my_teams])
        options = {f"{w['name']}  ({w['use_case']}, {w['shape']}×{w['nodes']})": w["id"]
                   for w in reversed(mine)}
        NEW = "something new…"
        pick = st.selectbox("What do you want to run?", [*options, NEW] if options else [NEW])

        if pick == NEW:
            if len(my_teams) == 1:
                team = my_teams[0]
            else:
                team_name = st.selectbox("Team", [t.name for t in my_teams])
                team = next(t for t in my_teams if t.name == team_name)
            ucs = [u.name for u in team.use_cases]
            c1, c2 = st.columns(2)
            use_case = c1.selectbox("Use case", ucs) if ucs else ""
            name = c2.text_input("Name it (so you can re-run it later)")
            c3, c4, c5 = st.columns(3)
            kind = c3.selectbox("Kind", ["notebook", "air_yaml"])
            shape = c4.selectbox("GPU", _shapes())
            nodes = c5.number_input("Nodes", 1, 20, 1)
            ref = st.text_input("Notebook workspace path" if kind == "notebook"
                                else "Workload YAML repo path")
            needs_torch = st.checkbox("Needs torch")
            if st.button("Run it", type="primary", disabled=not (name and ref)):
                try:
                    wid = broker.register_workload(
                        user=user, team=team.name, use_case=use_case, name=name.strip(),
                        kind=kind, ref=ref.strip(), shape=shape, nodes=int(nodes),
                        needs_torch=needs_torch)
                    rid = broker.request_run(user=user, workload_id=wid)
                    st.success(f"run {rid} created")
                    st.rerun()
                except hubq.GateError as e:
                    st.error(str(e))
        else:
            if st.button("Run it", type="primary"):
                try:
                    rid = broker.request_run(user=user, workload_id=options[pick])
                    st.success(f"run {rid} created")
                    st.rerun()
                except hubq.GateError as e:
                    st.error(str(e))

        st.divider()
        team_names = {t.name for t in my_teams}
        my_runs = [r for r in reversed(broker.runs())
                   if r["requested_by"] == user or r["team"] in team_names]
        active = [r for r in my_runs if r["state"] in ("QUEUED", "SUBMITTED", "RUNNING")]
        done = [r for r in my_runs if r["state"] not in ("QUEUED", "SUBMITTED", "RUNNING")]
        st.subheader("Active")
        if active:
            st.dataframe([{k: r[k] for k in ("id", "name", "use_case", "shape", "nodes",
                                             "state", "requested_by", "run_id")}
                          for r in active], use_container_width=True, hide_index=True)
        else:
            st.caption("Nothing queued or running.")
        st.subheader("Past runs")
        if done:
            st.dataframe([{k: r[k] for k in ("id", "name", "use_case", "shape", "state",
                                             "requested_by", "detail")}
                          for r in done], use_container_width=True, hide_index=True)
        else:
            st.caption("No finished runs yet.")

else:  # Overview (management)
    st.title("Overview")
    res = cfg.reservation
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reserved nodes", res.total_nodes)
    c2.metric("GPUs", res.total_nodes * res.gpus_per_node, res.accelerator_type)
    c3.metric("Quota allocated", f"{cfg.allocated_nodes} nodes")
    c4.metric("Unallocated", f"{res.total_nodes - cfg.allocated_nodes} nodes")

    st.subheader("Capacity now")
    cols = st.columns(len(_shapes()))
    for col, shape in zip(cols, _shapes()):
        c = broker.capacity(shape)
        cap_total = c.platform_quota_nodes or c.reserved_nodes or 0
        col.metric(shape, f"{c.admittable} free",
                   f"{c.in_flight} in flight of {cap_total or 'unknown'}")

    st.subheader("Teams")
    rows = []
    for t in cfg.teams:
        used = broker.in_flight(cfg.reservation.accelerator_type, t.name)
        rows.append({"team": t.name, "quota_nodes": t.quota_nodes, "in_flight": used,
                     "members": len(t.members),
                     "use_cases": ", ".join(u.name for u in t.use_cases)})
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("All runs")
    all_runs = broker.runs()
    if all_runs:
        st.dataframe([{k: r[k] for k in ("id", "team", "use_case", "name", "shape",
                                         "nodes", "requested_by", "state", "run_id",
                                         "detail")} for r in reversed(all_runs)],
                     use_container_width=True, hide_index=True)
    else:
        st.caption("No runs yet.")

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
                st.warning(f"{unmapped:,.0f} DBUs unattributed — principals missing from "
                           "teams.yaml, or aggregate reserved-pool billing records "
                           "(known product gap: per-workload tagging in pools).")
    except Exception as e:
        st.error(f"Usage query unavailable: {e}")

    if dispatch_events:
        st.caption("Dispatcher: " + "; ".join(dispatch_events))
