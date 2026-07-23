"""Training Hub — fleet visibility + template-based submission for a shared AIR pool."""

import tempfile

import streamlit as st

from hub import config, templates

st.set_page_config(page_title="Training Hub", page_icon="🛠", layout="wide")
st.title("Training Hub")

cfg = config.load()
fleet_tab, submit_tab = st.tabs(["Fleet", "Submit a workload"])


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
