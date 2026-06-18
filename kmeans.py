def run_kmeans():
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


FIXED_POINTS = np.array([
    [1.4, 4.6], [2.6, 5.4], [1.7, 5.6], [2.8, 4.5], [1.5, 5.2],
    [2.3, 4.3], [1.9, 5.8], [2.5, 5.1], [1.6, 4.8], [2.2, 5.7],
    [2.0, 4.4], [2.7, 5.3], [1.3, 5.0], [2.4, 4.7], [2.1, 5.5],
    [4.4, 8.6], [5.6, 9.4], [4.7, 9.6], [5.7, 8.7], [4.3, 9.1],
    [5.3, 8.4], [4.8, 9.7], [5.8, 9.2], [4.6, 8.8], [5.2, 9.5],
    [4.9, 8.5], [5.9, 9.3], [4.2, 9.0], [5.5, 8.6], [5.1, 9.6],
    [8.4, 6.6], [9.6, 7.4], [8.7, 7.6], [9.7, 6.7], [8.3, 7.1],
    [9.3, 6.4], [8.8, 7.7], [9.8, 7.2], [8.5, 6.8], [9.2, 7.5],
    [8.9, 6.5], [9.6, 7.3], [8.2, 7.0], [9.5, 6.6], [9.1, 7.6],
    [8.4, 2.6], [9.6, 3.4], [8.7, 3.6], [9.7, 2.7], [8.3, 3.1],
    [9.3, 2.4], [8.8, 3.7], [9.8, 3.2], [8.5, 2.8], [9.2, 3.5],
    [8.9, 2.5], [9.6, 3.3], [8.2, 3.0], [9.5, 2.6], [9.1, 3.6],
    [4.4, 1.1], [5.6, 1.9], [4.7, 2.1], [5.7, 1.2], [4.3, 1.6],
    [5.3, 0.9], [4.8, 2.2], [5.8, 1.7], [4.6, 1.3], [5.2, 2.0],
    [4.9, 1.0], [5.9, 1.8], [4.2, 1.5], [5.5, 1.1], [5.1, 2.1],
])

CLUSTER_COLORS = ["#E24B4A", "#185FA5", "#1D9E75", "#EF9F27", "#7F77DD"]
CLUSTER_NAMES  = ["A", "B", "C", "D", "E"]


def get_random_centroids(k):
    pts = FIXED_POINTS[:15 * k]
    indices = np.random.choice(len(pts), size=k, replace=False)
    return pts[indices].copy()


def euclidean(p, c):
    return np.sqrt(np.sum((p - c) ** 2))


def assign_points(points, centroids):
    labels = []
    for p in points:
        dists = [euclidean(p, c) for c in centroids]
        labels.append(int(np.argmin(dists)))
    return np.array(labels)


def move_centroids(points, labels, k, current_centroids):
    new_centroids = []
    for i in range(k):
        cluster_pts = points[labels == i]
        if len(cluster_pts) == 0:
            new_centroids.append(current_centroids[i])
        else:
            new_centroids.append(cluster_pts.mean(axis=0))
    return np.array(new_centroids)


def has_converged(old_c, new_c, tol=0.05):
    return np.max([euclidean(o, n) for o, n in zip(old_c, new_c)]) < tol


def build_steps(k, init_c):
    pts = FIXED_POINTS[: 15 * k]
    steps = []

    steps.append(dict(
        phase="init", iteration=0,
        centroids=init_c.copy(), old_centroids=None,
        labels=np.full(len(pts), -1),
        title="Step 1 — Place initial centroids (random)",
        explanation=(
            f"We randomly pick **{k} data points** and use them as our "
            "starting centroids (shown as ✕ marks). "
            "This is the standard random-initialisation strategy for K-Means. "
            "No points have been assigned yet — all dots are grey.\n\n"
            "*Tip: click **🎲 Re-randomize** in the sidebar to try different starting positions.*"
        ),
        formula=None, converged=False,
    ))

    centroids = init_c.copy()
    step_num = 1
    iteration = 1

    for iteration in range(1, 7):
        step_num += 1
        labels = assign_points(pts, centroids)
        steps.append(dict(
            phase="assign", iteration=iteration,
            centroids=centroids.copy(), old_centroids=None,
            labels=labels.copy(),
            title=f"Step {step_num} — Assign points to nearest centroid  (Iteration {iteration})",
            explanation=(
                "Every point is now coloured by its **nearest centroid**. "
                "We compute the **Euclidean distance** from each point to every centroid "
                "and assign the point to the cluster with the smallest distance."
            ),
            formula="distance(p, c) = √[ (x₁ − x₂)² + (y₁ − y₂)² ]",
            converged=False,
        ))

        step_num += 1
        new_centroids = move_centroids(pts, labels, k, centroids)
        conv = has_converged(centroids, new_centroids)
        steps.append(dict(
            phase="move", iteration=iteration,
            centroids=new_centroids.copy(), old_centroids=centroids.copy(),
            labels=labels.copy(),
            title=f"Step {step_num} — Move centroids to cluster mean  (Iteration {iteration})",
            explanation=(
                "✅ **Converged!** Centroids barely moved — the algorithm is done. "
                "The final clusters are stable."
                if conv else
                "Each centroid moves to the **mean position** of all points assigned to it. "
                "The dashed arrows show how far each centroid shifted. "
                "Then we will re-assign points with the new positions."
            ),
            formula="new_centroid = mean( x values in cluster ),  mean( y values in cluster )",
            converged=conv,
        ))

        if conv:
            break
        centroids = new_centroids

    steps.append(dict(
        phase="done", iteration=iteration,
        centroids=centroids.copy(), old_centroids=None,
        labels=assign_points(pts, centroids),
        title=f"Done — Final clusters  (k = {k})",
        explanation=(
            f"K-Means has converged. All **{k} clusters** are stable. "
            "Every point belongs to its nearest centroid and every centroid "
            "sits at the mean of its cluster."
        ),
        formula=None, converged=True,
    ))

    return pts, steps


def run_kmeans():
    st.title("K-Means Clustering — Step-by-Step Visualizer")
    st.caption("Walk through every iteration of K-Means to understand exactly how it works.")

    if "step_index" not in st.session_state:
        st.session_state.step_index = 0
    if "k" not in st.session_state:
        st.session_state.k = 3
    if "initial_centroids" not in st.session_state:
        st.session_state.initial_centroids = get_random_centroids(st.session_state.k)

    st.sidebar.header("Controls")
    k = st.sidebar.slider("Number of clusters (k)", 2, 5, st.session_state.k)

    if k != st.session_state.k:
        st.session_state.k = k
        st.session_state.step_index = 0
        st.session_state.initial_centroids = get_random_centroids(k)

    if st.sidebar.button("↺ Reset to Step 1"):
        st.session_state.step_index = 0

    if st.sidebar.button("🎲 Re-randomize centroids"):
        st.session_state.initial_centroids = get_random_centroids(st.session_state.k)
        st.session_state.step_index = 0

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Re-randomize** picks new random starting centroids from the data. "
        "Different starts can lead to different final clusters — this is why "
        "K-Means++ and multiple restarts are used in practice."
    )

    pts, steps = build_steps(st.session_state.k, st.session_state.initial_centroids)
    total = len(steps)
    idx   = st.session_state.step_index
    step  = steps[idx]

    PHASE_LABELS = {
        "init":   "Initialize",
        "assign": f"Assign  (iter {step['iteration']})",
        "move":   f"Move  (iter {step['iteration']})",
        "done":   "Done ✓",
    }

    phase_cols = st.columns(len(steps))
    for i, (col, s) in enumerate(zip(phase_cols, steps)):
        label = "✓" if i < idx else ("●" if i == idx else "○")
        col.markdown(
            f"<div style='text-align:center; font-size:11px; "
            f"color:{'#1D9E75' if i < idx else ('#185FA5' if i == idx else '#888')};'>"
            f"{label}<br>{PHASE_LABELS.get(s['phase'], s['phase'])}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    chart_col, panel_col = st.columns([2, 1], gap="medium")

    with chart_col:
        fig = go.Figure()

        if step["phase"] == "move" and step["old_centroids"] is not None:
            for i, (oc, nc) in enumerate(zip(step["old_centroids"], step["centroids"])):
                dist_moved = np.sqrt(np.sum((oc - nc) ** 2))
                if dist_moved > 0.05:
                    fig.add_annotation(
                        ax=oc[0], ay=oc[1], x=nc[0], y=nc[1],
                        xref="x", yref="y", axref="x", ayref="y",
                        showarrow=True, arrowhead=3, arrowsize=1.5,
                        arrowwidth=2, arrowcolor=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                        opacity=0.7,
                    )

        unique_labels = sorted(set(step["labels"]))
        for lbl in unique_labels:
            mask = step["labels"] == lbl
            color = CLUSTER_COLORS[lbl % len(CLUSTER_COLORS)] if lbl >= 0 else "#AAAAAA"
            name  = f"Cluster {CLUSTER_NAMES[lbl]}" if lbl >= 0 else "Unassigned"
            fig.add_trace(go.Scatter(
                x=pts[mask, 0], y=pts[mask, 1],
                mode="markers", name=name,
                marker=dict(size=9, color=color, opacity=0.7, line=dict(color=color, width=1.5)),
            ))

        for i, c in enumerate(step["centroids"]):
            fig.add_trace(go.Scatter(
                x=[c[0]], y=[c[1]],
                mode="markers+text",
                name=f"Centroid {CLUSTER_NAMES[i]}",
                text=[CLUSTER_NAMES[i]],
                textposition="top right",
                textfont=dict(size=13, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
                marker=dict(
                    symbol="x", size=18,
                    color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
                    line=dict(width=3, color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]),
                ),
                showlegend=False,
            ))

        fig.update_layout(
            template="plotly_dark", height=500,
            title=dict(text=step["title"], font=dict(size=14)),
            xaxis_title="Feature 1", yaxis_title="Feature 2",
            xaxis=dict(range=[0, 11]), yaxis=dict(range=[0, 11]),
            legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
            margin=dict(t=80, b=40, l=40, r=20),
        )

        st.plotly_chart(fig, use_container_width=True)

        prev_col, info_col, next_col = st.columns([1, 2, 1])
        with prev_col:
            if st.button("← Prev", disabled=(idx == 0), use_container_width=True):
                st.session_state.step_index -= 1
                st.rerun()
        with info_col:
            st.markdown(
                f"<p style='text-align:center;color:#888;font-size:13px;margin-top:8px'>"
                f"Step {idx + 1} of {total}</p>",
                unsafe_allow_html=True,
            )
        with next_col:
            label = "Done ✓" if idx == total - 1 else "Next →"
            if st.button(label, disabled=(idx == total - 1), use_container_width=True):
                st.session_state.step_index += 1
                st.rerun()

    with panel_col:
        phase_color = {"init": "#185FA5", "assign": "#EF9F27", "move": "#E24B4A", "done": "#1D9E75"}
        phase_label = {"init": "Initialization", "assign": "Assignment", "move": "Update", "done": "Converged"}
        st.markdown(
            f"<span style='background:{phase_color.get(step['phase'],'#555')};color:#fff;"
            f"padding:3px 12px;border-radius:12px;font-size:12px;font-weight:600'>"
            f"Iteration {step['iteration']} · {phase_label.get(step['phase'],'')}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(step["explanation"])

        if step["formula"]:
            st.code(step["formula"], language=None)

        if step["converged"]:
            st.success("✅ Centroids have converged — algorithm complete!")

        st.markdown("---")
        st.markdown("**Centroid positions**")
        centroid_df = pd.DataFrame(
            [(CLUSTER_NAMES[i], round(c[0], 2), round(c[1], 2))
             for i, c in enumerate(step["centroids"])],
            columns=["Cluster", "X", "Y"],
        )
        st.dataframe(centroid_df, hide_index=True, use_container_width=True)

        if np.any(step["labels"] >= 0):
            st.markdown("**Points per cluster**")
            counts = pd.Series(step["labels"]).value_counts().sort_index()
            count_df = pd.DataFrame({
                "Cluster": [CLUSTER_NAMES[i] for i in counts.index],
                "Count":   counts.values,
            })
            st.dataframe(count_df, hide_index=True, use_container_width=True)
