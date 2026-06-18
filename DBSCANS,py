def run_dbscan():
    import streamlit as st
import numpy as np
import plotly.graph_objects as go
import time


st.set_page_config(page_title="DBSCAN Visualizer", layout="wide")


# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
.stApp { background-color: #0e1117; }
.title-block { padding: 8px 0 4px 0; }
.title-block h1 { font-size: 2rem; font-weight: 700; color: #ffffff; margin: 0; }
.title-block p  { color: #888; margin: 2px 0 0 0; font-size: 0.9rem; }
.status-box {
    background: #1a1d27; border-radius: 10px; padding: 14px 18px;
    margin-bottom: 10px; border-left: 4px solid #1D9E75;
}
.stat-row { display: flex; gap: 20px; flex-wrap: wrap; margin-top: 8px; }
.stat-item { text-align: center; }
.stat-num  { font-size: 1.6rem; font-weight: 700; }
.stat-lbl  { font-size: 0.75rem; color: #888; }
div[data-testid="stButton"] button {
    border-radius: 8px; font-weight: 600; font-size: 1rem;
    padding: 8px 24px;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════════
def jitter_ring(cx, cy, r, n, rng, radial_jitter=0.12, angular_jitter=0.0):
    pts = []
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    for a in angles:
        aa = a + rng.normal(0, angular_jitter)
        rr = r + rng.normal(0, radial_jitter)
        pts.append([cx + rr*np.cos(aa), cy + rr*np.sin(aa)])
    return pts


def make_smiley():
    rng = np.random.RandomState(7)
    pts = []
    pts += jitter_ring(5, 5, 4.2, 90, rng, radial_jitter=0.10)
    pts += jitter_ring(3.2, 6.6, 0.55, 30, rng, radial_jitter=0.05)
    pts += jitter_ring(6.8, 6.6, 0.55, 30, rng, radial_jitter=0.05)
    mouth_angles = np.linspace(np.pi + 0.35, 2*np.pi - 0.35, 60)
    for a in mouth_angles:
        rr = 2.6 + rng.normal(0, 0.08)
        pts.append([5 + rr*np.cos(a), 5 + rr*np.sin(a) - 0.35])
    arr = np.array(pts) + rng.normal(0, 0.06, (len(pts), 2))
    noise = rng.uniform(0.5, 9.5, (12, 2))
    return np.vstack([arr, noise])


def make_rings():
    rng = np.random.RandomState(42)
    pts = []
    pts += jitter_ring(5, 5, 1.0, 50, rng, radial_jitter=0.10)
    pts += jitter_ring(5, 5, 2.6, 90, rng, radial_jitter=0.12)
    pts += jitter_ring(5, 5, 4.3, 130, rng, radial_jitter=0.12)
    noise = rng.uniform(0.5, 9.5, (14, 2))
    return np.vstack([np.array(pts), noise])


def make_moons():
    rng = np.random.RandomState(0)
    pts = []
    for i in range(110):
        a = np.pi * i / 109
        pts.append([np.cos(a)*3.5 + 2.8 + rng.normal(0,0.12),
                    np.sin(a)*2.8 + 5.2 + rng.normal(0,0.12)])
    for i in range(110):
        a = np.pi + np.pi * i / 109
        pts.append([np.cos(a)*3.5 + 6.2 + rng.normal(0,0.12),
                    np.sin(a)*2.8 + 4.0 + rng.normal(0,0.12)])
    noise = rng.uniform(0.5, 9.5, (12, 2))
    return np.vstack([np.array(pts), noise])


def make_blobs():
    rng = np.random.RandomState(1)
    centers = [[2,2],[7.5,2],[2,8],[7.5,8],[4.8,5]]
    pts = []
    for cx, cy in centers:
        for _ in range(48):
            pts.append([cx + rng.normal(0,0.48), cy + rng.normal(0,0.48)])
    noise = rng.uniform(0.5,9.5,(12,2))
    return np.vstack([np.array(pts), noise])


def make_spiral():
    rng = np.random.RandomState(3)
    pts = []
    for i in range(110):
        a = 0.30 * i * np.pi / 10
        r = 0.22 + 0.10*i
        pts.append([5 + r*np.cos(a) + rng.normal(0,0.08),
                    5 + r*np.sin(a) + rng.normal(0,0.08)])
        pts.append([5 - r*np.cos(a) + rng.normal(0,0.08),
                    5 - r*np.sin(a) + rng.normal(0,0.08)])
    noise = rng.uniform(0.5,9.5,(12,2))
    return np.clip(np.vstack([np.array(pts), noise]), 0.3, 9.7)


DATASETS = {
    "😊 Smiley Face":      make_smiley,
    "⭕ Concentric Rings": make_rings,
    "🌙 Two Moons":        make_moons,
    "🔵 Gaussian Blobs":   make_blobs,
    "🌀 Spiral":           make_spiral,
}


CLUSTER_COLORS = [
    "#E24B4A","#185FA5","#1D9E75","#EF9F27",
    "#9B59B6","#E67E22","#1ABC9C","#E91E63"
]
NOISE_COLOR    = "#FF6B6B"
UNVISITED_COLOR = "#444455"
ACTIVE_COLOR    = "#EF9F27"


# ══════════════════════════════════════════════════════════════════════════════
# DBSCAN — returns full animation frames
# ══════════════════════════════════════════════════════════════════════════════
def compute_frames(points, eps, min_pts):
    N = len(points)
    nbrs = [
        [j for j in range(N) if np.linalg.norm(points[i] - points[j]) <= eps]
        for i in range(N)
    ]
    is_core = [len(nbrs[i]) >= min_pts for i in range(N)]

    labels  = np.full(N, -2, dtype=int)
    ptypes  = np.full(N, "unvisited", dtype=object)
    visited = np.zeros(N, dtype=bool)
    frames  = []
    cid     = 0

    def snap(active, msg):
        frames.append((labels.copy(), ptypes.copy(), active, msg))

    snap(-1, "All points unvisited. Click ▶ Start to begin clustering.")

    for i in range(N):
        if visited[i]:
            continue
        visited[i] = True
        snap(i, f"Visiting point {i} — checking its neighborhood (ε = {eps})…")

        if not is_core[i]:
            labels[i]  = -1
            ptypes[i]  = "noise"
            snap(i, f"Point {i} has only {len(nbrs[i])} neighbor(s) < minPts ({min_pts}) → marked **Noise** ✕")
            continue

        ptypes[i] = "core"
        labels[i] = cid
        snap(i, f"Point {i} has {len(nbrs[i])} neighbors ≥ minPts → **Core point** ⭐  Seeds **Cluster {cid+1}**")

        seeds = [j for j in nbrs[i] if j != i]
        si = 0
        while si < len(seeds):
            q = seeds[si]
            si += 1
            if not visited[q]:
                visited[q] = True
            if labels[q] in (-2, -1):
                labels[q] = cid
                ptypes[q] = "core" if is_core[q] else "border"
                snap(q, f"Point {q} added to **Cluster {cid+1}** as {'core ⭐' if is_core[q] else 'border ◯'}")
                if is_core[q]:
                    new = [j for j in nbrs[q] if j not in seeds and not visited[j]]
                    seeds += new
                    if new:
                        snap(q, f"Point {q} is core → expanding cluster, adding {len(new)} more seed(s)")
        cid += 1

    snap(-1, f"✅ Done! Found **{cid} cluster(s)** and **{int(np.sum(labels==-1))} noise point(s)**.")
    return frames, nbrs, is_core, cid


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
for k, v in [("frame_idx",0),("playing",False),("frames",None),
             ("nbrs",None),("is_core",None),("n_clusters",0),
             ("dataset","😊 Smiley Face"),("eps",0.8),("min_pts",3),
             ("points",None),("speed",0.12)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Controls")

ds = st.sidebar.selectbox(
    "Dataset Shape",
    list(DATASETS.keys()),
    index=list(DATASETS.keys()).index(st.session_state.dataset)
)
if ds != st.session_state.dataset:
    st.session_state.dataset = ds
    st.session_state.frames = None
    st.session_state.frame_idx = 0
    st.session_state.playing = False

eps_val = st.sidebar.slider("ε  (neighborhood radius)", 0.2, 3.0, st.session_state.eps, 0.05)
mp_val  = st.sidebar.slider("minPts  (core threshold)",  2, 8,   st.session_state.min_pts, 1)

if eps_val != st.session_state.eps or mp_val != st.session_state.min_pts:
    st.session_state.eps = eps_val
    st.session_state.min_pts = mp_val
    st.session_state.frames = None
    st.session_state.frame_idx = 0
    st.session_state.playing = False

speed = st.sidebar.slider(
    "Animation Speed (seconds per frame)",
    0.1, 3.0, 0.5, 0.1,
    format="%.1f s/frame",
    help="Higher = slower animation"
)
st.session_state.speed = speed

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Point types:**\n\n"
    "⭐ **Core** — ≥ minPts neighbors in ε\n\n"
    "◯ **Border** — inside a core's ε, < minPts own\n\n"
    "✕ **Noise** — unreachable from any core\n\n"
    "· **Unvisited** — not yet examined"
)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="title-block">
  <h1>DBSCAN — Live Clustering Visualizer</h1>
  <p>Created by <b>Shiven Watts</b> &nbsp;·&nbsp; Under guidance of <b>Dr. Abhinay Kumar</b></p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# COMPUTE / CACHE FRAMES
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.frames is None:
    POINTS = DATASETS[st.session_state.dataset]()
    st.session_state.points = POINTS
    with st.spinner("Computing clustering frames…"):
        frames, nbrs, is_core, nc = compute_frames(
            POINTS, st.session_state.eps, st.session_state.min_pts
        )
    st.session_state.frames = frames
    st.session_state.nbrs = nbrs
    st.session_state.is_core = is_core
    st.session_state.n_clusters = nc
    st.session_state.frame_idx = 0

POINTS   = st.session_state.points
frames   = st.session_state.frames
nbrs     = st.session_state.nbrs
is_core  = st.session_state.is_core
N        = len(POINTS)
TOTAL_F  = len(frames)


# ══════════════════════════════════════════════════════════════════════════════
# PLAYBACK CONTROLS
# ══════════════════════════════════════════════════════════════════════════════
ctrl1, ctrl2, ctrl3, ctrl4, ctrl5 = st.columns([1,1,1,1,3])

with ctrl1:
    if st.button("⏮ Reset"):
        st.session_state.frame_idx = 0
        st.session_state.playing = False
        st.rerun()

with ctrl2:
    if st.session_state.playing:
        if st.button("⏸ Pause", use_container_width=True):
            st.session_state.playing = False
            st.rerun()
    else:
        if st.button("▶ Start", use_container_width=True):
            st.session_state.playing = True
            st.rerun()

with ctrl3:
    if st.button("⏭ Skip End"):
        st.session_state.frame_idx = TOTAL_F - 1
        st.session_state.playing = False
        st.rerun()

with ctrl4:
    if st.button("🔄 New Data"):
        st.session_state.frames = None
        st.session_state.frame_idx = 0
        st.session_state.playing = False
        st.rerun()

with ctrl5:
    fi = st.slider("Frame", 0, TOTAL_F-1, st.session_state.frame_idx, label_visibility="collapsed")
    if fi != st.session_state.frame_idx:
        st.session_state.frame_idx = fi
        st.session_state.playing = False


# ══════════════════════════════════════════════════════════════════════════════
# CURRENT FRAME DATA
# ══════════════════════════════════════════════════════════════════════════════
fi = st.session_state.frame_idx
lbl, pty, active_pt, msg = frames[fi]

n_clusters_now = int(np.max(lbl)+1) if np.any(lbl >= 0) else 0
noise_now      = int(np.sum(lbl == -1))
core_now       = int(np.sum(pty == "core"))
border_now     = int(np.sum(pty == "border"))
unvisited_now  = int(np.sum(lbl == -2))


# ══════════════════════════════════════════════════════════════════════════════
# CHART
# ══════════════════════════════════════════════════════════════════════════════
def make_chart(lbl, pty, active_pt):
    fig = go.Figure()
    t = np.linspace(0, 2*np.pi, 80)
    ev = st.session_state.eps

    def hex_rgba(h, a=0.15):
        h = h.lstrip("#")
        r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{a})"

    from scipy.spatial import ConvexHull
    for cid in range(n_clusters_now):
        mask = lbl == cid
        if np.sum(mask) >= 3:
            pts_xy = POINTS[mask]
            try:
                hull = ConvexHull(pts_xy)
                hx = list(pts_xy[hull.vertices,0]) + [pts_xy[hull.vertices[0],0]]
                hy = list(pts_xy[hull.vertices,1]) + [pts_xy[hull.vertices[0],1]]
                color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
                fig.add_trace(go.Scatter(
                    x=hx, y=hy, fill="toself",
                    fillcolor=hex_rgba(color, 0.18),
                    line=dict(color=color, width=1.5),
                    hoverinfo="skip", showlegend=False
                ))
            except Exception:
                pass

    if active_pt >= 0:
        cx, cy = POINTS[active_pt]
        fig.add_trace(go.Scatter(
            x=cx + ev*np.cos(t), y=cy + ev*np.sin(t),
            mode="lines", hoverinfo="skip",
            line=dict(color="rgba(239,159,39,0.85)", width=2.5, dash="dot"),
            showlegend=False
        ))

    for i in range(N):
        x, y = POINTS[i]
        l, p = lbl[i], pty[i]

        if l >= 0:
            color = CLUSTER_COLORS[l % len(CLUSTER_COLORS)]
        elif l == -1:
            color = NOISE_COLOR
        else:
            color = UNVISITED_COLOR

        is_active = (i == active_pt)
        size = 16 if is_active else 11
        symbol = "x" if p == "noise" else ("circle-open" if p == "border" else "circle")
        border_c = "#ffffff" if is_active else color
        bw = 3 if is_active else 1.2

        label_str = f"Cluster {l+1}" if l >= 0 else ("Noise" if l == -1 else "Unvisited")
        fig.add_trace(go.Scatter(
            x=[x], y=[y], mode="markers", showlegend=False,
            hovertemplate=(
                f"<b>Point {i}</b><br>({x:.2f}, {y:.2f})<br>"
                f"Status: {label_str}<br>Type: {p}<br>"
                f"Neighbors in ε: {len(nbrs[i])}<extra></extra>"
            ),
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(color=border_c, width=bw), opacity=0.95)
        ))

    fig.update_layout(
        template="plotly_dark", height=530,
        xaxis=dict(range=[-0.5,10.5], showgrid=True,
                   gridcolor="rgba(255,255,255,0.06)", title="Feature X"),
        yaxis=dict(range=[-0.5,10.5], showgrid=True,
                   gridcolor="rgba(255,255,255,0.06)", title="Feature Y"),
        margin=dict(t=20, b=40, l=40, r=20),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
chart_col, info_col = st.columns([3, 2], gap="large")

with chart_col:
    chart_placeholder = st.empty()
    chart_placeholder.plotly_chart(make_chart(lbl, pty, active_pt), use_container_width=True)

    pct = int(fi / (TOTAL_F-1) * 100) if TOTAL_F > 1 else 100
    st.markdown(
        f"<div style='margin-top:4px'>"
        f"<div style='background:#222;border-radius:4px;height:6px'>"
        f"<div style='background:#1D9E75;width:{pct}%;height:6px;border-radius:4px;transition:width 0.2s'></div>"
        f"</div>"
        f"<div style='text-align:right;font-size:11px;color:#666;margin-top:2px'>Frame {fi+1} / {TOTAL_F}</div>"
        f"</div>", unsafe_allow_html=True
    )

with info_col:
    st.markdown(
        f"<div class='status-box'>"
        f"<div style='font-size:0.85rem;color:#aaa;margin-bottom:4px'>Status</div>"
        f"<div style='font-size:1rem;font-weight:500;color:#fff'>{msg}</div>"
        f"</div>", unsafe_allow_html=True
    )

    st.markdown("**Live Stats**")
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Clusters", n_clusters_now)
    s2.metric("Noise", noise_now)
    s3.metric("Core", core_now)
    s4.metric("Unvisited", unvisited_now)

    st.markdown("---")

    if fi == TOTAL_F - 1:
        st.markdown("### 🎉 Final Result")
        nc = st.session_state.n_clusters
        for cid in range(nc):
            mask = lbl == cid
            n_c = int(np.sum((lbl == cid) & (pty == "core")))
            n_b = int(np.sum((lbl == cid) & (pty == "border")))
            color = CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]
            st.markdown(
                f"<div style='border-left:4px solid {color};padding:6px 12px;"
                f"margin:6px 0;background:#1a1d27;border-radius:4px'>"
                f"<b style='color:{color}'>Cluster {cid+1}</b> — "
                f"{int(np.sum(mask))} pts &nbsp;·&nbsp; "
                f"{n_c} core &nbsp;·&nbsp; {n_b} border"
                f"</div>", unsafe_allow_html=True
            )
        if noise_now > 0:
            st.markdown(
                f"<div style='border-left:4px solid {NOISE_COLOR};padding:6px 12px;"
                f"margin:6px 0;background:#1a1d27;border-radius:4px'>"
                f"<b style='color:{NOISE_COLOR}'>Noise</b> — {noise_now} point(s)"
                f"</div>", unsafe_allow_html=True
            )
    else:
        st.markdown("**Point Types Explained**")
        st.markdown(
            "| Symbol | Type | Meaning |\n|---|---|---|\n"
            "| ⭐ filled | Core | ≥ minPts neighbors |\n"
            "| ◯ open | Border | Inside a core's ε |\n"
            "| ✕ cross | Noise | Isolated point |\n"
            "| · grey | Unvisited | Not yet checked |"
        )
        st.markdown("---")
        st.markdown("💡 **Tip:** Use the slider to scrub through frames manually, or hit **▶ Start** to watch it animate live.")


# ══════════════════════════════════════════════════════════════════════════════
# AUTO-PLAY LOOP
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.playing:
    if st.session_state.frame_idx < TOTAL_F - 1:
        time.sleep(st.session_state.speed)
        st.session_state.frame_idx += 1
        st.rerun()
    else:
        st.session_state.playing = False
        st.rerun()
