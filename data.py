import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler, MinMaxScaler


# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI Preprocessing Visualizer",
    layout="wide"
)

st.title("AI Preprocessing Visualizer")
st.markdown("Created by **Shiven Watts**")
st.markdown("Under guidance of **Dr. Abhinay Kumar**")
st.markdown("---")


# =====================================================
# SIDEBAR CONTROLS
# =====================================================

st.sidebar.header("Dataset Controls")

NUM_POINTS = 100
st.sidebar.info("Number of Points: 100 (fixed)")

noise_sigma = st.sidebar.slider("Noise σ", 0.1, 5.0, 1.2)
outlier_rate = st.sidebar.slider("Outlier Rate %", 0, 30, 0)


# =====================================================
# IMPUTATION METHOD
# =====================================================

st.subheader("Imputation")
imputation_method = st.radio(
    "Choose Imputation Method", ["raw", "mean", "median"], horizontal=True
)


# =====================================================
# OUTLIER DETECTION METHOD
# =====================================================

st.subheader("Outlier Detection")
outlier_method = st.radio(
    "Choose Outlier Detection", ["off", "IQR x1.5", "z > 2.5"], horizontal=True
)


# =====================================================
# NORMALIZATION METHOD
# =====================================================

st.subheader("Normalization")
normalization_method = st.radio(
    "Choose Normalization",
    ["none", "z-score", "robust scaler", "min-max", "log"],
    horizontal=True,
)


# =====================================================
# DATA GENERATION
# Fixed seed dataset:
#   80 "normal" points around mean=50 (tight cluster)
#    8 HIGH outliers clearly above the mean (90–130)
#   12 LOW outliers clearly below the mean (5–25)
# All values ≥ 5 so log(x) is always safe and meaningful.
# =====================================================

FIXED_SEED = 42
np.random.seed(FIXED_SEED)

# Core cluster: 80 points, mean=50, std=5
main_data = np.random.normal(loc=50, scale=5, size=80)

# HIGH outliers: 8 points well above the main cluster
high_outliers = np.array([92.0, 97.0, 103.0, 108.0, 114.0, 119.0, 124.0, 130.0])

# LOW outliers: 12 points well below the main cluster
low_outliers = np.array([5.0, 7.5, 9.0, 11.0, 13.5, 15.0,
                          17.0, 19.0, 21.0, 22.5, 24.0, 25.5])

# Combine → 100 points, shuffle
data = np.concatenate([main_data, high_outliers, low_outliers])
np.random.seed(FIXED_SEED + 30)
np.random.shuffle(data)

# Add noise only to the main cluster (20–80) so outlier identity is preserved
np.random.seed(FIXED_SEED + 20)
noise = np.random.normal(0, noise_sigma, NUM_POINTS)
in_main = (data > 35) & (data < 65)
data = np.where(in_main, data + noise, data)

# Clip to ≥ 5 so log(x) is always defined and positive
data = np.clip(data, 5.0, None)


# =====================================================
# ADD EXTRA OUTLIERS (slider)
# =====================================================

num_extra_outliers = int((outlier_rate / 100) * NUM_POINTS)
np.random.seed(FIXED_SEED + 1)
outlier_indices = np.random.choice(NUM_POINTS, num_extra_outliers, replace=False)
for idx in outlier_indices:
    data[idx] = data[idx] * 3


# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(data, columns=["Data"])


# =====================================================
# IMPUTATION
# =====================================================

if imputation_method == "raw":
    df_processed = df.copy()
elif imputation_method == "mean":
    imputer = SimpleImputer(strategy="mean")
    df_processed = pd.DataFrame(imputer.fit_transform(df), columns=["Data"])
elif imputation_method == "median":
    imputer = SimpleImputer(strategy="median")
    df_processed = pd.DataFrame(imputer.fit_transform(df), columns=["Data"])


# =====================================================
# OUTLIER DETECTION
# Always compute z-scores for all points
# =====================================================

_det_mean   = df_processed["Data"].mean()
_det_std    = df_processed["Data"].std()
all_z_scores = (df_processed["Data"] - _det_mean) / _det_std

outliers = pd.DataFrame()

if outlier_method == "IQR x1.5":
    Q1 = df_processed["Data"].quantile(0.25)
    Q3 = df_processed["Data"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = df_processed[
        (df_processed["Data"] < lower) | (df_processed["Data"] > upper)
    ]
elif outlier_method == "z > 2.5":
    outliers = df_processed[abs(all_z_scores) > 2.5]


# =====================================================
# NORMALIZATION
# FIX: log transform applied BEFORE building scaled_data
#      so every downstream reference (plot, table) uses
#      the same transformed values.
# All raw values are ≥ 5, so log(x) is always safe.
# =====================================================

scaled_data = df_processed.copy()

if normalization_method == "z-score":
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df_processed.fillna(df_processed.mean()))
    z_vals = pd.DataFrame(scaled, columns=["Data"])
    scaled_data = z_vals.clip(lower=-2.5, upper=2.5)

elif normalization_method == "robust scaler":
    scaler = RobustScaler()
    scaled = scaler.fit_transform(df_processed.fillna(df_processed.mean()))
    scaled_data = pd.DataFrame(scaled, columns=["Data"])

elif normalization_method == "min-max":
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_processed.fillna(df_processed.mean()))
    scaled_data = pd.DataFrame(scaled, columns=["Data"])

elif normalization_method == "log":
    filled = df_processed.fillna(df_processed.mean())["Data"]
    log_vals = np.log(filled)
    # Reset index so iloc lookups align with 0..99
    scaled_data = pd.DataFrame(log_vals.values, columns=["Data"])


# =====================================================
# OUTLIER Y-VALUES IN THE SCALED SPACE
# FIX: map outlier indices into scaled_data so red
#      markers always sit on the correct transformed value.
# =====================================================

if len(outliers) > 0:
    outlier_scaled_y = scaled_data["Data"].iloc[outliers.index]
else:
    outlier_scaled_y = pd.Series(dtype=float)


# =====================================================
# COMPUTE STATISTICS (pre-normalization)
# =====================================================

raw_data   = df_processed["Data"].dropna()
stat_mean  = raw_data.mean()
stat_median= raw_data.median()
stat_std   = raw_data.std()
stat_q1    = raw_data.quantile(0.25)
stat_q3    = raw_data.quantile(0.75)
stat_iqr   = stat_q3 - stat_q1
iqr_lower  = stat_q1 - 1.5 * stat_iqr
iqr_upper  = stat_q3 + 1.5 * stat_iqr
z_lower    = stat_mean - 2.5 * stat_std
z_upper    = stat_mean + 2.5 * stat_std
stat_min   = raw_data.min()
stat_max   = raw_data.max()


# ── Row 1: core stats ──────────────────────────────
st.subheader("📊 Dataset Statistics (pre-normalization — same values used for outlier detection)")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean",     f"{stat_mean:.3f}")
c2.metric("Median",   f"{stat_median:.3f}")
c3.metric("Std Dev",  f"{stat_std:.3f}")
c4.metric("Q1 (25%)", f"{stat_q1:.3f}")
c5.metric("Q3 (75%)", f"{stat_q3:.3f}")

# ── Row 2: fence values ────────────────────────────
if outlier_method == "IQR x1.5":
    st.markdown("**IQR Fence Values** — a point is flagged if it falls outside these bounds:")
    f1, f2, f3 = st.columns(3)
    f1.metric("IQR", f"{stat_iqr:.3f}")
    f2.metric("Lower Fence (Q1 − 1.5×IQR)", f"{iqr_lower:.3f}",
              help="Points BELOW this are flagged")
    f3.metric("Upper Fence (Q3 + 1.5×IQR)", f"{iqr_upper:.3f}",
              help="Points ABOVE this are flagged")

elif outlier_method == "z > 2.5":
    st.markdown("**Z-Score Fence Values** — a point is flagged if its z-score is beyond ±2.5:")
    f1, f2, f3 = st.columns(3)
    f1.metric("Std Dev", f"{stat_std:.3f}")
    f2.metric("Lower Fence (mean − 2.5×std)", f"{z_lower:.3f}",
              help="Points BELOW this are flagged")
    f3.metric("Upper Fence (mean + 2.5×std)", f"{z_upper:.3f}",
              help="Points ABOVE this are flagged")


# =====================================================
# NORMALIZATION LIVE INFO BOX
# =====================================================

example_raw = round(float(stat_median), 3)

if normalization_method == "none":
    st.info("**No Normalization** — Raw values used as-is.")

elif normalization_method == "z-score":
    example_out = np.clip((example_raw - stat_mean) / stat_std, -2.5, 2.5)
    st.markdown("**📐 Z-Score (clipped to [−2.5, +2.5]) — How it works on YOUR data:**")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Formula", "clip((x − mean) / std, −2.5, 2.5)")
    n2.metric("Mean used", f"{stat_mean:.3f}")
    n3.metric("Std used", f"{stat_std:.3f}")
    n4.metric(f"Example: {example_raw} →", f"{example_out:.3f}")
    st.caption("✅ Clipping to ±2.5 means extreme outliers are capped rather than dominating the scale.")

elif normalization_method == "robust scaler":
    example_out = (example_raw - stat_median) / stat_iqr
    st.markdown("**📐 Robust Scaler — How it works on YOUR data:**")
    n1, n2, n3, n4 = st.columns(4)
    n1.metric("Formula", "(x − median) / IQR")
    n2.metric("Median used", f"{stat_median:.3f}")
    n3.metric("IQR used", f"{stat_iqr:.3f}")
    n4.metric(f"Example: {example_raw} →", f"{example_out:.3f}")
    st.caption("✅ Outliers don't affect median or IQR much — robust to extreme values.")

elif normalization_method == "min-max":
    example_out = (example_raw - stat_min) / (stat_max - stat_min)
    st.markdown("**📐 Min-Max — How it works on YOUR data:**")
    n1, n2, n3, n4, n5 = st.columns(5)
    n1.metric("Formula", "(x − min) / (max − min)")
    n2.metric("Min used", f"{stat_min:.3f}")
    n3.metric("Max used", f"{stat_max:.3f}")
    n4.metric("Range", f"{stat_max - stat_min:.3f}")
    n5.metric(f"Example: {example_raw} →", f"{example_out:.3f}")
    st.caption("⚠️ One extreme outlier becomes 0 or 1, squishing ALL normal points into a tiny range.")

elif normalization_method == "log":
    example_out = np.log(example_raw)
    # Derive accurate post-transform stats from the actual scaled data
    log_out_min  = scaled_data["Data"].min()
    log_out_max  = scaled_data["Data"].max()
    log_out_mean = scaled_data["Data"].mean()
    log_out_std  = scaled_data["Data"].std()
    st.markdown("**📐 Log Transform — How it works on YOUR data:**")
    n1, n2, n3, n4, n5, n6 = st.columns(6)
    n1.metric("Formula",             "ln(x)")
    n2.metric("Raw range",           f"{stat_min:.3f} → {stat_max:.3f}",
              help="Min and max of the raw (pre-transform) data")
    n3.metric("Log output min",      f"{log_out_min:.3f}",
              help=f"ln({stat_min:.3f}) — smallest transformed value")
    n4.metric("Log output max",      f"{log_out_max:.3f}",
              help=f"ln({stat_max:.3f}) — largest transformed value")
    n5.metric("Log mean / std",      f"{log_out_mean:.3f} / {log_out_std:.3f}",
              help="Mean and std of the log-transformed data")
    n6.metric(f"Example: {example_raw} →", f"{example_out:.3f}",
              help="ln(median raw value)")
    st.caption("✅ Compresses right-skewed data. All raw values are ≥ 5, so ln(x) is always defined and positive.")


# =====================================================
# Point picker
# =====================================================

st.markdown("---")
picked_input = st.text_input(
    "🔍 Enter a point index (0 – 99) to highlight on the graph and in the table:",
    value="0",
    placeholder="e.g. 42",
)

try:
    picked_idx = int(picked_input)
    if picked_idx < 0 or picked_idx >= NUM_POINTS:
        st.warning(f"Index must be between 0 and {NUM_POINTS - 1}. Defaulting to 0.")
        picked_idx = 0
except ValueError:
    st.warning("Please enter a valid integer index. Defaulting to 0.")
    picked_idx = 0


# =====================================================
# PLOTLY VISUALIZATION — main scatter
# FIX: outlier red markers now use outlier_scaled_y
#      so they sit on the transformed (log, z, etc.) value.
# =====================================================

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=list(range(NUM_POINTS)),
        y=scaled_data["Data"],
        mode="markers",
        name="observed",
        marker=dict(color="cyan", size=8),
    )
)

if len(outliers) > 0:
    fig.add_trace(
        go.Scatter(
            x=list(outliers.index),
            y=list(outlier_scaled_y),
            mode="markers",
            name="flagged outlier",
            marker=dict(color="red", size=14, symbol="circle-open"),
        )
    )

fig.add_trace(
    go.Scatter(
        x=[picked_idx],
        y=[scaled_data["Data"].iloc[picked_idx]],
        mode="markers",
        name=f"selected (#{picked_idx})",
        marker=dict(color="orange", size=16),
    )
)

fig.update_layout(
    template="plotly_dark",
    height=550,
    title="Interactive Dataset Visualization (Dots Only)",
    xaxis_title="Index",
    yaxis_title="Value",
)

st.plotly_chart(fig, use_container_width=True)
# =====================================================
# POST-GRAPH: PER-METHOD METRIC CARDS
# =====================================================

st.markdown("---")
st.subheader("📊 Stats Across All Normalizations")

_filled_all = df_processed["Data"].fillna(df_processed["Data"].mean())

_iqr_val = _filled_all.quantile(0.75) - _filled_all.quantile(0.25)

transforms = {
    "🔵 Raw": _filled_all,
    "📐 Z-Score (clipped ±2.5)": ((_filled_all - _filled_all.mean()) / _filled_all.std()).clip(-2.5, 2.5),
    "🛡️ Robust Scaler": (_filled_all - _filled_all.median()) / _iqr_val,
    "📏 Min-Max [0,1]": (_filled_all - _filled_all.min()) / (_filled_all.max() - _filled_all.min()),
    "📉 Log ln(x)": np.log(_filled_all),
}

active_map = {
    "none":          "🔵 Raw",
    "z-score":       "📐 Z-Score (clipped ±2.5)",
    "robust scaler": "🛡️ Robust Scaler",
    "min-max":       "📏 Min-Max [0,1]",
    "log":           "📉 Log ln(x)",
}
active_label = active_map[normalization_method]

for label, series in transforms.items():
    is_active = (label == active_label)
    if is_active:
        st.markdown(f"#### ✅ {label} ← *currently active*")
    else:
        st.markdown(f"#### {label}")

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Mean",   f"{series.mean():.3f}")
    c2.metric("Median", f"{series.median():.3f}")
    c3.metric("Std Dev", f"{series.std():.3f}")
    c4.metric("Min",    f"{series.min():.3f}")
    c5.metric("Max",    f"{series.max():.3f}")
    c6.metric("Q1",     f"{series.quantile(0.25):.3f}")
    c7.metric("Q3",     f"{series.quantile(0.75):.3f}")

# =====================================================
# Z-SCORE PLOT
# =====================================================

if outlier_method != "off":
    st.subheader("📉 Z-Score Plot — All Points")
    st.caption(
        "Each dot shows a point's z-score = (value − mean) / std. "
        "Dashed orange lines mark the ±2.5 threshold. "
        "Red dots have |z| > 2.5."
    )

    z_colors = ["red" if abs(z) > 2.5 else "cyan" for z in all_z_scores]

    fig_z = go.Figure()
    fig_z.add_trace(
        go.Scatter(
            x=list(range(NUM_POINTS)),
            y=all_z_scores.values,
            mode="markers",
            name="z-score",
            marker=dict(color=z_colors, size=8),
            hovertemplate="Index: %{x}<br>Z-score: %{y:.3f}<extra></extra>",
        )
    )
    fig_z.add_hline(y=2.5,  line_dash="dash", line_color="orange",
                    annotation_text="+2.5 threshold", annotation_position="top right")
    fig_z.add_hline(y=-2.5, line_dash="dash", line_color="orange",
                    annotation_text="−2.5 threshold", annotation_position="bottom right")
    fig_z.add_hline(y=0, line_dash="dot", line_color="gray")
    fig_z.add_trace(
        go.Scatter(
            x=[picked_idx],
            y=[all_z_scores.iloc[picked_idx]],
            mode="markers",
            name=f"selected (#{picked_idx})",
            marker=dict(color="orange", size=16),
        )
    )
    fig_z.update_layout(
        template="plotly_dark",
        height=400,
        title="Z-Score per Point  (red = |z| > 2.5)",
        xaxis_title="Index",
        yaxis_title="Z-Score",
    )
    st.plotly_chart(fig_z, use_container_width=True)


# =====================================================
# DATA TABLES
# =====================================================

col1, col2 = st.columns(2)

with col1:
    st.subheader("Processed Dataset")
    st.dataframe(df_processed)

with col2:
    st.subheader("Detected Outliers")
    if len(outliers) > 0:
        outliers_display = outliers.copy()
        outliers_display["Z-Score"] = all_z_scores[outliers.index].round(4)
        outliers_display["Direction"] = outliers_display["Z-Score"].apply(
            lambda z: "⬆ High outlier" if z > 0 else "⬇ Low outlier"
        )
        if outlier_method == "IQR x1.5":
            _iq1 = df_processed["Data"].quantile(0.25)
            _iq3 = df_processed["Data"].quantile(0.75)
            _iiqr = _iq3 - _iq1
            _ilower = _iq1 - 1.5 * _iiqr
            _iupper = _iq3 + 1.5 * _iiqr
            outliers_display["IQR Side"] = outliers_display["Data"].apply(
                lambda v: "Above upper fence" if v > _iupper else "Below lower fence"
            )
        st.dataframe(outliers_display)
    else:
        st.dataframe(outliers)


# =====================================================
# NORMALIZATION COMPARISON TABLE
# =====================================================

st.subheader("📋 Normalization Comparison Table — All Points")
st.caption("Heatmap per column shows low (blue) → high (red). Outlier rows highlighted in dark red.")

_filled  = df_processed["Data"].fillna(df_processed["Data"].mean())
_min     = _filled.min()
_max     = _filled.max()
_mean    = _filled.mean()
_std     = _filled.std()
_median  = _filled.median()
_q1      = _filled.quantile(0.25)
_q3      = _filled.quantile(0.75)
_iqr     = _q3 - _q1

zscore_vals = ((_filled - _mean) / _std).clip(-2.5, 2.5)
robust_vals = (_filled - _median) / _iqr
minmax_vals = (_filled - _min) / (_max - _min)
log_vals    = np.log(_filled)   # safe: all values ≥ 5

norm_table = pd.DataFrame({
    "Raw Value":           _filled.round(4),
    "Z-Score (clipped)":   zscore_vals.round(4),
    "Robust Scaler":       robust_vals.round(4),
    "Min-Max":             minmax_vals.round(4),
    "Log (ln(x))":         log_vals.round(4),
})

# Picked point metric strip
picked_row = norm_table.iloc[picked_idx]
st.markdown(f"**Point #{picked_idx} across all methods:**")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Raw Value",         f"{picked_row['Raw Value']:.4f}")
p2.metric("Z-Score (clipped)", f"{picked_row['Z-Score (clipped)']:.4f}")
p3.metric("Robust Scaler",     f"{picked_row['Robust Scaler']:.4f}")
p4.metric("Min-Max",           f"{picked_row['Min-Max']:.4f}")
p5.metric("Log (ln(x))",       f"{picked_row['Log (ln(x))']:.4f}")

st.dataframe(norm_table)
# =====================================================
# FEEDBACK FORM
# =====================================================

st.markdown("---")
st.subheader("💬 Feedback")
st.caption("We'd love to hear your thoughts!")

with st.form("feedback_form"):
    name = st.text_input("Your Name")
    email = st.text_input("Your Email (optional)")
    rating = st.select_slider("Rate this app", options=["⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐"], value="⭐⭐⭐")
    message = st.text_area("Your Feedback")
    submitted = st.form_submit_button("Submit Feedback")

if submitted:
    if message.strip() == "":
        st.warning("Please write something before submitting.")
    else:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart

        try:
            sender_email = st.secrets["EMAIL_ADDRESS"]
            sender_password = st.secrets["EMAIL_PASSWORD"]

            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = "shivenwatts@gmail.com"
            msg["Subject"] = "New Feedback — AI Preprocessing Visualizer"

            body = f"""
New feedback received:

Name: {name if name else 'Anonymous'}
Email: {email if email else 'Not provided'}
Rating: {rating}

Message:
{message}
            """
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
                server.login(sender_email, sender_password)
                server.sendmail(sender_email, "shivenwatts@gmail.com", msg.as_string())

            st.success("✅ Thank you! Your feedback has been sent.")
        except Exception as e:
            st.error(f"Something went wrong: {e}")
