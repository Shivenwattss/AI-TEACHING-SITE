import streamlit as st
import csv
import os
from datetime import datetime

from kmeans import run_kmeans
from DBSCANS import run_dbscan

st.set_page_config(page_title="Clustering Visualizer", layout="wide")
st.title("Clustering Visualizer")

algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ["K-Means", "DBSCAN"]
)

if algo == "K-Means":
    run_kmeans()
else:
    run_dbscan()

# ══════════════════════════════════════════════════════════════════════════════
# FEEDBACK
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("### 💬 Feedback")
st.caption(f"How helpful was the **{algo}** visualizer?")

rating = st.feedback("stars", key=f"rating_{algo}")

comment = st.text_area(
    "Any comments or suggestions? (optional)",
    placeholder="e.g. would love to see hierarchical clustering next...",
    key=f"comment_{algo}",
)

if st.button("Submit Feedback"):
    if rating is None:
        st.warning("Please select a star rating before submitting.")
    else:
        feedback_path = os.path.join(os.path.dirname(__file__), "feedback.csv")
        file_exists = os.path.isfile(feedback_path)
        with open(feedback_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["timestamp", "algorithm", "stars", "comment"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), algo, rating + 1, comment])
        st.success("Thanks for your feedback! 🙌")
