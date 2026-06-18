import streamlit as st
from kmeans import run_kmeans
from DBSCANS import run_dbscan

st.set_page_config(page_title="Clustering Visualizer")

st.title("Clustering Visualizer")

algo = st.sidebar.selectbox(
    "Choose Algorithm",
    ["K-Means", "DBSCAN"]
)

if algo == "K-Means":
    run_kmeans()

else:
    run_dbscan()
