import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="FlowSense", layout="wide")

# === File Upload ===
st.title("🧬 FlowSense: Smart Flow Cytometry Analysis")
st.markdown("### 1. Upload .fcs File")

uploaded_file = st.file_uploader("Upload a flow cytometry .fcs file", type=["fcs"])
if uploaded_file:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ File saved to: {file_path}")
else:
    st.info("Upload a .fcs file to proceed.")

# === Marker Assignment (Mock UI) ===
st.markdown("### 2. Assign Fluorochromes (Example Panel)")
fluor_map = {}
default_fluors = ["BV421-A", "BV510-A", "BV605-A", "BV650-A", "BB515-A", "PE-A", "APC-R700-A"]
for fluor in default_fluors:
    marker = st.text_input(f"{fluor}", key=fluor)
    if marker:
        fluor_map[fluor] = marker

if fluor_map:
    st.success("✅ Fluorochrome mapping saved!")
    if st.button("Show Mapping"):
        st.json(fluor_map)

# === Task Selection ===
st.markdown("### 3. Select Analysis Task")
task = st.radio("Choose a task:", [
    "CD4/CD8 Classification",
    "Anomaly Detection",
    "Marker Expression Visualization"
])

if st.button("Run Task"):
    st.info(f"🔧 Running: {task}...")

# === Download Links ===
st.markdown("### 4. Download Results")
st.download_button("Download Gated Data CSV", data="example", file_name="gated_data.csv")
st.download_button("Download Anomaly Table", data="example", file_name="anomalies.csv")

# === Marker Analysis (Post-Anomaly)
st.markdown("### 5. Marker Expression in Anomalies")

anomaly_csv_path = "data/processed/Flow_Tcell_anomalies.csv"
if os.path.exists(anomaly_csv_path):
    df_anom = pd.read_csv(anomaly_csv_path)
    marker_cols = ["CD3", "CD4", "CD8", "CD25", "IL2"]  # Can customize this

    for marker in marker_cols:
        if marker not in df_anom.columns:
            st.warning(f"⚠️ {marker} not found in data.")
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.violinplot(data=df_anom, x="anomaly", y=marker, palette=["gray", "red"])
        ax.set_title(f"{marker} Expression by Anomaly")
        st.pyplot(fig)
else:
    st.info("ℹ️ Run anomaly detection first to view marker shifts.")
