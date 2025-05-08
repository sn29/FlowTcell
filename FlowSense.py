import streamlit as st
import os
import pandas as pd
import json
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="FlowSense", layout="wide")
st.title("🧬 FlowSense: Smart Flow Cytometry Analysis")

# === Session state ===
if "map_confirmed" not in st.session_state:
    st.session_state.map_confirmed = False
if "fluor_map" not in st.session_state:
    st.session_state.fluor_map = {}

# === Step 1: Upload FCS File ===
st.header("1. Upload .fcs File")
fcs_file = st.file_uploader("Upload a flow cytometry .fcs file", type=["fcs"])

if fcs_file:
    temp_path = os.path.join("data", fcs_file.name)
    with open(temp_path, "wb") as f:
        f.write(fcs_file.read())
    st.success(f"Uploaded: {fcs_file.name}")

    # === Step 2: Map Fluorochromes ===
    st.header("2. Map Fluorochromes to Markers")
    import FlowCal
    fcs = FlowCal.io.FCSFile(temp_path)
    param_count = int(fcs.text['$PAR'])
    channel_names = []
    for i in range(1, param_count + 1):
        label = fcs.text.get(f"$P{i}S") or fcs.text.get(f"$P{i}N")
        if label:
            channel_names.append(label)

    st.markdown("Assign biological marker names for each detected channel:")
    for ch in channel_names:
        marker = st.text_input(f"{ch}", key=ch)
        if marker:
            st.session_state.fluor_map[ch] = marker

    if st.button("Confirm Mapping"):
        os.makedirs("data/processed", exist_ok=True)
        with open("data/processed/fluor_map.json", "w") as f:
            json.dump(st.session_state.fluor_map, f, indent=2)
        st.session_state.map_confirmed = True
        st.success("✅ Fluorochrome mapping saved!")

# === Step 3: Task Selection ===
if st.session_state.map_confirmed:
    st.header("3. Select Analysis Task")
    task = st.radio("Choose a task:", [
        "CD4/CD8 Classification",
        "Anomaly Detection",
        "Marker Expression Visualization"
    ])

    if st.button("Run Task"):
        with st.spinner("Running analysis..."):
            if task == "CD4/CD8 Classification":
                os.system("python3 models/gnn_model.py")
                st.image("plots/Flow_Tcell_umap_kmeans_3.png")

            elif task == "Anomaly Detection":
                os.system("python3 models/detect_anomalies_with_markers.py")
                os.system("python3 models/analyze_gnn_markers.py")
                st.image("plots/Flow_Tcell_anomalies_umap.png")

            elif task == "Marker Expression Visualization":
                os.system("python3 models/analyze_gnn_markers.py")
                for marker in st.session_state.fluor_map.values():
                    path = f"plots/marker_full_umap_{marker}.png"
                    if os.path.exists(path):
                        st.image(path)

    # === Downloadable Outputs ===
    st.header("4. Download Results")
    if os.path.exists("data/processed/gated_data.csv"):
        st.download_button("Download Gated Data CSV", open("data/processed/gated_data.csv", "rb"), file_name="gated_data.csv")
    if os.path.exists("data/processed/Flow_Tcell_anomalies.csv"):
        st.download_button("Download Anomaly Table", open("data/processed/Flow_Tcell_anomalies.csv", "rb"), file_name="Flow_Tcell_anomalies.csv")
