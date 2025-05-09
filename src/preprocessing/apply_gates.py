import os
import FlowCal
import numpy as np
import pandas as pd
import json

# === Setup Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "/Users/nididev/Documents/FlowTcell-MM/data"
processed_dir = os.path.join(data_dir, "processed")
map_path = os.path.join(processed_dir, "fluor_map.json")

# === Load Mapping ===
if not os.path.exists(map_path):
    raise FileNotFoundError("❌ fluor_map.json not found. Run gating_ui.py first.")

with open(map_path, "r") as f:
    fluor_map = json.load(f)

# === Process all FCS files ===
gated_all = []

fcs_files = [f for f in os.listdir(data_dir) if f.endswith(".fcs")]
if not fcs_files:
    raise FileNotFoundError("❌ No .fcs files found in /data")

for fname in fcs_files:
    fcs_path = os.path.join(data_dir, fname)
    print(f"📂 Loading: {fname}")
    data = FlowCal.io.FCSData(fcs_path)
    df_np = np.asarray(data).byteswap().newbyteorder()
    df = pd.DataFrame(df_np, columns=data.channels)

    # Rename using fluor_map
    df = df.rename(columns={fluor: marker for fluor, marker in fluor_map.items()})

    # Gating: FSC/SSC lymphocyte region
    if 'FSC-A' in df.columns and 'SSC-A' in df.columns:
        df = df[(df['FSC-A'] > 10000) & (df['FSC-A'] < 80000)]
        df = df[(df['SSC-A'] > 1000) & (df['SSC-A'] < 60000)]

    # Gating: Singlets
    if 'FSC-H' in df.columns and 'FSC-A' in df.columns:
        ratio = df['FSC-A'] / (df['FSC-H'] + 1e-6)
        df = df[(ratio > 0.85) & (ratio < 1.15)]

    # Drop unnamed or unmapped columns
    df = df[[col for col in df.columns if not col.startswith('Unnamed')]]

    # Track source file
    df["source_file"] = fname
    gated_all.append(df)

# === Save merged gated output ===
if gated_all:
    gated_df = pd.concat(gated_all, ignore_index=True)
    output_path = os.path.join(processed_dir, "gated_data.csv")
    gated_df.to_csv(output_path, index=False)
    print(f"\n✅ Merged gated data saved: {output_path}")
    print(f"🔢 Total cells: {gated_df.shape[0]} from {len(fcs_files)} files, {gated_df.shape[1]} features.")
else:
    print("⚠️ No valid gated data found.")
