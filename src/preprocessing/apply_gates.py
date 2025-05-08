import os
import FlowCal
import numpy as np
import pandas as pd
import json

# === Setup Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "/Users/nididev/Documents/FlowTcell-MM/data"
processed_dir = os.path.join(data_dir, "processed")
map_path = "/Users/nididev/Documents/FlowTcell-MM/data/processed/fluor_map.json"

# === Load Mapping ===
if not os.path.exists(map_path):
    raise FileNotFoundError("❌ fluor_map.json not found. Run gating_ui.py first.")

with open(map_path, "r") as f:
    fluor_map = json.load(f)

# === Load FCS File ===
fcs_files = [f for f in os.listdir(data_dir) if f.endswith(".fcs")]
if not fcs_files:
    raise FileNotFoundError("❌ No .fcs files found in /data.")
fcs_path = os.path.join(data_dir, fcs_files[0])

print(f"📂 Loading: {fcs_files[0]}")
data = FlowCal.io.FCSData(fcs_path)
df_np = np.asarray(data).byteswap().newbyteorder()
df = pd.DataFrame(df_np, columns=data.channels)


# === Rename columns ===
df = df.rename(columns={fluor: marker for fluor, marker in fluor_map.items()})

# === Gating: FSC/SSC lymphocyte region (very rough) ===
if 'FSC-A' in df.columns and 'SSC-A' in df.columns:
    df = df[(df['FSC-A'] > 10000) & (df['FSC-A'] < 80000)]
    df = df[(df['SSC-A'] > 1000) & (df['SSC-A'] < 60000)]

# === Gating: Singlets (FSC-H vs FSC-A) ===
if 'FSC-H' in df.columns and 'FSC-A' in df.columns:
    ratio = df['FSC-A'] / (df['FSC-H'] + 1e-6)
    df = df[(ratio > 0.85) & (ratio < 1.15)]

# === Drop unnamed/unmapped columns ===
df = df[[col for col in df.columns if not col.startswith('Unnamed')]]

# === Save to CSV ===
output_path = os.path.join(processed_dir, "gated_data.csv")
df.to_csv(output_path, index=False)
print(f"\n✅ Gated data saved to: {output_path}")
print(f"🔢 Final event count: {df.shape[0]} cells, {df.shape[1]} features.")
