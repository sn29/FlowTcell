import os
import FlowCal
import json

# === Setup Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = "/Users/nididev/Documents/FlowTcell-MM/data"
processed_dir = os.path.join(data_dir, "processed")
os.makedirs(processed_dir, exist_ok=True)

# === Find the first .fcs file ===
fcs_files = [f for f in os.listdir(data_dir) if f.endswith(".fcs")]
if not fcs_files:
    raise FileNotFoundError("❌ No .fcs files found in /data.")
fcs_path = os.path.join(data_dir, fcs_files[0])

# === Load only FCS header (fast) ===
print(f"📂 Loading: {fcs_path}")
fcsfile = FlowCal.io.FCSFile(fcs_path)
param_count = int(fcsfile.text['$PAR'])

# === Extract one name per parameter: prefer $PnS, fallback to $PnN ===
fluor_channels = []
for i in range(1, param_count + 1):
    label_key = f"$P{i}S"
    name_key = f"$P{i}N"
    channel_name = fcsfile.text.get(label_key) or fcsfile.text.get(name_key)
    if channel_name:
        fluor_channels.append(channel_name)

# === Deduplicate channel names ===
fluor_channels = sorted(set(fluor_channels))

# === Mapping Prompt ===
print("\n🔬 Detected Fluorochrome Channels:")
fluor_map = {}
for ch in fluor_channels:
    marker = input(f"Assign marker to '{ch}' (leave blank to skip): ").strip()
    if marker:
        fluor_map[ch] = marker

# === Save Mapping ===
map_path = os.path.join(processed_dir, "fluor_map.json")
with open(map_path, "w") as f:
    json.dump(fluor_map, f, indent=2)

print(f"\n✅ Mapping saved to: {map_path}")
print("🧪 Final Mappings:")
for k, v in fluor_map.items():
    print(f"  {k} → {v}")
