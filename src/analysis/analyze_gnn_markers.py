import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Setup paths ===
processed_dir = "/Users/nididev/Documents/FlowTcell-MM/data/processed"
plots_dir = "/Users/nididev/Documents/FlowTcell-MM/plots"
os.makedirs(plots_dir, exist_ok=True)

# === Load all .csvs from processed/ ===
csv_files = glob.glob(os.path.join(processed_dir, "*.csv"))
df_list = []

for path in csv_files:
    try:
        df = pd.read_csv(path)
        df["source_file"] = os.path.basename(path)
        df_list.append(df)
    except Exception as e:
        print(f"❌ Error reading {path}: {e}")

if not df_list:
    raise ValueError("❌ No valid CSVs found.")

df_all = pd.concat(df_list, ignore_index=True)

# === Must have 'anomaly' column ===
if "anomaly" not in df_all.columns:
    raise ValueError("❌ 'anomaly' column missing. Run detect_anomalies.py first.")

# === Markers to plot ===
marker_cols = ['CD3', 'CD4', 'CD8', 'CD25', 'IL2', 'CD62L', 'TNFa', 'IFNg']

# === Generate violin plots ===
for marker in marker_cols:
    if marker not in df_all.columns:
        print(f"⚠️ Skipping: {marker} not in data")
        continue

    plt.figure(figsize=(6, 4))
    sns.violinplot(data=df_all, x="anomaly", y=marker, palette=["gray", "red"])
    plt.title(f"{marker} Expression by Anomaly")
    plt.xlabel("Anomaly")
    plt.ylabel("Expression")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"violin_{marker}.png"), dpi=300)
    plt.close()

print(f"✅ Violin plots saved to: {plots_dir}")
