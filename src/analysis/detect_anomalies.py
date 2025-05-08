import os
import glob
import torch
import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from torch_geometric.data import Data

# === Paths ===
graph_path = "/Users/nididev/Documents/FlowTcell-MM/src/modeling/cell_graph.pt"
processed_dir = "/Users/nididev/Documents/FlowTcell-MM/data/processed"
out_csv = os.path.join(processed_dir, "Flow_Tcell_anomalies.csv")
out_umap = "/Users/nididev/Documents/FlowTcell-MM/plots/Flow_Tcell_anomalies_umap.png"

# === Load graph ===
data = torch.load(graph_path, weights_only=False)
X = data.x.numpy()

# === Load all CSVs
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
df_used = df_all.iloc[:X.shape[0]].reset_index(drop=True)

# === Isolation Forest
print("🔍 Detecting anomalies on GNN node features...")
iso = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
anomaly_labels = (iso.fit_predict(X) == -1).astype(int)

# === Save
df_used["anomaly"] = anomaly_labels
df_used["true_label"] = data.y.numpy()
df_used[[f"feat_{i}" for i in range(X.shape[1])]] = X
df_used.to_csv(out_csv, index=False)
print(f"✅ Anomaly-annotated CSV saved to: {out_csv}")

# === UMAP
print("📊 Visualizing anomalies in UMAP...")
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X)
colors = ['gray' if a == 0 else 'red' for a in anomaly_labels]

plt.figure(figsize=(10, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=8, alpha=0.7)
plt.title("UMAP of GNN Features with Anomalies Highlighted")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(out_umap, dpi=300)
plt.show()
