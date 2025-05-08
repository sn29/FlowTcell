import os
import sys
import glob
import pandas as pd
import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
import torch
from torch_geometric.data import Data

# === Setup paths ===
processed_dir = "/Users/nididev/Documents/FlowTcell-MM/data/processed"
output_path = "/Users/nididev/Documents/FlowTcell-MM/src/modeling/cell_graph.pt"

# === Load all CSVs ===
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
    raise ValueError("❌ No valid CSVs found in processed folder.")

combined_df = pd.concat(df_list, ignore_index=True)

# === Select marker columns ===
marker_cols = ['CD3', 'CD4', 'CD8', 'CD25', 'CD62L', 'IL2', 'TNFa', 'IFNg']
marker_cols = [col for col in marker_cols if col in combined_df.columns]

if not marker_cols:
    raise ValueError("❌ No known marker columns found.")

# === Apply CD4/CD8 gating mask
mask = (combined_df["CD4"] > 500) | (combined_df["CD8"] > 500)
filtered_df = combined_df[mask].copy()

X = filtered_df[marker_cols].fillna(0)
X_scaled = StandardScaler().fit_transform(X)
labels = np.where(filtered_df["CD4"] > filtered_df["CD8"], 0, 1)

# === Build kNN graph
knn_graph = kneighbors_graph(X_scaled, n_neighbors=15, mode='connectivity', include_self=False)
rows, cols = knn_graph.nonzero()
edge_index = torch.tensor([rows, cols], dtype=torch.long)

# === PyTorch Geometric Data object
x = torch.tensor(X_scaled, dtype=torch.float)
y = torch.tensor(labels, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)

# === Save
torch.save(data, output_path)
print(f"✅ Graph saved to: {output_path}")
print(f"🔢 Nodes: {x.size(0)}, Edges: {edge_index.size(1)}, Features: {x.size(1)}")
