import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils import resample

# ========== Setup ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, "..", "data", "processed")
plots_dir = os.path.join(script_dir, "..", "plots")
os.makedirs(plots_dir, exist_ok=True)

# ========== Load and Combine CSVs ==========
df_list = []
for fname in os.listdir(processed_dir):
    if fname.endswith(".csv"):
        path = os.path.join(processed_dir, fname)
        try:
            df = pd.read_csv(path)
            if df.empty:
                continue
            df["source_file"] = fname
            df_list.append(df)
        except:
            continue

combined_df = pd.concat(df_list, ignore_index=True)
X = combined_df.select_dtypes(include="number").dropna(axis=1)

# ========== Subsample, Scale, UMAP ==========
X_small = resample(X, n_samples=20000, random_state=42)
combined_df_small = combined_df.iloc[X_small.index]  # align metadata

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_small)

reducer = umap.UMAP(n_neighbors=10, min_dist=0.5, metric="cosine", random_state=42)
embedding = reducer.fit_transform(X_scaled)

# ========== KMeans Clustering ==========
k = 3
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)

# ========== Plot: UMAP with Cluster Labels ==========
plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=3, alpha=0.6)
plt.title(f"UMAP with {k} KMeans Clusters")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plot_path = os.path.join(plots_dir, f"Flow_Tcell_umap_kmeans_{k}.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"✅ Saved clustered plot to: {plot_path}")

# ========== Plot: UMAP Colored by Source File ==========
plt.figure(figsize=(10, 7))
colors = pd.factorize(combined_df_small["source_file"])[0]
plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, cmap="tab20", s=3, alpha=0.6)
plt.title("UMAP Colored by Source File")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
source_plot_path = os.path.join(plots_dir, "Flow_Tcell_umap_sourcefile.png")
plt.savefig(source_plot_path, dpi=300)
plt.show()
print(f"✅ Source-colored UMAP saved to: {source_plot_path}")

# ========== Marker Expression Heatmap per Cluster ==========

# Prep DataFrame with cluster labels
X_small_df = pd.DataFrame(X_scaled, columns=X.columns)
X_small_df["cluster"] = labels

# Compute cluster means
cluster_means = X_small_df.groupby("cluster").mean()

# Map fluors to real marker names (customize this for your panel)
marker_map = {
    'BV421-A': 'CD4',
    'BV510-A': 'CD8',
    'BV605-A': 'CD25',
    'BV650-A': 'FoxP3',
    'BV786-A': 'IFNg',
    'BB515-A': 'CD3',
    'PE-A': 'IL2',
    'APC-A': 'TNFa',
    'APC-R700-A': 'CD44',
    'APC-Cy7-A': 'CD62L'
}

# Rename marker columns
cluster_means_renamed = cluster_means.rename(columns={k: marker_map.get(k, k) for k in cluster_means.columns})

# Normalize (z-score per column)
heatmap_data = pd.DataFrame(
    StandardScaler().fit_transform(cluster_means_renamed),
    columns=cluster_means_renamed.columns,
    index=cluster_means_renamed.index
)

# Plot heatmap
plt.figure(figsize=(14, 6))
sns.heatmap(
    heatmap_data,
    cmap="RdBu_r",  # changed from "plasma"
    annot=True,
    fmt=".2f",
    cbar_kws={"label": "Z-score"},
    linewidths=0.4,
    linecolor='gray'
)
plt.title("Marker Expression per Cluster", fontsize=14, weight="bold")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()

heatmap_path = os.path.join(plots_dir, "Flow_Tcell_cluster_marker_heatmap_indigored.png")
plt.savefig(heatmap_path, dpi=300)
plt.show()
print(f"✅ Indigo-Red heatmap saved to: {heatmap_path}")

# ========== Save Clustered Dataset ==========
X_small_df["UMAP_1"] = embedding[:, 0]
X_small_df["UMAP_2"] = embedding[:, 1]
X_small_df["source_file"] = combined_df_small["source_file"].values

csv_path = os.path.join(processed_dir, "Flow_Tcell_clustered_data.csv")
X_small_df.to_csv(csv_path, index=False)
print(f"✅ Clustered data saved to: {csv_path}")

# ========== Summary Table ==========
summary = X_small_df.groupby(["source_file", "cluster"]).size().unstack(fill_value=0)
summary_path = os.path.join(processed_dir, "Flow_Tcell_cluster_summary.csv")
summary.to_csv(summary_path)
print(f"✅ Summary table saved to: {summary_path}")
print("\n📊 Cells per cluster per sample:")
print(summary)
