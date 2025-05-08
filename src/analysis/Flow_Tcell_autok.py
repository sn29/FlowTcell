import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

# ========== Setup Paths ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
processed_dir = os.path.join(script_dir, "..", "data", "processed")
plots_dir = os.path.join(script_dir, "..", "plots")
os.makedirs(plots_dir, exist_ok=True)

# ========== Load Combined CSVs ==========
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

# ========== Subsample & Scale ==========
X_small = resample(X, n_samples=20000, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_small)

# ========== Silhouette Scoring ==========
print("🔍 Running KMeans for different k values...")
k_range = range(2, 10)
scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)
    scores.append(score)
    print(f"k = {k}, silhouette score = {score:.4f}")

# ========== Plot Silhouette Scores ==========
plt.figure(figsize=(8, 5))
plt.plot(k_range, scores, marker='o')
plt.title("Silhouette Score vs Number of Clusters (k)")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(plots_dir, "Flow_Tcell_silhouette_scores.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"✅ Silhouette score plot saved to: {plot_path}")
