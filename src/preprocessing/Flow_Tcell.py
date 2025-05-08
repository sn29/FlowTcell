import os
import pandas as pd
import FlowCal # type: ignore
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import umap
import time

# ========== SETUP PATHS ==========
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "..", "data")
processed_dir = os.path.join(script_dir, "..", "data", "processed")
plots_dir = os.path.join(script_dir, "..", "plots")
os.makedirs(processed_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# ========== STEP 1: FCS → CSV ==========
for fname in os.listdir(data_dir):
    if fname.endswith(".fcs"):
        fcs_path = os.path.join(data_dir, fname)
        try:
            data = FlowCal.io.FCSData(fcs_path)
            df = pd.DataFrame(data, columns=data.channels)
            outname = fname.replace(".fcs", ".csv")
            outpath = os.path.join(processed_dir, outname)
            df.to_csv(outpath, index=False)
            print(f"✅ Converted: {fname} → {outname}")
        except Exception as e:
            print(f"❌ Failed to convert {fname}: {e}")

# ========== STEP 2: Combine CSVs ==========
df_list = []
for fname in os.listdir(processed_dir):
    if fname.endswith(".csv"):
        path = os.path.join(processed_dir, fname)
        try:
            df = pd.read_csv(path)
            if df.empty:
                print(f"⚠️ Skipping empty file: {fname}")
                continue
            df["source_file"] = fname
            df_list.append(df)
        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")

if not df_list:
    raise ValueError("❌ No valid CSVs to combine.")

combined_df = pd.concat(df_list, ignore_index=True)
print(f"✅ Combined shape: {combined_df.shape}")

# ========== STEP 3: UMAP on 50K Subsample ==========
X = combined_df.select_dtypes(include="number").dropna(axis=1)

# Subsample to 50,000 cells
X_small = resample(X, n_samples=50000, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_small)

print("🔄 Running UMAP on 50,000 cells...")
start = time.time()
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
embedding = reducer.fit_transform(X_scaled)
print(f"✅ UMAP completed in {time.time() - start:.2f} seconds")

# ========== STEP 4: Plot ==========
plt.figure(figsize=(10, 7))
plt.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.5)
plt.title("UMAP of Flow Cytometry Data")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()

plot_path = os.path.join(plots_dir, "Flow_Tcell_umap.png")
plt.savefig(plot_path, dpi=300)
plt.show()
print(f"✅ UMAP plot saved to: {plot_path}")

