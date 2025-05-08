import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap
from torch_geometric.nn import SAGEConv
import torch.serialization
from torch_geometric.data import Data

# === PyTorch 2.6+ fix ===
torch.serialization.add_safe_globals([Data])

# === Load Graph ===
script_dir = os.path.dirname(os.path.abspath(__file__))
graph_path = os.path.join(script_dir, "cell_graph.pt")
data = torch.load(graph_path, weights_only=False)

# === Load same model structure ===
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# === Initialize & Load Trained Model ===
model = GNN(data.num_node_features, 32, 2)
model.eval()

# === Forward Pass ===
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)

# === UMAP on Embeddings ===
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(logits.numpy())

# === Plot 1: True Labels ===
plt.figure(figsize=(10, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=data.y.numpy(), cmap="coolwarm", s=10, alpha=0.7)
plt.title("UMAP of GNN Embedding — Colored by TRUE Labels (CD4/CD8)")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "..", "plots", "GNN_UMAP_true.png"), dpi=300)
plt.show()

# === Plot 2: Predicted Labels ===
plt.figure(figsize=(10, 6))
plt.scatter(embedding[:, 0], embedding[:, 1], c=preds.numpy(), cmap="coolwarm", s=10, alpha=0.7)
plt.title("UMAP of GNN Embedding — Colored by PREDICTED Labels")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
plt.tight_layout()
plt.savefig(os.path.join(script_dir, "..", "plots", "GNN_UMAP_predicted.png"), dpi=300)
plt.show()
