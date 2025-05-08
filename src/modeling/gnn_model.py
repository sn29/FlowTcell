
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# === Fix for PyTorch 2.6+ deserialization ===
import torch.serialization
from torch_geometric.data import Data

# Whitelist the Data class for deserialization
torch.serialization.add_safe_globals([Data])


# === Load Graph ===
script_dir = os.path.dirname(os.path.abspath(__file__))
graph_path = os.path.join(script_dir, "cell_graph.pt")
data = torch.load(graph_path, weights_only=False)

# === Split train/test indices ===
idx = torch.arange(data.num_nodes)
train_idx, test_idx = train_test_split(
    idx, test_size=0.3, random_state=42, stratify=data.y)

# === GNN Model ===
class GNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index)
        return x

model = GNN(in_channels=data.num_node_features, hidden_channels=32, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# === Training Loop ===
model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[train_idx], data.y[train_idx])
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Evaluation ===
model.eval()
with torch.no_grad():
    logits = model(data.x, data.edge_index)
    preds = logits.argmax(dim=1)
    correct = (preds[test_idx] == data.y[test_idx]).sum().item()
    acc = correct / len(test_idx)

print(f"\n✅ Test Accuracy: {acc:.3f}")
