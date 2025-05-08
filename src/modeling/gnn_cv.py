import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import os
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedKFold
import torch.serialization

# === Fix for PyTorch 2.6+ ===
from torch_geometric.data import Data
torch.serialization.add_safe_globals([Data])

# === Load Graph ===
script_dir = os.path.dirname(os.path.abspath(__file__))
graph_path = os.path.join(script_dir, "cell_graph.pt")
data = torch.load(graph_path, weights_only=False)

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

# === Cross-validation ===
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
idx = torch.arange(data.num_nodes)
accs = []

print("🧪 Running 5-Fold Cross-Validation...")
for fold, (train_idx, test_idx) in enumerate(skf.split(idx, data.y)):
    model = GNN(data.num_node_features, 32, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    # Train
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        preds = logits.argmax(dim=1)
        correct = (preds[test_idx] == data.y[test_idx]).sum().item()
        acc = correct / len(test_idx)
        accs.append(acc)
        print(f"Fold {fold+1}: Accuracy = {acc:.3f}")

# === Summary ===
mean_acc = sum(accs) / len(accs)
print(f"\n✅ Mean Accuracy (5-fold CV): {mean_acc:.3f}")
