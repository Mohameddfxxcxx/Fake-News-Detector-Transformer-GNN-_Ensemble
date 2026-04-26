import torch
import torch.nn.functional as F
from models.gnn_model import GNNClassifier

# === Config ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/gnn_model.pt"

# === Sample Graph Input ===
# Example edge list: 3 nodes with edges 0-1 and 1-2
edge_index = torch.tensor([
    [0, 1, 1, 2],
    [1, 0, 2, 1]
], dtype=torch.long)  # shape [2, num_edges]

# Example node features: 3 nodes, each with 10 features
x = torch.randn((3, 10), dtype=torch.float)

# === Load Trained Model (use same hidden_channels used during training) ===
model = GNNClassifier(in_channels=10, hidden_channels=32, num_classes=2)  # 🔸 32 must match training
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Inference ===
with torch.no_grad():
    out = model(x.to(DEVICE), edge_index.to(DEVICE))  # logits
    prediction = torch.argmax(out, dim=1)             # predicted class for each node

print("🧠 Predicted classes for nodes:", prediction.tolist())
