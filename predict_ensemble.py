# predict_ensemble.py

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.utils.data import DataLoader

from models.transformer_model import TransformerClassifier
from models.gnn_model import GNNClassifier
from models.ensemble import EnsembleClassifier
from utils.preprocess import LIARDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_DIR = "checkpoints"

# Load models
transformer_model = TransformerClassifier().to(DEVICE)
transformer_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/transformer_model.pt", map_location=DEVICE))
transformer_model.eval()

gnn_model = GNNClassifier(in_channels=10, hidden_channels=32).to(DEVICE)
gnn_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/gnn_model.pt", map_location=DEVICE))
gnn_model.eval()

ensemble_model = EnsembleClassifier().to(DEVICE)
ensemble_model.load_state_dict(torch.load(f"{CHECKPOINT_DIR}/ensemble_model.pt", map_location=DEVICE))
ensemble_model.eval()

# Dummy graph
edge_index = torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,5,3,5,4]], dtype=torch.long)
gnn_x = torch.randn((6, 10), dtype=torch.float)
gnn_logits = gnn_model(gnn_x.to(DEVICE), edge_index.to(DEVICE))
gnn_logits_articles = gnn_logits[:3]

# Get one batch from validation set
val_dataset = LIARDataset("data/valid.tsv")
val_loader = DataLoader(val_dataset, batch_size=3)

for batch in val_loader:
    input_ids = batch["input_ids"].to(DEVICE)
    attention_mask = batch["attention_mask"].to(DEVICE)
    labels = batch["label"].to(DEVICE)

    with torch.no_grad():
        transformer_logits = transformer_model(input_ids, attention_mask)
        final_logits = ensemble_model(transformer_logits, gnn_logits_articles)
        predictions = torch.argmax(final_logits, dim=1)

    print("\n✅ Predictions:", predictions.tolist())
    print("📌 Ground Truth:", labels.tolist())
    break
