import torch
from transformers import BertTokenizer
from models.transformer_model import TransformerClassifier

# === Setup ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "checkpoints/transformer_model.pt"
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# === Load Model ===
model = TransformerClassifier()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Predict Function ===
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        predicted_class = torch.argmax(logits, dim=1).item()
        return "REAL" if predicted_class == 1 else "FAKE"

# === Try Example ===
if __name__ == "__main__":
    example = input("📰 Enter news article: ")
    prediction = predict(example)
    print(f"🧠 Prediction: {prediction}")
