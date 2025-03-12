import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

# Load Dataset (Example: Wikipedia)
dataset = load_dataset("wikipedia", "20220301.simple", split="train")

# Define Model
class StelAIModel(nn.Module):
    def __init__(self):
        super(StelAIModel, self).__init__()
        self.encoder = AutoModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(768, 1)  # Example output layer
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return self.fc(outputs.last_hidden_state[:, 0, :])  

# Initialize Model
model = StelAIModel()
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
loss_fn = nn.MSELoss()

# Training Loop
def train():
    model.train()
    for epoch in range(2):  # Train for 2 epochs
        for batch in dataset.shuffle().select(range(1000)):  # Example: 1000 samples
            text = batch['text']
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, torch.tensor([[0.0]]))  # Dummy target
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1} completed.")

# Run Training
if __name__ == "__main__":
    train()