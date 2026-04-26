#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import os

class BioPMGenerator(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.d_model = d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, d_model) # Projects back to 64-d token
        
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        # src shape: (Batch, Seq_Len, 64)
        seq_len = src.size(1)
        mask = self.generate_square_subsequent_mask(seq_len).to(src.device)
        out = self.transformer(src, mask=mask)
        return self.fc_out(out)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=str, default="features/biopm_tokens.npz")
    p.add_argument("--output", type=str, default="checkpoints/generator.pt")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Data
    data = np.load(args.tokens)
    tokens = data['tokens'] # Shape: (N, L, 64)
    
    # Create input (t) and target (t+1) sequences
    inputs = torch.tensor(tokens[:, :-1, :], dtype=torch.float32)
    targets = torch.tensor(tokens[:, 1:, :], dtype=torch.float32)
    
    dataset = TensorDataset(inputs, targets)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    # 2. Initialize Model
    model = BioPMGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 3. Train Loop
    print(f"Training on {device}...")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_inputs, batch_targets in dataloader:
            batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{args.epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        
    # 4. Save Model
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

if __name__ == "__main__":
    main()