import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models_diffusion import WaveformDecoder

def temporal_smoothness_loss(pred):
    """
    Penalizes large step-to-step jumps to encourage physically smooth signals.
    pred shape: (B, T, 3)
    """
    diff = pred[:, 1:, :] - pred[:, :-1, :]
    return torch.mean(diff ** 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="features/biopm_tokens.npz")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--smoothness_weight", type=float, default=0.005,
                        help="Weight for the temporal smoothness penalty")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="BioPM-Diffusion-Decoder", config=vars(args))

    print(f"Loading data from {args.data}...")
    npz = np.load(args.data)
    tokens = torch.tensor(npz['tokens'], dtype=torch.float32)
    raw_windows = torch.tensor(npz['raw_windows'], dtype=torch.float32)
    
    # Check sequence lengths
    B, L, _ = tokens.shape
    _, T, _ = raw_windows.shape
    print(f"Tokens: {tokens.shape}, Raw Windows: {raw_windows.shape}")

    dataset = TensorDataset(tokens, raw_windows)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=(args.device == "cuda"))

    model = WaveformDecoder(token_dim=64, hidden_dim=256, out_channels=3, target_length=T).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.MSELoss()

    print(f"Starting Waveform Decoder training on {args.device.upper()} for {args.epochs} epochs...")

    for epoch in range(args.epochs):
        model.train()
        epoch_mse = 0.0
        epoch_smooth = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{args.epochs}")
        for b_tokens, b_raw in pbar:
            b_tokens = b_tokens.to(args.device)
            b_raw = b_raw.to(args.device)

            optimizer.zero_grad()
            
            pred_raw = model(b_tokens)
            
            # Reconstruction Loss
            mse_loss = criterion(pred_raw, b_raw)
            
            # Physics-informed smoothness loss
            smooth_loss = temporal_smoothness_loss(pred_raw)
            
            loss = mse_loss + args.smoothness_weight * smooth_loss
            
            loss.backward()
            optimizer.step()

            epoch_mse += mse_loss.item()
            epoch_smooth += smooth_loss.item()
            
            pbar.set_postfix({
                "MSE": f"{mse_loss.item():.4f}",
                "Smooth": f"{smooth_loss.item():.4f}"
            })

        scheduler.step()

        if args.wandb:
            wandb.log({
                "mse_loss": epoch_mse / len(dataloader),
                "smoothness_loss": epoch_smooth / len(dataloader),
                "total_loss": (epoch_mse + args.smoothness_weight * epoch_smooth) / len(dataloader),
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch + 1
            })

    os.makedirs("checkpoints/diffusion", exist_ok=True)
    save_path = "checkpoints/diffusion/waveform_decoder.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Training finished! Model saved to {save_path}")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
