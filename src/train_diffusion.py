import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models_diffusion import TokenTransformerDiffusion, IMUDecoder

def get_ddpm_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="features/biopm_tokens.npz")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="BioPM-Diffusion")

    print(f"Loading continuous tokens from {args.data}...")
    npz = np.load(args.data)
    tokens = torch.tensor(npz['tokens'], dtype=torch.float32)
    masks = torch.tensor(npz['masks'], dtype=torch.bool)
    labels = torch.tensor(npz['labels'], dtype=torch.long)
    raw_patches = torch.tensor(npz['raw_patches'], dtype=torch.float32)

    B, L = tokens.shape[:2]
    num_classes = len(torch.unique(labels))
    patch_dim = np.prod(raw_patches.shape[2:])
    raw_patches = raw_patches.view(B, L, -1)

    dataset = TensorDataset(tokens, masks, labels, raw_patches)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    diffusion_model = TokenTransformerDiffusion(seq_len=L, token_dim=64, num_classes=num_classes).to(args.device)
    decoder_model = IMUDecoder(token_dim=64, patch_dim=patch_dim).to(args.device)

    opt = optim.Adam(list(diffusion_model.parameters()) + list(decoder_model.parameters()), lr=args.lr)
    alphas_cumprod = get_ddpm_schedule().to(args.device)
    timesteps = len(alphas_cumprod)

    print(f"Starting training on {args.device.upper()} for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        diffusion_model.train()
        decoder_model.train()
        
        epoch_diff_loss, epoch_dec_loss = 0.0, 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:02d}/{args.epochs}")
        for b_tokens, b_masks, b_labels, b_raw in pbar:
            b_tokens, b_masks = b_tokens.to(args.device), b_masks.to(args.device)
            b_labels, b_raw = b_labels.to(args.device), b_raw.to(args.device)
            
            # 1. IMU Decoder Step: Reconstruct physical sequences
            pred_raw = decoder_model(b_tokens)
            dec_loss = nn.functional.huber_loss(pred_raw[b_masks], b_raw[b_masks])
            
            # 2. Diffusion Step: Predict added Noise
            t = torch.randint(0, timesteps, (b_tokens.shape[0],), device=args.device).long()
            noise = torch.randn_like(b_tokens)
            a_cp = alphas_cumprod[t].view(-1, 1, 1)
            noisy_tokens = torch.sqrt(a_cp) * b_tokens + torch.sqrt(1 - a_cp) * noise
            
            pred_noise = diffusion_model(noisy_tokens, t, b_labels, b_masks)
            diff_loss = nn.functional.mse_loss(pred_noise[b_masks], noise[b_masks])
            
            # Optimize together
            loss = diff_loss + dec_loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            opt.step()
            
            epoch_diff_loss += diff_loss.item()
            epoch_dec_loss += dec_loss.item()
            pbar.set_postfix({"DiffLoss": f"{diff_loss.item():.4f}", "DecLoss": f"{dec_loss.item():.4f}"})
        
        if args.wandb:
            wandb.log({"diffusion_loss": epoch_diff_loss/len(dataloader), "decoder_loss": epoch_dec_loss/len(dataloader)})
            
    # Safely save off the resulting Models
    os.makedirs("checkpoints/diffusion", exist_ok=True)
    torch.save(diffusion_model.state_dict(), "checkpoints/diffusion/token_diff.pt")
    torch.save(decoder_model.state_dict(), "checkpoints/diffusion/imu_decoder.pt")
    print("Training finished! Models successfully saved to checkpoints/diffusion/")

if __name__ == "__main__":
    main()
