import os
import copy
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models_diffusion import TokenTransformerDiffusion

# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) — used for higher-quality sampling
# ---------------------------------------------------------------------------
class EMA:
    """
    Maintains a shadow copy of the model's parameters that is updated as a
    running average. At inference time, the shadow weights produce smoother
    and better-quality samples than the instantaneous training weights.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        for shadow_p, model_p in zip(self.shadow.parameters(), model.parameters()):
            shadow_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


def get_ddpm_schedule(timesteps=1000, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999), alphas_cumprod[1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="features/biopm_tokens.npz")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--diff_lr", type=float, default=2e-4,
                        help="Learning rate for the diffusion transformer")
    parser.add_argument("--ema_decay", type=float, default=0.9999)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="BioPM-Diffusion-Standard", config=vars(args))

    print(f"Loading continuous tokens from {args.data}...")
    npz = np.load(args.data)
    tokens = torch.tensor(npz['tokens'], dtype=torch.float32)
    masks = torch.tensor(npz['masks'], dtype=torch.bool)
    labels = torch.tensor(npz['labels'], dtype=torch.long)
    raw_patches = torch.tensor(npz['raw_patches'], dtype=torch.float32)

    B, L = tokens.shape[:2]
    num_classes = int(labels.max().item()) + 1
    
    # Normalization: Diffusion mathematically requires zero-mean unit-variance
    valid_tokens = tokens[masks]
    token_mean = valid_tokens.mean(dim=0)
    token_std = valid_tokens.std(dim=0)
    
    # Avoid division by zero for any flat dimensions
    token_std[token_std < 1e-6] = 1.0
    
    # Normalize all tokens
    tokens = (tokens - token_mean) / token_std
    
    os.makedirs("checkpoints/diffusion", exist_ok=True)
    torch.save({"mean": token_mean, "std": token_std}, "checkpoints/diffusion/token_scaler.pt")
    print(f"Saved token_scaler.pt. Data normalized.")

    dataset = TensorDataset(tokens, masks, labels, raw_patches)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=4 if args.device == "cuda" else 0, 
                            pin_memory=(args.device == "cuda"),
                            persistent_workers=(args.device == "cuda"))

    # Build model (Standard Conditioning: No +1 for null class)
    diffusion_model = TokenTransformerDiffusion(
        seq_len=L, token_dim=64, num_classes=num_classes
    ).to(args.device)

    # Hardware Optimization: Compile the model if requested
    if args.compile and args.device == "cuda":
        print("Compiling model for faster execution...")
        diffusion_model = torch.compile(diffusion_model)

    # Optimizer & Scheduler
    diff_opt = optim.AdamW(diffusion_model.parameters(), lr=args.diff_lr, weight_decay=1e-4)
    diff_scheduler = optim.lr_scheduler.CosineAnnealingLR(diff_opt, T_max=args.epochs, eta_min=args.diff_lr / 10)

    # EMA wrapper
    ema = EMA(diffusion_model, decay=args.ema_decay)

    # DDPM Schedule Setup
    betas, alphas_cumprod = get_ddpm_schedule()
    betas = betas.to(args.device)
    alphas_cumprod = alphas_cumprod.to(args.device)
    timesteps = len(alphas_cumprod)

    # Hardware Optimization: Setup Automatic Mixed Precision (AMP) Scaler
    scaler = torch.cuda.amp.GradScaler() if args.device == "cuda" else None

    print(f"Starting Standard Conditional training on {args.device.upper()} for {args.epochs} epochs...")
    print(f"  diff_lr={args.diff_lr}  EMA decay={args.ema_decay}")

    for epoch in range(args.epochs):
        diffusion_model.train()
        epoch_diff_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{args.epochs}")
        for b_tokens, b_masks, b_labels, _ in pbar:
            b_tokens = b_tokens.to(args.device, non_blocking=True)
            b_masks  = b_masks.to(args.device, non_blocking=True)
            b_labels = b_labels.to(args.device, non_blocking=True)

            # ---- Diffusion step ----
            t = torch.randint(0, timesteps, (b_tokens.shape[0],), device=args.device).long()
            noise = torch.randn_like(b_tokens)
            a_cp = alphas_cumprod[t].view(-1, 1, 1)
            noisy_tokens = torch.sqrt(a_cp) * b_tokens + torch.sqrt(1 - a_cp) * noise

            diff_opt.zero_grad(set_to_none=True)

            # Hardware Optimization: Automatic Mixed Precision Forward Pass
            with torch.autocast(device_type=args.device, dtype=torch.float16, enabled=(args.device == "cuda")):
                # Standard Conditioning: Pass exact b_labels 100% of the time
                pred_noise = diffusion_model(noisy_tokens, t, b_labels, b_masks)
                diff_loss = nn.functional.mse_loss(pred_noise[b_masks], noise[b_masks])

            # Hardware Optimization: AMP Backward Pass
            if scaler:
                scaler.scale(diff_loss).backward()
                scaler.unscale_(diff_opt)
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
                scaler.step(diff_opt)
                scaler.update()
            else:
                diff_loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
                diff_opt.step()

            # Update EMA shadow weights
            ema.update(diffusion_model)

            epoch_diff_loss += diff_loss.item()
            pbar.set_postfix({"DiffLoss": f"{diff_loss.item():.4f}"})

        diff_scheduler.step()

        if args.wandb:
            wandb.log({
                "diffusion_loss":  epoch_diff_loss / len(dataloader),
                "diff_lr":         diff_scheduler.get_last_lr()[0],
                "epoch":           epoch + 1,
            })

    # Save logic
    os.makedirs("checkpoints/diffusion", exist_ok=True)
    
    # Extract original model state if it was compiled
    model_to_save = diffusion_model._orig_mod if args.compile and args.device=="cuda" else diffusion_model
    
    torch.save(model_to_save.state_dict(), "checkpoints/diffusion/token_diff.pt")
    torch.save(ema.state_dict(), "checkpoints/diffusion/token_diff_ema.pt")
    
    print("Training finished! Models saved to checkpoints/diffusion/")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
