import os
import copy
import argparse
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
    running average.  At inference time, the shadow weights produce smoother
    and better-quality samples than the instantaneous training weights.
    """
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # Deep-copy so the shadow lives independently of the live model
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
    import math
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
    parser.add_argument("--cfg_drop_prob", type=float, default=0.1, 
                        help="Probability of dropping class condition for Classifier-Free Guidance")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="BioPM-Diffusion", config=vars(args))

    print(f"Loading continuous tokens from {args.data}...")
    npz = np.load(args.data)
    tokens = torch.tensor(npz['tokens'], dtype=torch.float32)
    masks = torch.tensor(npz['masks'], dtype=torch.bool)
    labels = torch.tensor(npz['labels'], dtype=torch.long)
    raw_patches = torch.tensor(npz['raw_patches'], dtype=torch.float32)

    B, L = tokens.shape[:2]
    num_classes = int(labels.max().item()) + 1
    patch_dim = int(np.prod(raw_patches.shape[2:]))
    raw_patches = raw_patches.view(B, L, -1)

    dataset = TensorDataset(tokens, masks, labels, raw_patches)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=(args.device == "cuda"))

    # Build models
    # We add +1 to num_classes for the "null" class used in Classifier-Free Guidance
    diffusion_model = TokenTransformerDiffusion(
        seq_len=L, token_dim=64, num_classes=num_classes + 1
    ).to(args.device)

    # Optimizer
    diff_opt = optim.AdamW(diffusion_model.parameters(),
                           lr=args.diff_lr, weight_decay=1e-4)

    # Cosine annealing over the full training run
    diff_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        diff_opt, T_max=args.epochs, eta_min=args.diff_lr / 10)

    # EMA wrapper for the diffusion model
    ema = EMA(diffusion_model, decay=args.ema_decay)

    betas, alphas_cumprod = get_ddpm_schedule()
    betas = betas.to(args.device)
    alphas_cumprod = alphas_cumprod.to(args.device)
    timesteps = len(alphas_cumprod)

    print(f"Starting training on {args.device.upper()} for {args.epochs} epochs...")
    print(f"  diff_lr={args.diff_lr}  EMA decay={args.ema_decay}")

    for epoch in range(args.epochs):
        diffusion_model.train()
        epoch_diff_loss = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1:03d}/{args.epochs}")
        for b_tokens, b_masks, b_labels, b_raw in pbar:
            b_tokens = b_tokens.to(args.device)
            b_masks  = b_masks.to(args.device)
            b_labels = b_labels.to(args.device)

            # ---- Diffusion step ----
            t = torch.randint(0, timesteps, (b_tokens.shape[0],),
                              device=args.device).long()
            noise = torch.randn_like(b_tokens)
            a_cp = alphas_cumprod[t].view(-1, 1, 1)
            noisy_tokens = torch.sqrt(a_cp) * b_tokens + torch.sqrt(1 - a_cp) * noise

            # Classifier-Free Guidance: randomly drop the class label
            drop_mask = torch.rand(b_tokens.shape[0], device=args.device) < args.cfg_drop_prob
            cfg_labels = b_labels.clone()
            cfg_labels[drop_mask] = num_classes  # null class index

            pred_noise = diffusion_model(noisy_tokens, t, cfg_labels, b_masks)
            diff_loss = nn.functional.mse_loss(pred_noise[b_masks], noise[b_masks])

            diff_opt.zero_grad()
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
            diff_opt.step()

            # Update EMA shadow weights
            ema.update(diffusion_model)

            epoch_diff_loss += diff_loss.item()
            pbar.set_postfix({
                "DiffLoss": f"{diff_loss.item():.4f}"
            })

        diff_scheduler.step()

        if args.wandb:
            wandb.log({
                "diffusion_loss":  epoch_diff_loss / len(dataloader),
                "diff_lr":         diff_scheduler.get_last_lr()[0],
                "epoch":           epoch + 1,
            })

    # Save both the live model and the EMA shadow
    os.makedirs("checkpoints/diffusion", exist_ok=True)
    torch.save(diffusion_model.state_dict(),
               "checkpoints/diffusion/token_diff.pt")
    torch.save(ema.state_dict(),
               "checkpoints/diffusion/token_diff_ema.pt")
    print("Training finished! Models saved to checkpoints/diffusion/")
    print("  token_diff.pt     — live model weights")
    print("  token_diff_ema.pt — EMA weights (use these for evaluation)")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
