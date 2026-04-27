import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from models_diffusion import TokenTransformerDiffusion, IMUDecoder

def get_ddpm_schedule(timesteps=1000):
    alphas = 1.0 - torch.linspace(1e-4, 0.02, timesteps)
    return torch.cumprod(alphas, dim=0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100) # Give it 100 on your GPU!
    args = parser.parse_args()

    npz = np.load("features/biopm_tokens.npz")
    tokens, masks, labels, raw = map(lambda x: torch.tensor(npz[x]), ['tokens', 'masks', 'labels', 'raw_patches'])
    
    num_classes = len(torch.unique(labels))
    NULL_CLASS = num_classes # The Drop-Label used for CFG
    
    loader = DataLoader(TensorDataset(tokens.float(), masks.bool(), labels.long(), raw.view(tokens.shape[0], tokens.shape[1], -1).float()), batch_size=64, shuffle=True)

    # Note: We provide num_classes + 1 to the architecture so it reserves an embedding for <NULL>
    diffusion = TokenTransformerDiffusion(seq_len=tokens.shape[1], num_classes=num_classes + 1).to(args.device)
    decoder = IMUDecoder(patch_dim=np.prod(raw.shape[2:])).to(args.device)

    opt = optim.Adam(list(diffusion.parameters()) + list(decoder.parameters()), lr=1e-3)
    alphas_cp = get_ddpm_schedule(1000).to(args.device)

    for epoch in range(args.epochs):
        diffusion.train()
        
        pbar = tqdm(loader, desc=f"CFG Epoch {epoch+1}")
        for b_tok, b_mask, b_lab, b_raw in pbar:
            b_tok, b_mask, b_lab, b_raw = b_tok.to(args.device), b_mask.to(args.device), b_lab.to(args.device), b_raw.to(args.device)
            
            dec_loss = nn.functional.huber_loss(decoder(b_tok)[b_mask], b_raw[b_mask])
            
            # Classifier-Free Guidance Random Probability Drop (10% of the time, drop the label to NULL)
            mask_drop = torch.rand(b_lab.shape, device=args.device) < 0.10
            cfg_labels = torch.where(mask_drop, torch.full_like(b_lab, NULL_CLASS), b_lab)
            
            t = torch.randint(0, 1000, (b_tok.shape[0],), device=args.device).long()
            noise = torch.randn_like(b_tok)
            a = alphas_cp[t].view(-1, 1, 1)
            noisy_tokens = torch.sqrt(a) * b_tok + torch.sqrt(1 - a) * noise
            
            pred_noise = diffusion(noisy_tokens, t, cfg_labels, b_mask)
            diff_loss = nn.functional.mse_loss(pred_noise[b_mask], noise[b_mask])
            
            loss = diff_loss + dec_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            
    os.makedirs("checkpoints/cfg", exist_ok=True)
    torch.save(diffusion.state_dict(), "checkpoints/cfg/diff_cfg.pt")
    torch.save(decoder.state_dict(), "checkpoints/cfg/dec_cfg.pt")

if __name__ == "__main__":
    main()
