import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import wandb
import time 

from models_diffusion import TokenTransformerDiffusion, IMUDecoder


# ---------------------------------------------------------------------------
# Correct DDPM reverse-process sampler
# ---------------------------------------------------------------------------
def get_ddpm_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas_cumprod


def sample_diffusion(diffusion_model, shape, classes, device, timesteps=1000):
    """
    Standard DDPM ancestral sampler (Ho et al. 2020).

    The key fix vs. the previous version: we precompute `betas` from the
    same linear schedule used during training and use them DIRECTLY in the
    reverse step.  The old code derived beta_t as (alpha_t / alpha_t_prev),
    which is a noisy approximation that compounds errors over 1000 steps and
    pushes samples to the outskirts of the latent space.
    """
    diffusion_model.eval()
    betas, alphas_cumprod = get_ddpm_schedule(timesteps)
    betas = betas.to(device)
    alphas = (1.0 - betas).to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    B, L, D = shape
    x = torch.randn(shape, device=device)

    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)),
                      desc="Reverse Denoising", total=timesteps):
            t_tensor = torch.full((B,), i, device=device, dtype=torch.long)
            full_mask = torch.ones((B, L), device=device, dtype=torch.bool)

            pred_noise = diffusion_model(x, t_tensor, classes, mask=full_mask)

            # Standard DDPM reverse step (Eq. 11 in Ho et al.)
            beta_t    = betas[i]
            alpha_t   = alphas[i]
            alpha_bar = alphas_cumprod[i]

            x_mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1.0 - alpha_bar)) * pred_noise
            )

            if i > 0:
                x = x_mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = x_mean  # no noise at the final step

    return x


# ---------------------------------------------------------------------------
# Waveform comparison plot
# ---------------------------------------------------------------------------
def plot_waveforms(real_window, syn_patches, class_id, save_path):
    """
    real_window  : (T, 3)  physical 3-axis acceleration window (g-force)
    syn_patches  : (L, 32) IMUDecoder output — normalised ME patches

    The BioPM zero-crossing extractor cycles through axes 0 → 1 → 2, so
    patches [0 : L//3] are X-axis MEs, [L//3 : 2L//3] are Y-axis MEs, etc.
    We flatten each group and plot them alongside the corresponding real axis.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    real_seq = real_raw.reshape(-1, 3)
    syn_seq = synthetic_raw.reshape(-1, 3)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(f"Physical IMU Waveform Comparison (Activity Class {class_id})", fontsize=16)
    
    axes[0].plot(real_seq[:, 0], label="Real X-Accel", color="blue", alpha=0.8)
    axes[0].plot(syn_seq[:, 0], label="Synthetic X-Accel", color="red", alpha=0.8)
    axes[0].legend(loc="upper right")
    
    axes[1].plot(real_seq[:, 1], label="Real Y-Accel", color="blue", alpha=0.8)
    axes[1].plot(syn_seq[:, 1], label="Synthetic Y-Accel", color="red", alpha=0.8)
    axes[1].legend(loc="upper right")
    
    axes[2].plot(real_seq[:, 2], label="Real Z-Accel", color="blue", alpha=0.8)
    axes[2].plot(syn_seq[:, 2], label="Synthetic Z-Accel", color="red", alpha=0.8)
    axes[2].legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",      type=str,
                        default="features/biopm_tokens.npz")
    parser.add_argument("--diff_ckpt", type=str,
                        default="checkpoints/diffusion/token_diff_ema.pt",
                        help="Path to diffusion checkpoint (EMA preferred)")
    parser.add_argument("--dec_ckpt",  type=str,
                        default="checkpoints/diffusion/imu_decoder.pt")
    parser.add_argument("--device",    type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen_n",     type=int, default=200,
                        help="Number of synthetic sequences to generate")
    args = parser.parse_args()

    print(f"Hardware Detected: {args.device.upper()}")

    npz = np.load(args.data)
    real_tokens  = npz['tokens']          # (N, L, 64)
    real_labels  = npz['labels']          # (N,)
    raw_patches  = npz['raw_patches']     # (N, L, 32) — normalised ME patches
    # Physical 3-axis windows saved by the updated extract_tokens.py
    raw_windows  = npz['raw_windows']     # (N, T, 3)

    B, L, _ = real_tokens.shape
    patch_dim = int(np.prod(raw_patches.shape[2:]))
    num_classes = int(real_labels.max()) + 1

    diffusion_model = TokenTransformerDiffusion(
        seq_len=L, token_dim=64, num_classes=num_classes
    ).to(args.device)
    decoder_model = IMUDecoder(token_dim=64, patch_dim=patch_dim).to(args.device)

    diffusion_model.load_state_dict(
        torch.load(args.diff_ckpt, map_location=args.device, weights_only=True))
    decoder_model.load_state_dict(
        torch.load(args.dec_ckpt,  map_location=args.device, weights_only=True))

    # 1. Generate synthetic token sequences
    print(f"Generating {args.gen_n} synthetic sequences...")
    gen_c = torch.randint(0, num_classes, (args.gen_n,), device=args.device)
    syn_tokens = sample_diffusion(
        diffusion_model, (args.gen_n, L, 64), gen_c, args.device)

    # 2. Decode to normalised ME patches
    with torch.no_grad():
        syn_patches_all = decoder_model(syn_tokens)  # (gen_n, L, 32)

    syn_tokens_np  = syn_tokens.cpu().numpy()
    syn_patches_np = syn_patches_all.cpu().numpy()

    # 3. t-SNE: real vs synthetic tokens (mean-pooled over sequence)
    print("Computing t-SNE (this may take a moment)...")
    X = np.concatenate([real_tokens, syn_tokens_np], axis=0).mean(axis=1)
    X_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)

    plt.figure(figsize=(9, 7))
    plt.scatter(X_emb[:B, 0],  X_emb[:B, 1],
                c='royalblue', label='Real Biological Tokens',
                alpha=0.3, s=10)
    plt.scatter(X_emb[B:, 0],  X_emb[B:, 1],
                c='tomato',    label='Synthetic Diffusion Tokens',
                alpha=0.8, s=25)
    plt.legend()
    plt.title("Latent Space Distribution: Real vs. Synthetic Generations")
    os.makedirs("plots", exist_ok=True)
    timestamp = int(time.time())
    plt.savefig(f"plots/tsne_latent_distribution_{timestamp}.png", dpi=200)
    # plt.savefig("plots/tsne_latent_distribution.png", dpi=200)
    plt.close()

    # 4. Waveform comparison: real physical window vs synthetic ME patches
    plots_saved = 0
    used_classes = set()
    for idx in np.random.permutation(args.gen_n):
        if plots_saved >= 3:
            break
        c_id = gen_c[idx].item()
        if c_id in used_classes:
            continue

        real_idx = np.where(real_labels == c_id)[0]
        if len(real_idx) > 0:
            real_samp = raw_patches[np.random.choice(real_idx)].reshape(-1)
            plot_waveforms(real_samp, syn_raw_np[idx], class_id=c_id, save_path=f"plots/waveform_cmp_demo_{i}.png")
    
    
    wandb.init(project="BioPM-Diffusion")
    print("Uploading plots to WandB...")
    wandb.log({"Latent/t-SNE": wandb.Image("plots/tsne_latent_distribution.png")})
    wf_logs = {}
    for i in range(plots_saved):
        wf_logs[f"Waveform/class_cmp_{i}"] = wandb.Image(
            f"plots/waveform_cmp_demo_{i}.png")
    wandb.log(wf_logs)
    wandb.finish()

    print("Evaluation complete! Plots saved to /plots")


if __name__ == "__main__":
    main()
