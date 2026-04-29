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
def get_ddpm_schedule(timesteps=1000, s=0.008):
    import math
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999), alphas_cumprod[1:]


def sample_diffusion(diffusion_model, shape, classes, device, timesteps=1000, w=1.5, num_classes=6):
    """
    Standard DDPM ancestral sampler with Classifier-Free Guidance.
    """
    diffusion_model.eval()
    betas, alphas_cumprod = get_ddpm_schedule(timesteps)
    betas = betas.to(device)
    alphas = (1.0 - betas).to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    B, L, D = shape
    x = torch.randn(shape, device=device)

    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)), desc="Reverse Denoising", total=timesteps):
            t_tensor = torch.full((B,), i, device=device, dtype=torch.long)
            full_mask = torch.ones((B, L), device=device, dtype=torch.bool)

            # CFG: Double batch (cond and uncond)
            null_classes = torch.full_like(classes, num_classes)
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            c_double = torch.cat([classes, null_classes], dim=0)
            m_double = torch.cat([full_mask, full_mask], dim=0)
            
            pred_noise_double = diffusion_model(x_double, t_double, c_double, mask=m_double)
            pred_cond, pred_uncond = pred_noise_double.chunk(2, dim=0)
            
            # Extrapolate away from unconditional prediction
            pred_noise = pred_uncond + w * (pred_cond - pred_uncond)

            # Standard DDPM reverse step
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
    L = syn_patches.shape[0]
    ppax = L // 3  # patches per axis

    syn_x = syn_patches[:ppax].reshape(-1)
    syn_y = syn_patches[ppax: 2 * ppax].reshape(-1)
    syn_z = syn_patches[2 * ppax:].reshape(-1)

    # Normalise real window to [0, 1] per axis so scales are comparable
    def norm01(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-10)

    real_x = norm01(real_window[:, 0])
    real_y = norm01(real_window[:, 1])
    real_z = norm01(real_window[:, 2])

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=False)
    fig.suptitle(
        f"IMU Waveform Comparison — Activity Class {class_id}\n"
        "(real normalised to [0,1]; synthetic = decoder ME patch output)",
        fontsize=13
    )

    for ax, real_sig, syn_sig, label in zip(
        axes,
        [real_x, real_y, real_z],
        [syn_x,  syn_y,  syn_z],
        ["X-Accel", "Y-Accel", "Z-Accel"],
    ):
        ax.plot(real_sig, label=f"Real {label} (norm.)",
                color="royalblue", alpha=0.85, linewidth=0.9)
        # Align synthetic to same time axis by resampling index
        syn_idx = np.linspace(0, len(syn_sig) - 1, len(real_sig))
        syn_resampled = np.interp(syn_idx, np.arange(len(syn_sig)), syn_sig)
        ax.plot(syn_resampled, label=f"Synthetic {label} (norm.)",
                color="tomato", alpha=0.85, linewidth=0.9)
        ax.set_ylabel("Norm. amplitude")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Time Step (30 Hz samples)")
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
        seq_len=L, token_dim=64, num_classes=num_classes + 1
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
        diffusion_model, (args.gen_n, L, 64), gen_c, args.device, w=1.5, num_classes=num_classes)

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
        if len(real_idx) == 0:
            continue

        real_win = raw_windows[np.random.choice(real_idx)]  # (T, 3) physical
        syn_pats = syn_patches_np[idx]                       # (L, 32) decoded

        plot_waveforms(
            real_win, syn_pats,
            class_id=c_id,
            save_path=f"plots/waveform_cmp_demo_{plots_saved}.png"
        )
        used_classes.add(c_id)
        plots_saved += 1

    wandb.init(project="BioPM-Diffusion")
    print("Uploading plots to WandB...")
    tsne_fname = f"plots/tsne_latent_distribution_{timestamp}.png"
    wandb.log({"Latent/t-SNE": wandb.Image(tsne_fname)})
    wf_logs = {}
    for i in range(plots_saved):
        wf_logs[f"Waveform/class_cmp_{i}"] = wandb.Image(
            f"plots/waveform_cmp_demo_{i}.png")
    wandb.log(wf_logs)
    wandb.finish()

    print("Evaluation complete! Plots saved to /plots")


if __name__ == "__main__":
    main()
