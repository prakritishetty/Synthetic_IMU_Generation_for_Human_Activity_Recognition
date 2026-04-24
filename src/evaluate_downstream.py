import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import wandb

from models_diffusion import TokenTransformerDiffusion, IMUDecoder

def get_ddpm_schedule(timesteps=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, timesteps)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)

def sample_diffusion(diffusion_model, shape, classes, device, timesteps=1000):
    diffusion_model.eval()
    alphas_cumprod = get_ddpm_schedule(timesteps).to(device)
    
    b, L, d = shape
    # Start with pure random Gaussian Noise
    x = torch.randn(shape, device=device)
    
    with torch.no_grad():
        for i in tqdm(reversed(range(timesteps)), desc="Reverse Denoising", total=timesteps):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            
            # Predict noise using the UNet/Transformer
            pred_noise = diffusion_model(x, t_tensor, classes, mask=torch.ones((b, L), device=device, dtype=torch.bool))
            
            alpha_t = alphas_cumprod[i]
            alpha_t_prev = alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0, device=device)
            beta_t = 1 - (alpha_t / alpha_t_prev)
            
            x_mean = (1 / torch.sqrt(1 - beta_t)) * (x - (beta_t / torch.sqrt(1 - alpha_t)) * pred_noise)
            
            if i > 0:
                noise = torch.randn_like(x)
                x = x_mean + torch.sqrt(beta_t) * noise
            else:
                x = x_mean
    return x

def plot_waveforms(real_raw, synthetic_raw, class_id, save_path):
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
    parser.add_argument("--data", type=str, default="features/biopm_tokens.npz")
    parser.add_argument("--diff_ckpt", type=str, default="checkpoints/diffusion/token_diff.pt")
    parser.add_argument("--dec_ckpt", type=str, default="checkpoints/diffusion/imu_decoder.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Hardware Detected: {args.device.upper()}")
    
    npz = np.load(args.data)
    real_tokens, real_labels, raw_patches = npz['tokens'], npz['labels'], npz['raw_patches']
    B, L, _ = real_tokens.shape
    patch_dim = np.prod(raw_patches.shape[2:])

    num_classes = len(np.unique(real_labels))
    diffusion_model = TokenTransformerDiffusion(seq_len=L, token_dim=64, num_classes=num_classes).to(args.device)
    decoder_model = IMUDecoder(token_dim=64, patch_dim=patch_dim).to(args.device)
    
    diffusion_model.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device, weights_only=True))
    decoder_model.load_state_dict(torch.load(args.dec_ckpt, map_location=args.device, weights_only=True))
    
    # 1. Hallucination Target: Generate 200 Novel sequences
    print("Generating pure Synthetic Sequence Hallucinations...")
    gen_n = 200
    gen_c = torch.randint(0, num_classes, (gen_n,), device=args.device)
    syn_tokens = sample_diffusion(diffusion_model, (gen_n, L, 64), gen_c, args.device)
    
    # 2. Decode back to the Physical Earth Dimensions
    with torch.no_grad():
        syn_raw = decoder_model(syn_tokens)

    syn_tokens_np = syn_tokens.cpu().numpy()
    syn_raw_np = syn_raw.cpu().numpy()
    
    # 3. Create Graphical Visualizations
    from sklearn.manifold import TSNE
    print("Computing t-SNE Plot Distributions (This might take a second)...")
    X = np.concatenate([real_tokens, syn_tokens_np], axis=0).mean(axis=1) # Flatten Sequence Temporal Average
    X_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X)
    
    plt.figure(figsize=(9, 7))
    plt.scatter(X_emb[:B, 0], X_emb[:B, 1], c='blue', label='Real Biological Tokens', alpha=0.3, s=10)
    plt.scatter(X_emb[B:, 0], X_emb[B:, 1], c='red', label='Artificial Diffusion Tokens', alpha=0.8, s=25)
    plt.legend()
    plt.title("Latent Space Distribution: Real vs. Synthetic Generations")
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/tsne_latent_distribution.png", dpi=200)
    plt.close()
    
    for i in range(3): # Plot 3 random classes
        idx = np.random.randint(gen_n)
        c_id = gen_c[idx].item()
        
        real_idx = np.where(real_labels == c_id)[0]
        if len(real_idx) > 0:
            real_samp = raw_patches[np.random.choice(real_idx)].reshape(-1)
            plot_waveforms(real_samp, syn_raw_np[idx], class_id=c_id, save_path=f"plots/waveform_cmp_demo_{i}.png")
    
    
    wandb.init(project="BioPM-Diffusion")
    
    print("Uploading plots to WandB...")
    
    # Send your t-SNE scatter plot
    wandb.log({"Latent Check/t-SNE Distribution": wandb.Image("plots/tsne_latent_distribution.png")})
    
    # Send your 3 physical Waveform graphs
    wandb.log({
        "Physical/Waveform_1": wandb.Image("plots/waveform_cmp_demo_0.png"),
        "Physical/Waveform_2": wandb.Image("plots/waveform_cmp_demo_1.png"),
        "Physical/Waveform_3": wandb.Image("plots/waveform_cmp_demo_2.png")
    })
    
    wandb.finish()

            
    print("Evaluation Suite Successful! Qualitative plots saved to the /plots folder.")

if __name__ == "__main__":
    main()
