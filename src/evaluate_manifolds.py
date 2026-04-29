import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
import time
from models_diffusion import TokenTransformerDiffusion

def mmd_rbf(X, Y, gamma=1.0):
    XX = rbf_kernel(X, X, gamma)
    YY = rbf_kernel(Y, Y, gamma)
    XY = rbf_kernel(X, Y, gamma)
    return XX.mean() + YY.mean() - 2 * XY.mean()

# (Include the get_ddpm_schedule and sample_diffusion functions from above here)
def get_ddpm_schedule(timesteps=1000, s=0.008):
    import math
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999), alphas_cumprod[1:]

def sample_diffusion(diffusion_model, shape, classes, device, timesteps=1000, w=1.5, num_classes=6):
    diffusion_model.eval()
    betas, alphas_cumprod = get_ddpm_schedule(timesteps)
    betas = betas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)
    b, L, d = shape
    x = torch.randn(shape, device=device)
    with torch.no_grad():
        for i in reversed(range(timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            null_classes = torch.full_like(classes, num_classes)
            x_double = torch.cat([x, x], dim=0)
            t_double = torch.cat([t_tensor, t_tensor], dim=0)
            c_double = torch.cat([classes, null_classes], dim=0)
            m_double = torch.ones((2*b, L), device=device, dtype=torch.bool)
            
            pred_noise_double = diffusion_model(x_double, t_double, c_double, mask=m_double)
            pred_cond, pred_uncond = pred_noise_double.chunk(2, dim=0)
            pred_noise = pred_uncond + w * (pred_cond - pred_uncond)
            
            alpha_t, alpha_t_prev = alphas_cumprod[i], (alphas_cumprod[i-1] if i > 0 else torch.tensor(1.0, device=device))
            beta_t = 1 - (alpha_t / alpha_t_prev)
            x_mean = (1 / torch.sqrt(1 - beta_t)) * (x - (beta_t / torch.sqrt(1 - alpha_t)) * pred_noise)
            x = x_mean + (torch.sqrt(beta_t) * torch.randn_like(x) if i > 0 else 0)
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="features/biopm_tokens.npz")
    parser.add_argument("--diff_ckpt", type=str, default="checkpoints/diffusion/token_diff_ema.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--gen_n", type=int, default=1000)
    args = parser.parse_args()

    npz = np.load(args.data)
    real_tokens, real_labels = npz['tokens'], npz['labels']
    B, L, _ = real_tokens.shape
    num_classes = len(np.unique(real_labels))

    model = TokenTransformerDiffusion(seq_len=L, token_dim=64, num_classes=num_classes + 1).to(args.device)
    model.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device, weights_only=True))

    gen_c = torch.randint(0, num_classes, (args.gen_n,), device=args.device)
    syn_tokens = sample_diffusion(model, (args.gen_n, L, 64), gen_c, args.device, w=1.5, num_classes=num_classes)
    
    real_features = real_tokens.mean(axis=1)
    syn_features = syn_tokens.cpu().numpy().mean(axis=1)

    # Balance for accurate manifold plots
    idx = np.random.choice(len(real_features), args.gen_n, replace=False)
    real_balanced = real_features[idx]

    # MMD Calculation
    gamma = 1.0 / real_balanced.shape[1]
    mmd_score = mmd_rbf(real_balanced, syn_features, gamma)
    print(f"Maximum Mean Discrepancy (MMD) Score: {mmd_score:.5f}")

    combined_X = np.vstack([real_balanced, syn_features])
    labels = np.array(['Real'] * args.gen_n + ['Synthetic'] * args.gen_n)

    # PCA and t-SNE 
    pca_emb = PCA(n_components=2).fit_transform(combined_X)
    tsne_emb = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(combined_X)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(x=pca_emb[:,0], y=pca_emb[:,1], hue=labels, palette={'Real':'blue', 'Synthetic':'red'}, alpha=0.5, s=15, ax=axes[0])
    axes[0].set_title("PCA: Real vs Synthetic")
    
    sns.scatterplot(x=tsne_emb[:,0], y=tsne_emb[:,1], hue=labels, palette={'Real':'blue', 'Synthetic':'red'}, alpha=0.5, s=15, ax=axes[1])
    axes[1].set_title("t-SNE: Real vs Synthetic")
    
    os.makedirs("plots", exist_ok=True)
    stamp = int(time.time())
    plt.savefig(f"plots/manifold_projections_{stamp}.png", dpi=200)
    plt.close()

    # Feature KDE Distributions
    variances = np.var(real_balanced, axis=0)
    top_features = np.argsort(variances)[-4:]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, feat_idx in zip(axes.flatten(), top_features):
        sns.kdeplot(real_balanced[:, feat_idx], fill=True, color="blue", label="Real", ax=ax, alpha=0.3)
        sns.kdeplot(syn_features[:, feat_idx], fill=True, color="red", label="Synthetic", ax=ax, alpha=0.3)
        ax.set_title(f'Feature {feat_idx} Distribution')
        ax.legend()
    plt.savefig(f"plots/kde_distributions_{stamp}.png", dpi=200)

if __name__ == "__main__":
    main()
