import argparse
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from models_diffusion import TokenTransformerDiffusion

WISDM_LABELS = {0: 'Downstairs', 1: 'Jogging', 2: 'Sitting', 3: 'Standing', 4: 'Upstairs', 5: 'Walking'}

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
    args = parser.parse_args()

    npz = np.load(args.data)
    real_tokens, real_labels, pids = npz['tokens'], npz['labels'], npz['pids']
    B, L, _ = real_tokens.shape
    num_classes = len(np.unique(real_labels))

    real_features = real_tokens.mean(axis=1) 
    
    # SUBJECT-INDEPENDENT SPLIT (Addressing Professor's Feedback)
    unique_pids = np.unique(pids)
    np.random.seed(42)
    np.random.shuffle(unique_pids)
    split_idx = int(len(unique_pids) * 0.8)
    train_pids, test_pids = unique_pids[:split_idx], unique_pids[split_idx:]
    
    train_mask = np.isin(pids, train_pids)
    test_mask = np.isin(pids, test_pids)
    
    X_train_full, y_train_full = real_features[train_mask], real_labels[train_mask]
    X_test, y_test = real_features[test_mask], real_labels[test_mask]
    
    # Simulate Class Starvation in Training Set
    rare_class = 0
    rare_idx = np.where(y_train_full == rare_class)[0]
    keep_rare = np.random.choice(rare_idx, size=int(len(rare_idx)*0.05), replace=False)
    other_idx = np.where(y_train_full != rare_class)[0]
    imbalanced_idx = np.concatenate([keep_rare, other_idx])
    
    X_train_imb, y_train_imb = X_train_full[imbalanced_idx], y_train_full[imbalanced_idx]
    
    # Baseline Model
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_base.fit(X_train_imb, y_train_imb)
    f1_base = f1_score(y_test, clf_base.predict(X_test), average='macro')
    
    # Diffusion Repair
    model = TokenTransformerDiffusion(seq_len=L, token_dim=64, num_classes=num_classes + 1).to(args.device)
    model.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device, weights_only=True))
    
    gen_n = len(rare_idx) - len(keep_rare) 
    gen_c = torch.full((gen_n,), rare_class, device=args.device, dtype=torch.long)
    syn_tokens = sample_diffusion(model, (gen_n, L, 64), gen_c, args.device, w=1.5, num_classes=num_classes)
    syn_features = syn_tokens.cpu().numpy().mean(axis=1)
    
    X_train_repaired = np.concatenate([X_train_imb, syn_features])
    y_train_repaired = np.concatenate([y_train_imb, gen_c.cpu().numpy()])
    
    clf_repair = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_repair.fit(X_train_repaired, y_train_repaired)
    preds_repair = clf_repair.predict(X_test)
    f1_repair = f1_score(y_test, preds_repair, average='macro')

    print(f"\nSubject-Independent Repaired F1 Macro Score:  {f1_repair:.4f} (Baseline: {f1_base:.4f})")
    
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, preds, title, cmap in zip([0, 1], [clf_base.predict(X_test), preds_repair], [f"Baseline (Starved)", "Repaired (Synthetic)"], ['Blues', 'Greens']):
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[WISDM_LABELS.get(i, str(i)) for i in range(num_classes)]).plot(ax=ax[ax_idx], cmap=cmap, colorbar=False)
        ax[ax_idx].set_title(title)
        
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/subject_independent_confusion_{int(time.time())}.png", dpi=200)
    plt.close()

if __name__ == "__main__":
    main()
