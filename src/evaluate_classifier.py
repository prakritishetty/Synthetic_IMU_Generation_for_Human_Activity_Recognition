import argparse
import numpy as np
import torch
import time
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from models_diffusion import TokenTransformerDiffusion

# Human-readable labels mapped alphabetically from WISDM
WISDM_LABELS = {
    0: 'Downstairs',
    1: 'Jogging',
    2: 'Sitting',
    3: 'Standing',
    4: 'Upstairs',
    5: 'Walking'
}

def get_ddpm_schedule(timesteps=1000):
    alphas = 1.0 - torch.linspace(1e-4, 0.02, timesteps)
    return torch.cumprod(alphas, dim=0)

def sample_diffusion(diffusion_model, shape, classes, device, timesteps=1000, w=1.5, num_classes=6):
    diffusion_model.eval()
    alphas_cumprod = get_ddpm_schedule(timesteps).to(device)
    b, L, d = shape
    x = torch.randn(shape, device=device)
    with torch.no_grad():
        for i in reversed(range(timesteps)):
            t_tensor = torch.full((b,), i, device=device, dtype=torch.long)
            
            # CFG
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

def plot_error_waveform(raw_patch, true_label_str, pred_label_str, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    seq = raw_patch.reshape(-1, 3) # Reconstruct back to (Time, 3 Axes)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(f"Misclassification Analysis\nGround Truth: {true_label_str} | AI Guessed: {pred_label_str}", fontsize=14, color='darkred')
    
    axes[0].plot(seq[:, 0], color='blue')
    axes[0].set_ylabel("X-Accel (g)")
    axes[1].plot(seq[:, 1], color='orange')
    axes[1].set_ylabel("Y-Accel (g)")
    axes[2].plot(seq[:, 2], color='green')
    axes[2].set_ylabel("Z-Accel (g)")
    axes[2].set_xlabel("Time Step (30Hz Samples)")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="features/biopm_tokens.npz")
    parser.add_argument("--diff_ckpt", type=str, default="checkpoints/diffusion/token_diff.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    npz = np.load(args.data)
    real_tokens, real_labels, raw_patches = npz['tokens'], npz['labels'], npz['raw_patches']
    B, L, _ = real_tokens.shape
    num_classes = len(np.unique(real_labels))

    # Average tokens temporally
    real_features = real_tokens.mean(axis=1) 
    
    # Obliterate 95% of 'Downstairs' (Class 0)
    rare_class = 0
    np.random.seed(42)
    rare_idx = np.where(real_labels == rare_class)[0]
    keep_rare = np.random.choice(rare_idx, size=int(len(rare_idx)*0.05), replace=False)
    other_idx = np.where(real_labels != rare_class)[0]
    imbalanced_idx = np.concatenate([keep_rare, other_idx])
    
    X_imb = real_features[imbalanced_idx]
    y_imb = real_labels[imbalanced_idx]
    raw_imb = raw_patches[imbalanced_idx] # Keep the raw physical data tied to the split!
    
    # Split the arrays
    X_train, X_test, y_train, y_test, raw_train, raw_test = train_test_split(
        X_imb, y_imb, raw_imb, test_size=0.2, random_state=42
    )
    
    # 1. Baseline Model
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_base.fit(X_train, y_train)
    preds_base = clf_base.predict(X_test)
    f1_base = f1_score(y_test, preds_base, average='macro')
    
    # 2. Diffusion Repair Model
    model = TokenTransformerDiffusion(seq_len=L, token_dim=64, num_classes=num_classes + 1).to(args.device)
    model.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device, weights_only=True))
    
    gen_n = len(rare_idx) - len(keep_rare) 
    gen_c = torch.full((gen_n,), rare_class, device=args.device, dtype=torch.long)
    syn_tokens = sample_diffusion(model, (gen_n, L, 64), gen_c, args.device, w=1.5, num_classes=num_classes)
    syn_features = syn_tokens.cpu().numpy().mean(axis=1)
    
    X_train_repaired = np.concatenate([X_train, syn_features])
    y_train_repaired = np.concatenate([y_train, gen_c.cpu().numpy()])
    
    clf_repair = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_repair.fit(X_train_repaired, y_train_repaired)
    preds_repair = clf_repair.predict(X_test)
    f1_repair = f1_score(y_test, preds_repair, average='macro')

    # Eval Output
    print("\n" + "="*50 + "\nClass Imbalance Resolution Experiment:")
    print(f"Base F1 Macro Score:      {f1_base:.4f}")
    print(f"Repaired F1 Macro Score:  {f1_repair:.4f}")
    
    # Confusion Matrix
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, preds, title, cmap in zip([0, 1], [preds_base, preds_repair], [f"Baseline (Starved Class {rare_class})", "Repaired (Synthetic Added)"], ['Blues', 'Greens']):
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[WISDM_LABELS.get(i, str(i)) for i in range(num_classes)]).plot(ax=ax[ax_idx], cmap=cmap, colorbar=False, xticks_rotation=45)
        ax[ax_idx].set_title(title)
        
    plt.suptitle("Downstream Activity Classifier Confusion Matrices", fontsize=16)
    plt.tight_layout()
    stamp = int(time.time())
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/confusion_matrix_{stamp}.png", dpi=200)
    plt.close()

    # Detailed Error Analysis & Plotting
    print("\n--- Diagnostic Misclassification Trace ---")
    misclassified_idx = np.where(y_test != preds_repair)[0]
    print(f"Total Test Set Mistakes: {len(misclassified_idx)} out of {len(y_test)}")
    
    os.makedirs(f"plots/errors_{stamp}", exist_ok=True)
    
    for count, i in enumerate(misclassified_idx[:10]): 
        true_label = int(y_test[i])
        pred_label = int(preds_repair[i])
        
        true_str = WISDM_LABELS.get(true_label, "Unknown")
        pred_str = WISDM_LABELS.get(pred_label, "Unknown")
        
        print(f"  [Failure Plot Created] Test Index {i:04d}: Ground Truth = {true_str} | AI Guessed = {pred_str}")
        
        # Save a physical graph of the failure case
        save_path = f"plots/errors_{stamp}/error_{count}_true_{true_str}_pred_{pred_str}.png"
        plot_error_waveform(raw_test[i], true_str, pred_str, save_path)
        
if __name__ == "__main__":
    main()
