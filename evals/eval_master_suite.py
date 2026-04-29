import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.decomposition import PCA
import scipy.stats as stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from models_diffusion import TokenTransformerDiffusion, WaveformDecoder
from evaluate_downstream import sample_diffusion

WISDM_LABELS = {0: 'Downstairs', 1: 'Jogging', 2: 'Sitting', 3: 'Standing', 4: 'Upstairs', 5: 'Walking'}

def expected_obs(title, text):
    print(f"\n[{title}] -> Expected Ideal Observation:")
    print(f"    {text}\n")

def eval_simple_classification(real_feats, syn_feats, real_y, syn_y, out_dir, use_wandb=False):
    import wandb
    expected_obs("Simple Classification", "The classifier trained on Real+Synthetic data should match or slightly exceed the F1-score of the Real Only classifier, proving the synthetic data acts as valid, physically accurate data augmentation.")
    
    # Filter to Walking(5), Jogging(1), Sitting(2)
    classes = [5, 1, 2]
    
    def filter_data(X, y, cls_list):
        mask = np.isin(y, cls_list)
        return X[mask], y[mask]
        
    Xr_f, yr_f = filter_data(real_feats, real_y, classes)
    Xs_f, ys_f = filter_data(syn_feats, syn_y, classes)
    
    from sklearn.model_selection import train_test_split
    Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr_f, yr_f, test_size=0.2, random_state=42)
    
    # 1. Real Only
    clf_real = RandomForestClassifier(random_state=42).fit(Xr_train, yr_train)
    pred_real = clf_real.predict(Xr_test)
    f1_real = f1_score(yr_test, pred_real, average='macro')
    
    # 2. Synthetic Only
    clf_syn = RandomForestClassifier(random_state=42).fit(Xs_f, ys_f)
    pred_syn = clf_syn.predict(Xr_test)
    f1_syn = f1_score(yr_test, pred_syn, average='macro')
    
    # 3. Real + Synthetic
    Xr_comb = np.vstack((Xr_train, Xs_f))
    yr_comb = np.concatenate((yr_train, ys_f))
    clf_comb = RandomForestClassifier(random_state=42).fit(Xr_comb, yr_comb)
    pred_comb = clf_comb.predict(Xr_test)
    f1_comb = f1_score(yr_test, pred_comb, average='macro')
    
    print(f"Real Only F1:       {f1_real:.4f}")
    print(f"Synthetic Only F1:  {f1_syn:.4f}")
    print(f"Real+Synthetic F1:  {f1_comb:.4f}")

    if use_wandb:
        wandb.log({
            "Eval/SimpleClass/Real_F1": f1_real,
            "Eval/SimpleClass/Syn_F1": f1_syn,
            "Eval/SimpleClass/RealSyn_F1": f1_comb
        })

def eval_physical_sanity(real_waveforms, syn_waveforms, real_y, syn_y, out_dir, use_wandb=False):
    import wandb
    expected_obs("Physical Sanity", "The histograms of amplitude distributions between Real and Synthetic data for a given class (e.g., Jogging) should be nearly identical. Additionally, the temporal smoothness (step-to-step L2 distance) should be comparable, meaning synthetic signals aren't jittery.")
    
    classes_to_plot = [1, 2] # Jogging (dynamic), Sitting (static)
    
    os.makedirs(f"{out_dir}/physical_sanity", exist_ok=True)
    
    for cls in classes_to_plot:
        rw = real_waveforms[real_y == cls]
        sw = syn_waveforms[syn_y == cls]
        
        if len(rw) == 0 or len(sw) == 0: continue
        
        # 1. Amplitude Distribution (flatten to compare all values)
        plt.figure(figsize=(8, 5))
        plt.hist(rw.flatten(), bins=50, alpha=0.5, density=True, label='Real Amplitude', color='blue')
        plt.hist(sw.flatten(), bins=50, alpha=0.5, density=True, label='Synthetic Amplitude', color='red')
        plt.title(f"Amplitude Distribution: {WISDM_LABELS[cls]}")
        plt.legend()
        plot_path = f"{out_dir}/physical_sanity/amp_dist_class_{cls}.png"
        plt.savefig(plot_path)
        plt.close()
        
        # 2. Temporal Smoothness
        # Calculate step-to-step L2 distance
        rw_smooth = np.mean(np.linalg.norm(rw[:, 1:, :] - rw[:, :-1, :], axis=-1))
        sw_smooth = np.mean(np.linalg.norm(sw[:, 1:, :] - sw[:, :-1, :], axis=-1))
        print(f"Class {WISDM_LABELS[cls]} | Real Smoothness: {rw_smooth:.4f} | Synthetic Smoothness: {sw_smooth:.4f}")

        if use_wandb:
            wandb.log({
                f"Eval/Sanity/AmpDist_Class_{cls}": wandb.Image(plot_path),
                f"Eval/Sanity/RealSmooth_Class_{cls}": rw_smooth,
                f"Eval/Sanity/SynSmooth_Class_{cls}": sw_smooth
            })

def eval_distributional_shifts(real_feats, syn_feats, out_dir, use_wandb=False):
    import wandb
    expected_obs("Distributional Shifts (PCA)", "The PCA plot should show the Synthetic (red) cluster completely overlapping the Real (blue) cluster, proving that the synthetic data has safely interpolated within the real data manifold without extrapolating into physically impossible spaces.")
    
    os.makedirs(f"{out_dir}/distribution", exist_ok=True)
    
    pca = PCA(n_components=2)
    X_comb = np.vstack((real_feats, syn_feats))
    X_pca = pca.fit_transform(X_comb)
    
    N_real = len(real_feats)
    
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:N_real, 0], X_pca[:N_real, 1], alpha=0.4, label='Real', color='blue', s=10)
    plt.scatter(X_pca[N_real:, 0], X_pca[N_real:, 1], alpha=0.6, label='Synthetic', color='red', s=15)
    plt.title("PCA Manifold: Real vs Synthetic")
    plt.legend()
    plot_path = f"{out_dir}/distribution/pca_manifold.png"
    plt.savefig(plot_path)
    plt.close()

    if use_wandb:
        wandb.log({"Eval/Distribution/PCA_Manifold": wandb.Image(plot_path)})

def plot_comparative_waveforms(real_waveforms, syn_waveforms, real_y, syn_y, out_dir, use_wandb=False):
    import wandb
    expected_obs("Comparative Waveforms", "Side-by-side or overlaid plots of Real vs Synthetic waveforms should look physically identical in amplitude range and wave structure for each specific class.")
    os.makedirs(f"{out_dir}/waveforms", exist_ok=True)
    
    for cls in WISDM_LABELS.keys():
        rw = real_waveforms[real_y == cls]
        sw = syn_waveforms[syn_y == cls]
        
        if len(rw) == 0 or len(sw) == 0: continue
        
        # Pick a random sample
        real_sample = rw[np.random.randint(len(rw))]
        syn_sample = sw[np.random.randint(len(sw))]
        
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle(f"Waveform Comparison: {WISDM_LABELS[cls]}\nReal (Blue) vs Synthetic (Red)")
        
        for ax, rx, sx, label in zip(axes, real_sample.T, syn_sample.T, ["X", "Y", "Z"]):
            ax.plot(rx, color='blue', alpha=0.7, label='Real')
            ax.plot(sx, color='red', alpha=0.7, label='Synthetic')
            ax.set_ylabel(f"{label}-Accel")
            ax.legend(loc='upper right')
            
        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()
        plot_path = f"{out_dir}/waveforms/waveform_cmp_{cls}.png"
        plt.savefig(plot_path)
        plt.close()
        
        if use_wandb:
            wandb.log({f"Eval/Waveforms/Class_{cls}": wandb.Image(plot_path)})

def eval_class_imbalance_repair(real_feats, real_y, model, L, num_classes, args, out_dir, use_wandb=False):
    import wandb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    expected_obs("Class Imbalance Repair", "The 'Repaired' F1 Score should be significantly higher than the 'Baseline' F1 Score, and the Repaired Confusion Matrix should show fewer errors for the starved class (Class 0 / Downstairs).")
    
    rare_class = 0
    np.random.seed(42)
    rare_idx = np.where(real_y == rare_class)[0]
    keep_rare = np.random.choice(rare_idx, size=int(len(rare_idx)*0.05), replace=False)
    other_idx = np.where(real_y != rare_class)[0]
    imbalanced_idx = np.concatenate([keep_rare, other_idx])
    
    X_imb = real_feats[imbalanced_idx]
    y_imb = real_y[imbalanced_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(X_imb, y_imb, test_size=0.2, random_state=42)
    
    # Baseline
    clf_base = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    preds_base = clf_base.predict(X_test)
    f1_base = f1_score(y_test, preds_base, average='macro')
    
    # Repair (Generate synthetic data to replace missing rare_class samples)
    gen_n = len(rare_idx) - len(keep_rare)
    print(f"Generating {gen_n} synthetic samples to repair starved Class {rare_class}...")
    gen_c = torch.full((gen_n,), rare_class, device=args.device, dtype=torch.long)
    syn_tokens = sample_diffusion(model, (gen_n, L, 64), gen_c, args.device, w=args.cfg_weight, num_classes=num_classes)
    syn_rare = syn_tokens.cpu().numpy().mean(axis=1)
        
    X_train_repaired = np.concatenate([X_train, syn_rare])
    y_train_repaired = np.concatenate([y_train, np.full(len(syn_rare), rare_class)])
    
    clf_repair = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_repaired, y_train_repaired)
    preds_repair = clf_repair.predict(X_test)
    f1_repair = f1_score(y_test, preds_repair, average='macro')
    
    print(f"Imbalance Baseline F1: {f1_base:.4f}")
    print(f"Imbalance Repaired F1: {f1_repair:.4f}")
    
    if use_wandb:
        wandb.log({"Eval/Imbalance/Base_F1": f1_base, "Eval/Imbalance/Repaired_F1": f1_repair})
        
    # Plot Confusion Matrices
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for ax_idx, preds, title, cmap in zip([0, 1], [preds_base, preds_repair], [f"Baseline (Starved {rare_class})", "Repaired (Syn Added)"], ['Blues', 'Greens']):
        cm = confusion_matrix(y_test, preds)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[WISDM_LABELS.get(i, str(i)) for i in range(num_classes)]).plot(ax=ax[ax_idx], cmap=cmap, colorbar=False, xticks_rotation=45)
        ax[ax_idx].set_title(title)
        
    plt.suptitle("Downstream Activity Classifier Confusion Matrices", fontsize=16)
    plt.tight_layout()
    os.makedirs(f"{out_dir}/imbalance", exist_ok=True)
    plot_path = f"{out_dir}/imbalance/confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()
    
    if use_wandb:
        wandb.log({"Eval/Imbalance/ConfusionMatrix": wandb.Image(plot_path)})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../features/biopm_tokens.npz")
    parser.add_argument("--diff_ckpt", type=str, default="../checkpoints/diffusion/token_diff_ema.pt")
    parser.add_argument("--dec_ckpt", type=str, default="../checkpoints/diffusion/waveform_decoder.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--cfg_weight", type=float, default=1.5, help="Guidance scale for Classifier-Free Guidance (CFG). Set to 0.0 for unconditional testing.")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    args = parser.parse_args()

    if args.wandb:
        import wandb
        wandb.init(project="BioPM-Diffusion-Eval", config=vars(args))

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading data...")
    npz = np.load(args.data)
    real_tokens = npz['tokens']
    real_labels = npz['labels']
    real_raw_windows = npz['raw_windows'] # (N, 300, 3)
    
    B, L, _ = real_tokens.shape
    num_classes = int(real_labels.max()) + 1

    # Ensure models exist before running, otherwise we just use dummy data for now
    if not os.path.exists(args.diff_ckpt) or not os.path.exists(args.dec_ckpt):
        print("Models not found! Please run train_diffusion.py and train_waveform_decoder.py first.")
        return

    model = TokenTransformerDiffusion(seq_len=L, token_dim=64, num_classes=num_classes + 1).to(args.device)
    model.load_state_dict(torch.load(args.diff_ckpt, map_location=args.device, weights_only=True))
    
    decoder = WaveformDecoder(token_dim=64, hidden_dim=256, out_channels=3, target_length=real_raw_windows.shape[1]).to(args.device)
    decoder.load_state_dict(torch.load(args.dec_ckpt, map_location=args.device, weights_only=True))
    
    print("Generating Synthetic Data with CFG...")
    gen_n = 500
    gen_c = torch.randint(0, num_classes, (gen_n,), device=args.device)
    
    # CFG generation
    syn_tokens = sample_diffusion(model, (gen_n, L, 64), gen_c, args.device, w=args.cfg_weight, num_classes=num_classes)
    
    with torch.no_grad():
        syn_waveforms = decoder(syn_tokens).cpu().numpy()
        
    syn_tokens_np = syn_tokens.cpu().numpy()
    syn_labels_np = gen_c.cpu().numpy()
    
    # Average tokens for features
    real_features = real_tokens.mean(axis=1)
    syn_features = syn_tokens_np.mean(axis=1)

    print("--- Starting Evaluations ---")
    eval_simple_classification(real_features, syn_features, real_labels, syn_labels_np, args.out_dir, args.wandb)
    eval_physical_sanity(real_raw_windows, syn_waveforms, real_labels, syn_labels_np, args.out_dir, args.wandb)
    eval_distributional_shifts(real_features, syn_features, args.out_dir, args.wandb)
    plot_comparative_waveforms(real_raw_windows, syn_waveforms, real_labels, syn_labels_np, args.out_dir, args.wandb)
    eval_class_imbalance_repair(real_features, real_labels, model, L, num_classes, args, args.out_dir, args.wandb)
    print("--- Evaluations Complete ---")

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main()
