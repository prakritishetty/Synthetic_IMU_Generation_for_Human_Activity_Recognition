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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models_diffusion import TokenTransformerDiffusion, WaveformDecoder
from src.evaluate_downstream import sample_diffusion

WISDM_LABELS = {0: 'Downstairs', 1: 'Jogging', 2: 'Sitting', 3: 'Standing', 4: 'Upstairs', 5: 'Walking'}

def expected_obs(title, text):
    print(f"\n[{title}] -> Expected Ideal Observation:")
    print(f"    {text}\n")

def eval_simple_classification(real_feats, syn_feats, real_y, syn_y, out_dir):
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

def eval_physical_sanity(real_waveforms, syn_waveforms, real_y, syn_y, out_dir):
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
        plt.savefig(f"{out_dir}/physical_sanity/amp_dist_class_{cls}.png")
        plt.close()
        
        # 2. Temporal Smoothness
        # Calculate step-to-step L2 distance
        rw_smooth = np.mean(np.linalg.norm(rw[:, 1:, :] - rw[:, :-1, :], axis=-1))
        sw_smooth = np.mean(np.linalg.norm(sw[:, 1:, :] - sw[:, :-1, :], axis=-1))
        print(f"Class {WISDM_LABELS[cls]} | Real Smoothness: {rw_smooth:.4f} | Synthetic Smoothness: {sw_smooth:.4f}")

def eval_distributional_shifts(real_feats, syn_feats, out_dir):
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
    plt.savefig(f"{out_dir}/distribution/pca_manifold.png")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../features/biopm_tokens.npz")
    parser.add_argument("--diff_ckpt", type=str, default="../checkpoints/diffusion/token_diff.pt")
    parser.add_argument("--dec_ckpt", type=str, default="../checkpoints/diffusion/waveform_decoder.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

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
    
    decoder = WaveformDecoder(token_dim=64, hidden_dim=128, out_channels=3, target_length=real_raw_windows.shape[1]).to(args.device)
    decoder.load_state_dict(torch.load(args.dec_ckpt, map_location=args.device, weights_only=True))
    
    print("Generating Synthetic Data with CFG...")
    gen_n = 500
    gen_c = torch.randint(0, num_classes, (gen_n,), device=args.device)
    
    # CFG generation
    syn_tokens = sample_diffusion(model, (gen_n, L, 64), gen_c, args.device, w=1.5, num_classes=num_classes)
    
    with torch.no_grad():
        syn_waveforms = decoder(syn_tokens).cpu().numpy()
        
    syn_tokens_np = syn_tokens.cpu().numpy()
    syn_labels_np = gen_c.cpu().numpy()
    
    # Average tokens for features
    real_features = real_tokens.mean(axis=1)
    syn_features = syn_tokens_np.mean(axis=1)

    print("--- Starting Evaluations ---")
    eval_simple_classification(real_features, syn_features, real_labels, syn_labels_np, args.out_dir)
    eval_physical_sanity(real_raw_windows, syn_waveforms, real_labels, syn_labels_np, args.out_dir)
    eval_distributional_shifts(real_features, syn_features, args.out_dir)
    print("--- Evaluations Complete ---")

if __name__ == "__main__":
    main()
