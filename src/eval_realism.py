#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import os
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_tokens", type=str, default="features/biopm_tokens.npz")
    p.add_argument("--syn_tokens", type=str, default="features/synthetic_tokens.npz")
    p.add_argument("--output_plot", type=str, default="results/tsne_comparison.png")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_plot), exist_ok=True)

    print("Loading data for visual evaluation...")
    real_data = np.load(args.real_tokens)
    syn_data = np.load(args.syn_tokens)

    # Average the tokens across the 10-second window to get a flat 64-d vector per sample
    # real_features = real_data['tokens'].mean(axis=1) 
    # syn_features = syn_data['tokens'].mean(axis=1)
    mid = real_data['tokens'].shape[1] // 2
    real_features = real_data['tokens'][:, mid-5:mid+5, :].mean(axis=1)
    syn_features = syn_data['tokens'][:, mid-5:mid+5, :].mean(axis=1)

    # Combine them for the TSNE transformation
    X_combined = np.vstack((real_features, syn_features))
    
    # Create labels just for the plot (0 for Real, 1 for Synthetic)
    y_plot = np.concatenate([np.zeros(len(real_features)), np.ones(len(syn_features))])

    print("Running t-SNE dimensionality reduction (this might take a minute)...")
    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X_combined)

    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[y_plot == 0, 0], X_2d[y_plot == 0, 1], alpha=0.5, label='Real Data', c='blue')
    plt.scatter(X_2d[y_plot == 1, 0], X_2d[y_plot == 1, 1], alpha=0.5, label='Synthetic Data', c='red')
    
    plt.title('t-SNE Projection: Real vs. Synthetic Bio-PM Tokens')
    plt.legend()
    plt.savefig(args.output_plot)
    print(f"Success! Visual plot saved to {args.output_plot}")

if __name__ == "__main__":
    main()