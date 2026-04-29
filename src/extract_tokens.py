#!/usr/bin/env python3
"""
extract_tokens.py — Extract unpooled and pooled BioPM tokens for generative training.
"""
import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.biopm import load_pretrained_encoder
from src.data.preprocessing import load_preprocessed_h5

def parse_args():
    p = argparse.ArgumentParser(description="Extract raw BioPM tokens for generation")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with Data_MeLabel_*.h5 files")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to 50MR checkpoint.pt")
    p.add_argument("--output", type=str, default="features/biopm_tokens.npz",
                   help="Output .npz file path")
    p.add_argument("--batch_size", type=int, default=128) # Bumped up for Colab GPU
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    print("=" * 65)
    print(f"🚀 BioPM Token Extraction Engine (Hardware: {args.device.upper()})")
    print("=" * 65)

    print(f"Loading preprocessed data from {args.data_dir}...")
    (X, pos_info, add_emb, labels, pids, X_grav, raw_acc) = load_preprocessed_h5(args.data_dir)
    
    print(f"Loading model checkpoint from {args.checkpoint}...")
    model = load_pretrained_encoder(args.checkpoint, device=args.device)
    model.eval()

    all_tokens, all_masks, all_labels, all_pids = [], [], [], []
    all_raw_patches, all_pos_info, all_raw_windows = [], [], []

    N = len(X)
    print(f"Extracting tokens for {N} sequences...")
    
    with torch.inference_mode():
        for i in tqdm(range(0, N, args.batch_size), desc="Processing Batches"):
            end_idx = min(i + args.batch_size, N)
            
            X_batch = torch.from_numpy(X[i:end_idx]).float().to(args.device)
            pos_batch = torch.from_numpy(pos_info[i:end_idx]).float().to(args.device)
            add_batch = torch.from_numpy(add_emb[i:end_idx]).float().to(args.device)
            Y_batch = labels[i:end_idx]
            PID_batch = pids[i:end_idx]
            raw_batch = raw_acc[i:end_idx]
            bs = X_batch.shape[0]

            mask = torch.zeros(bs, X_batch.shape[1], device=args.device)
            acc_tokens = model.encoder_acc(X_batch, pos_batch, mask, add_batch)
            
            input_mask = ~torch.isnan(X_batch).any(dim=-1)
            token_mask = torch.nn.functional.interpolate(
                input_mask.unsqueeze(1).float(), 
                size=acc_tokens.shape[1], 
                mode='nearest'
            ).squeeze(1).bool()

            all_tokens.append(acc_tokens.cpu().numpy())
            all_masks.append(token_mask.cpu().numpy())
            all_labels.append(Y_batch)
            all_pids.append(PID_batch)
            all_raw_patches.append(X_batch.cpu().numpy())
            all_pos_info.append(pos_batch.cpu().numpy())
            all_raw_windows.append(raw_batch)

    tokens = np.concatenate(all_tokens, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pids = np.concatenate(all_pids, axis=0)
    raw_patches = np.concatenate(all_raw_patches, axis=0)
    pos_info = np.concatenate(all_pos_info, axis=0)
    raw_windows = np.concatenate(all_raw_windows, axis=0)

    # Calculate Pooled Tokens for our PCA/KDE/MMD visualization scripts
    pooled_tokens = np.zeros((N, 64), dtype=np.float32)
    for b_idx in range(N):
        valid_len = masks[b_idx].sum()
        if valid_len > 0:
            pooled_tokens[b_idx] = tokens[b_idx, :valid_len, :].mean(axis=0)
        else:
            pooled_tokens[b_idx] = 0.0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output,
             tokens=tokens,
             pooled_tokens=pooled_tokens, 
             masks=masks,
             labels=labels,
             pids=pids,
             raw_patches=raw_patches,
             pos_info=pos_info,
             raw_windows=raw_windows)
    
    print("\n" + "=" * 65)
    print(f"Extraction complete! Safely locked into: {args.output}")
    print("=" * 65)

if __name__ == "__main__":
    main()
