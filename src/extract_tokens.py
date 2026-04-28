#!/usr/bin/env python3
"""
extract_tokens.py — Extract unpooled (N, L, 64) BioPM tokens for generative training.
"""

import os
import sys
import argparse
import numpy as np
import torch

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
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()

def main():
    args = parse_args()
    print("=" * 60)
    print("BioPM Token Extraction (Generative Pipeline)")
    print("=" * 60)

    # 1. Load Data and Model
    print(f"Loading preprocessed data from {args.data_dir}...")
    (X, pos_info, add_emb, labels, pids, X_grav, raw_acc) = load_preprocessed_h5(args.data_dir)
    
    print(f"Loading model checkpoint from {args.checkpoint} onto {args.device}...")
    model = load_pretrained_encoder(args.checkpoint, device=args.device)

    all_tokens, all_masks, all_labels, all_pids = [], [], [], []
    # ME patches (L, 32) for decoder training, and physical windows (T, 3) for waveform plots
    all_raw_patches, all_pos_info, all_raw_windows = [], [], []

    print("Extracting full sequence tokens...")
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            end_idx = min(i + args.batch_size, N)
            
            # Slice batch
            X_batch = torch.from_numpy(X[i:end_idx]).float().to(args.device)
            pos_batch = torch.from_numpy(pos_info[i:end_idx]).float().to(args.device)
            add_batch = torch.from_numpy(add_emb[i:end_idx]).float().to(args.device)

            # Padding mask: boolean tensor telling us which patches are valid
            # padding_mask = ~torch.isnan(my_X).any(dim=-1)

            # No mask modeling logic here, just forward.
            mask = torch.zeros(bs, my_X.shape[1], device=args.device)
            acc_tokens = model.encoder_acc(my_X, my_pos, mask, my_add)
            input_mask = ~torch.isnan(my_X).any(dim=-1)
            token_mask = torch.nn.functional.interpolate(
                input_mask.unsqueeze(1).float(), 
                size=acc_tokens.shape[1], 
                mode='nearest'
            ).squeeze(1).bool()

            all_tokens.append(acc_tokens.cpu().numpy())
            all_masks.append(token_mask.cpu().numpy())
            all_labels.append(my_Y.numpy())
            all_pids.append(my_PID.numpy())
            all_raw_patches.append(my_X.cpu().numpy())
            all_pos_info.append(my_pos.cpu().numpy())
            all_raw_windows.append(raw_batch.numpy())  # physical (bs, T, 3) windows

    tokens = np.concatenate(all_tokens, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pids = np.concatenate(all_pids, axis=0)
    raw_patches = np.concatenate(all_raw_patches, axis=0)
    pos_info = np.concatenate(all_pos_info, axis=0)
    raw_windows = np.concatenate(all_raw_windows, axis=0)

    np.savez(args.output,
             tokens=tokens,
             masks=masks,
             labels=labels,
             pids=pids,
             raw_patches=raw_patches,
             pos_info=pos_info,
             raw_windows=raw_windows)
    
    print("=" * 60)
    print(f"Extraction complete! Saved to {args.output}")
    print(f"  Tokens shape:    {all_tokens.shape}")
    print(f"  Pad Masks shape: {pad_masks.shape}")

if __name__ == "__main__":
    main()