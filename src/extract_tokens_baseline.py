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
    model.eval()

    N = X.shape[0]
    L = X.shape[1]
    
    # We will store the extracted tokens and pad masks here
    all_tokens = np.zeros((N, L, 64), dtype=np.float32)
    pad_masks = np.zeros((N, L), dtype=bool)

    print(f"Total samples: {N}, Sequence length: {L}")
    print("Extracting tokens in batches...")

    # 2. Process in Batches
    with torch.no_grad():
        for i in range(0, N, args.batch_size):
            end_idx = min(i + args.batch_size, N)
            
            # Slice batch
            X_batch = torch.from_numpy(X[i:end_idx]).float().to(args.device)
            pos_batch = torch.from_numpy(pos_info[i:end_idx]).float().to(args.device)
            add_batch = torch.from_numpy(add_emb[i:end_idx]).float().to(args.device)

            # Identify valid patches (where data is not NaN)
            valid_mask = ~torch.isnan(X_batch).any(dim=-1)
            pad_masks[i:end_idx] = valid_mask.cpu().numpy()

            # Zero-mask (we want to see everything)
            mask_no = torch.zeros(X_batch.shape[0], L, device=args.device)

            # Extract!
            tokens = model.encoder_acc(X_batch, pos_batch, mask_no, add_batch)
            all_tokens[i:end_idx] = tokens.cpu().numpy()

            if i % (args.batch_size * 5) == 0:
                print(f"Processed {i}/{N} samples...")

    # 3. Save to disk
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, 
             tokens=all_tokens, 
             pad_masks=pad_masks, 
             labels=labels, 
             pids=pids)

    print("=" * 60)
    print(f"Extraction complete! Saved to {args.output}")
    print(f"  Tokens shape:    {all_tokens.shape}")
    print(f"  Pad Masks shape: {pad_masks.shape}")

if __name__ == "__main__":
    main()