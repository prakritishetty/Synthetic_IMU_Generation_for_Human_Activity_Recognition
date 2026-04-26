#!/usr/bin/env python3
"""
extract_tokens.py — Extract continuous BioPM token sequences.

Instead of heavily pooling the outputs to 1028-d vectors, this saves the full 
(N, L, 64) token matrices and the boolean padding masks. This is the exact 
continuous space we will use to train our Diffusion generator.
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.biopm import load_pretrained_encoder
from src.data.dataset import MovementElementDataset
from src.data.preprocessing import load_preprocessed_h5

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Directory with Data_MeLabel_*.h5 files")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to 50MR checkpoint.pt")
    p.add_argument("--output", type=str, default="features/biopm_tokens.npz")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading data from {args.data_dir}...")
    (X, pos_info, add_emb, labels, pids, X_grav, raw_acc) = load_preprocessed_h5(args.data_dir)
    
    dataset = MovementElementDataset(
        X=X, X_grav=raw_acc, y=labels, pos_info=pos_info,
        additional_embedding=add_emb, pid=pids,
        name="extract_tokens", is_label=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    print(f"Loaded {X.shape[0]} windows. Pad size (L) = {X.shape[1]}")
    model = load_pretrained_encoder(args.checkpoint, device=args.device)

    all_tokens, all_masks, all_labels, all_pids = [], [], [], []
    # Additionally capturing raw patches info to train the IMU decoding later!
    all_raw_patches, all_pos_info = [], []

    print("Extracting full sequence tokens...")
    with torch.no_grad():
        for batch in loader:
            my_X, my_Y, my_PID, raw_batch, my_pos, my_add = batch
            bs = my_X.shape[0]

            my_X = my_X.to(args.device, dtype=torch.float)
            my_pos = my_pos.to(args.device, dtype=torch.float)
            my_add = my_add.to(args.device, dtype=torch.float)

            # Padding mask: boolean tensor telling us which patches are valid
            padding_mask = ~torch.isnan(my_X).any(dim=-1)

            # No mask modeling logic here, just forward.
            mask = torch.zeros(bs, my_X.shape[1], device=args.device)
            acc_tokens = model.encoder_acc(my_X, my_pos, mask, my_add)

            all_tokens.append(acc_tokens.cpu().numpy())
            all_masks.append(padding_mask.cpu().numpy())
            all_labels.append(my_Y.numpy())
            all_pids.append(my_PID.numpy())
            # all_raw_patches.append(my_X.cpu().numpy())
            all_raw_patches.append(raw_batch.cpu().numpy())
            all_pos_info.append(my_pos.cpu().numpy())

    tokens = np.concatenate(all_tokens, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    pids = np.concatenate(all_pids, axis=0)
    raw_patches = np.concatenate(all_raw_patches, axis=0)
    pos_info = np.concatenate(all_pos_info, axis=0)

    np.savez(args.output, 
             tokens=tokens, 
             masks=masks, 
             labels=labels, 
             pids=pids,
             raw_patches=raw_patches,
             pos_info=pos_info)
    
    print("=" * 60)
    print("Tokens saved successfully!")
    print(f"  Tokens shape:  {tokens.shape}")
    print(f"  Valid length:  {masks.sum(axis=1).mean():.1f} avg valid patches per window")
    print(f"  Saved to:      {args.output}")

if __name__ == "__main__":
    main()
