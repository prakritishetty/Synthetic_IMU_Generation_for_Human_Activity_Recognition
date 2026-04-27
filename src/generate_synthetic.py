#!/usr/bin/env python3
import argparse
import numpy as np
import torch
import os
from train_generator import BioPMGenerator

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generator", type=str, default="checkpoints/generator.pt")
    p.add_argument("--real_tokens", type=str, default="features/biopm_tokens.npz")
    p.add_argument("--output", type=str, default="features/synthetic_tokens.npz")
    p.add_argument("--num_samples", type=int, default=300) # Increased sample count
    p.add_argument("--seed_len", type=int, default=3)      
    p.add_argument("--target_class", type=int, default=0)  # ONLY generate this class
    p.add_argument("--noise_std", type=float, default=0.1) # Add variance to prevent collapse
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading generator from {args.generator} onto {device}...")
    model = BioPMGenerator().to(device)
    model.load_state_dict(torch.load(args.generator, map_location=device))
    model.eval() 
    
    data = np.load(args.real_tokens)
    real_tokens = data['tokens']
    labels = data['labels']
    max_len = real_tokens.shape[1] 
    
    # FILTER: Only get seeds from our target class!
    target_indices = np.where(labels == args.target_class)[0]
    if len(target_indices) == 0:
        print(f"Error: No samples found for class {args.target_class}")
        return

    synthetic_tokens = []
    synthetic_labels = []
    
    print(f"Hallucinating {args.num_samples} sequences for Class {args.target_class}...")
    
    with torch.no_grad():
        for i in range(args.num_samples):
            # Pick a random seed from ONLY our target class
            idx = np.random.choice(target_indices)
            seed_seq = real_tokens[idx, :args.seed_len, :]
            
            current_seq = torch.tensor(seed_seq, dtype=torch.float32).unsqueeze(0).to(device)
            
            for _ in range(max_len - args.seed_len):
                predictions = model(current_seq)
                next_token = predictions[:, -1:, :] 
                
                # INJECT NOISE: This prevents mode collapse so the dots spread out!
                noise = torch.randn_like(next_token) * args.noise_std
                next_token = next_token + noise
                next_token = torch.clamp(next_token, -3.0, 3.0)
                
                current_seq = torch.cat([current_seq, next_token], dim=1)
            
            synthetic_tokens.append(current_seq.squeeze(0).cpu().numpy())
            synthetic_labels.append(args.target_class)
            
            if (i + 1) % 50 == 0:
                print(f"Generated {i + 1}/{args.num_samples} sequences...")
                
    synthetic_tokens = np.array(synthetic_tokens)
    synthetic_labels = np.array(synthetic_labels)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez(args.output, tokens=synthetic_tokens, labels=synthetic_labels)
    
    print("=" * 60)
    print(f"Done! Saved synthetic tokens to {args.output}")

if __name__ == "__main__":
    main()