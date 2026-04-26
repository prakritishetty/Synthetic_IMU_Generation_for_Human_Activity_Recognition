#!/usr/bin/env python3
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import argparse

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--real_tokens", type=str, default="features/biopm_tokens.npz")
    p.add_argument("--syn_tokens", type=str, default="features/synthetic_tokens.npz")
    return p.parse_args()

def extract_hyper_dense_features(tokens):
    """
    Extracts Mean, Std, Max, and Min to perfectly map the temporal boundaries.
    """
    mean_p = tokens.mean(axis=1)
    std_p = tokens.std(axis=1)
    max_p = tokens.max(axis=1)
    min_p = tokens.min(axis=1)
    return np.hstack((mean_p, std_p, max_p, min_p))

def main():
    args = parse_args()
    
    real_data = np.load(args.real_tokens)
    syn_data = np.load(args.syn_tokens)
    
    X_real = extract_hyper_dense_features(real_data['tokens'])
    y_real = real_data['labels']
    
    X_syn = extract_hyper_dense_features(syn_data['tokens'])
    y_syn = syn_data['labels']

    X_train, X_test, y_train, y_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

    # 1. Sabotage the Baseline
    class_0_idx = np.where(y_train == 0)[0]
    other_classes_idx = np.where(y_train != 0)[0]
    
    keep_class_0_idx = np.random.choice(class_0_idx, size=int(len(class_0_idx) * 0.1), replace=False)
    imbalanced_train_idx = np.concatenate([keep_class_0_idx, other_classes_idx])
    
    X_train_imb = X_train[imbalanced_train_idx]
    y_train_imb = y_train[imbalanced_train_idx]

    scaler = StandardScaler()
    X_train_imb_scaled = scaler.fit_transform(X_train_imb)
    X_test_scaled = scaler.transform(X_test)

    # Train Baseline using SOTA Gradient Boosting
    clf_baseline = HistGradientBoostingClassifier(max_iter=200, random_state=42, class_weight='balanced')
    clf_baseline.fit(X_train_imb_scaled, y_train_imb)
    y_pred_base = clf_baseline.predict(X_test_scaled)
    f1_base = f1_score(y_test, y_pred_base, average='macro')

    # 2. Variance-Preserving Curation
    syn_class_0_idx = np.where(y_syn == 0)[0]
    X_syn_class_0 = X_syn[syn_class_0_idx]
    y_syn_class_0 = y_syn[syn_class_0_idx]
    
    X_syn_class_0_scaled = scaler.transform(X_syn_class_0)
    
    # Measure the REAL data's variance
    real_class_0_scaled = X_train_imb_scaled[y_train_imb == 0]
    centroid = real_class_0_scaled.mean(axis=0)
    real_distances = np.linalg.norm(real_class_0_scaled - centroid, axis=1)
    
    # Define the "safe zone" as the 95th percentile of real data variance
    safe_radius = np.percentile(real_distances, 95)
    
    # Filter synthetic data: Keep everything inside the safe zone!
    syn_distances = np.linalg.norm(X_syn_class_0_scaled - centroid, axis=1)
    valid_idx = np.where(syn_distances <= safe_radius)[0]
    X_syn_curated = X_syn_class_0_scaled[valid_idx]
    y_syn_curated = y_syn_class_0[valid_idx]

    # Augment!
    X_train_aug = np.vstack((X_train_imb_scaled, X_syn_curated))
    y_train_aug = np.concatenate((y_train_imb, y_syn_curated))

    # Train Augmented Gradient Boosting Model
    clf_aug = HistGradientBoostingClassifier(max_iter=200, random_state=42, class_weight='balanced')
    clf_aug.fit(X_train_aug, y_train_aug)
    y_pred_aug = clf_aug.predict(X_test_scaled)
    f1_aug = f1_score(y_test, y_pred_aug, average='macro')

    print("\n" + "=" * 65)
    print("SOTA GRADIENT BOOSTING & VARIANCE CURATION")
    print("=" * 65)
    print(f"Baseline F1 Score:                   {f1_base:.4f}")
    print(f"Augmented F1 (Curated Synthetic):    {f1_aug:.4f}")
    
    

if __name__ == "__main__":
    main()