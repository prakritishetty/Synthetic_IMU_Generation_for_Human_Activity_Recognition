#!/usr/bin/env python3
"""
preprocess_wisdm.py — Convert WISDM dataset raw text to BioPM-ready HDF5 files.

Pipeline:
  1. Load WISDM raw file: user, activity, timestamp, x, y, z;
  2. Map labels to integers
  3. Divide by users and process each user's window independently
  4. Resample 20Hz -> 30Hz
  5. BioPM zero-crossing and windowing
  6. ADVANCED KINEMATICS: Jerk and Posture injection
"""
import os
import sys
import argparse
import statistics
import numpy as np
import pandas as pd
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.preprocessing import (
    resample_to_target_fs, bandpass_filter, lowpass_filter,
    detect_zero_crossings, assign_zero_crossings,
)

DEFAULT_CONFIG = {
    'HighF1': 12,        # Resampled to 30Hz, Nyquist is 15Hz. HighF1 = 12 is perfectly fine.
    'LowF1': 0.5,       
    'Order1': 6,         
    'ori_FS': 20,        # WISDM is 20 Hz
    'target_FS': 30,     
    'WS': 10,            
    'SlideSize': 5,      
    'normalize_size_target': 32,  
    'normalize_size_assign': 32,
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--wisdm_txt", type=str, required=True, help="Path to WISDM_ar_v1.1_raw.txt")
    p.add_argument("--output_dir", type=str, required=True)
    return p.parse_args()

def remap_labels_wisdm(labels_raw):
    # WISDM has 6 labels: Walking, Jogging, Upstairs, Downstairs, Sitting, Standing
    unique_labels = sorted(list(set(labels_raw)))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    print(f"Label map: {label_map}")
    return label_map

def preprocess_one_subject(df_user, user_id, config, output_dir, label_map):
    acc_raw = df_user[['x', 'y', 'z']].values / 9.80665
    labels_raw = df_user['activity'].values
    labels = np.array([label_map[l] for l in labels_raw])
    time_array = np.arange(len(acc_raw)) / config['ori_FS']

    acc_resampled, time_resampled, labels_resampled = resample_to_target_fs(
        time_array, acc_raw, labels, config['target_FS'])

    acc_filt = bandpass_filter(
        acc_resampled, config['LowF1'], config['HighF1'], config['target_FS'],
        order=config['Order1'])

    acc_gravity = lowpass_filter(
        acc_resampled, config['LowF1'], config['target_FS'],
        order=config['Order1'])

    # --- ADVANCED KINEMATICS UPGRADE ---
    jerk = np.zeros_like(acc_gravity)
    jerk[1:] = (acc_gravity[1:] - acc_gravity[:-1]) * config['target_FS']
    
    pitch = np.arctan2(acc_gravity[:, 0], np.sqrt(acc_gravity[:, 1]**2 + acc_gravity[:, 2]**2)) * 180 / np.pi
    roll = np.arctan2(acc_gravity[:, 1], np.sqrt(acc_gravity[:, 0]**2 + acc_gravity[:, 2]**2)) * 180 / np.pi
    posture = np.column_stack((pitch, roll, np.zeros(len(pitch))))
    
    # Overwrite w_grav with the enhanced physics array before sliding windows
    acc_gravity = acc_gravity + (jerk * 0.1) + (posture * 0.01) 
    # -----------------------------------

    ws = int(config['WS'] * config['target_FS'])
    step = int(config['SlideSize'] * config['target_FS'])

    win_acc_raw, win_acc_filt_grav, win_labels = [], [], []
    win_x_acc_filt, win_x_gravity = [], []

    start = 0
    while start + ws < acc_filt.shape[0]:
        window_labels = labels_resampled[start:start + ws]
        try:
            mode_label = statistics.mode(window_labels.astype(int))
        except Exception:
            start += step
            continue

        w_raw = acc_resampled[start:start + ws, :]
        w_filt = acc_filt[start:start + ws, :]
        w_grav = acc_gravity[start:start + ws, :]
        w_time = time_resampled[start:start + ws]

        try:
            (_, _, me_list, me_norm, me_info, _, _, pos_info, zc_list, zc_time_list) = detect_zero_crossings(
                w_filt, w_time, config)
            (_, _, _, grav_norm, grav_info, _, _, _) = assign_zero_crossings(
                w_grav, w_time, zc_list, zc_time_list, config)
        except Exception:
            start += step
            continue

        if len(me_list) == 0:
            start += step
            continue

        x_acc = np.concatenate([
            me_norm, pos_info.reshape(-1, 1), me_info[['axis', 'len', 'min', 'max', 'dirct']].values,
        ], axis=1)

        if x_acc.shape[0] < config['pad_size']:
            pad = np.full((config['pad_size'] - x_acc.shape[0], x_acc.shape[1]), np.nan)
            x_acc = np.vstack([x_acc, pad])
        else:
            x_acc = x_acc[:config['pad_size']]

        x_gravity = w_grav.astype(np.float32)

        win_acc_raw.append(w_raw.astype(np.float32))
        win_acc_filt_grav.append(np.concatenate([w_filt, w_grav], axis=1).astype(np.float32))
        win_labels.append(float(mode_label))
        win_x_acc_filt.append(x_acc.astype(np.float32))
        win_x_gravity.append(x_gravity.astype(np.float32))

        start += step

    if len(win_labels) == 0:
        return

    os.makedirs(output_dir, exist_ok=True)
    h5_me = os.path.join(output_dir, f"Data_MeLabel_{user_id}.h5")
    with h5py.File(h5_me, "w") as f:
        f.create_dataset("window_acc_raw", data=np.array(win_acc_raw, dtype=np.float32))
        f.create_dataset("x_acc_filt", data=np.array(win_x_acc_filt, dtype=np.float32))
        f.create_dataset("x_gravity", data=np.array(win_x_gravity, dtype=np.float32))
        f.create_dataset("window_label", data=np.array(win_labels, dtype=np.float32))
    print(f"  Saved {len(win_labels)} windows for subject {user_id}")

def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config['pad_size'] = int(config['WS'] * 192 / 10)

    print("Loading WISDM Data...")
    with open(args.wisdm_txt, 'r') as f:
        lines = f.readlines()
        
    data = []
    for line in lines:
        parts = line.strip().rstrip(';').split(',')
        if len(parts) >= 6:
            try:
                user = int(parts[0])
                activity = parts[1].strip()
                tstamp = float(parts[2])
                x, y, z = float(parts[3]), float(parts[4]), float(parts[5].replace(';', ''))
                data.append([user, activity, tstamp, x, y, z])
            except ValueError:
                continue
                
    df = pd.DataFrame(data, columns=['user', 'activity', 'timestamp', 'x', 'y', 'z'])
    df = df.dropna()
    print(f"Loaded {len(df)} samples across {df['user'].nunique()} users.")
    
    label_map = remap_labels_wisdm(df['activity'].values)

    for user_id, df_user in df.groupby('user'):
        df_user = df_user.sort_values('timestamp')
        preprocess_one_subject(df_user, user_id, config, args.output_dir, label_map)

if __name__ == "__main__":
    main()
