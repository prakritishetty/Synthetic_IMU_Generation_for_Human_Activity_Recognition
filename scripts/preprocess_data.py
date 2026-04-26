#!/usr/bin/env python3
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
    'HighF1': 12,        
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
    p.add_argument("--raw_data_dir", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--window_sec", type=int, default=10)
    p.add_argument("--slide_sec", type=int, default=5)
    p.add_argument("--ori_fs", type=int, default=20)
    p.add_argument("--target_fs", type=int, default=30)
    return p.parse_args()

def load_raw_data(data_dir):
    """Custom parser to load the WISDM v1.1 dataset."""
    file_path = os.path.join(data_dir, "WISDM", "WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
    print(f"Loading WISDM data from {file_path}...")
    
    columns = ['user_id', 'activity', 'timestamp', 'x', 'y', 'z']
    df = pd.read_csv(file_path, header=None, names=columns, on_bad_lines='skip')
    
    df['z'] = df['z'].astype(str).str.replace(';', '').astype(float)
    df.dropna(inplace=True)
    
    label_map = {'Walking': 0, 'Jogging': 1, 'Upstairs': 2, 'Downstairs': 3, 'Sitting': 4, 'Standing': 5}
    df['label'] = df['activity'].map(label_map)
    df = df.dropna(subset=['label']) 
    
    df['x'] = df['x'] / 9.80665
    df['y'] = df['y'] / 9.80665
    df['z'] = df['z'] / 9.80665
    
    data_dict = {}
    for uid in df['user_id'].unique():
        user_data = df[df['user_id'] == uid]
        acc = user_data[['x', 'y', 'z']].values
        labels = user_data['label'].values.astype(int)
        
        if len(acc) > 100:
            data_dict[int(uid)] = {'acc': acc, 'label': labels}
            
    print(f"Successfully loaded data for {len(data_dict)} subjects.")
    return data_dict

def remap_labels(labels_raw, skip_labels=None):
    if skip_labels is None:
        skip_labels = set() # We don't need to skip any for WISDM
    unique_old = sorted(set(int(l) for l in np.unique(labels_raw)) - skip_labels)
    old_to_new = {old: new for new, old in enumerate(unique_old)}
    return old_to_new, skip_labels

def preprocess_one_subject(acc_raw, labels, subject_id, config, output_dir):
    """Run full preprocessing for one subject."""
    # Create an artificial time array since WISDM doesn't have reliable timestamps
    time_array = np.arange(len(acc_raw)) / config['ori_FS']

    acc_resampled, time_resampled, labels_resampled = resample_to_target_fs(
        time_array, acc_raw, labels, config['target_FS'])

    acc_filt = bandpass_filter(acc_resampled, config['LowF1'], config['HighF1'], config['target_FS'], order=config['Order1'])
    acc_gravity = lowpass_filter(acc_resampled, config['LowF1'], config['target_FS'], order=config['Order1'])

    ws = int(config['WS'] * config['target_FS'])  
    step = int(config['SlideSize'] * config['target_FS'])
    label_map, skip_labels = remap_labels(labels_resampled)

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

        if mode_label in skip_labels:
            start += step
            continue

        w_raw = acc_resampled[start:start + ws, :]
        w_filt = acc_filt[start:start + ws, :]
        w_grav = acc_gravity[start:start + ws, :]
        w_time = time_resampled[start:start + ws]

        try:
            (_, _, me_list, me_norm, me_info, _, _, pos_info, zc_list, zc_time_list) = detect_zero_crossings(w_filt, w_time, config)
            (_, _, _, grav_norm, grav_info, _, _, _) = assign_zero_crossings(w_grav, w_time, zc_list, zc_time_list, config)
        except Exception as e:
            start += step
            continue

        if len(me_list) == 0:
            start += step
            continue

        x_acc = np.concatenate([
            me_norm,
            pos_info.reshape(-1, 1),
            me_info[['axis', 'len', 'min', 'max', 'dirct']].values,
        ], axis=1)

        if x_acc.shape[0] < config['pad_size']:
            pad = np.full((config['pad_size'] - x_acc.shape[0], x_acc.shape[1]), np.nan)
            x_acc = np.vstack([x_acc, pad])
        else:
            x_acc = x_acc[:config['pad_size']]

        x_gravity = w_grav.astype(np.float32)
        mapped_label = label_map.get(mode_label, mode_label)

        win_acc_raw.append(w_raw.astype(np.float32))
        win_acc_filt_grav.append(np.concatenate([w_filt, w_grav], axis=1).astype(np.float32))
        win_labels.append(float(mapped_label))
        win_x_acc_filt.append(x_acc.astype(np.float32))
        win_x_gravity.append(x_gravity.astype(np.float32))

        start += step

    if len(win_labels) == 0:
        print(f"  WARNING: no valid windows for subject {subject_id}")
        return

    win_acc_raw = np.array(win_acc_raw, dtype=np.float32)
    win_acc_filt_grav = np.array(win_acc_filt_grav, dtype=np.float32)
    win_x_acc_filt = np.array(win_x_acc_filt, dtype=np.float32)
    win_x_gravity = np.array(win_x_gravity, dtype=np.float32)
    win_labels = np.array(win_labels, dtype=np.float32)

    os.makedirs(output_dir, exist_ok=True)

    h5_acc = os.path.join(output_dir, f"Data_AccLabel_{subject_id}.h5")
    with h5py.File(h5_acc, "w") as f:
        f.create_dataset("window_acc_raw", data=win_acc_raw)
        f.create_dataset("window_acc_filt_gravity", data=win_acc_filt_grav)
        f.create_dataset("window_label", data=win_labels)

    h5_me = os.path.join(output_dir, f"Data_MeLabel_{subject_id}.h5")
    with h5py.File(h5_me, "w") as f:
        f.create_dataset("window_acc_raw", data=win_acc_raw)
        f.create_dataset("x_acc_filt", data=win_x_acc_filt)
        f.create_dataset("x_gravity", data=win_x_gravity)
        f.create_dataset("window_label", data=win_labels)

    print(f"  Saved {len(win_labels)} windows for subject {subject_id}")

def main():
    args = parse_args()
    config = DEFAULT_CONFIG.copy()
    config['WS'] = args.window_sec
    config['SlideSize'] = args.slide_sec
    config['ori_FS'] = args.ori_fs
    config['target_FS'] = args.target_fs
    config['pad_size'] = int(config['WS'] * 192 / 10)

    print("=" * 60)
    print("BioPM Data Preprocessing (WISDM Version)")
    print("=" * 60)

    # Load all WISDM data into our dictionary
    data_dict = load_raw_data(args.raw_data_dir)

    if not data_dict:
        print("ERROR: No data loaded.")
        sys.exit(1)

    # Loop through our dictionary instead of looking for files
    for subject_id, user_data in data_dict.items():
        print(f"Processing subject {subject_id} ...")
        acc_raw = user_data['acc']
        labels = user_data['label']
        
        try:
            preprocess_one_subject(acc_raw, labels, subject_id, config, args.output_dir)
        except Exception as e:
            print(f"  ERROR processing subject {subject_id}: {e}")
            continue

    print("\nPreprocessing complete!")
    print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()