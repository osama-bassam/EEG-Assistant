# Copyright (c) 2025 Osama Bassam
# All Rights Reserved.
# Unauthorized use, reproduction, or distribution of this code is prohibited.

import os
import mne
import torch
import torch.nn as nn
import joblib
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.signal import welch
import pywt
from pycatch22 import catch22_all
from torch.nn.functional import softmax
from tqdm import tqdm
from io import BytesIO
from mne.io import (
    read_raw_edf, read_raw_bdf, read_raw_brainvision,
    read_raw_cnt, read_raw_eeglab
)
from mne.io import read_raw_gdf  # If pyEDFlib installed
# === 1. Load EDF file from uploaded object ===

def load_raw_auto(file_path_or_buffer):
    """Auto-detect EEG format and load raw object."""
    file_name = file_path_or_buffer.name if hasattr(file_path_or_buffer, 'name') else str(file_path_or_buffer)
    file_name = file_name.lower()

    if file_name.endswith('.edf'):
        raw = read_raw_edf(file_path_or_buffer, preload=True)
    elif file_name.endswith('.bdf'):
        raw = read_raw_bdf(file_path_or_buffer, preload=True)
    elif file_name.endswith('.vhdr'):
        raw = read_raw_brainvision(file_path_or_buffer, preload=True)
    elif file_name.endswith('.cnt'):
        raw = read_raw_cnt(file_path_or_buffer, preload=True)
    elif file_name.endswith('.set'):
        raw = read_raw_eeglab(file_path_or_buffer, preload=True)
    elif file_name.endswith('.trc'):
        raise ValueError(".trc Micromed format is not supported in this MNE version.")

    elif file_name.endswith('.gdf'):
        raw = read_raw_gdf(file_path_or_buffer, preload=True)

    else:
        raise ValueError("Unsupported EEG file format.")

    # ✅ Common preprocessing for all formats
    raw.pick_types(eeg=True)
    raw.filter(0.5, 40.)
    raw._data *= 1e6  # Convert to microvolts
    return raw


# === 2. Preprocess EEG (select fixed channels & create windows) ===
def preprocess(raw, window_size=1000, stride=1000, sampling_rate=100):
    fixed_channels = [
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF',
        'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF',
        'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', 'EEG F8-REF',
        'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF',
        'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF']
    raw.pick_channels([ch for ch in fixed_channels if ch in raw.ch_names])
    X = raw.get_data()
    X_windows = []
    num_windows = (X.shape[1] - window_size) // stride
    for i in range(num_windows):
        start = i * stride
        end = start + window_size
        win = X[:, start:end]
        if win.shape[1] == window_size:
            X_windows.append(torch.tensor(win, dtype=torch.float32))
    return torch.stack(X_windows), raw

# === NEW: Single-window feature extractor ===
def single_window_feats(window, selected_feature_names, sampling_rate=100):
    features = []
    for ch_idx, ch in enumerate(window):
        ch = np.nan_to_num(ch)
        if np.all(ch == ch[0]) or np.std(ch) < 1e-6:
            features.extend([0.0 for name in selected_feature_names if name.startswith(f"ch_{ch_idx}_")])
            continue
        if any("_time_" in f for f in selected_feature_names):
            if f"ch_{ch_idx}_time_mean" in selected_feature_names: features.append(ch.mean())
            if f"ch_{ch_idx}_time_std" in selected_feature_names: features.append(ch.std())
            if f"ch_{ch_idx}_time_max" in selected_feature_names: features.append(ch.max())
            if f"ch_{ch_idx}_time_min" in selected_feature_names: features.append(ch.min())
            if f"ch_{ch_idx}_time_skew" in selected_feature_names: features.append(skew(ch))
            if f"ch_{ch_idx}_time_kurtosis" in selected_feature_names: features.append(kurtosis(ch))
        if any("_psd_" in f for f in selected_feature_names):
            freqs, psd = welch(ch, fs=sampling_rate, nperseg=256)
            bands = {
                "delta": psd[(freqs >= 0.5) & (freqs < 4)].mean(),
                "theta": psd[(freqs >= 4) & (freqs < 8)].mean(),
                "alpha": psd[(freqs >= 8) & (freqs < 13)].mean(),
                "beta": psd[(freqs >= 13) & (freqs < 30)].mean(),
                "gamma": psd[(freqs >= 30) & (freqs < 40)].mean()
            }
            features += [
                bands['delta'] if f"ch_{ch_idx}_psd_delta" in selected_feature_names else None,
                bands['theta'] if f"ch_{ch_idx}_psd_theta" in selected_feature_names else None,
                bands['alpha'] if f"ch_{ch_idx}_psd_alpha" in selected_feature_names else None,
                bands['beta'] if f"ch_{ch_idx}_psd_beta" in selected_feature_names else None,
                bands['gamma'] if f"ch_{ch_idx}_psd_gamma" in selected_feature_names else None
            ]
        if any("_hjorth_" in f for f in selected_feature_names):
            d1, d2 = np.diff(ch), np.diff(np.diff(ch))
            features += [
                np.var(ch) if f"ch_{ch_idx}_hjorth_var" in selected_feature_names else None,
                np.std(d1)/(np.std(ch)+1e-8) if f"ch_{ch_idx}_hjorth_mob" in selected_feature_names else None,
                np.std(d2)/(np.std(d1)+1e-8) if f"ch_{ch_idx}_hjorth_comp" in selected_feature_names else None
            ]
        if any("_wavelet_" in f for f in selected_feature_names):
            coeffs = pywt.wavedec(ch, 'db4', level=3)
            for i, c in enumerate(coeffs):
                key = f"ch_{ch_idx}_wavelet_cD{i}"
                if key in selected_feature_names:
                    features.append(np.sqrt(np.sum(c ** 2)))
        if any("_catch22_" in f for f in selected_feature_names):
            c22 = catch22_all(ch)["values"]
            for i, val in enumerate(c22):
                key = f"ch_{ch_idx}_catch22_{i}"
                if key in selected_feature_names:
                    features.append(np.nan_to_num(val))
    return np.array([f for f in features if f is not None], dtype=np.float32)

# === 3. Extract Features ===
def extract_features(X_eval, selected_feature_names, sampling_rate=100):
    X_feat = [single_window_feats(w.numpy(), selected_feature_names, sampling_rate) for w in tqdm(X_eval, desc="🧪 Extracting features")]
    return torch.tensor(np.stack(X_feat), dtype=torch.float32)

# === Load Model ===
class MLPOnly(nn.Module):
    def __init__(self, feat_dim=168, num_classes=2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feat_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.model(x)

def load_model(model_path, input_dim=168):
    model = MLPOnly(feat_dim=input_dim)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# === 4. Predict ===
def predict(model, X_feat, scaler):
    X_scaled = torch.tensor(scaler.transform(X_feat.numpy()), dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_scaled)
        probs = softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values
    return preds, confidences

# === 5. Generate annotations from predictions ===
def generate_annotations(preds, confidences, threshold=0.90, window_size=1000, sampling_rate=100):
    abnormal_windows = (preds == 1) & (confidences > threshold)
    onset_times = (abnormal_windows.nonzero().squeeze().numpy()) * (window_size / sampling_rate)
    if onset_times.ndim == 0:
        onset_times = [onset_times.item()]
    durations = [window_size / sampling_rate] * len(onset_times)
    descriptions = ['abnormal'] * len(onset_times)
    return mne.Annotations(onset=onset_times, duration=durations, description=descriptions)

# === 6. Save results to DB (placeholder) ===
def save_results_to_db(results_dict):
    print("[✅] Saving results to DB:", results_dict)
    return True

# === 7. Plot abnormal window with context, centered annotation ===
def plot_window(raw, abnormal_start_time, selected_duration, channels):
    # Ensure the plot window does not go beyond the raw data
    max_time = raw.times[-1]  # This is the total duration of the raw EEG data
 # ❌ Remove the annotation code
    # annotation_onset = plot_start + (plot_duration / 2) - (selected_duration / 2)
    # annotation = mne.Annotations(
    #     onset=[annotation_onset],
    #     duration=[selected_duration],
    #     description=['abnormal']
    # )
    # raw.set_annotations(annotation)
    # Adjust the abnormal_start_time if it exceeds the max_time
    if abnormal_start_time + selected_duration > max_time:
        abnormal_start_time = max_time - selected_duration

    # Now plot the window
    fig = raw.plot(
        start=abnormal_start_time,
        duration=selected_duration,
        picks=channels,
        scalings='auto',
        title=f'EEG at {abnormal_start_time}s - ({selected_duration}s)',
        show=False,
        n_channels=len(channels),
        clipping='transparent',
        show_scrollbars=True,
        block=False
    )
    return fig

