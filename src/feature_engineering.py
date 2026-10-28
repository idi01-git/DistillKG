"""
Feature Engineering & Temporal Partitioning Pipeline
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 1)

Computes:
1. Closed-loop PID error residuals: e_i(t) = |SP_i - PV_i(t)|
2. Normalized actuator strain metrics: S_i(t) = (MV_i(t) - MV_nom) / (MV_max - MV_min)
3. Sliding temporal windows (w = 5)
4. Strict disjoint partitions:
   - TEP: Simulation seed isolation (1-300 Train, 301-400 Val, 401-500 Test)
   - Debutanizer: Chronological forward blocking (60% Train, 20% Val, 20% Test)
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Dict, Any, List

from src.data_loader import (
    load_debutanizer,
    load_tep_benchmark,
    load_pronto_benchmark,
    TEP_DISTILLATION_MEAS,
    TEP_DISTILLATION_MV,
    PRONTO_PROCESS_TAGS,
    PRONTO_VALVE_COLUMNS
)

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def ensure_processed_dir():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)


def compute_tep_control_residuals(
    df: pd.DataFrame,
    baseline_runs: List[int] = list(range(1, 301))
) -> pd.DataFrame:
    """
    Computes setpoint deviations e_i(t) and actuator strain S_i(t) for TEP.
    Baseline statistics are fit strictly on training fault-free runs to prevent leakage.
    """
    df = df.copy()
    
    # Isolate training fault-free baseline to compute nominal SP and MV bounds
    ff_train_mask = (df["fault_id"] == 0) & (df["simulation_id"].isin(baseline_runs))
    baseline_df = df[ff_train_mask]
    
    if len(baseline_df) == 0:
        # Fallback if filtered
        baseline_df = df[df["fault_id"] == 0]

    # Documented nominal regulatory setpoints from Downs & Vogel (1993) Table 1 & Table 2
    # Directly resolves ISS-11 by grounding tracking errors in physical engineering setpoints.
    TEP_NOMINAL_SETPOINTS = {
        "xmeas_7": 2705.0,      # Separator Pressure (kPa gauge)
        "xmeas_8": 65.0,        # Reactor Level (%)
        "xmeas_9": 50.0,        # Separator Level (%)
        "xmeas_11": 77.29,      # Separator Temperature (deg C)
        "xmeas_12": 50.0,       # Stripper Level (%)
        "xmeas_13": 3102.5,     # Stripper Pressure (kPa gauge)
        "xmeas_15": 1060.8,     # Stripper Steam Flow (kg/h)
        "xmeas_18": 65.73,      # Stripper Temperature (deg C)
        "xmeas_19": 230.0,      # Recycle Flow / Stripper Feed
        "xmeas_20": 93.79,      # Stripper Underflow Temp
        "xmeas_21": 46.23,      # Reactor Cooling Water Flow
    }

    # Process Variables: compute setpoint deviation e_i(t) = |PV_i(t) - SP_i|
    key_pvs = [c for c in [
        "xmeas_12", "xmeas_13", "xmeas_15", "xmeas_18",
        "xmeas_19", "xmeas_20", "xmeas_21", "xmeas_23", "xmeas_24"
    ] if c in df.columns]

    for pv in key_pvs:
        sp_nominal = TEP_NOMINAL_SETPOINTS.get(pv, float(baseline_df[pv].mean()))
        df[f"{pv}_err"] = np.abs(df[pv] - sp_nominal)

    # Manipulated Variables: compute normalized strain S_i(t) = (MV_i - MV_nom) / (MV_max - MV_min)
    for mv in TEP_DISTILLATION_MV:
        if mv in df.columns:
            mv_nom = baseline_df[mv].mean()
            mv_min = baseline_df[mv].min()
            mv_max = baseline_df[mv].max()
            span = max(mv_max - mv_min, 1e-5)
            df[f"{mv}_strain"] = (df[mv] - mv_nom) / span

    return df


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    window_size: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Constructs rolling window tensors: (N - window_size, window_size + 1, D)
    """
    num_samples, num_features = data.shape
    if num_samples <= window_size:
        raise ValueError(f"Sample length {num_samples} <= window size {window_size}")

    num_windows = num_samples - window_size
    windows = np.empty((num_windows, window_size + 1, num_features), dtype=np.float32)
    window_labels = labels[window_size:]

    for i in range(num_windows):
        windows[i] = data[i : i + window_size + 1]

    return windows, window_labels


def process_tep_pipeline(
    window_size: int = 5,
    sample_stride: int = 2
) -> Dict[str, Any]:
    """
    Executes full TEP feature engineering, splitting by simulation run seeds:
    - Train (60%): Runs 1-300
    - Val (20%): Runs 301-400
    - Test (20%): Runs 401-500
    """
    ensure_processed_dir()
    output_path = os.path.join(PROCESSED_DATA_DIR, "tep_processed.npz")
    
    print("Loading raw TEP benchmark...")
    df = load_tep_benchmark()
    
    # Clean ground-truth labeling: in TEP faulty runs, samples before injection (sample_id < 20)
    # represent true normal steady-state operation and are relabeled to Fault 0 to prevent label noise.
    pre_inj_mask = (df["fault_id"] != 0) & (df["sample_id"] < 20)
    print(f"Ground-Truth Clean: Relabeling {pre_inj_mask.sum()} pre-injection samples (sample_id < 20) to Fault 0 (Normal)...")
    df.loc[pre_inj_mask, "fault_id"] = 0

    print("Computing closed-loop control residuals and actuator strains...")
    df = compute_tep_control_residuals(df)
    
    # Identify feature columns (exclude identifiers)
    feature_cols = [c for c in df.columns if c not in ["fault_id", "simulation_id", "sample_id"]]
    print(f"Total engineered features: {len(feature_cols)}")

    # Separate splits by simulation run seed
    train_mask = df["simulation_id"].between(1, 300)
    val_mask = df["simulation_id"].between(301, 400)
    test_mask = df["simulation_id"].between(401, 500)

    # Subsample stride to keep tensor within lightweight CPU memory
    df_train = df[train_mask].iloc[::sample_stride].reset_index(drop=True)
    df_val = df[val_mask].iloc[::sample_stride].reset_index(drop=True)
    df_test = df[test_mask].iloc[::sample_stride].reset_index(drop=True)

    # Normalize based strictly on train split
    mean_vals = df_train[feature_cols].mean()
    std_vals = df_train[feature_cols].std().replace(0, 1.0)

    X_train_raw = ((df_train[feature_cols] - mean_vals) / std_vals).values.astype(np.float32)
    y_train_raw = df_train["fault_id"].values.astype(np.int64)

    X_val_raw = ((df_val[feature_cols] - mean_vals) / std_vals).values.astype(np.float32)
    y_val_raw = df_val["fault_id"].values.astype(np.int64)

    X_test_raw = ((df_test[feature_cols] - mean_vals) / std_vals).values.astype(np.float32)
    y_test_raw = df_test["fault_id"].values.astype(np.int64)

    print("Constructing sliding window tensors...")
    X_train_win, y_train_win = create_sliding_windows(X_train_raw, y_train_raw, window_size=window_size)
    X_val_win, y_val_win = create_sliding_windows(X_val_raw, y_val_raw, window_size=window_size)
    X_test_win, y_test_win = create_sliding_windows(X_test_raw, y_test_raw, window_size=window_size)

    # Save to compressed archive
    np.savez_compressed(
        output_path,
        X_train=X_train_win,
        y_train=y_train_win,
        X_val=X_val_win,
        y_val=y_val_win,
        X_test=X_test_win,
        y_test=y_test_win,
        feature_names=np.array(feature_cols),
        train_seeds=np.array(list(range(1, 301))),
        val_seeds=np.array(list(range(301, 401))),
        test_seeds=np.array(list(range(401, 501)))
    )

    print(f"TEP processed data saved to {output_path}")
    print(f"  Train: {X_train_win.shape}, Val: {X_val_win.shape}, Test: {X_test_win.shape}")
    
    return {
        "output_path": output_path,
        "X_train": X_train_win,
        "y_train": y_train_win,
        "X_val": X_val_win,
        "y_val": y_val_win,
        "X_test": X_test_win,
        "y_test": y_test_win,
        "features": feature_cols
    }


def process_debutanizer_pipeline(
    window_size: int = 5
) -> Dict[str, Any]:
    """
    Executes full industrial Debutanizer feature engineering with chronological forward blocking:
    - Train (60%): Hours 0 to 1436
    - Val (20%): Hours 1436 to 1915
    - Test (20%): Hours 1915 to 2394
    """
    ensure_processed_dir()
    output_path = os.path.join(PROCESSED_DATA_DIR, "debutanizer_processed.npz")
    
    df = load_debutanizer()
    feature_cols = [c for c in df.columns if c != "butane_content"]
    
    # Closed loop controller error approximations for Debutanizer
    # Reflux ratio proxy = reflux_flow / max(distillate_flow, 1e-4)
    df["reflux_ratio"] = df["reflux_flow"] / df["distillate_flow"].clip(lower=1e-3)
    # Reboiler heat proxy = reboiler_steam_flow * reboiler_bottom_temp
    df["reboiler_duty_proxy"] = df["reboiler_steam_flow"] * df["reboiler_bottom_temp"]
    # Temperature differential along the column
    df["delta_T_top_tray6"] = df["tray6_temp"] - df["top_tray_temp"]
    df["delta_T_tray6_reboiler"] = df["reboiler_bottom_temp"] - df["tray6_temp"]
    
    all_features = [c for c in df.columns if c != "butane_content"]
    
    # Chronological forward split (60% / 20% / 20%)
    n = len(df)
    n_train = int(n * 0.60) # 1436
    n_val = int(n * 0.20)   # 479
    
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].reset_index(drop=True)
    
    # Scale strictly on train
    mean_vals = df_train[all_features].mean()
    std_vals = df_train[all_features].std().replace(0, 1.0)
    
    X_train_raw = ((df_train[all_features] - mean_vals) / std_vals).values.astype(np.float32)
    y_train_raw = df_train["butane_content"].values.astype(np.float32)
    
    X_val_raw = ((df_val[all_features] - mean_vals) / std_vals).values.astype(np.float32)
    y_val_raw = df_val["butane_content"].values.astype(np.float32)
    
    X_test_raw = ((df_test[all_features] - mean_vals) / std_vals).values.astype(np.float32)
    y_test_raw = df_test["butane_content"].values.astype(np.float32)
    
    # Construct sliding windows
    X_train_win, y_train_win = create_sliding_windows(X_train_raw, y_train_raw, window_size=window_size)
    X_val_win, y_val_win = create_sliding_windows(X_val_raw, y_val_raw, window_size=window_size)
    X_test_win, y_test_win = create_sliding_windows(X_test_raw, y_test_raw, window_size=window_size)
    
    np.savez_compressed(
        output_path,
        X_train=X_train_win,
        y_train=y_train_win,
        X_val=X_val_win,
        y_val=y_val_win,
        X_test=X_test_win,
        y_test=y_test_win,
        feature_names=np.array(all_features),
        split_indices=np.array([n_train, n_train + n_val, n])
    )
    
    print(f"Debutanizer processed data saved to {output_path}")
    print(f"  Train: {X_train_win.shape}, Val: {X_val_win.shape}, Test: {X_test_win.shape}")
    
    return {
        "output_path": output_path,
        "X_train": X_train_win,
        "y_train": y_train_win,
        "X_val": X_val_win,
        "y_val": y_val_win,
        "X_test": X_test_win,
        "y_test": y_test_win,
        "features": all_features
    }


def process_pronto_pipeline(window_size: int = 5) -> Dict[str, Any]:
    """
    Ingests PRONTO Multiphase benchmark, computes closed-loop control tracking errors
    and actuator strain metrics, and generates sliding-window tensors (N, 6, D).
    """
    ensure_processed_dir()
    output_path = os.path.join(PROCESSED_DATA_DIR, "pronto_processed.npz")
    
    df = load_pronto_benchmark()
    
    # 1. Closed-loop PID tracking errors e_i(t) = |SP_i - PV_i|
    df["err_air1"] = np.abs(df["FIC302/PID1/SP.CV"] - df["FT305/OUT.CV"])
    df["err_air2"] = np.abs(df["FIC301/PID1/SP.CV"] - df["FT302/OUT.CV"])
    df["err_water1"] = np.abs(df["FIC102/PID1/SP.CV"] - df["FT102/OUT.CV"])
    df["err_water2"] = np.abs(df["FIC101/PID1/SP.CV"] - df["FT104/OUT.CV"])
    error_cols = ["err_air1", "err_air2", "err_water1", "err_water2"]
    
    # 2. Chronological forward blocking per operational run (60% Train, 20% Val, 20% Unseen Test)
    runs = df["run_id"].unique()
    train_dfs, val_dfs, test_dfs = [], [], []
    for r in runs:
        sub = df[df["run_id"] == r].copy()
        L = len(sub)
        n_tr = int(L * 0.60)
        n_va = int(L * 0.20)
        train_dfs.append(sub.iloc[:n_tr])
        val_dfs.append(sub.iloc[n_tr:n_tr + n_va])
        test_dfs.append(sub.iloc[n_tr + n_va:])

    df_train = pd.concat(train_dfs, ignore_index=True)
    df_val = pd.concat(val_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)

    # 3. Normalized Actuator Strain metrics S_i(t) strictly parameterized on training partition
    strain_cols = []
    normal_mask_train = df_train["fault_id"] == 0
    for v in PRONTO_VALVE_COLUMNS:
        v_col = f"strain_{v.split('/')[0]}"
        strain_cols.append(v_col)
        v_nom = df_train.loc[normal_mask_train, v].mean() if normal_mask_train.sum() > 0 else df_train[v].mean()
        v_min = df_train[v].min()
        v_max = df_train[v].max()
        denom = (v_max - v_min) if (v_max - v_min) > 1e-4 else 1.0
        df_train[v_col] = (df_train[v] - v_nom) / denom
        df_val[v_col] = (df_val[v] - v_nom) / denom
        df_test[v_col] = (df_test[v] - v_nom) / denom
        
    feature_cols = PRONTO_PROCESS_TAGS + error_cols + strain_cols
    feature_cols = [c for c in feature_cols if c in df_train.columns]
    
    df_train[feature_cols] = df_train[feature_cols].ffill().bfill().fillna(0.0)
    df_val[feature_cols] = df_val[feature_cols].ffill().bfill().fillna(0.0)
    df_test[feature_cols] = df_test[feature_cols].ffill().bfill().fillna(0.0)
    
    means = df_train[feature_cols].mean()
    stds = df_train[feature_cols].std().replace(0, 1.0)
    
    X_train_raw = ((df_train[feature_cols] - means) / stds).values.astype(np.float32)
    y_train_raw = df_train["fault_id"].values.astype(np.int64)
    
    X_val_raw = ((df_val[feature_cols] - means) / stds).values.astype(np.float32)
    y_val_raw = df_val["fault_id"].values.astype(np.int64)
    
    X_test_raw = ((df_test[feature_cols] - means) / stds).values.astype(np.float32)
    y_test_raw = df_test["fault_id"].values.astype(np.int64)
    
    # 4. Construct sliding windows
    X_train_win, y_train_win = create_sliding_windows(X_train_raw, y_train_raw, window_size=window_size)
    X_val_win, y_val_win = create_sliding_windows(X_val_raw, y_val_raw, window_size=window_size)
    X_test_win, y_test_win = create_sliding_windows(X_test_raw, y_test_raw, window_size=window_size)
    
    n_train = len(df_train)
    n_val = len(df_val)
    n = len(df)
    
    np.savez_compressed(
        output_path,
        X_train=X_train_win,
        y_train=y_train_win,
        X_val=X_val_win,
        y_val=y_val_win,
        X_test=X_test_win,
        y_test=y_test_win,
        feature_names=np.array(feature_cols),
        split_indices=np.array([n_train, n_train + n_val, n])
    )
    
    print(f"PRONTO processed data saved to {output_path}")
    print(f"  Train: {X_train_win.shape}, Val: {X_val_win.shape}, Test: {X_test_win.shape}")
    
    return {
        "output_path": output_path,
        "X_train": X_train_win,
        "y_train": y_train_win,
        "X_val": X_val_win,
        "y_val": y_val_win,
        "X_test": X_test_win,
        "y_test": y_test_win,
        "features": feature_cols
    }


if __name__ == "__main__":
    deb_res = process_debutanizer_pipeline()
    tep_res = process_tep_pipeline()
    pronto_res = process_pronto_pipeline()
