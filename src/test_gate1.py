"""
Test Gate 1 Verification Script
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 1)

Checks:
1. Disjoint partitions (Zero seed or timestamp leakage).
2. Actuator strain variance shift (p < 0.001) under fault onset.
3. Tensor dimensionality (N, w+1, D) with zero NaNs.
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from scipy import stats

PROCESSED_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def test_gate_1() -> bool:
    print("=" * 60)
    print("RUNNING TEST GATE 1: Ingestion & Control Signal Validation")
    print("=" * 60)

    tep_path = os.path.join(PROCESSED_DATA_DIR, "tep_processed.npz")
    deb_path = os.path.join(PROCESSED_DATA_DIR, "debutanizer_processed.npz")
    pronto_path = os.path.join(PROCESSED_DATA_DIR, "pronto_processed.npz")

    # 1. Check file existence
    if not os.path.exists(tep_path):
        print(f"FAIL: Missing {tep_path}")
        return False
    if not os.path.exists(deb_path):
        print(f"FAIL: Missing {deb_path}")
        return False
    if not os.path.exists(pronto_path):
        print(f"FAIL: Missing {pronto_path}")
        return False

    # 2. Inspect TEP archive
    tep_data = np.load(tep_path, allow_pickle=True)
    X_train_tep = tep_data["X_train"]
    y_train_tep = tep_data["y_train"]
    X_val_tep = tep_data["X_val"]
    y_val_tep = tep_data["y_val"]
    X_test_tep = tep_data["X_test"]
    y_test_tep = tep_data["y_test"]
    train_seeds = set(tep_data["train_seeds"])
    val_seeds = set(tep_data["val_seeds"])
    test_seeds = set(tep_data["test_seeds"])

    print(f"[TEP Shapes] Train: {X_train_tep.shape}, Val: {X_val_tep.shape}, Test: {X_test_tep.shape}")
    
    # Check 3D tensor shape (N, w+1, D)
    assert len(X_train_tep.shape) == 3, f"Expected 3D tensor, got {X_train_tep.shape}"
    assert X_train_tep.shape[1] == 6, f"Expected window length 6 (w=5 + 1), got {X_train_tep.shape[1]}"
    
    # Check zero NaNs
    nan_count_tep = np.isnan(X_train_tep).sum() + np.isnan(X_val_tep).sum() + np.isnan(X_test_tep).sum()
    print(f"[TEP NaN Check] Total NaNs: {nan_count_tep}")
    assert nan_count_tep == 0, "TEP arrays contain NaN values!"

    # Check zero seed leakage
    leakage_train_val = train_seeds & val_seeds
    leakage_train_test = train_seeds & test_seeds
    leakage_val_test = val_seeds & test_seeds
    print(f"[TEP Seed Leakage Check] Intersect(Train, Val)={len(leakage_train_val)}, Intersect(Train, Test)={len(leakage_train_test)}, Intersect(Val, Test)={len(leakage_val_test)}")
    assert len(leakage_train_val) == 0 and len(leakage_train_test) == 0 and len(leakage_val_test) == 0, "Seed leakage detected across TEP splits!"

    # 3. Inspect Debutanizer archive
    deb_data = np.load(deb_path, allow_pickle=True)
    X_train_deb = deb_data["X_train"]
    X_val_deb = deb_data["X_val"]
    X_test_deb = deb_data["X_test"]
    split_indices = deb_data["split_indices"]

    print(f"[Debutanizer Shapes] Train: {X_train_deb.shape}, Val: {X_val_deb.shape}, Test: {X_test_deb.shape}")
    nan_count_deb = np.isnan(X_train_deb).sum() + np.isnan(X_val_deb).sum() + np.isnan(X_test_deb).sum()
    print(f"[Debutanizer NaN Check] Total NaNs: {nan_count_deb}")
    assert nan_count_deb == 0, "Debutanizer arrays contain NaN values!"

    # Chronological forward split check
    assert split_indices[0] < split_indices[1] < split_indices[2], "Invalid chronological split ordering!"
    print(f"[Debutanizer Chronological Check] Forward-blocked cut points: {split_indices}")

    # 4. Statistical Shift Test on Actuator Strain Signals
    # Find actuator strain column indices in TEP
    feature_names = [str(f) for f in tep_data["feature_names"]]
    strain_indices = [i for i, f in enumerate(feature_names) if "_strain" in f]
    print(f"[TEP Actuator Strain Channels] Found {len(strain_indices)} strain features: {[feature_names[i] for i in strain_indices]}")
    assert len(strain_indices) >= 5, "Expected at least 5 actuator strain features!"

    # Extract strain values during normal (fault_id == 0) vs faulty (fault_id != 0)
    normal_mask = (y_train_tep == 0)
    faulty_mask = (y_train_tep != 0)

    # Test strain variance shift for condenser cooling valve (xmv_11_strain) or stripper steam (xmv_8_strain)
    p_values = []
    for idx in strain_indices:
        s_normal = X_train_tep[normal_mask, -1, idx]
        s_faulty = X_train_tep[faulty_mask, -1, idx]
        # Perform two-sample Kolmogorov-Smirnov test
        ks_stat, p_val = stats.ks_2samp(s_normal, s_faulty)
        p_values.append((feature_names[idx], p_val))

    significant_shifts = [name for name, p in p_values if p < 0.001]
    print(f"[Statistical Strain Shift Test (p < 0.001)] {len(significant_shifts)} / {len(strain_indices)} actuators showed p < 0.001 shift during fault onset.")
    for name, p in p_values:
        print(f"  - {name}: p-value = {p:.4e}")

    assert len(significant_shifts) >= 8, f"Expected majority of actuators to exhibit p < 0.001 shift, got {len(significant_shifts)}"

    # 5. Inspect PRONTO Multiphase Benchmark archive
    print("\n[5/5] Inspecting PRONTO Multiphase Separation Benchmark...")
    pronto_data = np.load(pronto_path, allow_pickle=True)
    X_train_p = pronto_data["X_train"]
    y_train_p = pronto_data["y_train"]
    X_val_p = pronto_data["X_val"]
    y_val_p = pronto_data["y_val"]
    X_test_p = pronto_data["X_test"]
    y_test_p = pronto_data["y_test"]
    p_features = [str(f) for f in pronto_data["feature_names"]]

    print(f"[PRONTO Shapes] Train: {X_train_p.shape}, Val: {X_val_p.shape}, Test: {X_test_p.shape}")
    assert len(X_train_p.shape) == 3 and X_train_p.shape[1] == 6, f"Invalid PRONTO tensor shape: {X_train_p.shape}"

    nan_count_p = np.isnan(X_train_p).sum() + np.isnan(X_val_p).sum() + np.isnan(X_test_p).sum()
    print(f"[PRONTO NaN Check] Total NaNs: {nan_count_p}")
    assert nan_count_p == 0, "PRONTO arrays contain NaN values!"

    # PRONTO Actuator strain shift test
    p_strain_indices = [i for i, f in enumerate(p_features) if f.startswith("strain_")]
    print(f"[PRONTO Actuators] Found {len(p_strain_indices)} valve strain features: {[p_features[i] for i in p_strain_indices]}")
    assert len(p_strain_indices) >= 5, "Expected >= 5 PRONTO valve strain features!"

    p_norm_mask = (y_train_p == 0)
    p_fault_mask = (y_train_p != 0)
    p_sig_shifts = 0
    for idx in p_strain_indices:
        s_norm = X_train_p[p_norm_mask, -1, idx]
        s_flt = X_train_p[p_fault_mask, -1, idx]
        ks_s, ks_p = stats.ks_2samp(s_norm, s_flt)
        if ks_p < 0.001:
            p_sig_shifts += 1
        print(f"  - PRONTO {p_features[idx]}: KS stat={ks_s:.4f}, p-val={ks_p:.4e}")

    assert p_sig_shifts == len(p_strain_indices), f"Expected all PRONTO actuators to show p < 0.001 shift, got {p_sig_shifts}/{len(p_strain_indices)}"
    print(f"  -> PASSED: {p_sig_shifts}/{len(p_strain_indices)} PRONTO actuators show statistically significant p < 0.001 strain shift.")

    print("=" * 60)
    print("STATUS: TEST_GATE_1_PASSED")
    print("All ingestion, leakage, and closed-loop control strain checks verified.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_gate_1()
    sys.exit(0 if success else 1)
