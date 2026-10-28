"""
Test Gate 3 Verification Script
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 3)

Checks:
1. All three trained model checkpoints exist (LightGBM, 1D-CNN, ST-GAT).
2. All three models achieve Macro-F1 >= 92% and PR-AUC >= 0.90 on unseen test set.
3. Subsystem-grouped TreeSHAP attributes >= 65% of attribution mass to the physically damaged subsystem for benchmark faults.
"""

import os
import sys
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model_trainer import (
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    LABEL_MAP,
    NUM_CLASSES,
    extract_tabular_features,
    load_and_subsample_data,
    Distillation1DCNN,
    DistillationSTGAT,
    construct_pid_adjacency
)
from src.xai_explainer import SubsystemGroupedExplainer


def test_gate_3() -> bool:
    print("=" * 65)
    print("RUNNING TEST GATE 3: SOTA Multi-Model Suite & Grouped TreeSHAP")
    print("=" * 65)

    lgbm_path = os.path.join(MODELS_DIR, "lgbm_model.bin")
    cnn_path = os.path.join(MODELS_DIR, "cnn_model.pt")
    gat_path = os.path.join(MODELS_DIR, "gat_model.pt")

    # 1. Check Model Checkpoint Existence
    for p, name in [(lgbm_path, "LightGBM"), (cnn_path, "1D-CNN"), (gat_path, "ST-GAT")]:
        if not os.path.exists(p):
            print(f"FAIL: Missing {name} checkpoint at {p}")
            return False

    print("[Check 1: Checkpoints] All 3 model checkpoints verified on disk.")

    # 2. Evaluate Models on Unseen Test Split
    data_dict = load_and_subsample_data(train_samples_per_class=5000, eval_samples_per_class=1500)
    X_te = data_dict["X_test"]
    y_te = data_dict["y_test"]
    y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))

    # Evaluate LightGBM
    with open(lgbm_path, "rb") as f:
        lgbm = pickle.load(f)
    X_te_tab = extract_tabular_features(X_te)
    lgbm_preds = lgbm.predict(X_te_tab)
    lgbm_probs = lgbm.predict_proba(X_te_tab)
    lgbm_f1 = f1_score(y_te, lgbm_preds, average="macro")
    lgbm_pr_auc = average_precision_score(y_bin, lgbm_probs, average="macro")
    print(f"\n[Model 1: LightGBM] Macro-F1 = {lgbm_f1:.4f}, PR-AUC = {lgbm_pr_auc:.4f}")
    assert lgbm_f1 >= 0.92, f"LightGBM Macro-F1 {lgbm_f1:.4f} < 0.92 threshold!"
    assert lgbm_pr_auc >= 0.90, f"LightGBM PR-AUC {lgbm_pr_auc:.4f} < 0.90 threshold!"

    # Evaluate 1D-CNN
    cnn = Distillation1DCNN(in_channels=55, num_classes=NUM_CLASSES)
    cnn.load_state_dict(torch.load(cnn_path, weights_only=True))
    cnn.eval()
    X_te_tensor = torch.tensor(X_te.transpose(0, 2, 1), dtype=torch.float32)
    with torch.no_grad():
        cnn_logits = cnn(X_te_tensor)
        cnn_probs = F.softmax(cnn_logits, dim=-1).cpu().numpy()
        cnn_preds = np.argmax(cnn_probs, axis=-1)
    cnn_f1 = f1_score(y_te, cnn_preds, average="macro")
    cnn_pr_auc = average_precision_score(y_bin, cnn_probs, average="macro")
    print(f"[Model 2: 1D-CNN]   Macro-F1 = {cnn_f1:.4f}, PR-AUC = {cnn_pr_auc:.4f}")
    assert cnn_f1 >= 0.92, f"1D-CNN Macro-F1 {cnn_f1:.4f} < 0.92 threshold!"
    assert cnn_pr_auc >= 0.90, f"1D-CNN PR-AUC {cnn_pr_auc:.4f} < 0.90 threshold!"

    # Evaluate ST-GAT
    adj = construct_pid_adjacency(data_dict["feature_names"])
    gat = DistillationSTGAT(adj=adj, in_channels=6, hidden_dim=64, num_classes=NUM_CLASSES)
    gat.load_state_dict(torch.load(gat_path, weights_only=True))
    gat.eval()
    with torch.no_grad():
        gat_logits = gat(X_te_tensor)
        gat_probs = F.softmax(gat_logits, dim=-1).cpu().numpy()
        gat_preds = np.argmax(gat_probs, axis=-1)
    gat_f1 = f1_score(y_te, gat_preds, average="macro")
    gat_pr_auc = average_precision_score(y_bin, gat_probs, average="macro")
    print(f"[Model 3: ST-GAT]   Macro-F1 = {gat_f1:.4f}, PR-AUC = {gat_pr_auc:.4f}")
    assert gat_f1 >= 0.92, f"ST-GAT Macro-F1 {gat_f1:.4f} < 0.92 threshold!"
    assert gat_pr_auc >= 0.90, f"ST-GAT PR-AUC {gat_pr_auc:.4f} < 0.90 threshold!"

    print("\n[Check 2: SOTA Benchmarks] PASSED: All 3 models achieve Macro-F1 >= 92% and PR-AUC >= 0.90.")

    # 3. Subsystem-Grouped TreeSHAP Localization Check
    print("\n[Check 3: Subsystem-Grouped TreeSHAP] Evaluating attribution mass concentration...")
    explainer = SubsystemGroupedExplainer(model_path=lgbm_path)
    shap_results = explainer.benchmark_subsystem_localization(test_samples_per_fault=150)

    masses = []
    for fault_id, res in shap_results.items():
        mass = res["damaged_subsystem_mass"]
        expected_sub = res["expected_subsystem"]
        masses.append(mass)
        dist = res["full_distribution"]
        # Verify that the damaged subsystem is the dominant subsystem
        dominant_sub = max(dist.keys(), key=lambda k: dist[k])
        print(f"  - Fault {fault_id:>2} ({expected_sub}): Attributed Mass = {mass:.2f}% | Dominant Subsystem = {dominant_sub}")
        assert dominant_sub == expected_sub, f"Fault {fault_id} dominant subsystem {dominant_sub} != expected {expected_sub}!"

    avg_mass = float(np.mean(masses))
    print(f"\n[Check 3: Grouped TreeSHAP] Mean Attribution Mass across benchmark faults: {avg_mass:.2f}% (Threshold >= 65.0%)")
    assert avg_mass >= 65.0, f"Mean attribution mass {avg_mass:.2f}% < 65.0% threshold!"
    print("[Check 3: Grouped TreeSHAP] PASSED: Collinear multi-tray smearing eliminated (Mean attribution mass >= 65% on damaged subsystems).")

    # 4. PRONTO LightGBM Benchmark Check
    pronto_path = os.path.join(MODELS_DIR, "lgbm_pronto.bin")
    assert os.path.exists(pronto_path), f"Missing PRONTO LightGBM checkpoint at {pronto_path}"
    with open(pronto_path, "rb") as f:
        pronto_clf = pickle.load(f)
    test_save_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pronto_test_blocksplit.npz")
    assert os.path.exists(test_save_path), f"Missing PRONTO test split at {test_save_path}"
    pronto_data = np.load(test_save_path, allow_pickle=True)
    X_pr_te = pronto_data["X_test"]
    y_pr_te = pronto_data["y_test"]
    pr_preds = pronto_clf.predict(X_pr_te)
    pr_probs = pronto_clf.predict_proba(X_pr_te)
    pr_f1 = f1_score(y_pr_te, pr_preds, average="macro")
    num_pr_classes = len(np.unique(y_pr_te))
    y_pr_bin = label_binarize(y_pr_te, classes=list(range(num_pr_classes)))
    pr_pr_auc = average_precision_score(y_pr_bin, pr_probs, average="macro")
    print(f"\n[Check 4: PRONTO LightGBM Benchmark] Macro-F1 = {pr_f1:.4f}, PR-AUC = {pr_pr_auc:.4f}")
    assert pr_f1 >= 0.90, f"PRONTO LightGBM Macro-F1 {pr_f1:.4f} < 0.90 threshold!"
    assert pr_pr_auc >= 0.95, f"PRONTO LightGBM PR-AUC {pr_pr_auc:.4f} < 0.95 threshold!"
    print("[Check 4: PRONTO LightGBM] PASSED: PRONTO Macro-F1 >= 90% and PR-AUC >= 0.95 verified.")

    print("\n" + "=" * 65)
    print("STATUS: TEST_GATE_3_PASSED")
    print("Multi-model training and subsystem-grouped explainability verified.")
    print("=" * 65)
    return True


if __name__ == "__main__":
    success = test_gate_3()
    sys.exit(0 if success else 1)
