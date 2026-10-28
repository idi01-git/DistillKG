"""
SOTA Multi-Model Suite for Distillation Column FDD
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 3)

Trains and benchmarks three diverse architectures on multi-seed TEP tensors:
1. LightGBM (Gradient Boosted Trees with Tabular Temporal Aggregates)
2. 1D-CNN (Temporal Convolutional Network)
3. Spatio-Temporal Graph Attention Network (ST-GAT leveraging P&ID Adjacency)
"""

import os
import sys
import pickle
import numpy as np
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, average_precision_score, accuracy_score
from sklearn.preprocessing import label_binarize
from typing import Dict, Any, Tuple, List

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.kg_manager import DistillationKGManager

PROCESSED_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "tep_processed.npz")
PRONTO_PROCESSED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pronto_processed.npz")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

TARGET_FAULTS = [0, 1, 2, 5, 6, 11]
LABEL_MAP = {f: i for i, f in enumerate(TARGET_FAULTS)}
INV_LABEL_MAP = {i: f for i, f in enumerate(TARGET_FAULTS)}
NUM_CLASSES = len(TARGET_FAULTS)


def ensure_models_dir():
    os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_subsample_data(
    train_samples_per_class: int = 5000,
    eval_samples_per_class: int = 1500,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Loads TEP processed tensors and creates balanced splits across the verified core distillation faults:
    Fault 0: Fault-Free normal operation
    Fault 1: Feed A/C composition step disturbance
    Fault 2: Component B composition shift (VLE displacement)
    Fault 5: Condenser cooling water temperature step
    Fault 6: Feed A loss (Column weeping risk)
    Fault 11: Actuator cooling water valve sticking
    """
    data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
    X_train_raw = data["X_train"] # (N, 6, 55)
    y_train_raw = data["y_train"]
    X_val_raw = data["X_val"]
    y_val_raw = data["y_val"]
    X_test_raw = data["X_test"]
    y_test_raw = data["y_test"]
    feature_names = [str(f) for f in data["feature_names"]]

    def balance_split(X, y, n_per_class, split_seed):
        rng = np.random.RandomState(split_seed)
        indices = []
        for raw_cls in TARGET_FAULTS:
            cls_idx = np.where(y == raw_cls)[0]
            if len(cls_idx) > n_per_class:
                selected = rng.choice(cls_idx, size=n_per_class, replace=False)
            else:
                selected = cls_idx
            indices.extend(selected)
        indices = np.array(indices)
        rng.shuffle(indices)
        y_mapped = np.array([LABEL_MAP[val] for val in y[indices]], dtype=np.int64)
        return X[indices], y_mapped

    X_train, y_train = balance_split(X_train_raw, y_train_raw, train_samples_per_class, seed)
    X_val, y_val = balance_split(X_val_raw, y_val_raw, eval_samples_per_class, seed + 1)
    X_test, y_test = balance_split(X_test_raw, y_test_raw, eval_samples_per_class, seed + 2)

    print(f"Dataset Balanced for Core Benchmark Faults {TARGET_FAULTS}:")
    print(f"  Train: {X_train.shape}, classes={np.bincount(y_train)}")
    print(f"  Val:   {X_val.shape}, classes={np.bincount(y_val)}")
    print(f"  Test:  {X_test.shape}, classes={np.bincount(y_test)}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "feature_names": feature_names
    }


def extract_tabular_features(X_window: np.ndarray) -> np.ndarray:
    """
    Extracts multi-moment features across time window: (N, 6, 55) -> (N, 165)
    1. Current values: X[:, -1, :] (55)
    2. Window mean: X.mean(axis=1) (55)
    3. Window differential: X[:, -1, :] - X[:, 0, :] (55)
    """
    curr = X_window[:, -1, :]
    mean = X_window.mean(axis=1)
    diff = X_window[:, -1, :] - X_window[:, 0, :]
    return np.hstack([curr, mean, diff])


# ==============================================================================
# Model 1: LightGBM Classifier
# ==============================================================================
def train_lightgbm(
    data_dict: Dict[str, Any]
) -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    print("\n--- Training Model 1: LightGBM ---")
    X_tr_tab = extract_tabular_features(data_dict["X_train"])
    X_val_tab = extract_tabular_features(data_dict["X_val"])
    X_te_tab = extract_tabular_features(data_dict["X_test"])
    
    y_tr = data_dict["y_train"]
    y_val = data_dict["y_val"]
    y_te = data_dict["y_test"]

    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=NUM_CLASSES,
        n_estimators=150,
        learning_rate=0.08,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    clf.fit(
        X_tr_tab, y_tr,
        eval_set=[(X_val_tab, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
    )

    preds = clf.predict(X_te_tab)
    probs = clf.predict_proba(X_te_tab)

    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")
    y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))
    pr_auc = average_precision_score(y_bin, probs, average="macro")

    metrics = {"accuracy": float(acc), "macro_f1": float(f1), "pr_auc": float(pr_auc)}
    print(f"LightGBM Test Results: Accuracy={acc:.4f}, Macro-F1={f1:.4f}, PR-AUC={pr_auc:.4f}")

    ensure_models_dir()
    save_path = os.path.join(MODELS_DIR, "lgbm_model.bin")
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved LightGBM model to {save_path}")

    return clf, metrics


# ==============================================================================
# Model 2: 1D-CNN (Temporal Convolutional Network)
# ==============================================================================
class Distillation1DCNN(nn.Module):
    def __init__(self, in_channels: int = 55, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.25)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = self.dropout(h)
        h = self.pool(h).squeeze(-1)
        return self.fc(h)


def train_cnn(
    data_dict: Dict[str, Any],
    epochs: int = 14,
    batch_size: int = 128
) -> Tuple[Distillation1DCNN, Dict[str, float]]:
    print("\n--- Training Model 2: 1D-CNN (Temporal Convolutions) ---")
    
    X_tr = torch.tensor(data_dict["X_train"].transpose(0, 2, 1), dtype=torch.float32)
    y_tr = torch.tensor(data_dict["y_train"], dtype=torch.long)
    
    X_val = torch.tensor(data_dict["X_val"].transpose(0, 2, 1), dtype=torch.float32)
    y_val = torch.tensor(data_dict["y_val"], dtype=torch.long)
    
    X_te = torch.tensor(data_dict["X_test"].transpose(0, 2, 1), dtype=torch.float32)
    y_te = data_dict["y_test"]

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = Distillation1DCNN(in_channels=55, num_classes=NUM_CLASSES)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                val_loss += criterion(out, by).item() * len(bx)
            val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")
    y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))
    pr_auc = average_precision_score(y_bin, probs, average="macro")

    metrics = {"accuracy": float(acc), "macro_f1": float(f1), "pr_auc": float(pr_auc)}
    print(f"1D-CNN Test Results: Accuracy={acc:.4f}, Macro-F1={f1:.4f}, PR-AUC={pr_auc:.4f}")

    ensure_models_dir()
    save_path = os.path.join(MODELS_DIR, "cnn_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved 1D-CNN checkpoint to {save_path}")

    return model, metrics


# ==============================================================================
# Model 3: Spatio-Temporal Graph Attention Network (ST-GAT)
# ==============================================================================
def construct_pid_adjacency(feature_names: List[str]) -> torch.Tensor:
    kg = DistillationKGManager()
    g = kg.graph
    D = len(feature_names)
    A = np.eye(D, dtype=np.float32)

    for i, f1 in enumerate(feature_names):
        base_f1 = f1.replace("_err", "").replace("_strain", "")
        for j, f2 in enumerate(feature_names):
            if i == j:
                continue
            base_f2 = f2.replace("_err", "").replace("_strain", "")
            if g.has_node(base_f1) and g.has_node(base_f2):
                eq1 = g.nodes[base_f1].get("equipment", "")
                eq2 = g.nodes[base_f2].get("equipment", "")
                if eq1 and eq2 and (eq1 == eq2 or g.has_edge(eq1, eq2) or g.has_edge(eq2, eq1)):
                    A[i, j] = 1.0

    deg = A.sum(axis=1)
    deg_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    deg_inv_sqrt[deg == 0] = 0.0
    A_norm = deg_inv_sqrt[:, None] * A * deg_inv_sqrt[None, :]
    return torch.tensor(A_norm, dtype=torch.float32)


class DistillationSTGAT(nn.Module):
    def __init__(self, adj: torch.Tensor, in_channels: int = 6, hidden_dim: int = 64, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.register_buffer("adj", adj)
        self.num_nodes = adj.shape[0] # 55
        
        # Temporal convolution over window
        self.tconv = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Spatial Graph Convolutions with residual shortcuts
        self.gconv1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(self.num_nodes)
        self.gconv2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(self.num_nodes)
        
        # Attention pooling across nodes
        self.att = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (N, 55, 6)
        x_t = x.permute(0, 2, 1) # (N, 6, 55)
        h_t = F.relu(self.tconv(x_t)).permute(0, 2, 1) # (N, 55, hidden)
        
        h = torch.matmul(self.adj, h_t)
        h = F.relu(self.bn1(self.gconv1(h))) + h_t
        h = torch.matmul(self.adj, h)
        h = F.relu(self.bn2(self.gconv2(h))) + h
        
        scores = F.softmax(self.att(h), dim=1)
        pooled = torch.sum(h * scores, dim=1)
        return self.fc(pooled)


def train_gat(
    data_dict: Dict[str, Any],
    epochs: int = 18,
    batch_size: int = 128
) -> Tuple[DistillationSTGAT, Dict[str, float]]:
    print("\n--- Training Model 3: Spatio-Temporal GAT (Physical P&ID Topology) ---")
    
    adj = construct_pid_adjacency(data_dict["feature_names"])
    
    X_tr = torch.tensor(data_dict["X_train"].transpose(0, 2, 1), dtype=torch.float32)
    y_tr = torch.tensor(data_dict["y_train"], dtype=torch.long)
    
    X_val = torch.tensor(data_dict["X_val"].transpose(0, 2, 1), dtype=torch.float32)
    y_val = torch.tensor(data_dict["y_val"], dtype=torch.long)
    
    X_te = torch.tensor(data_dict["X_test"].transpose(0, 2, 1), dtype=torch.float32)
    y_te = data_dict["y_test"]

    train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    model = DistillationSTGAT(adj=adj, in_channels=6, hidden_dim=64, num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_weights = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(bx)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, by in val_loader:
                out = model(bx)
                val_loss += criterion(out, by).item() * len(bx)
            val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = model.state_dict().copy()

    if best_weights:
        model.load_state_dict(best_weights)
    model.eval()
    with torch.no_grad():
        logits = model(X_te)
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)

    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")
    y_bin = label_binarize(y_te, classes=list(range(NUM_CLASSES)))
    pr_auc = average_precision_score(y_bin, probs, average="macro")

    metrics = {"accuracy": float(acc), "macro_f1": float(f1), "pr_auc": float(pr_auc)}
    print(f"ST-GAT Test Results: Accuracy={acc:.4f}, Macro-F1={f1:.4f}, PR-AUC={pr_auc:.4f}")

    ensure_models_dir()
    save_path = os.path.join(MODELS_DIR, "gat_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Saved ST-GAT checkpoint to {save_path}")

    return model, metrics


def train_pronto_lightgbm() -> Tuple[lgb.LGBMClassifier, Dict[str, float]]:
    print("\n--- Training LightGBM on PRONTO Multiphase Benchmark ---")
    from sklearn.model_selection import train_test_split
    from src.data_loader import load_pronto_benchmark

    df = load_pronto_benchmark()

    # Fluid dynamic differential pressure drops across multiphase infrastructure
    df["dP_riser"] = df["PT417/OUT.CV"] - df["PT408/OUT.CV"]
    df["dP_pipe"] = df["PT312/OUT.CV"] - df["PT417/OUT.CV"]
    df["dP_sep"] = df["PT403/OUT.CV"] - df["PT501/OUT.CV"]
    df["err_air1"] = np.abs(df["FIC302/PID1/SP.CV"] - df["FT305/OUT.CV"]) / (df["FIC302/PID1/SP.CV"].clip(lower=1.0))
    df["err_water1"] = np.abs(df["FIC102/PID1/SP.CV"] - df["FT102/OUT.CV"]) / (df["FIC102/PID1/SP.CV"].clip(lower=0.01))

    # Multiphase hydrodynamic oscillation metrics (15-step rolling window)
    for col in ["PT417/OUT.CV", "PT408/OUT.CV", "dP_riser"]:
        df[col + "_roll_std"] = df[col].rolling(15, min_periods=1).std().fillna(0)
        df[col + "_roll_diff"] = df[col].diff().fillna(0)

    feat_cols = [
        "dP_riser", "dP_pipe", "dP_sep", "err_air1", "err_water1",
        "PIC501/PID1/OUT.CV", "LVC502-SR/PID1/OUT.CV", "FIC302/PID1/OUT.CV", "FIC102/PID1/OUT.CV",
        "PT417/OUT.CV_roll_std", "PT408/OUT.CV_roll_std", "dP_riser_roll_std",
        "PT417/OUT.CV_roll_diff", "PT408/OUT.CV_roll_diff"
    ]

    # Block splitting: 30-second contiguous physical process blocks
    df["block_id"] = np.arange(len(df)) // 30
    blocks = df.groupby(["block_id", "fault_id"]).first().reset_index()

    train_blocks, test_blocks = train_test_split(
        blocks["block_id"], test_size=0.25, random_state=42, stratify=blocks["fault_id"]
    )

    train_mask = df["block_id"].isin(train_blocks)
    test_mask = df["block_id"].isin(test_blocks)

    X_tr = df.loc[train_mask, feat_cols].values.astype(np.float32)
    y_tr = df.loc[train_mask, "fault_id"].values.astype(np.int64)
    X_te = df.loc[test_mask, feat_cols].values.astype(np.float32)
    y_te = df.loc[test_mask, "fault_id"].values.astype(np.int64)

    num_classes = len(np.unique(y_tr))
    clf = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=num_classes,
        n_estimators=100,
        learning_rate=0.05,
        verbose=-1,
        random_state=42
    )

    clf.fit(X_tr, y_tr)

    preds = clf.predict(X_te)
    probs = clf.predict_proba(X_te)

    acc = accuracy_score(y_te, preds)
    f1 = f1_score(y_te, preds, average="macro")
    y_bin = label_binarize(y_te, classes=list(range(num_classes)))
    pr_auc = average_precision_score(y_bin, probs, average="macro")

    metrics = {"accuracy": float(acc), "macro_f1": float(f1), "pr_auc": float(pr_auc)}
    print(f"PRONTO LightGBM Test Results: Accuracy={acc:.4f}, Macro-F1={f1:.4f}, PR-AUC={pr_auc:.4f}")

    ensure_models_dir()
    save_path = os.path.join(MODELS_DIR, "lgbm_pronto.bin")
    with open(save_path, "wb") as f:
        pickle.dump(clf, f)
    print(f"Saved PRONTO LightGBM model to {save_path}")

    # Save test partition for standalone test gate verification
    test_save_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "pronto_test_blocksplit.npz")
    np.savez_compressed(test_save_path, X_test=X_te, y_test=y_te, feature_names=np.array(feat_cols))

    return clf, metrics


def train_all_models() -> Dict[str, Dict[str, float]]:
    data_dict = load_and_subsample_data()
    
    lgbm_model, lgbm_metrics = train_lightgbm(data_dict)
    cnn_model, cnn_metrics = train_cnn(data_dict)
    gat_model, gat_metrics = train_gat(data_dict)
    pronto_model, pronto_metrics = train_pronto_lightgbm()

    print("\n" + "=" * 65)
    print("SOTA MULTI-MODEL BENCHMARK RESULTS (TEST SET):")
    print("=" * 65)
    print(f"{'Model Architecture':<25} | {'Accuracy':<10} | {'Macro-F1':<10} | {'PR-AUC':<10}")
    print("-" * 65)
    print(f"{'1. LightGBM (Gradient Boost)':<25} | {lgbm_metrics['accuracy']:<10.4f} | {lgbm_metrics['macro_f1']:<10.4f} | {lgbm_metrics['pr_auc']:<10.4f}")
    print(f"{'2. 1D-CNN (Temporal Conv)':<25} | {cnn_metrics['accuracy']:<10.4f} | {cnn_metrics['macro_f1']:<10.4f} | {cnn_metrics['pr_auc']:<10.4f}")
    print(f"{'3. ST-GAT (P&ID Graph)':<25} | {gat_metrics['accuracy']:<10.4f} | {gat_metrics['macro_f1']:<10.4f} | {gat_metrics['pr_auc']:<10.4f}")
    print(f"{'4. LightGBM (PRONTO)':<25} | {pronto_metrics['accuracy']:<10.4f} | {pronto_metrics['macro_f1']:<10.4f} | {pronto_metrics['pr_auc']:<10.4f}")
    print("=" * 65)

    return {
        "lgbm": lgbm_metrics,
        "cnn": cnn_metrics,
        "gat": gat_metrics,
        "pronto": pronto_metrics
    }


if __name__ == "__main__":
    results = train_all_models()
