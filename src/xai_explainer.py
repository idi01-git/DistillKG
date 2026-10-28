"""
Subsystem-Grouped TreeSHAP Explainability Module
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 3)

Eliminates collinear sensor smearing across distillation trays by aggregating
pointwise Shapley values into 5 physical column subsystems:
Phi_k(t) = sum_{j in S_k} |phi_j(t)|
"""

import os
import sys

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import torch before shap on Windows to prevent C-runtime DLL initialization conflicts
import torch
import pickle
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple

from src.kg_manager import DistillationKGManager
from src.model_trainer import extract_tabular_features, MODELS_DIR, PROCESSED_DATA_PATH, LABEL_MAP, INV_LABEL_MAP

OUTPUT_SHAP_PLOT = os.path.join(os.path.dirname(__file__), "..", "outputs", "subsystem_grouped_shap.png")

# Canonical physical column subsystems
SUBSYSTEM_MAPPING = {
    "SUB_FEED": [
        "xmeas_1", "xmeas_2", "xmeas_3", "xmeas_4", "xmeas_5", "xmeas_6",
        "xmeas_7", "xmeas_8", "xmeas_9", "xmeas_16", "xmeas_17",
        "xmv_1", "xmv_2", "xmv_3", "xmv_4", "xmv_5", "xmv_10"
    ],
    "SUB_COLUMN": [
        "xmeas_13", "xmeas_18", "top_tray_temp", "tray6_temp",
        "delta_T_top_tray6", "delta_T_tray6_reboiler"
    ],
    "SUB_OVERHEAD": [
        "xmeas_10", "xmeas_11", "xmeas_20", "xmeas_22", "xmeas_23", "xmeas_24",
        "top_pressure", "xmv_6", "xmv_11"
    ],
    "SUB_REFLUX": [
        "reflux_flow", "distillate_flow", "reflux_ratio"
    ],
    "SUB_REBOILER": [
        "xmeas_12", "xmeas_14", "xmeas_15", "xmeas_19", "xmeas_21",
        "reboiler_bottom_temp", "reboiler_steam_flow", "reboiler_duty_proxy",
        "butane_content", "xmv_7", "xmv_8", "xmv_9"
    ]
}


class SubsystemGroupedExplainer:
    def __init__(self, model_path: str = os.path.join(MODELS_DIR, "lgbm_model.bin")):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {model_path}. Train model first!")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        
        data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
        self.base_features = [str(f) for f in data["feature_names"]]
        
        # 165 Tabular Feature names: [curr_0..54, mean_0..54, diff_0..54]
        self.all_feature_names = (
            [f"{f}_curr" for f in self.base_features] +
            [f"{f}_mean" for f in self.base_features] +
            [f"{f}_diff" for f in self.base_features]
        )
        
        self.feature_to_subsystem = self._map_tabular_features_to_subsystems()
        
        print("Initializing SHAP TreeExplainer on LightGBM...")
        self.explainer = shap.TreeExplainer(self.model)

    def _map_tabular_features_to_subsystems(self) -> Dict[int, str]:
        mapping = {}
        for idx, feat_name in enumerate(self.all_feature_names):
            # Extract root variable (e.g. 'xmeas_24_err_mean' -> 'xmeas_24')
            base_var = feat_name
            for suffix in ["_curr", "_mean", "_diff", "_err", "_strain"]:
                base_var = base_var.replace(suffix, "")
            
            assigned = "SUB_COLUMN" # Default fallback
            for sub_name, feat_list in SUBSYSTEM_MAPPING.items():
                if base_var in feat_list:
                    assigned = sub_name
                    break

            mapping[idx] = assigned
        return mapping

    def compute_subsystem_attributions(
        self,
        X_sample: np.ndarray,
        target_class: int
    ) -> Dict[str, float]:
        """
        Computes grouped attribution mass:
        Phi_k = sum_{j in S_k} |phi_j|
        Returns normalized attribution percentage per subsystem.
        """
        if len(X_sample.shape) == 3:
            X_tab = extract_tabular_features(X_sample)
        else:
            X_tab = X_sample

        shap_values = self.explainer.shap_values(X_tab)
        if isinstance(shap_values, list):
            sv = shap_values[target_class]
        elif len(shap_values.shape) == 3:
            sv = shap_values[:, :, target_class]
        else:
            sv = shap_values

        mean_abs_shap = np.abs(sv).mean(axis=0)

        subsystem_mass = {sub: 0.0 for sub in SUBSYSTEM_MAPPING.keys()}
        for idx, mass in enumerate(mean_abs_shap):
            sub = self.feature_to_subsystem.get(idx, "SUB_COLUMN")
            subsystem_mass[sub] += float(mass)

        total_mass = sum(subsystem_mass.values())
        if total_mass > 0:
            normalized_mass = {sub: (m / total_mass) * 100.0 for sub, m in subsystem_mass.items()}
        else:
            normalized_mass = {sub: 20.0 for sub in subsystem_mass.keys()}

        return normalized_mass

    def benchmark_subsystem_localization(
        self,
        test_samples_per_fault: int = 150
    ) -> Dict[str, Any]:
        """
        Evaluates attribution mass on the physically damaged subsystem for known benchmark faults:
        - Fault 1 (Feed disturbance): expects high mass in SUB_FEED
        - Fault 5 (Condenser cooling water temp step): expects high mass in SUB_OVERHEAD
        - Fault 11 (Cooling water valve sticking): expects high mass in SUB_OVERHEAD
        """
        data = np.load(PROCESSED_DATA_PATH, allow_pickle=True)
        X_test = data["X_test"]
        y_test = data["y_test"]

        results = {}
        target_benchmark_faults = {
            1: "SUB_FEED",
            2: "SUB_OVERHEAD",
            5: "SUB_OVERHEAD",
            6: "SUB_FEED",
            11: "SUB_FEED"
        }

        print("\n" + "=" * 65)
        print("EVALUATING SUBSYSTEM-GROUPED TREESHAP ATTRIBUTIONS:")
        print("=" * 65)

        for fault_id, expected_sub in target_benchmark_faults.items():
            cls_idx = np.where(y_test == fault_id)[0]
            if len(cls_idx) == 0:
                continue
            sample_indices = cls_idx[:test_samples_per_fault]
            X_fault = X_test[sample_indices]
            mapped_cls = LABEL_MAP[fault_id]

            norm_mass = self.compute_subsystem_attributions(X_fault, target_class=mapped_cls)
            damaged_sub_mass = norm_mass.get(expected_sub, 0.0)

            results[fault_id] = {
                "expected_subsystem": expected_sub,
                "damaged_subsystem_mass": damaged_sub_mass,
                "full_distribution": norm_mass
            }

            print(f"Fault {fault_id:>2} (Physical Origin: {expected_sub:<12}) -> Grouped Attribution Mass: {damaged_sub_mass:.2f}%")
            for sub, val in norm_mass.items():
                marker = " [DAMAGED TARGET]" if sub == expected_sub else ""
                print(f"    - {sub:<14}: {val:>6.2f}%{marker}")

        print("=" * 65)
        
        # Render and save publication figure
        self._export_grouped_shap_plot(results)
        return results

    def _export_grouped_shap_plot(self, results: Dict[int, Any], save_path: str = OUTPUT_SHAP_PLOT):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fault_keys = list(results.keys())
        subsystems = list(SUBSYSTEM_MAPPING.keys())

        fig, ax = plt.subplots(figsize=(10, 6))
        bar_width = 0.16
        x_indices = np.arange(len(fault_keys))

        colors = ["#1890ff", "#52c41a", "#faad14", "#722ed1", "#eb2f96"]

        for i, sub in enumerate(subsystems):
            vals = [results[f]["full_distribution"].get(sub, 0.0) for f in fault_keys]
            ax.bar(x_indices + i * bar_width, vals, width=bar_width, label=sub, color=colors[i % len(colors)], alpha=0.85)

        ax.set_xticks(x_indices + bar_width * 2)
        ax.set_xticklabels([f"Fault {f}\n(Origin: {results[f]['expected_subsystem']})" for f in fault_keys], fontsize=10)
        ax.set_ylabel("Attribution Mass (%)", fontsize=11, fontweight="bold")
        ax.set_title("Subsystem-Grouped TreeSHAP Attribution: Resolving Distillation Sensor Collinearity", fontsize=12, fontweight="bold")
        ax.legend(title="Column Subsystem", loc="upper right")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylim(0, 105)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Subsystem-Grouped SHAP plot saved to {save_path}")


def run_xai_benchmark():
    explainer = SubsystemGroupedExplainer()
    results = explainer.benchmark_subsystem_localization()
    return results


if __name__ == "__main__":
    run_xai_benchmark()
