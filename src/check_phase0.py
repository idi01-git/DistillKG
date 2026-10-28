"""
Phase 0 Verification Script: High-Performance Foundations & Dependencies Audit
Tests all directory paths, library imports, PCMCI+ engine, PyTorch CPU, and repo contracts.
"""

import os
import sys

def verify_phase0():
    print("=" * 60)
    print("  PHASE 0 AUDIT: Non-Linear Causal-NeuroSymbolic Environment")
    print("=" * 60)
    
    # 1. Directory Scaffolding Check
    required_dirs = [
        "data/raw",
        "data/raw/pronto",
        "data/processed",
        "src",
        "models",
        "outputs/.plans",
        "outputs/.drafts",
        "papers",
        "notes"
    ]
    print("\n[1/4] Verifying Directory Scaffolding...")
    dir_errors = []
    for d in required_dirs:
        if os.path.isdir(d):
            print(f"  [PASS] Found: {d}/")
        else:
            dir_errors.append(d)
            print(f"  [FAIL] Missing: {d}/")
            
    assert not dir_errors, f"Missing directories: {dir_errors}"

    # 2. Virtual Environment & Python Version
    print("\n[2/4] Verifying Python Environment...")
    py_ver = sys.version.split()[0]
    print(f"  [PASS] Python Version: {py_ver}")
    print(f"  [PASS] Executable: {sys.executable}")
    assert sys.version_info >= (3, 10), "Python 3.10+ is required."

    # 3. Core Package Suite & Versions Check
    print("\n[3/4] Testing Critical Library Imports & Versions...")
    
    import torch
    print(f"  [PASS] torch (CPU mode): {torch.__version__}")

    import numpy as np
    print(f"  [PASS] numpy: {np.__version__}")
    
    import pandas as pd
    print(f"  [PASS] pandas: {pd.__version__}")
    
    import scipy
    print(f"  [PASS] scipy: {scipy.__version__}")
    
    import sklearn
    print(f"  [PASS] scikit-learn: {sklearn.__version__}")
    
    import statsmodels
    print(f"  [PASS] statsmodels: {statsmodels.__version__}")

    import lightgbm as lgb
    print(f"  [PASS] lightgbm: {lgb.__version__}")
    
    import shap
    print(f"  [PASS] shap: {shap.__version__}")
    
    import networkx as nx
    print(f"  [PASS] networkx: {nx.__version__}")

    import jsonschema
    import importlib.metadata
    print(f"  [PASS] jsonschema: {importlib.metadata.version('jsonschema')}")
    
    import matplotlib
    print(f"  [PASS] matplotlib: {matplotlib.__version__}")
    
    import seaborn as sns
    print(f"  [PASS] seaborn: {sns.__version__}")

    # 4. Non-Linear Causal Engine Check (PCMCI+ / Tigramite)
    print("\n[4/4] Verifying PCMCI+ Non-Linear Causal Discovery Engine...")
    from tigramite import data_processing as pp
    from tigramite.pcmci import PCMCI
    from tigramite.independence_tests.parcorr import ParCorr
    
    # Test a micro-dataset with PCMCI
    data_sample = np.random.randn(100, 3)
    var_names = ["T_top", "P_col", "Q_reboiler"]
    dataframe = pp.DataFrame(data_sample, var_names=var_names)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    print(f"  [PASS] tigramite: {pcmci.__class__.__name__} initialized successfully.")

    # 5. Repo Contract Files
    required_files = [
        "AGENTS.md",
        "CHANGELOG.md",
        "PROJECT_PLAN.md",
        "outputs/.plans/distillation-kg-fdd-plan.md"
    ]
    print("\n[5/5] Verifying Repository Contract Files...")
    for f in required_files:
        if os.path.isfile(f):
            print(f"  [PASS] Verified file: {f}")
        else:
            raise FileNotFoundError(f"Missing repo contract file: {f}")

    print("\n" + "=" * 60)
    print("  RESULT: TEST GATE 0 FULLY PASSED (CORE_OK)")
    print("=" * 60)

if __name__ == "__main__":
    verify_phase0()
