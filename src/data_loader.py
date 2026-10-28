"""
Data Ingestion & Benchmark Preprocessing Module
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phase 1)

Ingests and validates:
1. Industrial Petroleum Debutanizer Column Benchmark (Fortuna et al., 2007)
2. Harvard Dataverse Modernized Tennessee Eastman Process (Rieth et al., 2017)
3. PRONTO Multiphase Benchmark Facility (Stief et al., 2019, Cranfield/Imperial)
"""

import os
import io
import urllib.request
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any, List

RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PRONTO_RAW_DIR = os.path.join(RAW_DATA_DIR, "pronto")
DEBUTANIZER_URL = "https://raw.githubusercontent.com/sj823774188/Debutanizer-Column-Data/master/debutanizer_data.txt"

# Distillation-specific TEP variable indices (1-indexed matching engineering specs)
TEP_DISTILLATION_MEAS = [
    "xmeas_1", "xmeas_2", "xmeas_3", "xmeas_4", "xmeas_5", "xmeas_6",
    "xmeas_7", "xmeas_8", "xmeas_9", "xmeas_10", "xmeas_11", "xmeas_12",
    "xmeas_13", "xmeas_14", "xmeas_15", "xmeas_16", "xmeas_17", "xmeas_18",
    "xmeas_19", "xmeas_20", "xmeas_21", "xmeas_22", "xmeas_23", "xmeas_24"
]
TEP_DISTILLATION_MV = [
    "xmv_1", "xmv_2", "xmv_3", "xmv_4", "xmv_5", "xmv_6",
    "xmv_7", "xmv_8", "xmv_9", "xmv_10", "xmv_11"
]

DEBUTANIZER_COLUMNS = [
    "top_tray_temp",        # u1: Top tray temperature
    "top_pressure",         # u2: Top pressure
    "reflux_flow",          # u3: Reflux flow
    "distillate_flow",      # u4: Distillate flow
    "tray6_temp",           # u5: 6th tray temperature
    "reboiler_bottom_temp", # u6: Reboiler bottom temperature
    "reboiler_steam_flow",  # u7: Reboiler steam flow
    "butane_content"        # y: Bottom butane impurity (C4)
]

PRONTO_PROCESS_TAGS = [
    "FT305/OUT.CV", "FT302/OUT.CV", "FT305/AI2/OUT.CV", "PT312/OUT.CV",
    "FT102/OUT.CV", "FT104/OUT.CV", "FT102/AI3/OUT.CV", "FT102/AI2/OUT.CV",
    "PT417/OUT.CV", "PT408/OUT.CV", "PT403/OUT.CV", "FT404/OUT.CV", "FT406/OUT.CV",
    "PT501/OUT.CV", "PIC501/PID1/OUT.CV", "LI502/OUT.CV", "LI503/OUT.CV",
    "LVC502-SR/PID1/OUT.CV", "LI101/OUT.CV", "FIC302/PID1/OUT.CV",
    "FIC302/PID1/SP.CV", "FIC302/PID1/PV.CV", "FIC301/PID1/OUT.CV",
    "FIC301/PID1/SP.CV", "FIC301/PID1/PV.CV", "FIC102/PID1/OUT.CV",
    "FIC102/PID1/SP.CV", "FIC102/PID1/PV.CV", "FIC101/PID1/OUT.CV",
    "FIC101/PID1/SP.CV", "FIC101/PID1/PV.CV"
]

PRONTO_VALVE_COLUMNS = [
    "PIC501/PID1/OUT.CV", "LVC502-SR/PID1/OUT.CV", "FIC302/PID1/OUT.CV",
    "FIC301/PID1/OUT.CV", "FIC102/PID1/OUT.CV", "FIC101/PID1/OUT.CV"
]


def ensure_raw_dir():
    os.makedirs(RAW_DATA_DIR, exist_ok=True)


def download_debutanizer() -> str:
    """Downloads the industrial Debutanizer dataset if not present."""
    ensure_raw_dir()
    filepath = os.path.join(RAW_DATA_DIR, "debutanizer_data.txt")
    if not os.path.exists(filepath):
        print(f"Downloading Debutanizer column dataset from {DEBUTANIZER_URL}...")
        req = urllib.request.Request(DEBUTANIZER_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read().decode("utf-8")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Saved Debutanizer data to {filepath}")
    else:
        print(f"Debutanizer data already exists at {filepath}")
    return filepath


def load_debutanizer() -> pd.DataFrame:
    """Loads and formats the industrial Debutanizer dataset."""
    filepath = download_debutanizer()
    df = pd.read_csv(filepath, sep=r"\s+", header=None)
    if df.shape[1] == 8:
        df.columns = DEBUTANIZER_COLUMNS
    else:
        raise ValueError(f"Unexpected shape for Debutanizer data: {df.shape}")
    
    # Assert zero missing values
    assert df.isnull().sum().sum() == 0, "Debutanizer data contains unexpected NaNs!"
    # Check length
    assert len(df) == 2394, f"Expected 2394 rows, found {len(df)}"
    print(f"Debutanizer loaded successfully: shape {df.shape}")
    return df


def load_tep_benchmark(
    target_faults: List[int] = [0, 1, 2, 5, 6, 11, 15],
    distillation_only: bool = True
) -> pd.DataFrame:
    """
    Loads TEP benchmark runs (Rieth et al. 2017) filtered to target distillation faults.
    Fault 0: Fault-Free normal operation.
    Fault 1: Feed A/C composition step disturbance.
    Fault 2: Component B composition shift (VLE change).
    Fault 5: Condenser cooling water temperature step.
    Fault 6: Feed A loss (Column weeping risk).
    Fault 11: Reactor cooling water valve sticking.
    Fault 15: Condenser cooling water valve stiction.
    """
    ensure_raw_dir()
    ff_path = os.path.join(RAW_DATA_DIR, "tep_fault_free_train.parquet")
    faulty_path = os.path.join(RAW_DATA_DIR, "tep_faulty_train.parquet")

    if not os.path.exists(ff_path):
        raise FileNotFoundError(f"Missing {ff_path}. Please download fault-free training parquet.")
    if not os.path.exists(faulty_path):
        raise FileNotFoundError(f"Missing {faulty_path}. Please download faulty training parquet.")

    print(f"Reading fault-free runs from {ff_path}...")
    df_ff = pd.read_parquet(ff_path)
    # Ensure fault_id is integer
    df_ff["fault_id"] = df_ff["fault_id"].astype(int)

    print(f"Reading target faulty runs {target_faults} from {faulty_path}...")
    # Read faulty parquet filtered to target faults
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    
    table_faulty = pq.read_table(
        faulty_path,
        filters=[("fault_id", "in", [f for f in target_faults if f != 0])]
    )
    df_faulty = table_faulty.to_pandas()
    df_faulty["fault_id"] = df_faulty["fault_id"].astype(int)

    # Combine
    df_combined = pd.concat([df_ff, df_faulty], ignore_index=True)

    if distillation_only:
        cols_to_keep = ["fault_id", "simulation_id", "sample_id"] + TEP_DISTILLATION_MEAS + TEP_DISTILLATION_MV
        # Keep only columns that exist
        available_cols = [c for c in cols_to_keep if c in df_combined.columns]
        df_combined = df_combined[available_cols]

    print(f"TEP benchmark loaded: {df_combined.shape}, fault distribution: {df_combined['fault_id'].value_counts().to_dict()}")
    return df_combined


def load_pronto_benchmark() -> pd.DataFrame:
    """
    Loads and standardizes the Cranfield/Imperial PRONTO Multiphase Flow Facility Benchmark
    (Stief et al., 2019, Journal of Process Control).
    
    Fault Legend:
      Fault 0: Normal baseline (Test 9, 0912Testday4)
      Fault 1: Slugging / Multiphase oscillatory instability (Test 10, 0912Testday4)
      Fault 2: Air Blockage (Tests 2 & 3, 0907Testday2)
      Fault 3: Air Leakage (Test 4, 0907Testday2; Tests 5 & 6, 0911Testday3)
      Fault 4: Diverted Flow bypass (Tests 7 & 8, 0911Testday3)
    """
    ensure_raw_dir()
    os.makedirs(PRONTO_RAW_DIR, exist_ok=True)
    
    file_map = [
        ("0912Testday4.csv", 0, 0, 7200),        # Normal Test 9
        ("0912Testday4.csv", 1, 7200, 14400),    # Slugging Test 10
        ("0907Testday2.csv", 2, 0, 11000),       # Air Blockage Tests 2/3
        ("0907Testday2.csv", 3, 11000, 18600),   # Air Leakage Test 4
        ("0911Testday3.csv", 3, 0, 12600),       # Air Leakage Tests 5/6
        ("0911Testday3.csv", 4, 12600, 25200),   # Diverted Flow Tests 7/8
    ]
    
    dfs = []
    for fname, fid, s_idx, e_idx in file_map:
        fpath = os.path.join(PRONTO_RAW_DIR, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing PRONTO raw file: {fpath}.")
        
        tags = list(pd.read_csv(fpath, header=None, nrows=2).iloc[1])
        tags[0] = "TIMESTAMP"
        
        raw_df = pd.read_csv(fpath, skiprows=2, header=None, low_memory=False)
        raw_df.columns = tags[:raw_df.shape[1]]
        
        segment = raw_df.iloc[s_idx:min(e_idx, len(raw_df))].copy()
        segment["fault_id"] = fid
        segment["run_id"] = f"{fname}_{fid}_{s_idx}"
        dfs.append(segment)
        
    combined = pd.concat(dfs, ignore_index=True)
    
    for col in PRONTO_PROCESS_TAGS:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")
    
    combined[PRONTO_PROCESS_TAGS] = combined[PRONTO_PROCESS_TAGS].ffill().bfill()
    assert combined[PRONTO_PROCESS_TAGS].isna().sum().sum() == 0, "PRONTO contains unexpected NaNs!"
    
    print(f"PRONTO benchmark loaded: {combined.shape}, fault distribution: {combined['fault_id'].value_counts().to_dict()}")
    return combined


if __name__ == "__main__":
    deb_df = load_debutanizer()
    print("Debutanizer summary statistics:")
    print(deb_df.describe().T[["mean", "std", "min", "max"]])
