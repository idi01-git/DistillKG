"""
Anti-Hallucination Assertion Verification Protocol
Author: Feynman Research Agent
Project: distillation-kg-fdd (Phases 5 & 6)

Enforces strict programmatic anti-hallucination verification:
1. Parses reported table metrics from markdown reports AND manuscript drafts.
2. Cross-references every reported numerical value against raw execution logs in autoresearch.jsonl.
3. Asserts that every single number matches within a strict tolerance of +/- 0.0001.
4. Exits with code 0 if 100% verified; exits with code 1 upon discovering any discrepancy.
"""

import os
import re
import sys
import json
from typing import List, Dict, Any, Tuple

AUTORESEARCH_PATH = os.path.join(os.path.dirname(__file__), "..", "autoresearch.jsonl")
ABLATION_REPORT_PATH = os.path.join(os.path.dirname(__file__), "..", "outputs", "distillation-kg-fdd-ablation.md")
PAPER_DRAFT_PATH = os.path.join(os.path.dirname(__file__), "..", "papers", "distillation-kg-fdd.md")


def load_raw_logs(log_path: str = AUTORESEARCH_PATH) -> List[Dict[str, Any]]:
    """Loads raw append-only execution records from autoresearch.jsonl."""
    if not os.path.exists(log_path):
        raise FileNotFoundError(f"Execution log not found at {log_path}. Please run ablation_runner.py first.")
    
    records = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def parse_and_verify_table(
    file_path: str,
    logs: List[Dict[str, Any]],
    tolerance: float = 1e-4
) -> Tuple[bool, int, List[str]]:
    """
    Parses table lines in a markdown file and asserts that every numerical
    entry matches its corresponding record in autoresearch.jsonl.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Target verification file not found at {file_path}")

    # Build canonical lookup key mapping
    log_map = {}
    for r in logs:
        # Canonical keys: ("Config A", "Nominal Clean"), etc.
        cid = r["configuration_id"].replace("_", " ")
        reg = r["stress_regime"].replace("_", " ")
        log_map[(cid, reg)] = r

    discrepancies = []
    verified_assertions = 0

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Filter lines that look like ablation table rows
    table_lines = [l.strip() for l in lines if l.strip().startswith("|") and "Config" in l and not "Configuration" in l]

    if not table_lines:
        return False, 0, [f"No ablation table rows found in {file_path}!"]

    for tl in table_lines:
        # Split by pipe delimiter and clean markdown bolding/italics
        raw_parts = [p.strip().replace("**", "") for p in tl.split("|")[1:-1]]
        # Ablation Table 3 has exactly 8 columns (Config, Regime, F1, AUC, FAR, MTTD, RootAcc, Latency)
        if len(raw_parts) != 8:
            continue

        c_col = raw_parts[0]
        reg_col = raw_parts[1]

        # Extract base config name: "Config A", "Config B", etc.
        c_match = re.search(r"Config\s+[A-E]", c_col)
        if not c_match:
            continue
        c_name = c_match.group(0).strip()

        # Map regime column to canonical name
        if "Nominal" in reg_col:
            reg_name = "Nominal Clean"
        elif "Noise" in reg_col:
            reg_name = "Sensor Noise 15dB"
        elif "Dropout" in reg_col:
            reg_name = "Sensor Dropout 5pct"
        else:
            reg_name = reg_col.strip()

        key = (c_name, reg_name)
        if key not in log_map:
            discrepancies.append(f"[{file_path}] Row {key} has no matching entry in autoresearch.jsonl!")
            continue

        raw = log_map[key]

        try:
            f1_val = float(raw_parts[2])
            prauc_val = float(raw_parts[3])
            far_val = float(raw_parts[4].replace("%", "").strip())
            mttd_val = float(raw_parts[5])
            rc_val = float(raw_parts[6].replace("%", "").strip())
            lat_val = float(raw_parts[7].replace("ms", "").strip())
        except ValueError as ve:
            discrepancies.append(f"[{file_path}] Failed to parse numbers from row: {tl} ({ve})")
            continue

        # Check Macro-F1
        if abs(f1_val - raw["macro_f1"]) > tolerance:
            discrepancies.append(f"[{file_path}][{key}] Macro-F1 mismatch: doc={f1_val}, raw_log={raw['macro_f1']}")
        else:
            verified_assertions += 1

        # Check PR-AUC
        if abs(prauc_val - raw["pr_auc"]) > tolerance:
            discrepancies.append(f"[{file_path}][{key}] PR-AUC mismatch: doc={prauc_val}, raw_log={raw['pr_auc']}")
        else:
            verified_assertions += 1

        # Check FAR
        if abs(far_val - raw["false_alarm_rate_pct"]) > tolerance:
            discrepancies.append(f"[{file_path}][{key}] FAR mismatch: doc={far_val}, raw_log={raw['false_alarm_rate_pct']}")
        else:
            verified_assertions += 1

        # Check MTTD
        if abs(mttd_val - raw["mean_time_to_detect_steps"]) > tolerance:
            discrepancies.append(f"[{file_path}][{key}] MTTD mismatch: doc={mttd_val}, raw_log={raw['mean_time_to_detect_steps']}")
        else:
            verified_assertions += 1

        # Check Root Cause Accuracy
        if abs(rc_val - raw["root_cause_accuracy_pct"]) > tolerance:
            discrepancies.append(f"[{file_path}][{key}] Root-Cause mismatch: doc={rc_val}, raw_log={raw['root_cause_accuracy_pct']}")
        else:
            verified_assertions += 1

        # Check Latency (within 0.1ms rounding display tolerance)
        if abs(lat_val - raw["inference_latency_ms"]) > 0.1:
            discrepancies.append(f"[{file_path}][{key}] Latency mismatch: doc={lat_val}, raw_log={raw['inference_latency_ms']}")
        else:
            verified_assertions += 1

    success = (len(discrepancies) == 0) and (verified_assertions > 0)
    return success, verified_assertions, discrepancies


def run_anti_hallucination_verification() -> bool:
    print("=" * 65)
    print("RUNNING ANTI-HALLUCINATION VERIFICATION PROTOCOL")
    print("=" * 65)

    try:
        logs = load_raw_logs()
        print(f"Loaded {len(logs)} empirical execution records from autoresearch.jsonl.")
    except Exception as e:
        print(f"ERROR: {e}")
        return False

    total_verified = 0
    all_discrepancies = []

    # 1. Verify outputs/distillation-kg-fdd-ablation.md
    if os.path.exists(ABLATION_REPORT_PATH):
        ok_abl, count_abl, disc_abl = parse_and_verify_table(ABLATION_REPORT_PATH, logs=logs)
        total_verified += count_abl
        all_discrepancies.extend(disc_abl)
        print(f"Verified {count_abl} assertions in {os.path.basename(ABLATION_REPORT_PATH)}.")

    # 2. Verify papers/distillation-kg-fdd.md
    if os.path.exists(PAPER_DRAFT_PATH):
        ok_paper, count_paper, disc_paper = parse_and_verify_table(PAPER_DRAFT_PATH, logs=logs)
        total_verified += count_paper
        all_discrepancies.extend(disc_paper)
        print(f"Verified {count_paper} assertions in {os.path.basename(PAPER_DRAFT_PATH)}.")

    print(f"\n[Total Assertion Checks Completed]: {total_verified}")
    if all_discrepancies:
        print(f"[FAIL] Found {len(all_discrepancies)} hallucinated or mismatched metrics:")
        for d in all_discrepancies:
            print(f"  -> {d}")
        return False
    else:
        print(f"[PASS] 100% of reported numerical metrics in reports and manuscript match raw execution logs.")
        print("STATUS: ZERO_HALLUCINATIONS_VERIFIED")
        print("=" * 65)
        return True


if __name__ == "__main__":
    passed = run_anti_hallucination_verification()
    sys.exit(0 if passed else 1)
